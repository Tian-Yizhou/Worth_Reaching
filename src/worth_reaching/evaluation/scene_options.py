"""Plan with latent options and a budget-conditioned value in Scene.

A `scene-regions` run supplies the world model and the latent-goal GCIVL executor, and
its `scene-executor` run supplies the observation-goal executor for direct execution.
This runner fits the option value on hindsight options from the Scene archive, whose
terminal success is the official scene predicate, or reuses it from an earlier run.
Online rounds run the planner on training resets of the evaluated tasks, with seeds
disjoint from the development and test resets, and continue fitting the value on the
executor's attempts. The environment steps they use are reported with each round.

Each arm then plays the official development resets in closed loop. Scene has no
privileged path planner, so direct execution is the only bound.
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import load_state_archive
from worth_reaching.evaluation import options as shared
from worth_reaching.evaluation.options import (
    OFFLINE,
    ONLINE,
    OptionRunConfig,
    option_arms,
)
from worth_reaching.evaluation.scene_reachability import scene_success
from worth_reaching.evaluation.scene_regions import load_executor
from worth_reaching.models.continuous_completion import (
    CompletionConfig,
    LearnedCompletion,
    PredictorStopping,
)
from worth_reaching.models.gcivl import GCIVL, GCIVLConfig
from worth_reaching.models.option_value import (
    NEVER,
    OptionValue,
    fit_option_value,
    label_options,
    value_range,
)
from worth_reaching.models.training import (
    WEIGHTS_FILE,
    describe,
    load_weights,
    restore_weights,
    save_weights,
    stream_seed,
    weights,
)
from worth_reaching.planning.options import OptionPlanner, surprise_threshold
from worth_reaching.tasks.ogbench_manipulation.environment import Manipulation
from worth_reaching.tasks.ogbench_manipulation.spec import (
    ACTION_SIZE,
    SCENE,
    SEED_OFFSETS,
    TASKS,
)

SCHEMA = "scene_options_v1"
OPTION_WEIGHTS = "option-weights.pt"
TRAIN_SEED_OFFSET = 20_000
"""Training resets start here, between the development and test seed ranges."""


@dataclass(frozen=True, slots=True)
class SceneOptionConfig(OptionRunConfig):
    def __post_init__(self) -> None:
        OptionRunConfig.__post_init__(self)
        if self.online.rounds * self.online.episodes > (
            SEED_OFFSETS["test"] - TRAIN_SEED_OFFSET
        ):
            raise ValueError("Keep training resets below the test seeds")


def load_regions(run: Path) -> tuple[LearnedCompletion, GCIVL, dict]:
    """The world model and latent-goal executor of a scene-regions run."""
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema") != "scene_regions_v1":
        raise ValueError(f"Expected a scene-regions run in {run}")
    saved = manifest["regions"]
    settings = dict(saved["completion"])
    if settings.get("predictor_stopping") is not None:
        settings["predictor_stopping"] = PredictorStopping(
            **settings["predictor_stopping"]
        )
    values = load_weights(run / WEIGHTS_FILE)
    completion = LearnedCompletion.from_weights(CompletionConfig(**settings), values)
    executor_config = GCIVLConfig(**saved["latent_executor"])
    executor = GCIVL(
        SCENE.observation_size,
        completion.config.latent_size,
        ACTION_SIZE,
        executor_config,
    )
    restore_weights(executor, values["latent_executor"])
    return completion, executor, manifest


def arms(config: SceneOptionConfig) -> dict[str, shared.Arm]:
    return option_arms(config, (shared.DIRECT,))


def _load(trained: Path, regions_sha: str, latent_size: int, config: SceneOptionConfig):
    manifest = json.loads((trained / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected a scene-options run in {trained}")
    if manifest["recipe"] != config.recipe:
        raise ValueError(f"The value recipe differs from {trained}")
    if manifest["regions_weights_sha256"] != regions_sha:
        raise ValueError(f"{trained} was trained on a different scene-regions run")
    values = load_weights(trained / OPTION_WEIGHTS)
    restored = {
        name: restore_weights(
            OptionValue(latent_size, SCENE.action_limit, config.value), values[name]
        )
        for name in (OFFLINE, ONLINE)
    }
    return restored, manifest["rounds"], manifest["labels"]


_WORKER: dict = {}


def _start_worker(completion, executor, latent_executor, values, arm_configs, limit):
    torch.set_num_threads(1)
    _WORKER.update(
        env=Manipulation(SCENE),
        completion=completion,
        executor=executor,
        latent_executor=latent_executor,
        values=values,
        arms=arm_configs,
        limit=limit,
    )


def _policy(arm: str, task: int, seed: int):
    worker = _WORKER
    value, setting = worker["arms"][arm]
    if setting == shared.DIRECT:
        return worker["executor"].act
    completion = worker["completion"]
    return OptionPlanner(
        completion,
        worker["values"][value],
        worker["latent_executor"].act,
        stream_seed(
            completion.config.seed, task, seed, list(worker["arms"]).index(arm)
        ),
        worker["limit"],
        setting,
        action_size=ACTION_SIZE,
    )


def _play(batch: list[tuple[str, int, int]], collect: bool = False):
    env, limit = _WORKER["env"], _WORKER["limit"]
    rows, collected = [], []
    for arm, task, seed in batch:
        start = env.capture(task, seed)
        policy = _policy(arm, task, seed)
        observation, trajectory = start.observation, [start.observation]
        success, step = False, 0
        for step in range(1, limit + 1):
            observation, success, done = env.step(policy(observation, start.goal))
            trajectory.append(observation)
            if success or done:
                break
        first = step if success else None
        rows.append(shared.record(arm, seed, task, policy, success, step, first))
        if collect:
            attempt = shared.attempts(
                policy, np.stack(trajectory), start.goal, first, limit
            )
            if attempt is not None:
                collected.append(attempt)
    return rows, collected


def _pool(workers, completion, executor, latent_executor, values, arm_configs, limit):
    return ProcessPoolExecutor(
        workers,
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(completion, executor, latent_executor, values, arm_configs, limit),
    )


def _shards(planned: list, workers: int) -> list[list]:
    return [planned[index::workers] for index in range(min(workers, len(planned)))]


def run(
    data_dir: Path,
    regions_run: Path,
    out: Path,
    config: SceneOptionConfig,
    *,
    split: str,
    tasks: tuple[int, ...],
    repeats: int,
    workers: int = 1,
    trained: Path | None = None,
) -> dict:
    if (
        not 0 < repeats <= TRAIN_SEED_OFFSET
        or workers < 1
        or not tasks
        or not set(tasks) <= set(TASKS)
    ):
        raise ValueError("Use positive repeats, workers and official Scene tasks")
    train, validation = (
        load_state_archive(
            data_dir / f"{SCENE.dataset}{suffix}.npz",
            observation_size=SCENE.observation_size,
            action_size=ACTION_SIZE,
        )
        for suffix in ("", "-val")
    )
    completion, latent_executor, regions = load_regions(regions_run)
    if regions["train_sha256"] != train.fingerprint:
        raise ValueError("The scene-regions run used a different Scene archive")
    executor_run = Path(regions["executor_run"])
    executor, _ = load_executor(executor_run)
    regions_sha = sha256((regions_run / WEIGHTS_FILE).read_bytes()).hexdigest()
    limit, model = SCENE.action_limit, completion.config
    out.mkdir(parents=True, exist_ok=False)
    windows = validation.sample_windows(
        np.random.default_rng(801),
        512,
        history=model.history,
        queries=model.queries,
        max_horizon=model.max_horizon,
    )
    surprise = surprise_threshold(
        completion, windows, stream_seed(model.seed, 7200), config.surprise_quantile
    )
    if trained is None:
        latents = completion.encode_observations(train.observations)
        options = label_options(
            train,
            scene_success,
            config.value.options,
            max_budget=limit,
            max_horizon=model.max_horizon,
            queries=model.queries,
            config=config.value,
            rng=np.random.default_rng([model.seed, 7100]),
        )
        offline = fit_option_value(
            options,
            latents,
            limit,
            config.value,
            seed=model.seed,
            out=out / "checkpoints" / "value",
        )
        collection = {
            "collect": (ONLINE, replace(config.collection, surprise=surprise))
        }
        episodes = config.online.episodes
        train_base = SCENE.seed_base + TRAIN_SEED_OFFSET

        def collect(value, index):
            planned = [
                ("collect", tasks[episode % len(tasks)], train_base + seed)
                for episode, seed in enumerate(
                    range(index * episodes, (index + 1) * episodes)
                )
            ]
            shards = _shards(planned, workers)
            with _pool(
                len(shards),
                completion,
                executor,
                latent_executor,
                {ONLINE: value},
                collection,
                limit,
            ) as pool:
                results = list(pool.map(partial(_play, collect=True), shards))
            return (
                [row for shard, _ in results for row in shard],
                [attempt for _, shard in results for attempt in shard],
            )

        online, rounds = shared.correct(
            collect,
            offline,
            options,
            latents,
            limit,
            config,
            seed=lambda index: stream_seed(model.seed, 7300, index),
            out=out,
        )
        values = {OFFLINE: offline.eval(), ONLINE: online}
        save_weights(
            out / OPTION_WEIGHTS,
            {name: weights(value) for name, value in values.items()},
        )
        weights_path = out / OPTION_WEIGHTS
        labels = {
            "labeled_options": int(len(options.first)),
            "labeled_goal_reached_share": float((options.first < NEVER).mean()),
            "offline_value_range": value_range(
                values[OFFLINE], options, latents, config.eval_costs[-1]
            ),
        }
    else:
        values, rounds, labels = _load(trained, regions_sha, model.latent_size, config)
        weights_path = trained / OPTION_WEIGHTS
    recipe = config.recipe
    config = replace(config, planner=replace(config.planner, surprise=surprise))
    arm_configs = arms(config)
    start_seed = SCENE.seed_base + SEED_OFFSETS[split]
    seeds = list(range(start_seed, start_seed + repeats))
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "environment": SCENE.environment,
            "train_sha256": train.fingerprint,
            "versions": {
                name: version(name) for name in ("ogbench", "mujoco", "torch")
            },
            "regions_run": str(regions_run),
            "regions_weights_sha256": regions_sha,
            "executor_run": str(executor_run),
            "trained_run": str(trained) if trained else None,
            "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
            "recipe": recipe,
            "labels": labels,
            "rounds": rounds,
            "training_resets": {
                "tasks": list(tasks),
                "seeds": [
                    SCENE.seed_base + TRAIN_SEED_OFFSET,
                    SCENE.seed_base
                    + TRAIN_SEED_OFFSET
                    + config.online.rounds * config.online.episodes,
                ],
            },
            "planner": asdict(config.planner),
            "surprise_threshold": surprise,
            "evaluation": {
                "split": split,
                "tasks": list(tasks),
                "seeds": seeds,
                "action_limit": limit,
            },
            "arms": {
                name: {
                    "value": value,
                    "planner": setting if isinstance(setting, str) else asdict(setting),
                }
                for name, (value, setting) in arm_configs.items()
            },
            "compute": {**describe(), "eval_workers": workers},
            "code_revision": code_revision(),
        },
    )
    planned = [
        (arm, task, seed) for task in tasks for seed in seeds for arm in arm_configs
    ]
    shards = _shards(planned, workers)
    with _pool(
        len(shards), completion, executor, latent_executor, values, arm_configs, limit
    ) as pool:
        rows = [row for shard, _ in pool.map(_play, shards) for row in shard]
    rows.sort(key=lambda row: (row["task"], row["arm"], row["seed"]))
    with gzip.open(out / "attempts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "labels": labels,
        "rounds": rounds,
        "surprise_threshold": surprise,
        "evaluation_environment_steps": sum(row["steps"] for row in rows),
        **shared.summarize(rows),
        "per_task": {
            str(task): shared.summarize([row for row in rows if row["task"] == task])
            for task in tasks
        },
    }
    write_json(out / "summary.json", summary)
    return summary
