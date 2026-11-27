"""Plan with latent options and a budget-conditioned value in Maze2D.

A frozen pointmaze-latent run supplies the world model. This runner trains a latent-goal
GCIVL executor on frozen encodings of archive states and an option value on hindsight
options from the same archive, or reuses both from an earlier run. The archive never
pursued the terminal goal on purpose, so the offline value misjudges options the
executor would pursue. Online rounds therefore run the planner on training resets,
pursuing a random candidate at some decisions, and continue fitting the value on the
executor's attempts mixed with the hindsight options.

Each arm then plays the development resets in closed loop. Nothing defines
intermediate goals or arrival: the planner picks among sampled latent futures and the
encoded terminal goal, and the executor pursues the choice until the planner switches
or samples again. The offline and corrected values are each evaluated with routes
proposed under the remaining budget only and under shorter budgets as well.

Direct tracking of the terminal goal with Diffuser's position and velocity tracker is
the floor and a privileged shortest-path oracle is the ceiling, as in E1. Direct
pursuit of the encoded goal with the latent executor isolates the executor.
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import StateArchive
from worth_reaching.evaluation import options as shared
from worth_reaching.evaluation.closed_loop import (
    ORACLE,
    ClosedLoop,
    Controls,
    RegionModels,
)
from worth_reaching.evaluation.options import (
    OFFLINE,
    ONLINE,
    Arm,
    OptionRunConfig,
    option_arms,
)
from worth_reaching.models.gcivl import GCIVL, GCIVLConfig, fit_gcivl
from worth_reaching.models.option_value import (
    NEVER,
    OptionValue,
    fit_option_value,
    label_options,
    value_range,
)
from worth_reaching.models.training import (
    describe,
    load_weights,
    restore_weights,
    save_weights,
    stream_seed,
    weights,
)
from worth_reaching.planning.options import (
    OptionPlanner,
    surprise_threshold,
)
from worth_reaching.tasks.pointmaze.dataset import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    load_dataset,
    split_trajectories,
)
from worth_reaching.tasks.pointmaze.energy import FrozenRun
from worth_reaching.tasks.pointmaze.environment import (
    GOAL_RADIUS,
    SEED_BASE,
    PointMaze,
)
from worth_reaching.tasks.pointmaze.experiment import rollout, track
from worth_reaching.tasks.pointmaze.geometry import MazeGeometry
from worth_reaching.tasks.pointmaze.spec import TASKS

SCHEMA = "pointmaze_options_v2"
WEIGHTS_FILE = "option-weights.pt"
TRAIN_SEED_BASE = 9001
"""Training resets for online rounds, disjoint from the development and test seeds."""


@dataclass(frozen=True, slots=True)
class MazeOptionConfig(OptionRunConfig):
    executor: GCIVLConfig = field(default_factory=lambda: GCIVLConfig(updates=300_000))
    oracle_cells: int = 1
    oracle_speed: float = 2.0

    def __post_init__(self) -> None:
        OptionRunConfig.__post_init__(self)
        if (
            self.oracle_cells < 1
            or self.oracle_speed <= 0
            or self.online.rounds * self.online.episodes
            > SEED_BASE["dev"] - TRAIN_SEED_BASE
        ):
            raise ValueError(
                "Use a positive oracle and keep training resets below the "
                "development seeds"
            )

    @property
    def recipe(self) -> dict:
        """The settings that determine the trained executor and values."""
        return {
            "executor": json.loads(json.dumps(asdict(self.executor))),
            **OptionRunConfig.recipe.fget(self),
        }


def reached(states: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """PointMaze's sparse success predicate on public positions."""
    return np.linalg.norm(states[..., :2] - goals[..., :2], axis=-1) <= GOAL_RADIUS


def arms(config: MazeOptionConfig) -> dict[str, Arm]:
    return option_arms(config, (shared.DIRECT, ORACLE))


def _archive(trajectories, fingerprint: str) -> StateArchive:
    finals = np.zeros(len(trajectories.observations), dtype=bool)
    finals[trajectories.ends] = True
    return StateArchive(
        trajectories.observations,
        trajectories.actions,
        trajectories.ends,
        np.flatnonzero(~finals),
        fingerprint,
    )


def _label(model, train, limit, config):
    return label_options(
        train,
        reached,
        config.value.options,
        max_budget=limit,
        max_horizon=model.config.max_horizon,
        queries=model.config.queries,
        config=config.value,
        rng=np.random.default_rng([model.config.seed, 7100]),
    )


def _train(frozen, train, latents, options, limit, config, out, fingerprint):
    seed = frozen.model.config.seed
    executor = fit_gcivl(
        _archive(train, fingerprint),
        ACTION_SIZE,
        config.executor,
        seed=seed,
        out=out / "checkpoints" / "executor",
        goal_features=latents,
    )
    value = fit_option_value(
        options,
        latents,
        limit,
        config.value,
        seed=seed,
        out=out / "checkpoints" / "value",
    )
    return executor, value


def _load(trained: Path, frozen: FrozenRun, limit: int, config: MazeOptionConfig):
    manifest = json.loads((trained / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected a pointmaze-options run in {trained}")
    if manifest["recipe"] != config.recipe:
        raise ValueError(f"The executor or value recipe differs from {trained}")
    if manifest["source_weights_sha256"] != frozen.weights_sha256:
        raise ValueError(f"{trained} was trained on a different world model")
    values = load_weights(trained / WEIGHTS_FILE)
    latent_size = frozen.model.config.latent_size
    executor = GCIVL(OBSERVATION_SIZE, latent_size, ACTION_SIZE, config.executor)
    restore_weights(executor, values["executor"])
    restored = {}
    for name in (OFFLINE, ONLINE):
        restored[name] = restore_weights(
            OptionValue(latent_size, limit, config.value), values[name]
        )
    return executor, restored, manifest.get("rounds", [])


_WORKER: dict = {}


def _start_worker(
    data_dir, task_name, completion, executor, values, arm_configs, config, limit
):
    torch.set_num_threads(1)
    task = TASKS[task_name]
    dataset, _ = load_dataset(data_dir, task.dataset_id)
    native = dataset.recover_environment(eval_env=True)
    env, geometry = PointMaze(native, task), MazeGeometry(native.unwrapped.maze)
    _WORKER.update(
        env=env,
        completion=completion,
        executor=executor,
        values=values,
        arms=arm_configs,
        limit=limit,
        oracle=Controls(
            act=track,
            act_latent=track,
            terminal=track,
            arrived=env.arrived,
            oracle=lambda observation, goal: geometry.waypoint(
                observation, goal, config.oracle_cells, config.oracle_speed
            ),
        ),
    )


def _policy(arm: str, seed: int):
    worker = _WORKER
    value, setting = worker["arms"][arm]
    completion, limit = worker["completion"], worker["limit"]
    key = stream_seed(completion.config.seed, seed, list(worker["arms"]).index(arm))
    if setting == shared.DIRECT:
        return track
    if setting == ORACLE:
        return ClosedLoop(
            RegionModels(completion, None),
            worker["oracle"],
            ORACLE,
            "forecast",
            key,
            limit,
            stages=1,
            region_samples=1,
            action_size=ACTION_SIZE,
        )
    return OptionPlanner(
        completion,
        worker["values"][value],
        worker["executor"].act,
        key,
        limit,
        setting,
        action_size=ACTION_SIZE,
    )


def _play(batch: list[tuple[str, int]], collect: bool = False):
    rows, collected = [], []
    env, limit = _WORKER["env"], _WORKER["limit"]
    for arm, seed in batch:
        policy = _policy(arm, seed)
        outcome, trajectory = rollout(env, seed, limit, policy)
        first = outcome["first_success_step"]
        rows.append(
            shared.record(
                arm, seed, None, policy, outcome["success"], outcome["steps"], first
            )
        )
        if collect:
            goal = np.asarray(outcome["goal"], dtype=np.float32)
            attempt = shared.attempts(policy, trajectory, goal, first, limit)
            if attempt is not None:
                collected.append(attempt)
    return rows, collected


def _pool(
    workers,
    data_dir,
    task_name,
    completion,
    executor,
    values,
    arm_configs,
    config,
    limit,
):
    return ProcessPoolExecutor(
        workers,
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(
            data_dir,
            task_name,
            completion,
            executor,
            values,
            arm_configs,
            config,
            limit,
        ),
    )


def _correct(
    frozen,
    task_name,
    data_dir,
    executor,
    offline,
    options,
    latents,
    limit,
    config,
    surprise,
    workers,
    out,
):
    """Collect the executor's attempts on training resets and correct the value."""
    episodes = config.online.episodes
    arm_configs = {"collect": (ONLINE, replace(config.collection, surprise=surprise))}

    def collect(value, index):
        seeds = range(
            TRAIN_SEED_BASE + index * episodes, TRAIN_SEED_BASE + (index + 1) * episodes
        )
        planned = [("collect", seed) for seed in seeds]
        shards = [planned[i::workers] for i in range(workers)]
        with _pool(
            len(shards),
            data_dir,
            task_name,
            frozen.model,
            executor,
            {ONLINE: value},
            arm_configs,
            config,
            limit,
        ) as pool:
            results = list(pool.map(partial(_play, collect=True), shards))
        return (
            [row for shard, _ in results for row in shard],
            [attempt for _, shard in results for attempt in shard],
        )

    return shared.correct(
        collect,
        offline,
        options,
        latents,
        limit,
        config,
        seed=lambda index: stream_seed(frozen.model.config.seed, 7300, index),
        out=out,
    )


def run(
    data_dir: Path,
    run_path: Path,
    out: Path,
    config: MazeOptionConfig,
    *,
    split: str,
    repeats: int,
    workers: int = 1,
    trained: Path | None = None,
) -> dict:
    frozen = FrozenRun.load(run_path)
    manifest = frozen.manifest
    limit = manifest["evaluation"]["action_limit"]
    if not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"] or workers < 1:
        raise ValueError("Use a positive number of reset seeds and workers")
    task = TASKS[manifest["task"]]
    dataset, hashes = load_dataset(data_dir, task.dataset_id)
    if hashes != manifest["dataset_files_sha256"]:
        raise ValueError("Dataset files disagree with the frozen run")
    train, validation, _ = split_trajectories(dataset)
    out.mkdir(parents=True, exist_ok=False)
    model = frozen.model
    windows = validation.sample_windows(
        np.random.default_rng(801),
        512,
        history=model.config.history,
        queries=model.config.queries,
        max_horizon=model.config.max_horizon,
        max_action_budget=model.config.max_action_budget,
    )
    surprise = surprise_threshold(
        model, windows, stream_seed(model.config.seed, 7200), config.surprise_quantile
    )
    if trained is None:
        latents = model.encode_observations(train.observations)
        options = _label(model, train, limit, config)
        executor, offline = _train(
            frozen,
            train,
            latents,
            options,
            limit,
            config,
            out,
            hashes["main_data.hdf5"],
        )
        online, rounds = _correct(
            frozen,
            manifest["task"],
            data_dir,
            executor,
            offline,
            options,
            latents,
            limit,
            config,
            surprise,
            workers,
            out,
        )
        values = {OFFLINE: offline, ONLINE: online}
        save_weights(
            out / WEIGHTS_FILE,
            {"executor": weights(executor)}
            | {name: weights(value) for name, value in values.items()},
        )
        weights_path = out / WEIGHTS_FILE
        labeled = {
            "labeled_options": int(len(options.first)),
            "labeled_goal_reached_share": float((options.first < NEVER).mean()),
            "offline_value_range": value_range(
                offline.eval(), options, latents, config.eval_costs[-1]
            ),
        }
    else:
        executor, values, rounds = _load(trained, frozen, limit, config)
        weights_path = trained / WEIGHTS_FILE
        labeled = json.loads((trained / "manifest.json").read_text())["labels"]
    recipe = config.recipe
    config = replace(config, planner=replace(config.planner, surprise=surprise))
    arm_configs = arms(config)
    seeds = list(range(SEED_BASE[split], SEED_BASE[split] + repeats))
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "task": manifest["task"],
            "dataset": task.dataset_id,
            "dataset_files_sha256": hashes,
            "versions": {
                name: version(name)
                for name in (
                    "minari",
                    "gymnasium-robotics",
                    "gymnasium",
                    "mujoco",
                    "torch",
                )
            },
            "source_run": str(frozen.directory),
            "source_weights_sha256": frozen.weights_sha256,
            "trained_run": str(trained) if trained else None,
            "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
            "recipe": recipe,
            "labels": labeled,
            "rounds": rounds,
            "training_resets": [
                TRAIN_SEED_BASE,
                TRAIN_SEED_BASE + config.online.rounds * config.online.episodes,
            ],
            "planner": asdict(config.planner),
            "surprise_threshold": surprise,
            "action_limit": limit,
            "split": split,
            "seeds": seeds,
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
    planned = [(arm, seed) for seed in seeds for arm in arm_configs]
    shards = [planned[index::workers] for index in range(workers)]
    with _pool(
        len(shards),
        data_dir,
        manifest["task"],
        model,
        executor,
        values,
        arm_configs,
        config,
        limit,
    ) as pool:
        rows = [row for shard, _ in pool.map(_play, shards) for row in shard]
    rows.sort(key=lambda row: (row["arm"], row["seed"]))
    with gzip.open(out / "attempts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "labels": labeled,
        "rounds": rounds,
        "surprise_threshold": surprise,
        "evaluation_environment_steps": sum(row["steps"] for row in rows),
        **shared.summarize(rows),
    }
    write_json(out / "summary.json", summary)
    return summary
