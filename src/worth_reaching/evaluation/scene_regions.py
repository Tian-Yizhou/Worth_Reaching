"""Compare region goals with decoded point goals in closed-loop Scene control (E1).

A frozen `scene-executor` run supplies the GCIVL executor, which pursues observation
targets and the terminal goal. This runner trains a latent completion model, a region
decoder and a latent-goal GCIVL whose goals are frozen encodings of archive
observations. It calibrates the arrival rules on validation windows and runs every arm
in closed loop from the official seeded resets, with direct execution as the floor.
Scene has no privileged path planner, so there is no oracle ceiling.

Native arrival at a declared target applies the official scene predicate and a 4 cm
end-effector match after projecting the target's button states onto one-hot values.
A target makes progress when more goal components (cube, drawer, window and the two
buttons) match it than match the current observation.
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import load_state_archive
from worth_reaching.evaluation import regions as shared
from worth_reaching.evaluation.closed_loop import (
    ClosedLoop,
    Controls,
    Episode,
    RegionModels,
    episode_key,
    episodes,
    record,
    summarize,
)
from worth_reaching.evaluation.scene_reachability import (
    arrived,
    matched_parts,
    snap_buttons,
)
from worth_reaching.models.continuous_completion import (
    CompletionConfig,
    LearnedCompletion,
    fit,
)
from worth_reaching.models.gcivl import GCIVL, GCIVLConfig, fit_gcivl
from worth_reaching.models.region import RegionConfig, RegionDecoder
from worth_reaching.models.training import (
    WEIGHTS_FILE,
    describe,
    load_weights,
    restore_weights,
    save_weights,
    weights,
)
from worth_reaching.tasks.ogbench_manipulation.environment import Manipulation
from worth_reaching.tasks.ogbench_manipulation.spec import (
    ACTION_SIZE,
    SCENE,
    SEED_OFFSETS,
)

SCHEMA = "scene_regions_v1"


@dataclass(frozen=True, slots=True)
class SceneRegionConfig:
    completion: CompletionConfig = field(
        default_factory=lambda: CompletionConfig(max_horizon=SCENE.action_limit)
    )
    region: RegionConfig = field(default_factory=RegionConfig)
    latent_executor: GCIVLConfig = field(default_factory=GCIVLConfig)
    recall_levels: tuple[float, ...] = (0.5, 0.75, 0.9)
    region_samples: int = 32
    calibration_windows: int = 512
    decoder_windows: int = 20000
    """Training windows whose queries train the region decoder."""
    bandwidth: float = 0.25
    stages: int = 1
    sample_repeats: int = 2

    def __post_init__(self) -> None:
        if (
            not self.recall_levels
            or not all(0 < level < 1 for level in self.recall_levels)
            or min(self.region_samples, self.calibration_windows) < 2
            or min(self.stages, self.sample_repeats, self.decoder_windows) < 1
            or self.stages > self.completion.queries
            or self.bandwidth <= 0
        ):
            raise ValueError(
                "Use recall levels in (0, 1), at least two samples and calibration "
                "windows, positive counts within the query count and a positive bandwidth"
            )


def load_executor(run: Path) -> tuple[GCIVL, dict]:
    """Restore the GCIVL executor of a scene-executor run."""
    manifest = json.loads((run / "manifest.json").read_text())
    if manifest.get("schema") != "scene_executor_v1":
        raise ValueError(f"Expected a scene-executor run in {run}")
    settings = {k: v for k, v in manifest["executor"].items() if k != "kind"}
    config = GCIVLConfig(**settings)
    model = GCIVL(SCENE.observation_size, SCENE.observation_size, ACTION_SIZE, config)
    restore_weights(model, load_weights(run / WEIGHTS_FILE)["executor"])
    return model, manifest


def _train(train, validation, config: SceneRegionConfig, out: Path):
    completion_config = config.completion
    seed = completion_config.seed
    settings = dict(
        history=completion_config.history,
        queries=completion_config.queries,
        max_horizon=completion_config.max_horizon,
    )
    mean = train.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train.observations.std(axis=0, dtype=np.float64), 1e-3).astype(
        np.float32
    )
    windows = train.sample_windows(
        np.random.default_rng([seed, 800]), completion_config.windows, **settings
    )
    heldout = validation.sample_windows(np.random.default_rng(801), 512, **settings)
    completion = fit(
        windows, heldout, mean, scale, completion_config, out / "checkpoints"
    )
    decoder = shared.fit_decoder(
        shared.region_windows(
            completion,
            train.sample_windows(
                np.random.default_rng([seed, 6200]), config.decoder_windows, **settings
            ),
        ),
        config.region,
        seed=seed,
        out=out / "checkpoints" / "region",
    )
    latent_policy = fit_gcivl(
        train,
        ACTION_SIZE,
        config.latent_executor,
        seed=seed,
        out=out / "checkpoints" / "latent-executor",
        goal_features=completion.encode_observations(train.observations),
    )
    save_weights(
        out / WEIGHTS_FILE,
        {
            **completion.weights,
            "mean": mean,
            "scale": scale,
            "flow_scale": completion.flow_scale.numpy(),
            "decoder": decoder.weights,
            "latent_executor": weights(latent_policy),
        },
    )
    return (
        completion,
        decoder,
        latent_policy,
        completion.validate(heldout, compare_xy=False),
    )


def _load(models_run: Path, config: SceneRegionConfig):
    """Reuse the models of an earlier scene-regions run with the same recipe."""
    manifest = json.loads((models_run / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected a scene-regions run in {models_run}")
    saved = manifest["regions"]
    for name in ("completion", "region", "latent_executor"):
        if saved[name] != json.loads(json.dumps(asdict(getattr(config, name)))):
            raise ValueError(f"The {name} recipe differs from {models_run}")
    values = load_weights(models_run / WEIGHTS_FILE)
    completion = LearnedCompletion.from_weights(config.completion, values)
    decoder = RegionDecoder.from_weights(
        values["decoder"], config.region.sampling_steps
    )
    latent_policy = GCIVL(
        SCENE.observation_size,
        config.completion.latent_size,
        ACTION_SIZE,
        config.latent_executor,
    )
    restore_weights(latent_policy, values["latent_executor"])
    return completion, decoder, latent_policy, manifest.get("validation", {})


_WORKER: dict = {}


def _start_worker(models, executor, latent_policy, arms, config, limit):
    torch.set_num_threads(1)
    env = Manipulation(SCENE)

    def check(observation, target, goal):
        snapped = snap_buttons(target)
        return {
            "progress": matched_parts(snapped, goal) > matched_parts(observation, goal)
        }

    _WORKER.update(
        env=env,
        models=models,
        arms=arms,
        config=config,
        limit=limit,
        controls=Controls(
            act=executor.act,
            act_latent=latent_policy.act,
            terminal=executor.act,
            arrived=lambda observation, target: bool(
                arrived(observation, snap_buttons(target))
            ),
            check=check,
        ),
    )


def _play(batch: list[Episode]) -> list[dict]:
    env, models, config = _WORKER["env"], _WORKER["models"], _WORKER["config"]
    limit = _WORKER["limit"]
    rows = []
    for episode in batch:
        start = env.capture(episode.task, episode.seed)
        policy = ClosedLoop(
            models,
            _WORKER["controls"],
            _WORKER["arms"].get(episode.arm, episode.arm),
            episode.mode,
            episode_key(models, episode),
            limit,
            stages=config.stages,
            region_samples=config.region_samples,
            action_size=ACTION_SIZE,
        )
        observation, success, step = start.observation, False, 0
        for step in range(1, limit + 1):
            observation, success, done = env.step(policy(observation, start.goal))
            if success or done:
                break
        rows.append(record(episode, policy, success, step, step if success else None))
    return rows


def run(
    data_dir: Path,
    executor_run: Path,
    out: Path,
    config: SceneRegionConfig,
    *,
    split: str,
    tasks: tuple[int, ...],
    repeats: int,
    workers: int = 1,
    models_run: Path | None = None,
) -> dict:
    if not 0 < repeats <= SEED_OFFSETS["test"] or workers < 1:
        raise ValueError(
            "Use positive repeats within the split and at least one worker"
        )
    config = replace(
        config, completion=replace(config.completion, max_horizon=SCENE.action_limit)
    )
    train, validation = (
        load_state_archive(
            data_dir / f"{SCENE.dataset}{suffix}.npz",
            observation_size=SCENE.observation_size,
            action_size=ACTION_SIZE,
        )
        for suffix in ("", "-val")
    )
    executor, executor_manifest = load_executor(executor_run)
    if executor_manifest["train_sha256"] != train.fingerprint:
        raise ValueError("The executor was trained on a different Scene archive")
    out.mkdir(parents=True, exist_ok=False)
    if models_run is None:
        completion, decoder, latent_policy, validation_metrics = _train(
            train, validation, config, out
        )
    else:
        completion, decoder, latent_policy, validation_metrics = _load(
            models_run, config
        )
    models = RegionModels(completion, decoder)
    settings = dict(
        history=completion.config.history,
        queries=completion.config.queries,
        max_horizon=completion.config.max_horizon,
    )
    thresholds, calibration = shared.calibrate(
        decoder,
        models.encode,
        shared.region_windows(
            completion,
            validation.sample_windows(np.random.default_rng(801), 512, **settings),
        ),
        levels=config.recall_levels,
        samples=config.region_samples,
        bandwidth=config.bandwidth,
        limit=config.calibration_windows,
        seed=completion.config.seed,
    )
    arms = shared.arms(
        thresholds, config.recall_levels, models.encode, config.bandwidth
    )
    start_seed = SCENE.seed_base + SEED_OFFSETS[split]
    seeds = list(range(start_seed, start_seed + repeats))
    weights_path = (models_run or out) / WEIGHTS_FILE
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "environment": SCENE.environment,
            "train_sha256": train.fingerprint,
            "validation_sha256": validation.fingerprint,
            "versions": {
                name: version(name) for name in ("ogbench", "mujoco", "torch")
            },
            "executor_run": str(executor_run),
            "executor_weights_sha256": sha256(
                (executor_run / WEIGHTS_FILE).read_bytes()
            ).hexdigest(),
            "models_run": str(models_run) if models_run else None,
            "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
            "regions": asdict(config),
            "validation": validation_metrics,
            "arms": {name: shared.describe(arm) for name, arm in arms.items()},
            "evaluation": {
                "split": split,
                "tasks": tasks,
                "seeds": seeds,
                "action_limit": SCENE.action_limit,
            },
            "native_arrival": "official scene predicate and 4 cm effector match "
            "after snapping target buttons to one-hot states",
            "continuation": "frozen GCIVL executor toward the official goal",
            "compute": {**describe(), "eval_workers": workers},
            "code_revision": code_revision(),
        },
    )
    planned = episodes(
        list(arms), seeds, sample_repeats=config.sample_repeats, tasks=tasks
    )
    shards = [planned[index::workers] for index in range(workers)]
    with ProcessPoolExecutor(
        len(shards),
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(models, executor, latent_policy, arms, config, SCENE.action_limit),
    ) as pool:
        rows = [row for shard in pool.map(_play, shards) for row in shard]
    rows.sort(
        key=lambda row: (
            row["task"],
            row["arm"],
            row["mode"],
            row["seed"],
            row["repeat"],
        )
    )
    with gzip.open(out / "attempts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "validation": validation_metrics,
        "arrival_calibration_validation": calibration,
        "evaluation_environment_steps": sum(row["steps"] for row in rows),
        **summarize(rows),
        "per_task": {
            str(task): summarize([row for row in rows if row["task"] == task])
            for task in tasks
        },
    }
    write_json(out / "summary.json", summary)
    return summary
