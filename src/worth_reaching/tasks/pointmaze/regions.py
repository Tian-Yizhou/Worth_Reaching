"""Compare region goals with decoded point goals in closed-loop Maze2D control (E1).

A frozen pointmaze-latent run supplies the world model. This runner trains the region
decoder and a latent-conditioned executor on the same archive and calibrates the arrival
rules on validation windows. Every arm then plans in closed loop: from the public
history it declares the first queries of one completion as stages, pursues them until
each arrives or its deadline passes, and replans after the declared stages close. Near
the end of the budget, Diffuser's position and velocity tracker pursues the terminal
goal.

Two bounds use the same tracker. Direct tracking of the terminal goal is the floor. A
shortest-path oracle on the privileged layout replans one cell ahead on the same
schedule and is the ceiling. Plans are scored on the same layout for wall placement and
geodesic progress, which the models never see.
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.trajectory_windows import query_offsets
from worth_reaching.evaluation import regions as shared
from worth_reaching.evaluation.closed_loop import (
    DIRECT,
    ORACLE,
    ClosedLoop,
    Controls,
    Episode,
    RegionModels,
    episode_key,
    episodes,
    record,
    summarize,
)
from worth_reaching.models.executor import (
    ExecutorConfig,
    fit_policy,
    sample_hindsight,
)
from worth_reaching.models.region import RegionConfig
from worth_reaching.models.training import (
    describe,
    save_weights,
    stream_seed,
    tensor,
    weights,
)
from worth_reaching.tasks.pointmaze.dataset import (
    ACTION_SIZE,
    load_dataset,
    split_trajectories,
)
from worth_reaching.tasks.pointmaze.energy import FrozenRun
from worth_reaching.tasks.pointmaze.environment import SEED_BASE, PointMaze
from worth_reaching.tasks.pointmaze.experiment import rollout, track
from worth_reaching.tasks.pointmaze.geometry import MazeGeometry
from worth_reaching.tasks.pointmaze.spec import TASKS

SCHEMA = "pointmaze_regions_v2"


@dataclass(frozen=True, slots=True)
class MazeRegionConfig:
    region: RegionConfig = field(default_factory=RegionConfig)
    executor: ExecutorConfig = field(
        default_factory=lambda: ExecutorConfig(updates=4000, width=256, batch_size=256)
    )
    recall_levels: tuple[float, ...] = (0.5, 0.75, 0.9)
    region_samples: int = 32
    calibration_windows: int = 512
    executor_pairs: int = 200_000
    bandwidth: float = 0.25
    """Kernel width in observation units normalized by the training scale."""
    stages: int = 1
    """Queries of each completion declared as stages before the next replan."""
    sample_repeats: int = 2
    """Episodes per start for arms that draw a fresh flow sample at every replan."""
    oracle_cells: int = 1
    oracle_speed: float = 2.0

    def __post_init__(self) -> None:
        if (
            not self.recall_levels
            or not all(0 < level < 1 for level in self.recall_levels)
            or min(self.region_samples, self.calibration_windows) < 2
            or min(
                self.executor_pairs,
                self.stages,
                self.sample_repeats,
                self.oracle_cells,
            )
            < 1
            or self.bandwidth <= 0
            or self.oracle_speed <= 0
        ):
            raise ValueError(
                "Use recall levels in (0, 1), at least two samples and calibration "
                "windows, positive counts, bandwidth and oracle speed"
            )


def _fit(frozen: FrozenRun, train, validation, limit, config, out):
    model = frozen.model
    settings = dict(
        history=model.config.history,
        queries=model.config.queries,
        max_horizon=model.config.max_horizon,
        max_action_budget=model.config.max_action_budget,
    )
    seed = model.config.seed
    batch = train.sample_windows(
        np.random.default_rng([seed, 6200]),
        model.config.windows,
        full_span_windows=max(1, model.config.windows // 4),
        **settings,
    )
    decoder = shared.fit_decoder(
        shared.region_windows(model, batch),
        config.region,
        seed=seed,
        out=out / "checkpoints" / "region",
    )
    span = min(limit, model.config.max_horizon)
    offsets = query_offsets(np.array([span]), model.config.queries)[0]
    horizon = int(offsets[min(config.stages, len(offsets)) - 1])
    rng = np.random.default_rng([seed, 6201])
    observations, goals, actions = sample_hindsight(
        train.observations,
        train.actions,
        train.ends,
        config.executor_pairs,
        horizon,
        rng,
    )
    models = RegionModels(model, decoder)
    normalized, normalized_goals = (
        models.normalize(observations),
        models.normalize(goals),
    )
    latent_goals = model.codec.encode(tensor(normalized_goals)).detach().numpy()
    latent_policy = fit_policy(
        normalized,
        latent_goals,
        actions,
        None,
        config.executor,
        seed=stream_seed(seed, 6202),
        out=out / "checkpoints" / "latent-executor",
    )
    heldout = validation.sample_windows(
        np.random.default_rng(801), 512, full_span_windows=128, **settings
    )
    thresholds, calibration = shared.calibrate(
        decoder,
        models.encode,
        shared.region_windows(model, heldout),
        levels=config.recall_levels,
        samples=config.region_samples,
        bandwidth=config.bandwidth,
        limit=config.calibration_windows,
        seed=seed,
    )
    check_obs, check_goals, check_actions = sample_hindsight(
        validation.observations,
        validation.actions,
        validation.ends,
        4096,
        horizon,
        np.random.default_rng([seed, 6203]),
    )
    predicted = np.stack(
        [
            latent_policy.act(o, g)
            for o, g in zip(
                models.normalize(check_obs),
                model.codec.encode(tensor(models.normalize(check_goals)))
                .detach()
                .numpy(),
                strict=True,
            )
        ]
    )
    diagnostics = {
        "arrival_calibration_validation": calibration,
        "latent_executor_validation_action_mse": float(
            np.mean(np.square(predicted - check_actions))
        ),
        "executor_goal_horizon": horizon,
    }
    save_weights(
        out / "region-weights.pt",
        {"decoder": decoder.weights, "latent_executor": weights(latent_policy)},
    )
    return models, latent_policy, thresholds, diagnostics


_WORKER: dict = {}


def _start_worker(data_dir, task_name, models, latent_policy, arms, config, limit):
    torch.set_num_threads(1)
    task = TASKS[task_name]
    dataset, _ = load_dataset(data_dir, task.dataset_id)
    native = dataset.recover_environment(eval_env=True)
    env, geometry = PointMaze(native, task), MazeGeometry(native.unwrapped.maze)
    _WORKER.update(
        env=env,
        models=models,
        arms=arms,
        config=config,
        limit=limit,
        controls=Controls(
            act=track,
            act_latent=lambda observation, latent: latent_policy.act(
                models.normalize(observation), latent
            ),
            terminal=track,
            arrived=env.arrived,
            oracle=lambda observation, goal: geometry.waypoint(
                observation, goal, config.oracle_cells, config.oracle_speed
            ),
            check=lambda observation, target, goal: {
                "in_wall": not geometry.free(target),
                "progress": geometry.progress(observation, target, goal),
            },
        ),
    )


def _play(batch: list[Episode]) -> list[dict]:
    env, models, config = _WORKER["env"], _WORKER["models"], _WORKER["config"]
    rows = []
    for episode in batch:
        policy = ClosedLoop(
            models,
            _WORKER["controls"],
            _WORKER["arms"].get(episode.arm, episode.arm),
            episode.mode,
            episode_key(models, episode),
            _WORKER["limit"],
            stages=config.stages,
            region_samples=config.region_samples,
            action_size=ACTION_SIZE,
        )
        outcome, _ = rollout(env, episode.seed, _WORKER["limit"], policy)
        rows.append(
            record(
                episode,
                policy,
                outcome["success"],
                outcome["steps"],
                outcome["first_success_step"],
            )
        )
    return rows


def run(
    data_dir: Path,
    run_path: Path,
    out: Path,
    config: MazeRegionConfig,
    *,
    split: str,
    repeats: int,
    workers: int = 1,
) -> dict:
    frozen = FrozenRun.load(run_path)
    manifest = frozen.manifest
    limit = manifest["evaluation"]["action_limit"]
    if config.stages > frozen.model.config.queries:
        raise ValueError("Declare at most as many stages as the model queries")
    if not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"] or workers < 1:
        raise ValueError("Use a positive number of reset seeds and workers")
    task = TASKS[manifest["task"]]
    dataset, hashes = load_dataset(data_dir, task.dataset_id)
    if hashes != manifest["dataset_files_sha256"]:
        raise ValueError("Dataset files disagree with the frozen run")
    train, validation, _ = split_trajectories(dataset)
    out.mkdir(parents=True, exist_ok=False)
    models, latent_policy, thresholds, diagnostics = _fit(
        frozen, train, validation, limit, config, out
    )
    arms = shared.arms(
        thresholds, config.recall_levels, models.encode, config.bandwidth
    )
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
            "action_limit": limit,
            "split": split,
            "seeds": seeds,
            "regions": asdict(config),
            "arms": {name: shared.describe(arm) for name, arm in arms.items()},
            "modes": {
                "forecast": "deterministic forecast at every replan",
                "sample": "one fresh flow sample at every replan",
            },
            "bounds": {
                "direct": "position and velocity tracker toward the terminal goal",
                "oracle": "privileged shortest-path waypoint on the same replanning "
                "schedule with native arrival",
            },
            "native_arrival": "PointMaze sparse goal predicate on public XY",
            "continuation": "position and velocity tracker toward the terminal goal",
            "compute": {**describe(), "eval_workers": workers},
            "code_revision": code_revision(),
            "region_weights_sha256": sha256(
                (out / "region-weights.pt").read_bytes()
            ).hexdigest(),
        },
    )
    planned = episodes(
        list(arms), seeds, sample_repeats=config.sample_repeats, bounds=(DIRECT, ORACLE)
    )
    shards = [planned[index::workers] for index in range(workers)]
    with ProcessPoolExecutor(
        len(shards),
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(
            data_dir,
            manifest["task"],
            models,
            latent_policy,
            arms,
            config,
            limit,
        ),
    ) as pool:
        rows = [row for shard in pool.map(_play, shards) for row in shard]
    rows.sort(key=lambda row: (row["arm"], row["mode"], row["seed"], row["repeat"]))
    with gzip.open(out / "attempts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        **diagnostics,
        "evaluation_environment_steps": sum(row["steps"] for row in rows),
        **summarize(rows),
    }
    write_json(out / "summary.json", summary)
    return summary
