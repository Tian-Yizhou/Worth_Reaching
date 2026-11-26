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
from collections import deque
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
from worth_reaching.planning.arrival import LatentGoal
from worth_reaching.planning.goals import GoalSequence, Stage
from worth_reaching.planning.receding_goals import completion_context
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
MODES = ("forecast", "sample")


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


@dataclass
class Models:
    completion: object
    decoder: object
    latent_policy: object

    def normalize(self, observation):
        return (observation - self.completion.mean) / self.completion.scale

    def denormalize(self, values):
        return values * self.completion.scale + self.completion.mean

    def encode(self, normalized: np.ndarray) -> np.ndarray:
        return (
            self.completion.codec.encode(tensor(normalized)[None]).detach().numpy()[0]
        )


def _region_windows(model, batch) -> shared.RegionWindows:
    _, forecast = model.forecast(batch.context)
    states = (batch.targets - model.mean) / model.scale
    encoded = model.codec.encode(tensor(states)).detach().numpy()
    contexts = np.repeat(
        ((batch.context.observations[:, -1] - model.mean) / model.scale)[:, None],
        states.shape[1],
        axis=1,
    )
    size = states.shape[-1]
    return shared.RegionWindows(
        forecast.numpy().reshape(-1, encoded.shape[-1]),
        encoded.reshape(-1, encoded.shape[-1]),
        contexts.reshape(-1, size).astype(np.float32),
        states.reshape(-1, size).astype(np.float32),
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
        _region_windows(model, batch),
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
    models = Models(model, decoder, None)
    normalized, normalized_goals = (
        models.normalize(observations),
        models.normalize(goals),
    )
    latent_goals = model.codec.encode(tensor(normalized_goals)).detach().numpy()
    models.latent_policy = fit_policy(
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
        _region_windows(model, heldout),
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
            models.latent_policy.act(o, g)
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
        {"decoder": decoder.weights, "latent_executor": weights(models.latent_policy)},
    )
    return models, thresholds, diagnostics


@dataclass(frozen=True, slots=True)
class Episode:
    """One closed-loop attempt: an arm, a proposal mode and a start."""

    arm: str
    mode: str
    seed: int
    repeat: int = 0


class ClosedLoop:
    """Plan, pursue declared stages and replan, then track the terminal goal."""

    def __init__(self, env, geometry, models, arm, mode, key, limit, config):
        """`arm` is a region arm, "oracle" for the shortest-path bound or None for
        direct tracking."""
        self.env, self.geometry = env, geometry
        self.models, self.arm, self.mode = models, arm, mode
        self.key, self.limit, self.config = key, limit, config
        self.history: deque = deque(maxlen=models.completion.config.history)
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.goals: GoalSequence | None = None
        self.step = 0
        self.plans: list[dict] = []

    def __call__(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        self.history.append((observation, self.last_action))
        stage = None
        if self.arm is not None:
            if self.goals is not None:
                stage = self.goals.advance(observation, self.step)
            if stage is None and self._can_plan():
                stage = self._replan(observation, goal)
        if stage is None:
            action = track(observation, goal)
        elif self.arm != "oracle" and self.arm.executor == "latent":
            action = self.models.latent_policy.act(
                self.models.normalize(observation), stage.target.latent
            )
        else:
            action = track(observation, stage.target.realization)
        self.last_action = np.asarray(action, dtype=np.float32)
        self.step += 1
        return action

    def _can_plan(self) -> bool:
        return self.limit - self.step > self.models.completion.config.queries

    def _offsets(self) -> np.ndarray:
        config = self.models.completion.config
        span = min(self.limit - self.step, config.max_horizon)
        return query_offsets(np.array([span]), config.queries)[0]

    def _replan(self, observation, goal):
        if self.arm == "oracle":
            stages, record = self._oracle(observation, goal)
        else:
            stages, record = self._propose(observation, goal)
        targets = record["targets"]
        record.update(
            step=self.step,
            in_wall=[not self.geometry.free(target) for target in targets],
            progress=[
                self.geometry.progress(observation, target, goal) for target in targets
            ],
        )
        self.plans.append(record)
        if self.goals is None:
            self.goals = GoalSequence(stages, self._arrived)
        else:
            self.goals.replace_remaining(stages)
        return self.goals.active

    def _oracle(self, observation, goal):
        target = self.geometry.waypoint(
            observation, goal, self.config.oracle_cells, self.config.oracle_speed
        )
        deadline = self.step + int(self._offsets()[0])
        stage = Stage(LatentGoal(np.zeros(0, np.float32), realization=target), deadline)
        return [stage], {"targets": [target.tolist()]}

    def _propose(self, observation, goal):
        models, completion = self.models, self.models.completion
        context = completion_context(
            list(self.history), goal, self.limit - self.step, completion.config
        )
        count = min(self.config.stages, completion.config.queries)
        if self.mode == "forecast":
            _, forecast = completion.forecast(context)
            latents = forecast.numpy()[0, :count]
        else:
            routes, _ = completion.propose(
                context, stream_seed(self.key, self.step), candidates=1
            )
            latents = routes[0, 0, :count]
        decoded = completion.decode(tensor(latents))
        draws = models.decoder.sample(
            latents,
            models.normalize(observation),
            self.config.region_samples + 1,
            stream_seed(self.key, self.step, 1),
        )
        realizations = models.denormalize(draws[:, -1])
        chosen = decoded if self.arm.target == "decoded" else realizations
        offsets = context.offsets[0, :count]
        stages = [
            Stage(
                LatentGoal(
                    latents[slot].astype(np.float32),
                    samples=draws[slot, :-1],
                    realization=chosen[slot].astype(np.float32),
                ),
                self.step + int(offsets[slot]),
            )
            for slot in range(count)
        ]
        # Latent arms are scored at the deterministic decoding of the goal they pursue.
        targets = decoded if self.arm.executor == "latent" else chosen
        return stages, {"targets": targets.tolist()}

    def _arrived(self, observation, goal: LatentGoal) -> bool:
        if self.arm == "oracle" or self.arm.arrival is None:
            return self.env.arrived(observation, goal.realization)
        return self.arm.arrival(self.models.normalize(observation), goal)


_WORKER: dict = {}


def _start_worker(data_dir, task_name, models, arms, config, limit):
    torch.set_num_threads(1)
    task = TASKS[task_name]
    dataset, _ = load_dataset(data_dir, task.dataset_id)
    native = dataset.recover_environment(eval_env=True)
    _WORKER.update(
        env=PointMaze(native, task),
        geometry=MazeGeometry(native.unwrapped.maze),
        models=models,
        arms=arms,
        config=config,
        limit=limit,
    )


def _play(episodes: list[Episode]) -> list[dict]:
    env, geometry = _WORKER["env"], _WORKER["geometry"]
    rows = []
    for episode in episodes:
        arm = {"direct": None, "oracle": "oracle"}.get(episode.arm)
        arm = arm or _WORKER["arms"].get(episode.arm)
        seed_key = stream_seed(
            _WORKER["models"].completion.config.seed,
            episode.seed,
            MODES.index(episode.mode),
            episode.repeat,
        )
        policy = ClosedLoop(
            env,
            geometry,
            _WORKER["models"],
            arm,
            episode.mode,
            seed_key,
            _WORKER["limit"],
            _WORKER["config"],
        )
        outcome, _ = rollout(env, episode.seed, _WORKER["limit"], policy)
        events = policy.goals.events if policy.goals is not None else []
        flags = [
            (wall, progress)
            for plan in policy.plans
            for wall, progress in zip(plan["in_wall"], plan["progress"], strict=True)
        ]
        rows.append(
            {
                **asdict(episode),
                "success": outcome["success"],
                "first_success_step": outcome["first_success_step"],
                "steps": outcome["steps"],
                "plans": len(policy.plans),
                "stages": len(events),
                "arrivals": sum(event.outcome == "arrived" for event in events),
                "targets_in_wall": sum(wall for wall, _ in flags),
                "targets_with_progress": sum(progress for _, progress in flags),
                "targets": len(flags),
                "first_plan": policy.plans[0] if policy.plans else None,
            }
        )
    return rows


def _episodes(arms, seeds, config) -> list[Episode]:
    episodes = [Episode("direct", "forecast", seed) for seed in seeds]
    episodes += [Episode("oracle", "forecast", seed) for seed in seeds]
    for name in arms:
        for seed in seeds:
            episodes.append(Episode(name, "forecast", seed))
            episodes += [
                Episode(name, "sample", seed, repeat)
                for repeat in range(config.sample_repeats)
            ]
    return episodes


def _label(row) -> str:
    return (
        row["arm"]
        if row["arm"] in ("direct", "oracle")
        else f"{row['arm']}/{row['mode']}"
    )


def _summarize(rows: list[dict], seeds: list[int]) -> dict:
    """Average repeats within each start, then compare arms paired by start."""
    by_arm: dict[str, dict[int, list[dict]]] = {}
    for row in rows:
        by_arm.setdefault(_label(row), {}).setdefault(row["seed"], []).append(row)
    draws = np.random.default_rng(20).integers(len(seeds), size=(2000, len(seeds)))

    def per_start(label):
        return np.array(
            [np.mean([r["success"] for r in by_arm[label][seed]]) for seed in seeds]
        )

    def ratio(trials, numerator, denominator):
        total = sum(t[denominator] for t in trials)
        return sum(t[numerator] for t in trials) / total if total else None

    direct, oracle = per_start("direct"), per_start("oracle")
    arms = {}
    for label, starts in by_arm.items():
        trials = [trial for seed in seeds for trial in starts[seed]]
        success = per_start(label)
        arrival_steps = [t["first_success_step"] for t in trials if t["success"]]
        summary = {
            "success": float(success.mean()),
            "mean_first_success_step": float(np.mean(arrival_steps))
            if arrival_steps
            else None,
            "plans_per_episode": float(np.mean([t["plans"] for t in trials])),
            "stage_arrival": ratio(trials, "arrivals", "stages"),
            "targets_in_wall": ratio(trials, "targets_in_wall", "targets"),
            "targets_with_progress": ratio(trials, "targets_with_progress", "targets"),
        }
        comparisons = {"direct": direct} if label != "direct" else {}
        if "/" in label and not label.startswith("point_native/"):
            comparisons["point_native"] = per_start(
                "point_native/" + label.split("/")[-1]
            )
        for name, baseline in comparisons.items():
            delta = success - baseline
            summary[f"difference_from_{name}"] = float(delta.mean())
            summary[f"difference_from_{name}_bootstrap_95"] = np.quantile(
                delta[draws].mean(axis=1), (0.025, 0.975)
            ).tolist()
        if label not in ("direct", "oracle"):
            gap = oracle.mean() - direct.mean()
            summary["share_of_oracle_gap"] = (
                float((success.mean() - direct.mean()) / gap) if gap > 0 else None
            )
        arms[label] = summary
    return {"starts": len(seeds), "arms": arms}


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
    models, thresholds, diagnostics = _fit(
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
    episodes = _episodes(arms, seeds, config)
    shards = [episodes[index::workers] for index in range(workers)]
    with ProcessPoolExecutor(
        len(shards),
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(data_dir, manifest["task"], models, arms, config, limit),
    ) as pool:
        rows = [row for shard in pool.map(_play, shards) for row in shard]
    rows.sort(key=lambda row: (row["arm"], row["mode"], row["seed"], row["repeat"]))
    with gzip.open(out / "attempts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        **diagnostics,
        "evaluation_environment_steps": sum(row["steps"] for row in rows),
        **_summarize(rows, seeds),
    }
    write_json(out / "summary.json", summary)
    return summary
