"""Compare region goals with decoded point goals in Maze2D (E1).

A frozen pointmaze-latent run supplies the world model. This runner trains the region
decoder and a latent-conditioned executor on the same archive, calibrates the arrival
rules on validation windows, and attempts every candidate from matched seeded resets.
The task controller is Diffuser's position and velocity tracker. It pursues state
targets for the observation arms and tracks the terminal goal after every stage, so
no arm hands control to a learned policy that already knows the task.
"""

import json
from dataclasses import asdict, dataclass, field
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

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
from worth_reaching.tasks.pointmaze.spec import TASKS

SCHEMA = "pointmaze_regions_v1"


@dataclass(frozen=True, slots=True)
class MazeRegionConfig:
    region: RegionConfig = field(default_factory=RegionConfig)
    executor: ExecutorConfig = field(
        default_factory=lambda: ExecutorConfig(updates=4000, width=256, batch_size=256)
    )
    recall_levels: tuple[float, ...] = (0.5, 0.75, 0.9)
    region_samples: int = 32
    coherence_realizations: int = 4
    calibration_windows: int = 512
    executor_pairs: int = 200_000
    bandwidth: float = 0.25
    """Kernel width in observation units normalized by the training scale."""

    def __post_init__(self) -> None:
        if (
            not self.recall_levels
            or not all(0 < level < 1 for level in self.recall_levels)
            or min(
                self.region_samples,
                self.coherence_realizations,
                self.calibration_windows,
            )
            < 2
            or self.executor_pairs < 1
            or self.bandwidth <= 0
        ):
            raise ValueError(
                "Use recall levels in (0, 1), at least two samples, realizations and "
                "calibration windows, and a positive bandwidth"
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


def _fit(frozen: FrozenRun, train, validation, limit, prefix, config, out):
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
    horizon = int(query_offsets(np.array([span]), model.config.queries)[0, prefix - 1])
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


def _declare(
    models: Models,
    seed,
    start,
    goal,
    limit,
    prefix,
    config: MazeRegionConfig,
    candidates,
):
    completion = models.completion
    context = completion_context(
        [(start, np.zeros(ACTION_SIZE, dtype=np.float32))],
        goal,
        limit,
        completion.config,
    )
    _, forecast = completion.forecast(context)
    key = stream_seed(completion.config.seed, seed)
    latents, _ = completion.propose(context, stream_seed(key, 0), candidates=candidates)
    pool = np.concatenate((forecast.numpy()[:, :prefix], latents[0, :, :prefix]))
    decoded = completion.decode(tensor(pool))
    origin = models.normalize(start)

    def draw(count, stream):
        values = models.decoder.sample(pool, origin, count, stream_seed(key, stream))
        return values.reshape(len(pool), prefix, count, -1)

    return {
        "seed": seed,
        "start": start.tolist(),
        "goal": goal.tolist(),
        "deadlines": context.offsets[0, :prefix].tolist(),
        "candidates": [
            {
                "id": "forecast" if index == 0 else f"flow_{index - 1}",
                "latents": pool[index].tolist(),
                "decoded": decoded[index].tolist(),
            }
            for index in range(len(pool))
        ],
    }, {
        "samples": draw(config.region_samples, 1),
        "realizations": draw(1, 2),
        "coherence": draw(config.coherence_realizations, 3),
    }


def _attempt(env, models, arm: shared.Arm, row, index, regions, limit, realization=0):
    candidate = row["candidates"][index]
    prefix = len(candidate["latents"])
    samples = regions["samples"][index]
    if arm.target == "sample":
        source = (
            regions["realizations"] if realization is None else regions["coherence"]
        )
        targets = models.denormalize(
            source[index][:, 0 if realization is None else realization]
        )
    elif arm.target == "decoded":
        targets = np.asarray(candidate["decoded"], np.float32)
    else:
        targets = np.asarray(candidate["latents"], np.float32)
    stages = [
        Stage(
            LatentGoal(
                np.asarray(candidate["latents"][slot], np.float32),
                samples=samples[slot],
                realization=targets[slot],
            ),
            row["deadlines"][slot],
        )
        for slot in range(prefix)
    ]

    def arrived(observation, goal):
        if arm.arrival is None:
            return env.arrived(observation, goal.realization)
        return arm.arrival(models.normalize(observation), goal)

    goals = GoalSequence(stages, arrived)
    step = 0

    def policy(observation, terminal):
        nonlocal step
        stage = goals.advance(observation, step)
        step += 1
        if stage is None:
            return track(observation, terminal)
        if arm.executor == "latent":
            return models.latent_policy.act(
                models.normalize(observation), stage.target.realization
            )
        return track(observation, stage.target.realization)

    episode, path = rollout(env, row["seed"], limit, policy)
    goals.finish(path[-1], step)
    return {
        "success": episode["success"],
        "arrived": goals.arrived_all,
        "attempt_steps": goals.handoff_step,
        "steps": episode["steps"],
    }


def _rates(trials):
    return {
        "success": float(np.mean([t["success"] for t in trials])),
        "arrival": float(np.mean([t["arrived"] for t in trials])),
        "success_after_miss": float(
            np.mean([t["success"] and not t["arrived"] for t in trials])
        ),
        "attempt_steps": float(np.mean([t["attempt_steps"] for t in trials])),
    }


def _summarize(outcomes: list[dict]) -> dict:
    names = list(outcomes[0]["arms"])
    draws = np.random.default_rng(20).integers(
        len(outcomes), size=(2000, len(outcomes))
    )

    def column(name, key="success"):
        return np.array([row["arms"][name][key] for row in outcomes])

    arms = {}
    for name in names:
        arms[name] = {
            key: float(column(name, key).mean())
            for key in ("success", "arrival", "success_after_miss", "attempt_steps")
        }
        if "/" in name and not name.startswith("point_native/"):
            baseline = "point_native/" + name.split("/")[-1]
            delta = column(name) - column(baseline)
            arms[name]["paired_difference_from_point_native"] = float(delta.mean())
            arms[name]["start_bootstrap_95"] = np.quantile(
                delta[draws].mean(axis=1), (0.025, 0.975)
            ).tolist()
    return {
        "starts": len(outcomes),
        "arms": arms,
        "coherence": {
            key: float(np.mean([row["coherence"][key] for row in outcomes]))
            for key in outcomes[0]["coherence"]
        },
    }


def run(
    data_dir: Path,
    run_path: Path,
    out: Path,
    config: MazeRegionConfig,
    *,
    split: str,
    repeats: int,
    candidates: int,
    prefix: int,
) -> dict:
    frozen = FrozenRun.load(run_path)
    manifest = frozen.manifest
    limit = manifest["evaluation"]["action_limit"]
    if not 1 <= prefix <= frozen.model.config.queries or candidates < 2:
        raise ValueError(
            "Use a prefix within the query count and at least two candidates"
        )
    if not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"]:
        raise ValueError("Use a positive number of reset seeds within the split")
    task = TASKS[manifest["task"]]
    dataset, hashes = load_dataset(data_dir, task.dataset_id)
    if hashes != manifest["dataset_files_sha256"]:
        raise ValueError("Dataset files disagree with the frozen run")
    train, validation, _ = split_trajectories(dataset)
    out.mkdir(parents=True, exist_ok=False)
    models, thresholds, diagnostics = _fit(
        frozen, train, validation, limit, prefix, config, out
    )
    arms = shared.arms(
        thresholds, config.recall_levels, models.encode, config.bandwidth
    )
    native = dataset.recover_environment(eval_env=True)
    try:
        env = PointMaze(native, task)
        seeds = range(SEED_BASE[split], SEED_BASE[split] + repeats)
        declared, arrays = [], {}
        for seed in seeds:
            start, goal = env.reset(seed)
            row, regions = _declare(
                models, seed, start, goal, limit, prefix, config, candidates
            )
            declared.append((row, regions))
            for name, values in regions.items():
                arrays[f"{seed}:{name}"] = values
        bank_path = out / "candidate-bank.json"
        write_json(bank_path, [row for row, _ in declared])
        regions_path = out / "candidate-regions.npz"
        np.savez_compressed(regions_path, **arrays)
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
                "seeds": list(seeds),
                "candidates": candidates,
                "prefix": prefix,
                "regions": asdict(config),
                "arms": {name: shared.describe(arm) for name, arm in arms.items()},
                "native_arrival": "PointMaze sparse goal predicate on public XY",
                "continuation": "position and velocity tracker toward the terminal goal",
                "compute": describe(),
                "code_revision": code_revision(),
                "region_weights_sha256": sha256(
                    (out / "region-weights.pt").read_bytes()
                ).hexdigest(),
                "candidate_bank_sha256": _digest(bank_path),
                "candidate_regions_sha256": _digest(regions_path),
            },
        )
        outcomes = []
        for row, regions in declared:
            direct, _ = rollout(env, row["seed"], limit, track)
            scored = {
                "direct": _rates([{**direct, "arrived": False, "attempt_steps": 0}])
            }
            steps = direct["steps"]
            for name, arm in arms.items():
                trials = [
                    _attempt(
                        env, models, arm, row, index, regions, limit, realization=None
                    )
                    for index in range(len(row["candidates"]))
                ]
                steps += sum(trial["steps"] for trial in trials)
                scored[f"{name}/forecast"] = _rates(trials[:1])
                scored[f"{name}/uniform"] = _rates(trials[1:])
            coherence_arm = shared.Arm("observation", "sample")
            values = []
            for index in range(1, len(row["candidates"])):
                region = []
                for realization in range(config.coherence_realizations):
                    trial = _attempt(
                        env,
                        models,
                        coherence_arm,
                        row,
                        index,
                        regions,
                        limit,
                        realization,
                    )
                    steps += trial["steps"]
                    region.append(float(trial["success"]))
                values.append(region)
            outcomes.append(
                {
                    "seed": row["seed"],
                    "arms": scored,
                    "coherence": {"values": values, **shared.coherence(values, 1)},
                    "evaluation_steps": steps,
                }
            )
    finally:
        native.close()
    with (out / "attempts.jsonl").open("w", encoding="utf-8") as stream:
        for row in outcomes:
            stream.write(json.dumps(row) + "\n")
    summary = {
        **diagnostics,
        "evaluation_environment_steps": sum(
            row["evaluation_steps"] for row in outcomes
        ),
        **_summarize(outcomes),
    }
    write_json(out / "summary.json", summary)
    return summary


def _digest(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()
