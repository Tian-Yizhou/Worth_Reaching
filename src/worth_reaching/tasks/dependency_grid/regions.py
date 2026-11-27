"""Compare region goals with decoded point goals on the same latent proposals (E1).

Every arm attempts the same latent candidates from matched starts and slip draws. Point
arms decode z once and require the executor to reach that cell. Latent arms give the
executor z itself and judge arrival in latent space, either by encoder consistency or
by support under the region decoder p(s | z). The terminal continuation is shared.
"""

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.models import flow
from worth_reaching.models.completion import TrainingConfig, pack_inputs, pack_windows
from worth_reaching.models.executor import (
    ExecutorConfig,
    GoalPolicy,
    accuracy,
    fit_policy,
    hindsight_pairs,
)
from worth_reaching.models.region import (
    RegionConfig,
    RegionDecoder,
    gaussian_support,
    interpolated_pairs,
    kernel_support,
    nearest_neighbor_distance,
)
from worth_reaching.models.training import (
    WEIGHTS_FILE,
    describe,
    save_weights,
    stream_seed,
    weights,
)
from worth_reaching.planning.arrival import (
    EncoderConsistency,
    GaussianSupport,
    LatentGoal,
    RegionSupport,
)
from worth_reaching.planning.goals import Stage
from worth_reaching.tasks.dependency_grid import dataset, experiment, latent
from worth_reaching.tasks.dependency_grid import environment as grid

SCHEMA = "dependency_grid_regions_v1"
# Half a cell in normalized coordinates sets the kernel width and the Gaussian floor.
BANDWIDTH = 0.5 / (grid.SIZE - 1)


@dataclass(frozen=True, slots=True)
class SweepConfig:
    recall_levels: tuple[float, ...] = (0.25, 0.5, 0.75, 0.9, 0.95)
    """Fractions of realized development states each arrival rule must accept."""
    region_samples: int = 32
    calibration_windows: int = 128
    coherence_realizations: int = 4
    """Realizations per flow candidate attempted as point goals for region coherence."""

    def __post_init__(self) -> None:
        if (
            not self.recall_levels
            or not all(0 < level < 1 for level in self.recall_levels)
            or self.region_samples < 2
            or self.calibration_windows < 2
            or self.coherence_realizations < 2
        ):
            raise ValueError(
                "Use recall levels in (0, 1) and at least two region samples, "
                "calibration windows and coherence realizations"
            )


@dataclass(frozen=True, slots=True)
class RegionRunConfig:
    region: RegionConfig = field(default_factory=RegionConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


@dataclass
class Models:
    completion: object
    decoder: RegionDecoder
    observation_policy: GoalPolicy
    latent_policy: GoalPolicy
    distance_unit: float


def _trajectories(out: Path, split: str) -> list[tuple[np.ndarray, np.ndarray]]:
    rows = []
    with (out / "trajectories.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            if row["split"] == split and row["success"]:
                rows.append(
                    (
                        np.asarray(row["observations"], np.float32),
                        np.asarray(row["actions"], np.int64),
                    )
                )
    return rows


def _fit_models(out, windows, training, flow_updates, config: RegionRunConfig):
    completion, _, fitted_flow, scale = latent.train_world_model(
        out, windows, training, flow_updates
    )
    packed, targets = pack_windows(windows["train"], latent.SPEC, training)
    predicted = completion.forecast(packed).numpy().reshape(-1, training.latent_size)
    states = targets.reshape(-1, dataset.OBSERVATION_SIZE)
    # Every window starts at the reset, which is the decision-time observation.
    contexts = np.repeat(packed.observations[:, 0], targets.shape[1], axis=0)
    encoded = completion.encode(states).numpy()
    latents, realized = interpolated_pairs(
        predicted,
        encoded,
        states,
        config.region.interpolations,
        stream_seed(training.seed, 6101),
    )
    decoder = RegionDecoder.fit(
        latents,
        np.tile(contexts, (config.region.interpolations, 1)),
        realized,
        config.region,
        seed=stream_seed(training.seed, 6102),
        out=out / "checkpoints" / "region",
    )

    train = _trajectories(out, "train")
    observations, goals, actions = hindsight_pairs(train, config.executor.horizon)
    policies = {}
    for name, goal_values in (
        ("observation", goals),
        ("latent", completion.encode(goals).numpy()),
    ):
        policies[name] = fit_policy(
            observations,
            goal_values,
            actions,
            grid.WAIT + 1,
            config.executor,
            seed=stream_seed(training.seed, 6103),
            out=out / "checkpoints" / f"{name}-executor",
        )
    all_states = np.concatenate([states for states, _ in train])
    models = Models(
        completion,
        decoder,
        policies["observation"],
        policies["latent"],
        nearest_neighbor_distance(completion.encode(all_states).numpy()),
    )
    dev_observations, dev_goals, dev_actions = hindsight_pairs(
        _trajectories(out, "dev"), config.executor.horizon
    )
    diagnostics = {
        "executor_dev_accuracy": {
            "observation": accuracy(
                models.observation_policy, dev_observations, dev_goals, dev_actions
            ),
            "latent": accuracy(
                models.latent_policy,
                dev_observations,
                completion.encode(dev_goals).numpy(),
                dev_actions,
            ),
        },
        "latent_distance_unit": models.distance_unit,
        "region_training_pairs": len(latents),
    }
    save_weights(
        out / "region-weights.pt",
        {
            "decoder": decoder.weights,
            "observation_executor": weights(models.observation_policy),
            "latent_executor": weights(models.latent_policy),
        },
    )
    thresholds, calibration = _calibrate(models, windows["dev"], training, config)
    diagnostics["arrival_calibration_dev"] = calibration
    return models, fitted_flow, scale, thresholds, diagnostics


def _calibrate(models: Models, windows, training, config: RegionRunConfig) -> dict:
    """Set each rule's threshold so it accepts a fixed share of realized dev states.

    Latent goals interpolate between the forecast and the realized encoding, as in
    decoder training. The same thresholds applied to a mismatched dev state give the
    false arrival rate at that recall level.
    """
    packed, targets = pack_windows(windows, latent.SPEC, training)
    predicted = models.completion.forecast(packed).numpy()
    encoded = models.completion.encode(targets).numpy()
    rng = np.random.default_rng(stream_seed(training.seed, 6104))
    sweep = config.sweep
    scores = {"latent_point": [], "region_gaussian": [], "region_flow": []}
    impostor = {name: [] for name in scores}
    for index in rng.permutation(len(windows))[: sweep.calibration_windows]:
        other = (index + 1 + rng.integers(len(windows) - 1)) % len(windows)
        for query in range(targets.shape[1]):
            alpha = rng.random()
            goal = predicted[index, query] + alpha * (
                encoded[index, query] - predicted[index, query]
            )
            samples = models.decoder.sample(
                goal,
                packed.observations[index, 0],
                sweep.region_samples,
                stream_seed(training.seed, 6105, int(index), query),
            )[0]
            for state, sink in (
                (targets[index, query], scores),
                (targets[other, query], impostor),
            ):
                latent_state = models.completion.encode(state[None]).numpy()[0]
                # Negate the distance so that a higher score always means arrival.
                sink["latent_point"].append(-np.linalg.norm(latent_state - goal))
                sink["region_gaussian"].append(
                    gaussian_support(state, samples, BANDWIDTH)
                )
                sink["region_flow"].append(kernel_support(state, samples, BANDWIDTH))
    thresholds, report = {}, {}
    for name, values in scores.items():
        for level in sweep.recall_levels:
            threshold = float(np.quantile(values, 1 - level))
            thresholds[name, level] = threshold
            report[f"{name}@{level:g}"] = {
                "threshold": -threshold if name == "latent_point" else threshold,
                "false_arrival": float(
                    np.mean(np.asarray(impostor[name]) >= threshold)
                ),
            }
    report["latent_point_radius_unit"] = models.distance_unit
    return thresholds, report


def _declare(
    models: Models,
    fitted_flow,
    scale,
    layout,
    depth: int,
    goal: str,
    terminal,
    *,
    candidates: int,
    repeats: int,
    region_samples: int,
    coherence_realizations: int,
):
    completion = models.completion
    public = dataset.public_input(layout, depth, goal)
    packed = pack_inputs([public], completion.spec, completion.config)
    forecast = completion.forecast(packed)
    conditions = flow.route_conditions(packed, forecast, len(grid.GOALS))
    case = f"{layout.id}:{depth}:{goal}"
    seed = int.from_bytes(sha256(case.encode()).digest()[:4], "little")
    sampled = flow.sample(
        fitted_flow,
        forecast,
        conditions,
        scale,
        stream_seed(completion.config.seed, seed, 0),
        candidates=candidates,
        steps=latent.FLOW_STEPS,
    )[0]
    pool = np.concatenate((forecast.numpy(), sampled.numpy()))[:, :depth]
    decoded = completion.decode(pool).numpy()
    reset = public.observations[0]
    samples = models.decoder.sample(
        pool, reset, region_samples, stream_seed(completion.config.seed, seed, 1)
    ).reshape(len(pool), depth, region_samples, -1)
    realizations = models.decoder.sample(
        pool, reset, repeats, stream_seed(completion.config.seed, seed, 2)
    ).reshape(len(pool), depth, repeats, -1)
    coherence = models.decoder.sample(
        pool,
        reset,
        coherence_realizations,
        stream_seed(completion.config.seed, seed, 3),
    ).reshape(len(pool), depth, coherence_realizations, -1)
    return {
        "layout_id": layout.id,
        "depth": depth,
        "goal": goal,
        "terminal": terminal,
        "candidates": [
            {
                "id": "forecast" if index == 0 else f"flow_{index - 1}",
                "latents": pool[index].tolist(),
                "decoded": decoded[index].tolist(),
                "point_targets": latent.decode_prefix(decoded[index], layout, terminal),
            }
            for index in range(len(pool))
        ],
    }, {"samples": samples, "realizations": realizations, "coherence": coherence}


@dataclass(frozen=True, slots=True)
class Arm:
    executor: str
    """bfs, observation or latent."""
    target: str
    """decoded (deterministic decoding of z), sample (one region realization per
    repeat) or latent (z itself)."""
    arrival: object = None
    """A latent arrival rule, or None to require the target's exact cell."""


def _arm_specs(models: Models, sweep: SweepConfig, thresholds: dict) -> dict:
    """Name each arm by its executor, target and recall-calibrated arrival rule."""

    def encode(observation):
        return models.completion.encode(observation[None]).numpy()[0]

    arms = {
        "point_bfs": Arm("bfs", "decoded"),
        "point_learned": Arm("observation", "decoded"),
        "sample_cell": Arm("observation", "sample"),
    }
    for level in sweep.recall_levels:
        tag = f"@{level:g}"
        consistency = EncoderConsistency(encode, -thresholds["latent_point", level])
        gaussian = GaussianSupport(BANDWIDTH, thresholds["region_gaussian", level])
        support = RegionSupport(BANDWIDTH, thresholds["region_flow", level])
        arms["latent_point" + tag] = Arm("latent", "latent", consistency)
        arms["region_gaussian" + tag] = Arm("latent", "latent", gaussian)
        arms["region_flow" + tag] = Arm("latent", "latent", support)
        arms["region_decoded" + tag] = Arm("observation", "sample", support)
        arms["decoded_support" + tag] = Arm("observation", "decoded", support)
    return arms


def _describe(rule) -> dict:
    if rule is None:
        return {"rule": "exact_cell"}
    if isinstance(rule, EncoderConsistency):
        return {"rule": "encoder_consistency", "radius": rule.epsilon}
    return {
        "rule": type(rule).__name__,
        "bandwidth": BANDWIDTH,
        "threshold": rule.threshold,
    }


def _stages(arm: Arm, layout, terminal, candidate, samples, realizations, repeat):
    """Declare one attempt's stages. Exact-cell arrival keeps the valid cell prefix."""
    depth = len(candidate["latents"])
    deadlines = experiment.DEADLINES
    if arm.target == "sample":
        vectors = realizations[:, repeat]
    elif arm.target == "decoded":
        vectors = np.asarray(candidate["decoded"], np.float32)
    else:
        vectors = np.asarray(candidate["latents"], np.float32)
    goals = [
        LatentGoal(
            np.asarray(candidate["latents"][slot], np.float32),
            samples=None if samples is None else samples[slot],
            realization=vectors[slot],
        )
        for slot in range(depth)
    ]
    if arm.arrival is not None:
        return [Stage(goals[slot], deadlines[slot]) for slot in range(depth)]
    cells = latent.decode_prefix(np.asarray(vectors), layout, terminal)
    return [
        Stage((cell, goals[slot]), deadlines[slot]) for slot, cell in enumerate(cells)
    ]


def _attempt(models, arm: Arm, layout, depth, terminal, stages, slips):
    def observe(state):
        return dataset.observe(layout, depth, state)

    if arm.executor == "bfs":
        targets = tuple(stage.target[0] for stage in stages)
        return experiment.attempt(layout, depth, terminal, targets, slips)
    policy = (
        models.latent_policy if arm.executor == "latent" else models.observation_policy
    )
    capped = [Stage(stage.target, min(stage.deadline, len(slips))) for stage in stages]
    if arm.arrival is None:
        arrived = lambda state, target: state.position == target[0]  # noqa: E731
        goal_of = lambda target: target[1]  # noqa: E731
    else:
        arrived = lambda state, target: arm.arrival(observe(state), target)  # noqa: E731
        goal_of = lambda target: target  # noqa: E731
    return experiment.pursue(
        layout,
        depth,
        terminal,
        capped,
        arrived,
        lambda state, target: policy.act(observe(state), goal_of(target).realization),
        slips,
    )


def _attempts(models, arms, layout, row, regions, slips_by_stream):
    depth, terminal = row["depth"], tuple(row["terminal"])
    results = {}
    for name, arm in arms.items():
        per_stream = {}
        for stream, repeats in slips_by_stream.items():
            per_candidate = []
            for index, candidate in enumerate(row["candidates"]):
                per_candidate.append(
                    [
                        _attempt(
                            models,
                            arm,
                            layout,
                            depth,
                            terminal,
                            _stages(
                                arm,
                                layout,
                                terminal,
                                candidate,
                                regions["samples"][index],
                                regions["realizations"][index],
                                repeat,
                            ),
                            slips,
                        )
                        for repeat, slips in enumerate(repeats)
                    ]
                )
            per_stream[stream] = per_candidate
        results[name] = per_stream
    return results


def _rates(trials) -> dict:
    count = len(trials)
    return {
        "success": sum(trial.success for trial in trials) / count,
        "arrival": sum(trial.arrived is True for trial in trials) / count,
        "success_after_miss": sum(
            trial.arrived is False and trial.success for trial in trials
        )
        / count,
        "attempt_steps": float(np.mean([trial.attempt_steps for trial in trials])),
    }


def _coherence(models, layout, row, regions, repeats) -> tuple[dict, int]:
    """Attempt several realizations of each flow region as fixed point goals.

    A coherent region has realizations of similar value. Compare the spread of value
    within each region with the spread between regions of the same start, and with
    the binomial noise of a finite number of repeats.
    """
    depth, terminal = row["depth"], tuple(row["terminal"])
    arm = Arm("observation", "sample")
    values, steps = [], 0
    for index, candidate in enumerate(row["candidates"][1:], start=1):
        draws = regions["coherence"][index]
        region = []
        for realization in range(draws.shape[1]):
            stages = _stages(arm, layout, terminal, candidate, None, draws, realization)
            trials = [
                _attempt(models, arm, layout, depth, terminal, stages, slips)
                for slips in repeats
            ]
            steps += sum(trial.environment_steps for trial in trials)
            region.append(np.mean([trial.success for trial in trials]))
        values.append(region)
    values = np.asarray(values)
    return {
        "values": values.tolist(),
        "within_region_variance": float(values.var(axis=1).mean()),
        "between_region_variance": float(values.mean(axis=1).var()),
        "binomial_noise_variance": float((values * (1 - values)).mean() / len(repeats)),
    }, steps


def _evaluate(models, arms, layout, row, regions, config) -> dict:
    slips = {
        "evaluation": latent._slips(layout, config, "evaluation", config.repeats),
        "reference": latent._slips(layout, config, "reference", config.repeats),
    }
    depth, terminal = row["depth"], tuple(row["terminal"])
    direct = [
        experiment.attempt(layout, depth, terminal, (), draws)
        for draws in slips["evaluation"]
    ]
    results = _attempts(models, arms, layout, row, regions, slips)
    scored = {"direct": _rates(direct)}
    steps = sum(trial.environment_steps for trial in direct)
    reference_steps = 0
    for name, streams in results.items():
        evaluation, reference = streams["evaluation"], streams["reference"]
        flows = evaluation[1:]
        best = 1 + max(
            range(len(flows)),
            key=lambda index: sum(trial.success for trial in reference[1 + index]),
        )
        scored[f"{name}/forecast"] = _rates(evaluation[0])
        scored[f"{name}/uniform"] = _rates(
            [trial for trials in flows for trial in trials]
        )
        scored[f"{name}/reference_best"] = _rates(evaluation[best])
        steps += sum(t.environment_steps for trials in evaluation for t in trials)
        reference_steps += sum(
            t.environment_steps for trials in reference for t in trials
        )
    coherence, coherence_steps = _coherence(
        models, layout, row, regions, slips["evaluation"]
    )
    return {
        "layout_id": layout.id,
        "depth": depth,
        "goal": row["goal"],
        "arms": scored,
        "coherence": coherence,
        "evaluation_steps": steps,
        "reference_steps": reference_steps,
        "coherence_steps": coherence_steps,
    }


def _summarize(outcomes: list[dict]) -> dict:
    summary = {}
    for depth in grid.DEPTHS:
        rows = [row for row in outcomes if row["depth"] == depth]
        if not rows:
            continue
        layout_count = len(rows) // len(grid.GOALS)
        draws = np.random.default_rng(20).integers(
            layout_count, size=(2000, layout_count)
        )

        def interval(delta):
            by_layout = delta.reshape(layout_count, len(grid.GOALS)).mean(axis=1)
            return np.quantile(by_layout[draws].mean(axis=1), (0.025, 0.975)).tolist()

        def column(name, key="success"):
            return np.array([row["arms"][name][key] for row in rows])

        arms = {}
        for name in rows[0]["arms"]:
            reference = name.split("/")[-1]
            baseline = f"point_learned/{reference}" if "/" in name else None
            arms[name] = {
                key: float(column(name, key).mean())
                for key in ("success", "arrival", "success_after_miss", "attempt_steps")
            }
            arms[name]["paired_difference_from_direct"] = float(
                (column(name) - column("direct")).mean()
            )
            if baseline is not None and baseline != name:
                delta = column(name) - column(baseline)
                arms[name]["paired_difference_from_point_learned"] = float(delta.mean())
                arms[name]["layout_bootstrap_95"] = interval(delta)
        summary[str(depth)] = {
            "cases": len(rows),
            "arms": arms,
            "coherence": {
                key: float(np.mean([row["coherence"][key] for row in rows]))
                for key in (
                    "within_region_variance",
                    "between_region_variance",
                    "binomial_noise_variance",
                )
            },
        }
    return summary


def run(
    out: Path,
    training: TrainingConfig,
    evaluation: latent.EvaluationConfig,
    config: RegionRunConfig,
    *,
    collection_seed: int,
    demonstrations: int,
    flow_updates: int,
) -> dict:
    layouts = grid.layouts(evaluation.split)
    if evaluation.layouts is not None and not 1 <= evaluation.layouts <= len(layouts):
        raise ValueError(f"Expected 1 to {len(layouts)} layouts for {evaluation.split}")
    if (
        collection_seed < 0
        or min(demonstrations, flow_updates) < 1
        or training.queries_per_window != len(dataset.QUERY_STEPS)
    ):
        raise ValueError(
            "Use a nonnegative collection seed, positive collection and flow counts, and two model queries"
        )
    layouts = layouts[: evaluation.layouts]
    out.mkdir(parents=True, exist_ok=False)
    windows, _, steps = latent._collect(
        out, demonstrations, collection_seed, evaluation.slip_probability
    )
    models, fitted_flow, scale, thresholds, diagnostics = _fit_models(
        out, windows, training, flow_updates, config
    )
    arms = _arm_specs(models, config.sweep, thresholds)

    declared = []
    region_arrays = {}
    for layout, depth, goal, lane in latent._cases(layouts):
        row, regions = _declare(
            models,
            fitted_flow,
            scale,
            layout,
            depth,
            goal,
            lane.goal,
            candidates=evaluation.candidates,
            repeats=evaluation.repeats,
            region_samples=config.sweep.region_samples,
            coherence_realizations=config.sweep.coherence_realizations,
        )
        declared.append((layout, row, regions))
        key = f"{layout.id}:{depth}:{goal}"
        region_arrays[f"{key}:samples"] = regions["samples"]
        region_arrays[f"{key}:realizations"] = regions["realizations"]
        region_arrays[f"{key}:coherence"] = regions["coherence"]
    bank_path = out / "candidate-bank.json"
    write_json(bank_path, [row for _, row, _ in declared])
    regions_path = out / "candidate-regions.npz"
    np.savez_compressed(regions_path, **region_arrays)
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "code_revision": code_revision(),
            "compute": describe(),
            "versions": {
                name: version(name) for name in ("torch", "lightning", "numpy")
            },
            "model": asdict(latent.SPEC),
            "training": asdict(training),
            "evaluation": asdict(evaluation),
            "regions": asdict(config),
            "arms": {
                name: {
                    "executor": arm.executor,
                    "target": arm.target,
                    "arrival": _describe(arm.arrival),
                }
                for name, arm in arms.items()
            },
            "kernel_bandwidth": BANDWIDTH,
            "collection_seed": collection_seed,
            "demonstrations_per_case": demonstrations,
            "flow_updates": flow_updates,
            "query_steps": dataset.QUERY_STEPS,
            "deadlines": experiment.DEADLINES,
            "layouts": [layout.id for layout in layouts],
            "weights_sha256": sha256((out / WEIGHTS_FILE).read_bytes()).hexdigest(),
            "region_weights_sha256": sha256(
                (out / "region-weights.pt").read_bytes()
            ).hexdigest(),
            "candidate_bank_sha256": sha256(bank_path.read_bytes()).hexdigest(),
            "candidate_regions_sha256": sha256(regions_path.read_bytes()).hexdigest(),
        },
    )
    outcomes = [
        _evaluate(models, arms, layout, row, regions, evaluation)
        for layout, row, regions in declared
    ]
    with (out / "attempts.jsonl").open("w", encoding="utf-8") as stream:
        for row in outcomes:
            stream.write(json.dumps(row) + "\n")
    summary = {
        "split": evaluation.split,
        "training_windows": {split: len(rows) for split, rows in windows.items()},
        "collection_environment_steps": dict(steps),
        "evaluation_environment_steps": sum(
            row["evaluation_steps"] for row in outcomes
        ),
        "reference_environment_steps": sum(row["reference_steps"] for row in outcomes),
        "coherence_environment_steps": sum(row["coherence_steps"] for row in outcomes),
        **diagnostics,
        "by_depth": _summarize(outcomes),
    }
    write_json(out / "summary.json", summary)
    return summary
