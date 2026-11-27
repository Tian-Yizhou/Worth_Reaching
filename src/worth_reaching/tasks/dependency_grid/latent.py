"""Compare frozen latent proposals and matched-time replay in the dependency grid."""

import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from random import Random

import numpy as np

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.completion import CompletionExample
from worth_reaching.evaluation.attempt_selection import fit_attempt_heads
from worth_reaching.models import flow
from worth_reaching.models.completion import (
    FrozenCompletion,
    LatentPredictor,
    ModelSpec,
    TrainingConfig,
    fit,
    pack_inputs,
    pack_windows,
    predict,
)
from worth_reaching.models.latent_energy import prefix_energy
from worth_reaching.models.training import (
    WEIGHTS_FILE,
    describe,
    save_weights,
    stream_seed,
    weights,
)
from worth_reaching.tasks.dependency_grid import dataset, experiment, outcome_scoring
from worth_reaching.tasks.dependency_grid import environment as grid

SCHEMA = "dependency_grid_latent_v1"
FLOW_STEPS = 16
SPEC = ModelSpec(dataset.OBSERVATION_SIZE, grid.WAIT + 1, grid.GOALS, experiment.BUDGET)


@dataclass(frozen=True, slots=True)
class EvaluationConfig:
    split: str = "dev"
    layouts: int | None = None
    seed: int = 43301
    repeats: int = 16
    candidates: int = 8
    density_samples: int = 128
    slip_probability: float = 0.10

    def __post_init__(self) -> None:
        if (
            self.seed < 0
            or self.repeats < 1
            or not 2 <= self.candidates <= len(grid.layouts("train"))
            or self.density_samples < 2
            or not 0 <= self.slip_probability < 1
        ):
            raise ValueError(
                "Use a nonnegative evaluation seed, positive repeats, 2–32 "
                "candidates, at least two density samples and slip in [0, 1)"
            )


def _cases(layouts):
    for layout in layouts:
        for depth in grid.DEPTHS:
            for goal, lane in zip(grid.GOALS, layout.lanes, strict=True):
                yield layout, depth, goal, lane


def _collect(out: Path, count: int, seed: int, slip: float):
    windows = {"train": [], "dev": []}
    replay = defaultdict(list)
    steps = Counter()
    with (out / "trajectories.jsonl").open("w", encoding="utf-8") as stream:
        for split in windows:
            for layout, depth, goal, lane in _cases(grid.layouts(split)):
                for index in range(count):
                    episode_id = f"{layout.id}:{depth}:{goal}:{index}"
                    rng = Random(f"{SCHEMA}:collection:{seed}:{episode_id}")
                    trace = dataset.trajectory(layout, depth, lane.goal, rng, slip)
                    steps[split] += len(trace.actions)
                    stream.write(
                        json.dumps(
                            {
                                "split": split,
                                "layout_id": layout.id,
                                "episode_id": episode_id,
                                "depth": depth,
                                "goal": goal,
                                "observations": trace.observations.tolist(),
                                "actions": trace.actions.tolist(),
                                "success": trace.success,
                            }
                        )
                        + "\n"
                    )
                    if trace.success:
                        window = trace.window(layout.id, episode_id, goal)
                        windows[split].append(window)
                        if split == "train" and (
                            not replay[depth, goal]
                            or replay[depth, goal][-1].world_id != layout.id
                        ):
                            replay[depth, goal].append(window)
    if not all(windows.values()):
        raise ValueError("Collection must supply successful train and dev windows")
    return windows, replay, steps


def decode_prefix(values: np.ndarray, layout: grid.Layout, terminal: grid.Position):
    """Round coordinates and retain the executable prefix before the first invalid cell."""
    if not np.isfinite(values).all():
        raise ValueError("Expected finite target observations")
    seen = {grid.START, terminal}
    positions = []
    for row, column in values[:, :2]:
        cell = (
            round(float(row) * (grid.SIZE - 1)),
            round(float(column) * (grid.SIZE - 1)),
        )
        if cell in seen or not layout.is_free(cell):
            break
        positions.append(cell)
        seen.add(cell)
    return tuple(positions)


def _declare(
    completion: FrozenCompletion,
    goal_blind: LatentPredictor,
    fitted_flow,
    scale,
    replay: list[CompletionExample],
    layout: grid.Layout,
    depth: int,
    goal: str,
    terminal: grid.Position,
    config: EvaluationConfig,
):
    public = dataset.public_input(layout, depth, goal)
    packed = pack_inputs([public], completion.spec, completion.config)
    forecast = completion.forecast(packed)
    conditions = flow.route_conditions(packed, forecast, len(grid.GOALS))
    case = f"{layout.id}:{depth}:{goal}"
    seed = int.from_bytes(sha256(case.encode()).digest()[:4], "little")

    def sample(stream: int, candidates: int):
        return flow.sample(
            fitted_flow,
            forecast,
            conditions,
            scale,
            stream_seed(completion.config.seed, seed, stream),
            candidates=candidates,
            steps=FLOW_STEPS,
        )[0]

    latents = sample(0, config.candidates)
    scores = prefix_energy(
        latents[:, :depth].numpy(),
        sample(1, config.density_samples)[:, :depth].numpy(),
        scale.numpy(),
    )
    errors = (latents[:, :depth] - forecast[0, :depth]).square().mean(dim=(1, 2))
    decoded = completion.decode(latents).numpy()[:, :depth]
    generated = []
    for observation, energy, error in zip(
        decoded, scores, errors.tolist(), strict=True
    ):
        generated.append(
            {
                "decoded_xy": observation[:, :2].tolist(),
                "targets": decode_prefix(observation, layout, terminal),
                "energy": float(energy),
                "forecast_mse": error,
            }
        )
    reset = public.observations[0]
    _, blind_decoded = predict(completion.codec, goal_blind, packed.without_goal())
    nearest = sorted(
        (source for source in replay if source.world_id != layout.id),
        key=lambda row: (
            float(np.square(row.inputs.observations[0] - reset).sum()),
            row.world_id,
        ),
    )[: config.candidates]
    if len(nearest) != config.candidates:
        raise ValueError(
            f"Too few successful replay layouts for depth {depth}, goal {goal}"
        )

    def proximity(candidate: dict) -> tuple[int, int]:
        targets = candidate["targets"]
        previous = (grid.START, *targets[:-1])
        distance = sum(
            abs(origin[0] - target[0]) + abs(origin[1] - target[1])
            for origin, target in zip(previous, targets)
        )
        return -len(targets), distance

    return {
        "layout_id": layout.id,
        "depth": depth,
        "goal": goal,
        "terminal": terminal,
        "public_observation": reset.tolist(),
        "flow": generated,
        "replay": [
            {
                "episode_id": source.episode_id,
                "targets": decode_prefix(
                    source.target.observations[:depth], layout, terminal
                ),
            }
            for source in nearest
        ],
        "forecast": decode_prefix(
            completion.decode(forecast[0, :depth]).numpy(), layout, terminal
        ),
        "goal_blind": decode_prefix(blind_decoded[0, :depth].numpy(), layout, terminal),
        "choices": {
            "lowest_energy": int(scores.argmin()),
            "highest_energy": int(scores.argmax()),
            "lowest_forecast_mse": int(errors.argmin()),
            "closest_prefix": min(
                range(len(generated)), key=lambda i: proximity(generated[i])
            ),
        },
    }


def _slips(
    layout: grid.Layout, config: EvaluationConfig, stream: str, repeats: int
) -> list[tuple[bool, ...]]:
    rng = Random(f"{SCHEMA}:{stream}:{config.seed}:{layout.id}")
    return [
        tuple(rng.random() < config.slip_probability for _ in range(experiment.BUDGET))
        for _ in range(repeats)
    ]


def _evaluate(layout: grid.Layout, row: dict, config: EvaluationConfig) -> dict:
    evaluation = _slips(layout, config, "evaluation", config.repeats)
    reference = _slips(layout, config, "reference", config.repeats)
    depth, goal = row["depth"], row["terminal"]

    def attempts(targets, repeats):
        return [
            experiment.attempt(layout, depth, goal, targets, slips) for slips in repeats
        ]

    achieved = dataset.trajectory(
        layout, depth, goal, Random(f"{SCHEMA}:achieved:{layout.id}"), 0
    )
    achieved_targets = decode_prefix(
        achieved.observations[
            np.minimum(dataset.QUERY_STEPS[:depth], len(achieved.actions))
        ],
        layout,
        goal,
    )
    arms = {
        "direct": attempts((), evaluation),
        "deterministic_forecast": attempts(row["forecast"], evaluation),
        "goal_blind_forecast": attempts(row["goal_blind"], evaluation),
        "achieved_replay": attempts(achieved_targets, evaluation),
    }
    outcomes = {
        name: [asdict(trial) for trial in trials] for name, trials in arms.items()
    }
    steps = sum(trial.environment_steps for trials in arms.values() for trial in trials)
    reference_steps = 0
    reference_choices = {}
    for source in ("flow", "replay"):
        estimates = [
            attempts(candidate["targets"], reference) for candidate in row[source]
        ]
        successes = [sum(trial.success for trial in trials) for trials in estimates]
        best = max(range(len(estimates)), key=successes.__getitem__)
        reference_choices[source] = {"index": best, "reference_successes": successes}
        results = [
            attempts(candidate["targets"], evaluation) for candidate in row[source]
        ]
        steps += sum(trial.environment_steps for trials in results for trial in trials)
        reference_steps += sum(
            trial.environment_steps for trials in estimates for trial in trials
        )
        arms[f"{source}_uniform"] = [trial for trials in results for trial in trials]
        arms[f"{source}_reference_best"] = results[best]
        outcomes[source] = [[asdict(trial) for trial in trials] for trials in results]
        if "outcome_scores" in row:
            chosen = row["outcome_scores"][source]["chosen_index"]
            arms[f"{source}_outcome"] = (
                arms["direct"] if chosen is None else results[chosen]
            )
        if source == "flow":
            arms.update(
                {choice: results[index] for choice, index in row["choices"].items()}
            )
        else:
            arms["replay_nearest"] = results[0]

    return {
        "layout_id": layout.id,
        "depth": depth,
        "goal": row["goal"],
        "arms": {
            name: {
                "success": sum(trial.success for trial in trials) / len(trials),
                "arrival": sum(trial.arrived is True for trial in trials) / len(trials),
                "success_after_miss": sum(
                    trial.arrived is False and trial.success for trial in trials
                )
                / len(trials),
            }
            for name, trials in arms.items()
        },
        "outcomes": outcomes,
        "achieved_targets": achieved_targets,
        "diagnostic_collection_steps": len(achieved.actions),
        "reference_choices": reference_choices,
        "evaluation_steps": steps,
        "reference_steps": reference_steps,
    }


def _summarize(outcomes: list[dict], bank: list[dict]) -> dict:
    summary = {}
    for depth in grid.DEPTHS:
        rows = [row for row in outcomes if row["depth"] == depth]
        direct = np.array([row["arms"]["direct"]["success"] for row in rows])
        layout_count = len(rows) // len(grid.GOALS)
        draws = np.random.default_rng(20).integers(
            layout_count, size=(2000, layout_count)
        )
        rates = {
            name: np.array([row["arms"][name]["success"] for row in rows])
            for name in rows[0]["arms"]
        }

        def interval(delta):
            by_layout = delta.reshape(layout_count, len(grid.GOALS)).mean(axis=1)
            return np.quantile(by_layout[draws].mean(axis=1), (0.025, 0.975)).tolist()

        arms = {}
        for name, successes in rates.items():
            scores = [row["arms"][name] for row in rows]
            delta = successes - direct
            arms[name] = {
                "terminal_success": float(successes.mean()),
                "arrival": float(np.mean([part["arrival"] for part in scores])),
                "success_after_miss": float(
                    np.mean([part["success_after_miss"] for part in scores])
                ),
                "paired_difference_from_direct": float(delta.mean()),
                "layout_bootstrap_95": interval(delta),
                "cases_better_than_direct": int((delta > 0).sum()),
            }
        contrasts = {}
        comparisons = (
            ("lowest_energy", "flow_uniform"),
            ("lowest_energy", "highest_energy"),
            ("lowest_forecast_mse", "flow_uniform"),
            ("flow_reference_best", "flow_uniform"),
            ("flow_reference_best", "replay_reference_best"),
            ("deterministic_forecast", "goal_blind_forecast"),
        )
        if "flow_outcome" in rates:
            comparisons += (
                ("flow_outcome", "flow_uniform"),
                ("flow_outcome", "lowest_energy"),
                ("flow_outcome", "flow_reference_best"),
                ("replay_outcome", "replay_nearest"),
                ("replay_outcome", "replay_reference_best"),
                ("flow_outcome", "replay_outcome"),
            )
        for method, control in comparisons:
            delta = rates[method] - rates[control]
            contrasts[f"{method}_vs_{control}"] = {
                "difference": float(delta.mean()),
                "layout_bootstrap_95": interval(delta),
            }
        summary[str(depth)] = {
            "cases": len(rows),
            "arms": arms,
            "paired_contrasts": contrasts,
            "prefix_lengths": {
                source: dict(
                    Counter(
                        len(candidate["targets"])
                        for row in bank
                        if row["depth"] == depth
                        for candidate in row[source]
                    )
                )
                for source in ("flow", "replay")
            },
        }
        if "flow_outcome" in rates:
            summary[str(depth)]["outcome_direct_choices"] = {
                source: sum(
                    row["outcome_scores"][source]["chosen_index"] is None
                    for row in bank
                    if row["depth"] == depth
                )
                for source in ("flow", "replay")
            }
    return summary


def run(
    out: Path,
    training: TrainingConfig,
    evaluation: EvaluationConfig,
    *,
    collection_seed: int,
    demonstrations: int,
    flow_updates: int,
    selector_repeats: int | None = None,
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
    train_layouts = grid.layouts("train")
    if selector_repeats is not None and (
        selector_repeats < 1 or evaluation.candidates >= len(train_layouts)
    ):
        raise ValueError(
            "Use positive selector repeats and leave one training layout out of each replay bank"
        )
    layouts = layouts[: evaluation.layouts]
    out.mkdir(parents=True, exist_ok=False)
    windows, replay, steps = _collect(
        out, demonstrations, collection_seed, evaluation.slip_probability
    )
    codec, predictor, goal_blind = fit(
        windows["train"], windows["dev"], SPEC, training, out / "checkpoints"
    )
    completion = FrozenCompletion(SPEC, training, {"kind": SCHEMA}, codec, predictor)
    packed, targets = pack_windows(windows["train"], SPEC, training)
    forecast = completion.forecast(packed)
    fitted_flow, scale = flow.fit(
        flow.route_conditions(packed, forecast, len(grid.GOALS)),
        completion.encode(targets) - forecast,
        seed=stream_seed(training.seed, 4153),
        updates=flow_updates,
        out=out / "checkpoints" / "flow",
    )
    weights_path = out / WEIGHTS_FILE
    save_weights(
        weights_path,
        {
            "codec": weights(codec),
            "conditioned": weights(predictor),
            "goal_blind": weights(goal_blind),
            "flow": weights(fitted_flow),
            "scale": scale.numpy(),
        },
    )

    def declare(case_layouts):
        for layout, depth, goal, lane in _cases(case_layouts):
            yield (
                layout,
                _declare(
                    completion,
                    goal_blind,
                    fitted_flow,
                    scale,
                    replay[depth, goal],
                    layout,
                    depth,
                    goal,
                    lane.goal,
                    evaluation,
                ),
            )

    selector = None
    selector_metadata = {}
    selector_steps = 0
    if selector_repeats is not None:
        training_attempts = [
            outcome_scoring.collect_attempts(
                layout,
                row,
                _slips(layout, evaluation, "selector_training", selector_repeats),
            )
            for layout, row in declare(train_layouts)
        ]
        with (out / "selector-training-attempts.jsonl").open(
            "w", encoding="utf-8"
        ) as stream:
            for row in training_attempts:
                stream.write(json.dumps(row) + "\n")
        selector = fit_attempt_heads(training_attempts)
        selector_path = out / "selector.npz"
        np.savez(
            selector_path,
            mean=selector.mean,
            scale=selector.scale,
            attempt=selector.attempt,
        )
        selector_steps = sum(
            trial["environment_steps"]
            for row in training_attempts
            for candidate in row["candidates"]
            for trial in candidate["trials"]
        )
        selector_metadata = {
            "outcome_selector": {
                "schema": outcome_scoring.FEATURE_SCHEMA,
                "repeats": selector_repeats,
                "cases": len(training_attempts),
                "weights_sha256": sha256(selector_path.read_bytes()).hexdigest(),
                "stream": "selector_training",
            },
        }
    declared = list(declare(layouts))
    bank = [row for _, row in declared]
    if selector is not None:
        for layout, row in declared:
            row["outcome_scores"] = outcome_scoring.score_pools(selector, layout, row)
    bank_path = out / "candidate-bank.json"
    write_json(bank_path, bank)
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            **selector_metadata,
            "code_revision": code_revision(),
            "compute": describe(),
            "versions": {
                name: version(name) for name in ("torch", "lightning", "numpy")
            },
            "model": asdict(SPEC),
            "observation_schema": "dependency_grid_public_xy_key_locations_depth_inventory_v1",
            "training": asdict(training),
            "evaluation": asdict(evaluation),
            "collection_seed": collection_seed,
            "query_steps": dataset.QUERY_STEPS,
            "deadlines": experiment.DEADLINES,
            "layout_split_seed": grid.LAYOUT_SPLIT_SEED,
            "layouts": {
                **{
                    split: [layout.id for layout in grid.layouts(split)]
                    for split in windows
                },
                "evaluation": [layout.id for layout in layouts],
            },
            "demonstrations_per_case": demonstrations,
            "flow_updates": flow_updates,
            "flow_sampling_steps": FLOW_STEPS,
            "trajectories_sha256": sha256(
                (out / "trajectories.jsonl").read_bytes()
            ).hexdigest(),
            "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
            "candidate_bank_sha256": sha256(bank_path.read_bytes()).hexdigest(),
        },
    )
    outcomes = [_evaluate(layout, row, evaluation) for layout, row in declared]
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
        "diagnostic_collection_environment_steps": sum(
            row["diagnostic_collection_steps"] for row in outcomes
        ),
        "by_depth": _summarize(outcomes, bank),
    }
    if selector is not None:
        summary["selector_training_environment_steps"] = selector_steps
    write_json(out / "summary.json", summary)
    return summary
