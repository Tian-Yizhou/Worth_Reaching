"""Update a latent route flow from matched attempts and score it with a fixed reranker."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.completion import CompletionInput
from worth_reaching.envs import grid_routes as grid
from worth_reaching.evaluation import grid_execution as execution
from worth_reaching.evaluation.attempt_selection import fit_attempt_heads
from worth_reaching.models import route_flow
from worth_reaching.models.completion import pack_inputs
from worth_reaching.models.parallel import describe

ARRIVAL_FLOOR = 0.25
SLIP = 0.15


@dataclass(frozen=True, slots=True)
class FeedbackConfig:
    seed: int = 7
    updates: int = 250
    candidates: int = 8
    repeats: int = 8
    passage_slip: float = 0.85
    split: str = "dev"
    layouts: int | None = None

    def __post_init__(self) -> None:
        execution.ExecutionConfig(
            self.seed,
            self.updates,
            self.candidates,
            self.repeats,
            SLIP,
            self.split,
            self.layouts,
        )
        if not 0 <= self.passage_slip <= 1:
            raise ValueError("Passage slip probability must be in [0, 1]")


def _public_features(
    public: CompletionInput, waypoint: execution.Waypoint | None
) -> list[int]:
    start, passages, goal = execution._geometry(public)
    if waypoint is None:
        prefix, length, deadline = (start,), 0, 0
    else:
        prefix, actions = grid.shortest_route(start, waypoint.cell, passages)
        length, deadline = len(actions), waypoint.deadline
        prefix = prefix[: deadline + 1]
    suffix, actions = grid.shortest_route(prefix[-1], goal, passages)
    path = (*prefix, *suffix[1:])
    risky_entries = sum(grid.risky_entry(a, b) for a, b in zip(path, path[1:]))
    return [
        len(path) - 1,
        length,
        deadline - length,
        len(actions),
        risky_entries,
        sum(grid.risky_entry(a, b) for a, b in zip(prefix, prefix[1:])),
    ]


@lru_cache(maxsize=128)
def _optimal_policy(
    target: grid.Position, passages: tuple[int, int], passage_slip: float
) -> np.ndarray:
    """Finite-horizon oracle with public transition probabilities."""
    shape = (grid.BUDGET + 1, grid.SIZE, grid.SIZE)
    values = np.zeros(shape)
    values[0, *target] = 1.0
    actions = np.full(shape, grid.WAIT, dtype=np.int32)
    cells = [
        (y, x)
        for y in range(grid.SIZE)
        for x in range(grid.SIZE)
        if grid.is_free((y, x), passages)
    ]
    for remaining in range(1, grid.BUDGET + 1):
        previous = values[remaining - 1]
        for cell in cells:
            for action in range(grid.ACTION_COUNT):
                next_cell = grid.move(cell, action, passages)
                wait = (
                    1 - (1 - SLIP) * (1 - passage_slip)
                    if grid.risky_entry(cell, next_cell)
                    else SLIP
                )
                value = wait * previous[cell] + (1 - wait) * previous[next_cell]
                if value > values[remaining, *cell]:
                    values[remaining, *cell] = value
                    actions[remaining, *cell] = action
    return actions


def _case_inputs(starts):
    return [
        (world.world_id, execution._public(world, reverse, goal))
        for world, reverse in starts
        for goal in grid.GOALS
    ]


def _condition(completion, inputs):
    packed = pack_inputs(inputs, completion.spec, completion.config)
    forecast = completion.forecast(packed)
    return forecast, route_flow.route_conditions(
        packed, forecast, len(completion.spec.goals)
    )


def _propose(completion, model, weights, scale, forecast, conditions, key, count):
    latents = route_flow.sample(
        model,
        weights,
        forecast,
        conditions,
        scale,
        key,
        candidates=count,
        steps=execution.FLOW_STEPS,
    )
    decoded = np.asarray(completion.decode(latents))
    if not np.isfinite(decoded).all():
        raise ValueError("Route prediction produced non-finite observations")
    return np.asarray(latents), decoded


def _streams(config: FeedbackConfig, cases: int, *, training: bool) -> np.ndarray:
    rng = np.random.default_rng([config.seed, 818 if training else 919])
    probabilities = np.array([SLIP, config.passage_slip])
    return rng.random((cases, config.repeats, grid.BUDGET, 2)) < probabilities


def _trials(public, waypoint, masks, policy=None) -> list[dict]:
    results = []
    for repeat in masks:
        attempt, _ = execution.rollout(
            public,
            waypoint,
            repeat[:, 0],
            passage_slips=repeat[:, 1],
            terminal_policy=policy,
        )
        results.append(
            {
                "success": attempt.success,
                "arrived": attempt.arrived_step is not None
                if waypoint is not None
                else None,
                "passage": attempt.passage,
            }
        )
    return results


def _plans(public, paths):
    return [execution._pick_waypoint(public, path) for path in paths]


def _training_bank(cases, forecast, latents, decoded, masks, scale):
    decisions, sources, targets, credit, weights = [], [], [], [], []
    for case, ((world, public), paths, repeat) in enumerate(
        zip(cases, decoded, masks, strict=True)
    ):
        candidates = []
        for index, proposal in enumerate(_plans(public, paths)):
            if proposal is None:
                continue
            step, waypoint = proposal
            trials = _trials(public, waypoint, repeat)
            candidates.append(
                {
                    "id": f"flow_{index}",
                    "features": _public_features(public, waypoint),
                    "step": step,
                    "waypoint": asdict(waypoint),
                    "trials": trials,
                }
            )
        decisions.append(
            {
                "world_id": world,
                "goal": public.goal,
                "start": public.observations[0].tolist(),
                "candidates": candidates,
            }
        )
        if not candidates:
            continue
        rates = np.asarray(
            [
                np.mean([trial["success"] for trial in row["trials"]])
                for row in candidates
            ]
        )
        for rate, candidate in zip(rates, candidates, strict=True):
            reached = np.mean([trial["arrived"] for trial in candidate["trials"]])
            advantage = rate - rates.mean()
            if reached < ARRIVAL_FLOOR or advantage <= 0:
                continue
            slot = candidate["step"]
            proposal_index = int(candidate["id"].removeprefix("flow_"))
            normalized = np.zeros(forecast.shape[1:], dtype=np.float32)
            normalized[slot] = (
                latents[case, proposal_index, slot] - forecast[case, slot]
            ) / scale
            mask = np.zeros_like(normalized)
            mask[slot] = 1
            targets.append(normalized.ravel())
            credit.append(mask.ravel())
            sources.append(case)
            weights.append(advantage * reached)
    if not any(decision["candidates"] for decision in decisions):
        raise ValueError("The frozen proposer produced no usable training waypoints")
    return decisions, sources, targets, credit, weights


def _evaluate(cases, decoded, masks, reranker, rare_families):
    rows, selected, successes, arrivals = [], defaultdict(list), [], []
    rare_cover = both_cover = invalid = missed_success = 0
    for (world, public), paths, repeat in zip(cases, decoded, masks, strict=True):
        plans = [
            proposal[1] if proposal is not None else None
            for proposal in _plans(public, paths)
        ]
        features = np.array(
            [_public_features(public, plan) for plan in plans], dtype=float
        )
        terminal, reach, *_ = reranker.predict(features)
        legal = [index for index, plan in enumerate(plans) if plan is not None]
        invalid += len(plans) - len(legal)
        choices = {
            "first": 0,
            "arrival": max(legal, key=reach.__getitem__, default=0),
            "terminal": max(legal, key=terminal.__getitem__, default=0),
        }
        # The selector receives public features before the evaluator runs any trial.
        trials = [_trials(public, plan, repeat) for plan in plans]
        family_targets = defaultdict(set)
        for plan, attempts in zip(plans, trials, strict=True):
            successes.extend(trial["success"] for trial in attempts)
            arrivals.extend(
                trial["arrived"] for trial in attempts if trial["arrived"] is not None
            )
            missed_success += sum(
                trial["success"] and trial["arrived"] is False for trial in attempts
            )
            if plan is not None:
                for family in ("near", "far"):
                    if sum(
                        trial["arrived"]
                        and trial["success"]
                        and trial["passage"] == family
                        for trial in attempts
                    ) >= execution.RELIABLE * len(attempts):
                        family_targets[family].add(plan)
        rare_cover += any(family_targets[family] for family in rare_families)
        both_cover += any(
            a != b for a in family_targets["near"] for b in family_targets["far"]
        )
        for strategy, index in choices.items():
            selected[strategy].append([trial["success"] for trial in trials[index]])
        rows.append(
            {
                "world_id": world,
                "goal": public.goal,
                "choices": choices,
                "waypoints": [
                    asdict(plan) if plan is not None else None for plan in plans
                ],
                "trials": trials,
            }
        )
    outcomes = {arm: np.array(values, dtype=float) for arm, values in selected.items()}
    return (
        rows,
        outcomes,
        {
            "terminal_success": {
                arm: float(values.mean()) for arm, values in outcomes.items()
            },
            "pool_terminal_success": float(np.mean(successes)),
            "pool_arrival": float(np.mean(arrivals)) if arrivals else None,
            "invalid_proposals": invalid,
            "pool_success_after_miss": missed_success,
            "cases_with_reliable_low_support_route": rare_cover,
            "cases_with_both_route_families": both_cover,
        },
    )


def _paired_interval(cases, first, second) -> list[float]:
    layout = [grid.layout_id(world) for world, _ in cases]
    grouped = defaultdict(list)
    for group, delta in zip(layout, np.mean(first - second, axis=1), strict=True):
        grouped[group].append(delta)
    effects = np.asarray([np.mean(values) for values in grouped.values()])
    draws = np.random.default_rng(32).integers(len(effects), size=(2000, len(effects)))
    return np.quantile(effects[draws].mean(axis=1), [0.025, 0.975]).tolist()


def _write_rows(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")


def run(checkpoint: Path, flow_run: Path, out: Path, config: FeedbackConfig) -> dict:
    completion = execution.load_grid_completion(checkpoint)
    flow_manifest = json.loads((flow_run / "manifest.json").read_text(encoding="utf-8"))
    checkpoint_hash = sha256((checkpoint / "weights.msgpack").read_bytes()).hexdigest()
    if (
        flow_manifest.get("schema") != "grid_latent_route_attempts_v1"
        or flow_manifest.get("checkpoint_weights_sha256") != checkpoint_hash
        or flow_manifest.get("layout_split_seed") != grid.SPLIT_SEED
    ):
        raise ValueError(
            "Expected a flow run trained with the supplied grid completion checkpoint"
        )
    flow_bytes = (flow_run / "flow.msgpack").read_bytes()
    payload = serialization.msgpack_restore(flow_bytes)
    weights, scale = payload["params"], payload["scale"]
    model = route_flow.RouteFlow((grid.BUDGET - 1) * completion.config.latent_size)
    starts = grid.grid_starts(
        include_test=config.split == "test", include_reference=True
    )
    train = _case_inputs(starts["train"])
    evaluation = starts[config.split]
    if config.layouts is not None:
        evaluation = evaluation[: config.layouts * len(grid.STARTS)]
    cases = _case_inputs(evaluation)
    key = jax.random.key(config.seed)
    train_forecast, train_context = _condition(
        completion, [public for _, public in train]
    )
    training_latents, training_paths = _propose(
        completion,
        model,
        weights,
        scale,
        train_forecast,
        train_context,
        jax.random.fold_in(key, 201),
        config.candidates,
    )
    train_masks = _streams(config, len(train), training=True)
    training_rows, sources, targets, credit, advantages = _training_bank(
        train,
        np.asarray(train_forecast),
        training_latents,
        training_paths,
        train_masks,
        scale,
    )
    reranker = fit_attempt_heads(training_rows)
    demonstration_public, observations, _ = execution._demonstrations(
        starts["train"], reference=False
    )
    demo_forecast, demo_context = _condition(completion, demonstration_public)
    demo_residual = ((completion.encode(observations) - demo_forecast) / scale).reshape(
        (len(observations), model.size)
    )
    credit_count = len(sources)
    if not sources:
        sources, targets, credit, advantages = (
            [0],
            [np.zeros(model.size)],
            [np.zeros(model.size)],
            [1.0],
        )
    probability = np.asarray(advantages) / sum(advantages)
    source_indices = np.asarray(sources)
    feedback_targets = np.asarray(targets, dtype=np.float32)
    credit_mask = np.asarray(credit, dtype=np.float32)
    continued, updated = route_flow.refine(
        model,
        weights,
        demo_context,
        demo_residual,
        train_context[source_indices],
        feedback_targets,
        credit_mask,
        probability,
        seed=config.seed,
        updates=config.updates,
    )
    _, _, reference = execution._demonstrations(starts["reference"], reference=True)
    rare = {
        key
        for key, count in reference.items()
        if count / reference.total() <= execution.LOW_SUPPORT
    }
    forecast, conditions = _condition(completion, [public for _, public in cases])
    evaluation_masks = _streams(config, len(cases), training=False)
    direct, optimal = [], []
    for (_, public), repeat in zip(cases, evaluation_masks, strict=True):
        _, passages, goal = execution._geometry(public)
        direct.extend(_trials(public, None, repeat))
        optimal.extend(
            _trials(
                public,
                None,
                repeat,
                _optimal_policy(goal, passages, config.passage_slip),
            )
        )
    out.mkdir(parents=True, exist_ok=False)
    arms, choices = {}, {}
    for name, params in (
        ("frozen", weights),
        ("demonstrations", continued),
        ("feedback", updated),
    ):
        _, decoded = _propose(
            completion,
            model,
            params,
            scale,
            forecast,
            conditions,
            jax.random.fold_in(key, 202),
            config.candidates,
        )
        rows, choices[name], arms[name] = _evaluate(
            cases, decoded, evaluation_masks, reranker, rare
        )
        _write_rows(out / f"{name}.jsonl", rows)
    _write_rows(out / "training_attempts.jsonl", training_rows)
    np.savez_compressed(
        out / "feedback_targets.npz",
        case_indices=source_indices[:credit_count],
        targets=feedback_targets[:credit_count],
        mask=credit_mask[:credit_count],
        probabilities=probability[:credit_count],
    )
    np.savez_compressed(
        out / "repeat_masks.npz",
        training=train_masks,
        evaluation=evaluation_masks,
        direct=np.asarray([trial["success"] for trial in direct]).reshape(
            len(cases), config.repeats
        ),
        risk_aware_direct=np.asarray([trial["success"] for trial in optimal]).reshape(
            len(cases), config.repeats
        ),
    )
    (out / "weights.msgpack").write_bytes(
        serialization.to_bytes(
            {
                "demonstrations": continued,
                "feedback": updated,
                "scale": scale,
            }
        )
    )
    np.savez(
        out / "reranker.npz",
        mean=reranker.mean,
        scale=reranker.scale,
        attempt=reranker.attempt,
        reach=reranker.reach,
    )
    write_json(
        out / "manifest.json",
        {
            "schema": "grid_route_feedback_v1",
            "compute": describe(),
            "code_revision": code_revision(),
            "config": asdict(config),
            "checkpoint_sha256": checkpoint_hash,
            "flow_checkpoint_sha256": sha256(flow_bytes).hexdigest(),
            "base_flow_run": str(flow_run.resolve()),
            "ordinary_slip": SLIP,
            "arrival_floor": ARRIVAL_FLOOR,
            "layouts": {
                name: sorted({grid.layout_id(world.world_id) for world, _ in group})
                for name, group in {
                    "train": starts["train"],
                    config.split: evaluation,
                    "reference": starts["reference"],
                }.items()
            },
        },
    )
    summary = {
        "evaluation_cases": len(cases),
        "training_cases": len(train),
        "credited_waypoints": credit_count,
        "reference_passages": dict(reference),
        "arms": arms,
        "direct_terminal_success": float(
            np.mean([trial["success"] for trial in direct])
        ),
        "risk_aware_direct_terminal_success": float(
            np.mean([trial["success"] for trial in optimal])
        ),
        "paired_terminal_delta_layout_bootstrap_95": {
            "feedback_minus_frozen": _paired_interval(
                cases, choices["feedback"]["terminal"], choices["frozen"]["terminal"]
            ),
            "feedback_minus_demonstrations": _paired_interval(
                cases,
                choices["feedback"]["terminal"],
                choices["demonstrations"]["terminal"],
            ),
            "frozen_terminal_minus_arrival": _paired_interval(
                cases, choices["frozen"]["terminal"], choices["frozen"]["arrival"]
            ),
        },
        "training_environment_steps": sum(
            len(row["candidates"]) for row in training_rows
        )
        * config.repeats
        * grid.BUDGET,
        "evaluation_environment_steps": len(cases)
        * config.repeats
        * grid.BUDGET
        * (3 * config.candidates + 2),
    }
    write_json(out / "summary.json", summary)
    return summary
