"""Generate multiple latent routes and test declared waypoint attempts on the grid."""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.completion import CompletionInput
from worth_reaching.envs import grid_routes as grid
from worth_reaching.models import route_flow
from worth_reaching.models.completion import ModelSpec, pack_inputs
from worth_reaching.models.parallel import describe

DEMONSTRATION_SEED = 8112
FAR_ROUTE_RATE = 0.25
LOW_SUPPORT = 0.25
RELIABLE = 0.5
FLOW_STEPS = 12


@dataclass(frozen=True, slots=True)
class ExecutionConfig:
    seed: int = 7
    updates: int = 800
    candidates: int = 8
    repeats: int = 8
    slip_probability: float = 0.15
    split: str = "dev"
    layouts: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.seed < 2**32:
            raise ValueError("Use an unsigned 32-bit seed")
        if min(self.updates, self.candidates, self.repeats) < 1 or (
            self.layouts is not None and self.layouts < 1
        ):
            raise ValueError(
                "Update, candidate, repeat and layout counts must be positive"
            )
        if not 0 <= self.slip_probability <= 1 or self.split not in ("dev", "test"):
            raise ValueError("Use a probability in [0, 1] and a dev or test split")


@dataclass(frozen=True, slots=True)
class Waypoint:
    cell: grid.Position
    deadline: int


@dataclass(frozen=True, slots=True)
class Attempt:
    waypoint_attempted: bool
    arrived_step: int | None
    success: bool
    passage: str | None


def _geometry(
    public: CompletionInput,
) -> tuple[grid.Position, tuple[int, int], grid.Position]:
    y, x, left, right, red_y, red_x, blue_y, blue_x = (
        int(value) for value in np.rint(public.observations[-1] * (grid.SIZE - 1))
    )
    goal = (red_y, red_x) if public.goal == grid.GOALS[0] else (blue_y, blue_x)
    return (y, x), (left, right), goal


def _passage_family(column: int, goal: grid.Position) -> str:
    return "near" if (column < grid.SIZE // 2) == (goal[1] < grid.SIZE // 2) else "far"


def rollout(
    public: CompletionInput,
    waypoint: Waypoint | None,
    slips: np.ndarray,
    *,
    replay: tuple[int, ...] | None = None,
    passage_slips: np.ndarray | None = None,
    terminal_policy: np.ndarray | None = None,
) -> tuple[Attempt, tuple[int, ...]]:
    """Execute a declared target and its continuation under a supplied repeat stream."""
    position, passages, goal = _geometry(public)
    budget = public.remaining_actions
    if (
        len(slips) != budget
        or (replay is not None and len(replay) != budget)
        or (passage_slips is not None and len(passage_slips) != budget)
    ):
        raise ValueError("Rollout and replay must cover the whole action budget")
    if waypoint is not None and (
        waypoint.cell in (position, goal)
        or not grid.is_free(waypoint.cell, passages)
        or not 0 < waypoint.deadline < budget
    ):
        raise ValueError(
            "Expected an unachieved intermediate cell and an interior deadline"
        )
    arrived = passage = None
    actions = []
    for index, slip in enumerate(slips):
        attempting = (
            waypoint is not None and arrived is None and index < waypoint.deadline
        )
        if replay is not None:
            action = replay[index]
        elif not attempting and terminal_policy is not None:
            action = int(terminal_policy[budget - index, *position])
        else:
            destination = waypoint.cell if attempting else goal
            _, path = grid.shortest_route(position, destination, passages)
            action = path[0] if path else grid.WAIT
        actions.append(action)
        next_position = grid.move(position, grid.WAIT if slip else action, passages)
        if (
            passage_slips is None
            or not passage_slips[index]
            or not grid.risky_entry(position, next_position)
        ):
            position = next_position
        if passage is None and position[0] == grid.WALL_ROW:
            passage = _passage_family(position[1], goal)
        if attempting and position == waypoint.cell:
            arrived = index + 1
    return Attempt(waypoint is not None, arrived, position == goal, passage), tuple(
        actions
    )


def _pick_waypoint(
    public: CompletionInput, decoded: np.ndarray
) -> tuple[int, Waypoint] | None:
    start, passages, goal = _geometry(public)
    seen = {start}
    eligible = []
    cells = np.rint(decoded[:, :2] * (grid.SIZE - 1)).astype(int)
    for index, (row, column) in enumerate(cells):
        cell = (int(row), int(column))
        if cell == goal:
            break
        if cell not in seen and grid.is_free(cell, passages):
            deadline = min(
                int(public.query_steps[index]) + 1, public.remaining_actions - 1
            )
            eligible.append((index, Waypoint(cell, deadline)))
            seen.add(cell)
    return eligible[len(eligible) // 2] if eligible else None


def _public(world: grid.GridWorld, reverse: bool, goal: str) -> CompletionInput:
    observation = grid.grid_observation(
        world.start, world.passages, world.markers(reverse)
    )
    return CompletionInput(
        observation[None],
        np.empty(0, dtype=np.int32),
        goal,
        grid.BUDGET,
        np.arange(1, grid.BUDGET, dtype=np.int32),
    )


def _demonstrations(
    starts: list[tuple[grid.GridWorld, bool]], *, reference: bool
) -> tuple[list[CompletionInput], np.ndarray, Counter]:
    rng = np.random.default_rng([DEMONSTRATION_SEED, int(reference)])
    inputs, targets, counts = [], [], Counter()
    for world, reverse in starts:
        markers = world.markers(reverse)
        for goal, destination in zip(grid.GOALS, markers, strict=True):
            include_far = rng.random() < FAR_ROUTE_RATE
            for passage in world.passages:
                family = _passage_family(passage, destination)
                if family == "far" and not include_far:
                    continue
                positions, actions = grid.shortest_route(
                    world.start, destination, (passage,)
                )
                waits = grid.BUDGET - len(actions)
                if waits < 0:
                    continue
                route = positions + (destination,) * waits
                inputs.append(_public(world, reverse, goal))
                targets.append(
                    np.stack(
                        [
                            grid.grid_observation(position, world.passages, markers)
                            for position in route[1:-1]
                        ]
                    )
                )
                counts[family] += 1
    return inputs, np.stack(targets), counts


def _candidate(
    name: str,
    public: CompletionInput,
    slips: np.ndarray,
    route: np.ndarray | None = None,
    decoded: np.ndarray | None = None,
) -> dict:
    picked = _pick_waypoint(public, decoded) if decoded is not None else None
    index, waypoint = picked if picked is not None else (None, None)
    trials = [asdict(rollout(public, waypoint, stream)[0]) for stream in slips]
    candidate = {
        "id": name,
        "waypoint": asdict(waypoint) if waypoint is not None else None,
        "latent_target": route[index].tolist() if index is not None else None,
        "trials": trials,
    }
    if name.startswith("flow_"):
        _, template = rollout(
            public, waypoint, np.zeros(public.remaining_actions, dtype=bool)
        )
        candidate["open_loop"] = [
            asdict(rollout(public, waypoint, stream, replay=template)[0])
            for stream in slips
        ]
    return candidate


def _rates(trials: list[dict]) -> dict[str, float | int | None]:
    attempted = sum(row["waypoint_attempted"] for row in trials)
    arrivals = sum(row["arrived_step"] is not None for row in trials)
    success = sum(row["success"] for row in trials)
    missed_success = sum(
        row["success"] and row["waypoint_attempted"] and row["arrived_step"] is None
        for row in trials
    )
    count = len(trials)
    return {
        "attempts": count,
        "waypoint_attempts": attempted,
        "arrival_rate": arrivals / attempted if attempted else None,
        "terminal_success_rate": success / count,
        "success_without_arrival": missed_success,
    }


def summarize(decisions: list[dict], reference: Counter) -> dict[str, object]:
    arms = {
        name: []
        for name in (
            "direct",
            "deterministic",
            "flow_first",
            "flow_feedback",
            "flow_replay",
        )
    }
    support = {name: reference[name] / reference.total() for name in ("near", "far")}
    metrics = Counter()
    by_layout: dict[str, list[float]] = {}
    for decision in decisions:
        direct, deterministic, *flow = decision["candidates"]
        arms["direct"].extend(direct["trials"])
        arms["deterministic"].extend(deterministic["trials"])
        arms["flow_first"].extend(flow[0]["trials"])
        distinct = set()
        reliable = {family: set() for family in support}
        successes_delta = []
        for candidate in flow:
            trials, replay = candidate["trials"], candidate["open_loop"]
            arms["flow_feedback"].extend(trials)
            arms["flow_replay"].extend(replay)
            successes_delta.extend(
                int(a["success"]) - int(b["success"])
                for a, b in zip(trials, replay, strict=True)
            )
            if candidate["waypoint"] is None:
                metrics["invalid_proposals"] += 1
                continue
            waypoint = candidate["waypoint"]
            target = (tuple(waypoint["cell"]), waypoint["deadline"])
            distinct.add(target)
            valid = Counter(
                row["passage"]
                for row in trials
                if row["arrived_step"] is not None and row["success"]
            )
            for family in support:
                if valid[family] >= RELIABLE * len(trials):
                    reliable[family].add(target)
        metrics["multiple_waypoints"] += len(distinct) >= 2
        metrics["both_reliable_routes"] += any(
            near != far for near in reliable["near"] for far in reliable["far"]
        )
        metrics["reliable_low_support_route"] += any(
            support[family] <= LOW_SUPPORT and targets
            for family, targets in reliable.items()
        )
        by_layout.setdefault(grid.layout_id(decision["world_id"]), []).append(
            float(np.mean(successes_delta))
        )
    layout_deltas = np.asarray([np.mean(values) for values in by_layout.values()])
    draws = np.random.default_rng(32).integers(
        len(layout_deltas), size=(2000, len(layout_deltas))
    )
    arms_summary = {name: _rates(trials) for name, trials in arms.items()}
    arms_summary["direct"] = {
        key: value
        for key, value in arms_summary["direct"].items()
        if key in ("attempts", "terminal_success_rate")
    }
    return {
        "cases": len(decisions),
        "layouts": len(by_layout),
        "reference_counts": dict(reference),
        "reference_support": support,
        "candidate_coverage": {
            key: metrics[key]
            for key in (
                "invalid_proposals",
                "multiple_waypoints",
                "both_reliable_routes",
                "reliable_low_support_route",
            )
        },
        "arms": arms_summary,
        "feedback_minus_replay_layout_bootstrap_95": np.quantile(
            layout_deltas[draws].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "evaluation_environment_steps": (
            len(arms["direct"])
            + len(arms["deterministic"])
            + len(arms["flow_feedback"])
            + len(arms["flow_replay"])
        )
        * grid.BUDGET,
    }


def load_grid_completion(checkpoint: Path) -> route_flow.FrozenCompletion:
    completion = route_flow.FrozenCompletion.load(checkpoint)
    spec = ModelSpec(8, grid.ACTION_COUNT, grid.GOALS, grid.BUDGET)
    if (
        completion.spec != spec
        or completion.config.queries_per_window < grid.BUDGET - 1
        or completion.source.get("kind") != "two_passage_grid_v2"
        or completion.source.get("split_seed") != grid.SPLIT_SEED
    ):
        raise ValueError(
            "Expected a grid checkpoint with matching splits and full masked routes"
        )
    return completion


def run(checkpoint: Path, out: Path, config: ExecutionConfig) -> dict[str, object]:
    completion = load_grid_completion(checkpoint)
    spec = completion.spec
    starts = grid.grid_starts(
        include_test=config.split == "test", include_reference=True
    )
    evaluation = starts[config.split]
    if config.layouts is not None:
        evaluation = evaluation[: config.layouts * len(grid.STARTS)]
    cases = [
        (world.world_id, _public(world, reverse, goal))
        for world, reverse in evaluation
        for goal in grid.GOALS
    ]
    training, targets, training_counts = _demonstrations(
        starts["train"], reference=False
    )
    _, _, reference_counts = _demonstrations(starts["reference"], reference=True)
    packed = pack_inputs(training, spec, completion.config)
    forecast = completion.forecast(packed)
    conditions = route_flow.route_conditions(packed, forecast, len(spec.goals))
    model, weights, scale = route_flow.fit(
        conditions,
        completion.encode(targets) - forecast,
        seed=config.seed,
        updates=config.updates,
    )
    packed = pack_inputs([public for _, public in cases], spec, completion.config)
    forecast = completion.forecast(packed)
    conditions = route_flow.route_conditions(packed, forecast, len(spec.goals))
    samples = route_flow.sample(
        model,
        weights,
        forecast,
        conditions,
        scale,
        jax.random.fold_in(jax.random.key(config.seed), 8113),
        candidates=config.candidates,
        steps=FLOW_STEPS,
    )
    decoded, sample_decoded = map(
        np.asarray, (completion.decode(forecast), completion.decode(samples))
    )
    if not np.isfinite(decoded).all() or not np.isfinite(sample_decoded).all():
        raise ValueError("Route prediction produced non-finite observations")
    forecast, samples = map(np.asarray, (forecast, samples))
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng([config.seed, 8114])
    decisions = []
    for index, (world_id, public) in enumerate(cases):
        slips = rng.random((config.repeats, grid.BUDGET)) < config.slip_probability
        candidates = [
            _candidate("direct", public, slips),
            _candidate("deterministic", public, slips, forecast[index], decoded[index]),
            *[
                _candidate(f"flow_{n}", public, slips, latent, observations)
                for n, (latent, observations) in enumerate(
                    zip(samples[index], sample_decoded[index], strict=True)
                )
            ],
        ]
        decisions.append(
            {
                "world_id": world_id,
                "goal": public.goal,
                "start": public.observations[0].tolist(),
                "repeat_slips": slips.tolist(),
                "candidates": candidates,
            }
        )
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for decision in decisions:
            stream.write(json.dumps(decision, separators=(",", ":")) + "\n")
    (out / "flow.msgpack").write_bytes(
        serialization.to_bytes({"params": weights, "scale": scale})
    )
    write_json(
        out / "manifest.json",
        {
            "schema": "grid_latent_route_attempts_v1",
            "compute": describe(),
            "config": asdict(config),
            "code_revision": code_revision(),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_weights_sha256": sha256(
                (checkpoint / "weights.msgpack").read_bytes()
            ).hexdigest(),
            "training_demonstrations": dict(training_counts),
            "demonstration_seed": DEMONSTRATION_SEED,
            "far_route_sampling_probability": FAR_ROUTE_RATE,
            "flow_steps": FLOW_STEPS,
            "layout_split_seed": grid.SPLIT_SEED,
            "low_support_threshold": LOW_SUPPORT,
            "joint_arrival_terminal_threshold": RELIABLE,
            "layouts": {
                key: sorted({grid.layout_id(world.world_id) for world, _ in rows})
                for key, rows in {
                    "train": starts["train"],
                    "reference": starts["reference"],
                    config.split: evaluation,
                }.items()
            },
        },
    )
    summary = summarize(decisions, reference_counts)
    write_json(out / "summary.json", summary)
    return summary
