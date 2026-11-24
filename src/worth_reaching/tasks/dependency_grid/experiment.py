"""Run predeclared waypoint interventions from common dependency-grid starts."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from platform import python_version
from random import Random

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.tasks.dependency_grid import environment as grid

BUDGET = 20
DEADLINES = (7, 14)


@dataclass(frozen=True, slots=True)
class Attempt:
    success: bool
    arrived: bool | None
    arrival_steps: tuple[int, ...]
    attempt_steps: int
    environment_steps: int


def attempt(
    layout: grid.Layout,
    depth: int,
    goal: grid.Position,
    targets: tuple[grid.Position, ...],
    slips: tuple[bool, ...],
) -> Attempt:
    state = grid.State()
    elapsed = 0
    arrivals = []

    def pursue(target: grid.Position, deadline: int) -> bool:
        nonlocal state, elapsed
        while elapsed < deadline and state.position not in (goal, target):
            action = (
                grid.WAIT
                if slips[elapsed]
                else grid.track(layout, state.position, target)
            )
            state = grid.step(layout, depth, state, action)
            elapsed += 1
        return state.position == target

    for index, target in enumerate(targets):
        deadline = DEADLINES[index]
        if not pursue(target, min(deadline, len(slips))):
            break
        arrivals.append(elapsed)
    attempt_steps = elapsed
    success = pursue(goal, len(slips))
    return Attempt(
        success,
        len(arrivals) == len(targets) if targets else None,
        tuple(arrivals),
        attempt_steps,
        elapsed,
    )


def _targets(layout: grid.Layout) -> dict[str, tuple[grid.Position, ...]]:
    candidates = {"direct": ()}
    for name, lane in zip(grid.GOALS, layout.lanes, strict=True):
        candidates[f"{name}_entry"] = (lane.entry,)
        candidates[f"{name}_inner"] = (lane.inner,)
        candidates[f"{name}_entry_then_inner"] = (lane.entry, lane.inner)
        candidates[f"{name}_inner_then_entry"] = (lane.inner, lane.entry)
    return candidates


def _summarize(decisions: list[dict]) -> dict:
    groups = defaultdict(lambda: defaultdict(list))
    switches = defaultdict(dict)
    for row in decisions:
        depth, goal = row["depth"], row["goal"]
        for name, trials in row["trials"].items():
            groups[(depth, goal)][name].extend(trials)
        reference = "entry" if depth == 1 else "entry_then_inner"
        left, right = (
            sum(trial.success for trial in row["trials"][f"{side}_{reference}"])
            for side in grid.GOALS
        )
        switches[(depth, row["layout_id"])][goal] = left - right

    by_depth: dict[str, dict] = {}
    for (depth, goal), arms in groups.items():
        direct_rate = sum(row.success for row in arms["direct"]) / len(arms["direct"])
        results = {}
        for name, trials in arms.items():
            successes = sum(row.success for row in trials)
            rate = successes / len(trials)
            results[name] = {
                "attempts": len(trials),
                "successes": successes,
                "terminal_success_rate": rate,
                "difference_from_direct": rate - direct_rate,
                "arrivals": sum(row.arrived is True for row in trials),
                "miss_then_successes": sum(
                    row.arrived is False and row.success for row in trials
                ),
            }
        by_depth.setdefault(str(depth), {})[goal] = results
    return {
        "decisions": len(decisions),
        "by_depth_and_goal": by_depth,
        "layouts_with_goal_dependent_rank_reversal": {
            str(depth): sum(
                deltas["left"] > 0 > deltas["right"]
                for (candidate_depth, _), deltas in switches.items()
                if candidate_depth == depth
            )
            for depth in grid.DEPTHS
        },
        "evaluation_environment_steps": sum(
            trial.environment_steps
            for row in decisions
            for trials in row["trials"].values()
            for trial in trials
        ),
    }


def run(
    out: Path,
    *,
    split: str,
    layout_limit: int | None,
    repeats: int,
    seed: int,
    slip_probability: float,
) -> dict:
    layouts = grid.layouts(split)
    if layout_limit is not None and not 1 <= layout_limit <= len(layouts):
        raise ValueError(f"Expected 1 to {len(layouts)} layouts for {split}")
    if repeats < 1 or seed < 0 or not 0 <= slip_probability < 1:
        raise ValueError(
            "Expected positive repeats, a nonnegative seed and slip in [0, 1)"
        )
    layouts = layouts[:layout_limit]
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "dependency_grid_attempts_v1",
            "config": {
                "split": split,
                "repeats": repeats,
                "seed": seed,
                "slip_probability": slip_probability,
                "budget": BUDGET,
                "deadlines": DEADLINES,
                "layout_split_seed": grid.LAYOUT_SPLIT_SEED,
            },
            "controller": "shortest geometric path with gated environment transitions",
            "code_revision": code_revision(),
            "python": python_version(),
            "layouts": {
                layout.id: {"lanes": [asdict(lane) for lane in layout.lanes]}
                for layout in layouts
            },
        },
    )
    decisions = []
    with (out / "decisions.jsonl").open("w", encoding="utf-8") as stream:
        for layout in layouts:
            rng = Random(f"{seed}:{layout.id}")
            slips = [
                tuple(rng.random() < slip_probability for _ in range(BUDGET))
                for _ in range(repeats)
            ]
            candidates = _targets(layout)
            for depth in grid.DEPTHS:
                for name, lane in zip(grid.GOALS, layout.lanes, strict=True):
                    trials = {
                        arm: [
                            attempt(layout, depth, lane.goal, targets, draws)
                            for draws in slips
                        ]
                        for arm, targets in candidates.items()
                    }
                    row = {
                        "layout_id": layout.id,
                        "start": grid.START,
                        "depth": depth,
                        "goal": name,
                        "goal_cell": lane.goal,
                        "candidates": candidates,
                        "trials": trials,
                    }
                    decisions.append(row)
                    logged_trials = {
                        arm: [asdict(trial) for trial in rows]
                        for arm, rows in trials.items()
                    }
                    stream.write(
                        json.dumps({**row, "trials": logged_trials, "slips": slips})
                        + "\n"
                    )
    result = {"layouts": len(layouts), **_summarize(decisions)}
    write_json(out / "summary.json", result)
    return result
