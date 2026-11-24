"""Train and apply one outcome-based selector to both dependency-grid pools."""

from dataclasses import asdict

import numpy as np

from worth_reaching.evaluation.attempt_selection import ScoreHeads
from worth_reaching.tasks.dependency_grid import environment as grid
from worth_reaching.tasks.dependency_grid.experiment import BUDGET, DEADLINES, attempt

FEATURE_SCHEMA = "dependency_grid_public_prefix_v1"


def _distance(first: grid.Position, second: grid.Position) -> int:
    return abs(first[0] - second[0]) + abs(first[1] - second[1])


def features(
    layout: grid.Layout,
    depth: int,
    terminal: grid.Position,
    targets: tuple[grid.Position, ...],
) -> list[float]:
    """Describe the intervention in public, goal-relative geometry."""
    side = 1 if terminal[1] > grid.START[1] else -1
    scale = grid.SIZE - 1

    def coordinates(cell: grid.Position) -> tuple[float, float]:
        return cell[0] / scale, side * (cell[1] - grid.START[1]) / scale

    landmarks = sorted(
        (cell for lane in layout.lanes for cell in (lane.entry, lane.inner)),
        key=lambda cell: (_distance(cell, terminal), coordinates(cell)),
    )
    result = [float(depth), *coordinates(terminal)]
    for landmark in landmarks:
        result.extend(coordinates(landmark))
    previous = grid.START
    for index, deadline in enumerate(DEADLINES):
        if index >= len(targets):
            result.extend([0.0] * (6 + 2 * len(landmarks)))
            continue
        target = targets[index]
        result.extend(
            (
                1.0,
                *coordinates(target),
                _distance(previous, target) / scale,
                _distance(target, terminal) / scale,
                deadline / BUDGET,
            )
        )
        for landmark in landmarks:
            result.extend(
                (_distance(target, landmark) / scale, float(target == landmark))
            )
        previous = target
    return result


def collect_attempts(
    layout: grid.Layout, declaration: dict, repeats: list[tuple[bool, ...]]
) -> dict:
    depth, terminal = declaration["depth"], declaration["terminal"]
    targets = dict.fromkeys(
        [
            (),
            *(
                candidate["targets"]
                for source in ("flow", "replay")
                for candidate in declaration[source]
            ),
        ]
    )
    return {
        "layout_id": layout.id,
        "depth": depth,
        "goal": declaration["goal"],
        "replay_episode_ids": [row["episode_id"] for row in declaration["replay"]],
        "candidates": [
            {
                "id": f"prefix_{index}" if prefix else "direct",
                "targets": prefix,
                "features": features(layout, depth, terminal, prefix),
                "trials": [
                    asdict(attempt(layout, depth, terminal, prefix, slips))
                    for slips in repeats
                ],
            }
            for index, prefix in enumerate(targets)
        ],
    }


def score_pools(heads: ScoreHeads, layout: grid.Layout, declaration: dict) -> dict:
    depth, terminal = declaration["depth"], declaration["terminal"]
    banks = [declaration["flow"], declaration["replay"]]
    design = np.asarray(
        [
            features(layout, depth, terminal, targets)
            for targets in ((), *(row["targets"] for bank in banks for row in bank))
        ],
        dtype=float,
    )
    success = heads.predict(design)[0]
    direct = float(success[0])
    results = {"direct_probability": direct}
    offset = 1
    for source, bank in zip(("flow", "replay"), banks, strict=True):
        scores = success[offset : offset + len(bank)]
        index = int(scores.argmax())
        results[source] = {
            "probabilities": scores.tolist(),
            "chosen_index": index if scores[index] > direct else None,
        }
        offset += len(bank)
    return results
