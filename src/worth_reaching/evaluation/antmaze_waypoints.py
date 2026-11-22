"""Matched AntMaze comparison of direct execution and oracle waypoint chains."""

from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import file_digest
from pathlib import Path

import numpy as np

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.envs import antmaze
from worth_reaching.evaluation.antmaze import (
    EvalConfig,
    Policy,
    evaluation_starts,
    paired_metrics,
    rollout,
    summarize_episodes,
)
from worth_reaching.models.gcbc import load_policy

NEIGHBORS = ((-1, 0), (0, -1), (1, 0), (0, 1))
Cell = tuple[int, int]


@dataclass(frozen=True, slots=True)
class WaypointConfig:
    spacing: int = 1
    steps: int = 150
    arrival_radius: float = 1.0

    def __post_init__(self) -> None:
        if self.spacing < 1 or self.steps < 1 or not 0 < self.arrival_radius < np.inf:
            raise ValueError(
                "Waypoint spacing, deadline and arrival radius must be positive"
            )


def shortest_cells(distances: np.ndarray, start: Cell, goal: Cell) -> list[Cell]:
    """Follow OGBench's shortest-path distance map in its canonical neighbor order."""
    if distances[start] < 0 or distances[goal] != 0:
        raise ValueError("The AntMaze oracle could not connect the start and goal")
    path = [start]
    while path[-1] != goal:
        y, x = path[-1]
        remaining = distances[y, x] - 1
        next_cell = next(
            (
                (y + dy, x + dx)
                for dy, dx in NEIGHBORS
                if 0 <= y + dy < distances.shape[0]
                and 0 <= x + dx < distances.shape[1]
                and distances[y + dy, x + dx] == remaining
            ),
            None,
        )
        if next_cell is None:
            raise ValueError("The AntMaze oracle returned a broken shortest path")
        path.append(next_cell)
    return path


def waypoint_cells(path: list[Cell], spacing: int) -> list[Cell]:
    """Subsample long straight corridors while keeping every corner."""
    waypoints = []
    last = 0
    for index, (before, cell, after) in enumerate(
        zip(path, path[1:], path[2:]), start=1
    ):
        corner = (after[0] - cell[0], after[1] - cell[1]) != (
            cell[0] - before[0],
            cell[1] - before[1],
        )
        if corner or index - last >= spacing:
            waypoints.append(cell)
            last = index
    return waypoints


class FollowWaypoints:
    """Pass a target to the same actor until arrival or deadline."""

    def __init__(
        self, actor: Policy, targets: list[np.ndarray], config: WaypointConfig
    ):
        self.actor = actor
        self.targets = targets
        self.config = config
        self.index = 0
        self.start_step = 0
        self.closest = float("inf")
        self.attempts: list[dict] = []

    def __call__(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        if self.index >= len(self.targets):
            return self.actor(observation, goal)
        target = self.targets[self.index]
        self.closest = min(
            self.closest, float(np.linalg.norm(observation[:2] - target[:2]))
        )
        return self.actor(observation, target)

    def _record(self, step: int, outcome: str) -> None:
        self.attempts.append(
            {
                "waypoint": self.index,
                "start_step": self.start_step,
                "end_step": step,
                "outcome": outcome,
                "closest_distance": self.closest,
            }
        )

    def observe(self, observation: np.ndarray, step: int) -> None:
        if self.index >= len(self.targets):
            return
        distance = float(np.linalg.norm(observation[:2] - self.targets[self.index][:2]))
        self.closest = min(self.closest, distance)
        if distance <= self.config.arrival_radius:
            self._record(step, "arrived")
            self.index += 1
            self.start_step = step
            self.closest = float("inf")
        elif step - self.start_step >= self.config.steps:
            self._record(step, "timeout")
            self.index = len(self.targets)

    def finish(self, step: int) -> None:
        if self.index < len(self.targets) and step > self.start_step:
            self._record(step, "episode_end")


def run(
    checkpoint: Path,
    out: Path,
    evaluation: EvalConfig,
    waypoint: WaypointConfig,
    start_bank: Path | None,
) -> dict:
    actor = load_policy(checkpoint)
    with (checkpoint / "weights.msgpack").open("rb") as stream:
        weights_sha256 = file_digest(stream, "sha256").hexdigest()
    with (checkpoint / "normalization.npz").open("rb") as stream:
        normalization_sha256 = file_digest(stream, "sha256").hexdigest()
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "antmaze_oracle_waypoints_v1",
            "environment": antmaze.ENVIRONMENT,
            "versions": dict(
                zip(("ogbench", "gymnasium", "mujoco"), antmaze.VERSIONS, strict=True)
            ),
            "evaluation": asdict(evaluation),
            "waypoint": asdict(waypoint),
            "checkpoint": str(checkpoint.resolve()),
            "weights_sha256": weights_sha256,
            "normalization_sha256": normalization_sha256,
            "start_source": str(start_bank.resolve()) if start_bank else None,
            "inference_backend": "pytorch_cpu",
            "code_revision": code_revision(),
        },
    )
    environment = antmaze.AntMaze()
    try:
        starts = evaluation_starts(environment, evaluation, start_bank)
        antmaze.save_starts(out / "starts.npz", starts)
        native = environment.environment.unwrapped
        direct, guided = [], []
        for start in starts:
            start_xy, goal_xy = start.observation[:2], start.goal[:2]
            _, distances = native.get_oracle_subgoal(start_xy, goal_xy)
            path = shortest_cells(
                distances, native.xy_to_ij(start_xy), native.xy_to_ij(goal_xy)
            )
            cells = waypoint_cells(path, waypoint.spacing)
            targets = []
            for cell in cells:
                goal = start.goal.copy()
                goal[:2] = native.ij_to_xy(cell)
                goal.flags.writeable = False
                targets.append(goal)
            direct.append(rollout(environment, start, actor, evaluation.action_limit))
            follower = FollowWaypoints(actor, targets, waypoint)
            episode = rollout(
                environment,
                start,
                follower,
                evaluation.action_limit,
                after_step=follower.observe,
            )
            follower.finish(episode["steps"])
            guided.append(
                {
                    **episode,
                    "path": path,
                    "waypoint_cells": cells,
                    "attempts": follower.attempts,
                }
            )
    finally:
        environment.close()

    counts = Counter(
        attempt["outcome"] for row in guided for attempt in row["attempts"]
    )
    write_json(out / "episodes.json", {"direct": direct, "waypoints": guided})
    paired = paired_metrics(direct, guided)
    paired["only_waypoints"] = paired.pop("only_guided")
    summary = {
        "direct": summarize_episodes(direct),
        "waypoints": summarize_episodes(guided),
        "paired": paired,
        "waypoint_attempts": counts,
        "all_waypoints_reached": sum(
            len(row["waypoint_cells"])
            == sum(attempt["outcome"] == "arrived" for attempt in row["attempts"])
            for row in guided
        ),
        "terminal_success_after_miss": sum(
            row["success"]
            and any(
                attempt["outcome"] in ("timeout", "episode_end")
                for attempt in row["attempts"]
            )
            for row in guided
        ),
        "evaluation_environment_steps": sum(row["steps"] for row in (*direct, *guided)),
    }
    write_json(out / "summary.json", summary)
    return summary
