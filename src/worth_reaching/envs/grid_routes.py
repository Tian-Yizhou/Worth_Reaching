"""Paired shortest-path demonstrations in a small grid with two passages."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import product

import numpy as np

from worth_reaching.data.completion import (
    CompletionExample,
    CompletionInput,
    CompletionTarget,
    WindowConfig,
)

SIZE = 7
BUDGET = 14
GOALS = ("reach_red", "reach_blue")
MOVES = ((1, 0), (-1, 0), (0, -1), (0, 1))
ACTION_COUNT = len(MOVES) + 1
WAIT = len(MOVES)
WALL_ROW = 3
SPLIT_SEED = 813
LAYOUT_COUNTS = (32, 8, 16)
STARTS = tuple(product(range(3), range(2, 5)))
Position = tuple[int, int]


def layout_id(world_id: str) -> str:
    return world_id.split("-start-", 1)[0]


def is_free(position: Position, passages: tuple[int, ...]) -> bool:
    y, x = position
    return 0 <= y < SIZE and 0 <= x < SIZE and (y != WALL_ROW or x in passages)


def risky_entry(start: Position, destination: Position) -> bool:
    return (
        start[0] != WALL_ROW and destination[0] == WALL_ROW and destination[1] in (1, 5)
    )


def move(position: Position, action: int, passages: tuple[int, ...]) -> Position:
    if action == WAIT:
        return position
    dy, dx = MOVES[action]
    destination = (position[0] + dy, position[1] + dx)
    return destination if is_free(destination, passages) else position


@lru_cache(maxsize=16384)
def shortest_route(
    start: Position, goal: Position, passages: tuple[int, ...]
) -> tuple[tuple[Position, ...], tuple[int, ...]]:
    queue = deque([(start, (start,), ())])
    seen = {start}
    while queue:
        position, positions, actions = queue.popleft()
        if position == goal:
            return positions, actions
        for action in range(len(MOVES)):
            next_position = move(position, action, passages)
            if next_position not in seen:
                seen.add(next_position)
                queue.append(
                    (next_position, (*positions, next_position), (*actions, action))
                )
    raise ValueError("Grid goal is unreachable")


def grid_observation(
    position: Position, passages: tuple[int, int], markers: tuple[Position, Position]
) -> np.ndarray:
    return np.asarray(
        (*position, *passages, *markers[0], *markers[1]), dtype=np.float32
    ) / (SIZE - 1)


@dataclass(frozen=True, slots=True)
class GridWorld:
    start: Position
    passages: tuple[int, int]
    left_goal: Position
    right_goal: Position

    @property
    def world_id(self) -> str:
        layout = "-".join(map(str, (*self.passages, *self.left_goal, *self.right_goal)))
        return f"grid-v2-{layout}-start-{self.start[0]}-{self.start[1]}"

    def route(self, goal: Position) -> tuple[tuple[Position, ...], tuple[int, ...]]:
        positions, actions = shortest_route(self.start, goal, self.passages)
        waits = BUDGET - len(actions)
        if waits < 0:
            raise ValueError("Grid route exceeds the action budget")
        return positions + (goal,) * waits, actions + (WAIT,) * waits

    def markers(self, reverse_colors: bool) -> tuple[Position, Position]:
        return (
            (self.right_goal, self.left_goal)
            if reverse_colors
            else (self.left_goal, self.right_goal)
        )

    def demonstrations(
        self, *, reverse_colors: bool, training: bool
    ) -> list[CompletionExample]:
        markers = self.markers(reverse_colors)
        windows = [(0, BUDGET - 1)]
        if training:
            windows.extend(
                WindowConfig(seed=SPLIT_SEED, windows_per_episode=3).sample(
                    self.world_id, BUDGET
                )[1:]
            )
        examples = []
        for goal, destination in zip(GOALS, markers, strict=True):
            positions, actions = self.route(destination)
            observations = np.stack(
                [
                    grid_observation(position, self.passages, markers)
                    for position in positions
                ]
            )
            action_ids = np.asarray(actions, dtype=np.int32)
            for start, span in windows:
                end = start + span
                examples.append(
                    CompletionExample(
                        self.world_id,
                        f"{self.world_id}-{goal}",
                        start,
                        CompletionInput(
                            observations[: start + 1].copy(),
                            action_ids[:start].copy(),
                            goal,
                            BUDGET - start,
                            np.arange(1, span + 1, dtype=np.int32),
                        ),
                        CompletionTarget(
                            observations[start + 1 : end + 1].copy(),
                            action_ids[start:end].copy(),
                        ),
                    )
                )
        return examples


def grid_starts(
    *, include_test: bool = False, include_reference: bool = False
) -> dict[str, list[tuple[GridWorld, bool]]]:
    layouts = list(
        product(
            product(range(2), range(5, 7)),
            product(range(5, 7), range(2)),
            product(range(5, 7), range(5, 7)),
        )
    )
    permutation = np.random.default_rng(SPLIT_SEED).permutation(len(layouts))
    splits = {}
    start = 0
    counts = (*LAYOUT_COUNTS, len(layouts) - sum(LAYOUT_COUNTS))
    for split, count in zip(("train", "dev", "test", "reference"), counts, strict=True):
        if (split != "test" or include_test) and (
            split != "reference" or include_reference
        ):
            splits[split] = [
                (GridWorld(spawn, *layouts[layout]), (map_index + spawn_index) % 2 == 1)
                for map_index, layout in enumerate(permutation[start : start + count])
                for spawn_index, spawn in enumerate(STARTS)
            ]
        start += count
    return splits


def grid_demonstrations(
    *, include_test: bool = False
) -> dict[str, list[CompletionExample]]:
    return {
        split: [
            example
            for world, reverse in starts
            for example in world.demonstrations(
                reverse_colors=reverse, training=split == "train"
            )
        ]
        for split, starts in grid_starts(include_test=include_test).items()
    }
