"""Paired shortest-path demonstrations in a small grid with two passages."""

from collections import deque
from dataclasses import dataclass
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
SPLIT_SEED = 813
LAYOUT_COUNTS = (32, 8, 16)
STARTS = tuple(product(range(3), range(2, 5)))
Position = tuple[int, int]


def layout_id(world_id: str) -> str:
    return world_id.split("-start-", 1)[0]


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
        queue = deque([(self.start, (self.start,), ())])
        seen = {self.start}
        while queue:
            position, positions, actions = queue.popleft()
            if position == goal:
                waits = BUDGET - len(actions)
                if waits < 0:
                    raise RuntimeError("Grid route exceeds the action budget")
                return positions + (goal,) * waits, actions + (len(MOVES),) * waits
            for action, (dy, dx) in enumerate(MOVES):
                y, x = next_position = (position[0] + dy, position[1] + dx)
                if not (0 <= y < SIZE and 0 <= x < SIZE) or next_position in seen:
                    continue
                if y == 3 and x not in self.passages:
                    continue
                seen.add(next_position)
                queue.append(
                    (next_position, (*positions, next_position), (*actions, action))
                )
        raise RuntimeError("Grid goal is unreachable")

    def demonstrations(
        self, *, reverse_colors: bool, training: bool
    ) -> list[CompletionExample]:
        markers = (
            (self.right_goal, self.left_goal)
            if reverse_colors
            else (self.left_goal, self.right_goal)
        )
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
            observations = np.asarray(
                [
                    (*position, *self.passages, *markers[0], *markers[1])
                    for position in positions
                ],
                dtype=np.float32,
            ) / (SIZE - 1)
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


def grid_demonstrations(
    *, include_test: bool = False
) -> dict[str, list[CompletionExample]]:
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
    for split, count in zip(("train", "dev", "test"), LAYOUT_COUNTS, strict=True):
        if split == "test" and not include_test:
            break
        splits[split] = [
            example
            for map_index, layout in enumerate(permutation[start : start + count])
            for spawn_index, spawn in enumerate(STARTS)
            for example in GridWorld(spawn, *layouts[layout]).demonstrations(
                reverse_colors=(map_index + spawn_index) % 2 == 1,
                training=split == "train",
            )
        ]
        start += count
    return splits
