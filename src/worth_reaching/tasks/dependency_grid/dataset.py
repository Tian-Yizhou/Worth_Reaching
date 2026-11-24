"""Collect terminal-directed trajectories and expose masked public windows."""

from collections import deque
from dataclasses import dataclass
from random import Random

import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.completion import (
    CompletionExample,
    CompletionInput,
    CompletionTarget,
)
from worth_reaching.tasks.dependency_grid import environment as grid
from worth_reaching.tasks.dependency_grid.experiment import BUDGET

QUERY_STEPS = (4, 8)
OBSERVATION_SIZE = 15


def observe(layout: grid.Layout, depth: int, state: grid.State) -> NDArray[np.float32]:
    key_cells = tuple(
        cell for lane in layout.lanes for cell in (lane.entry, lane.inner)
    )
    coordinates = [
        axis / (grid.SIZE - 1) for cell in (state.position, *key_cells) for axis in cell
    ]
    return np.asarray(
        [*coordinates, depth - 1, *(cell in state.keys for cell in key_cells)],
        dtype=np.float32,
    )


def public_input(layout: grid.Layout, depth: int, goal: str) -> CompletionInput:
    return CompletionInput(
        observe(layout, depth, grid.State())[None],
        np.empty(0, dtype=np.int32),
        goal,
        BUDGET,
        np.asarray(QUERY_STEPS, dtype=np.int32),
    )


def shortest_actions(
    layout: grid.Layout, depth: int, goal: grid.Position, action_order: list[int]
) -> tuple[int, ...]:
    """Use the simulator only to collect behavior labeled by its terminal goal."""
    start = grid.State()
    queue = deque([(start, ())])
    seen = {start}
    while queue:
        state, actions = queue.popleft()
        if state.position == goal:
            return actions
        for action in action_order:
            destination = grid.step(layout, depth, state, action)
            if destination not in seen:
                seen.add(destination)
                queue.append((destination, (*actions, action)))
    raise ValueError(f"Unreachable terminal goal for {layout.id}, depth {depth}")


@dataclass(frozen=True, slots=True)
class Trajectory:
    observations: NDArray[np.float32]
    actions: NDArray[np.int32]
    success: bool

    def window(self, layout_id: str, episode_id: str, goal: str) -> CompletionExample:
        queries = np.asarray(QUERY_STEPS, dtype=np.int32)
        # A finished episode stays at its final observation for later queries.
        return CompletionExample(
            layout_id,
            episode_id,
            0,
            CompletionInput(
                self.observations[:1].copy(),
                self.actions[:0].copy(),
                goal,
                BUDGET,
                queries,
            ),
            CompletionTarget(
                self.observations[np.minimum(queries, len(self.actions))],
                np.pad(
                    self.actions,
                    (0, max(0, QUERY_STEPS[-1] - len(self.actions))),
                    constant_values=grid.WAIT,
                )[queries - 1],
            ),
        )


def trajectory(
    layout: grid.Layout, depth: int, goal: grid.Position, rng: Random, slip: float
) -> Trajectory:
    order = list(range(len(grid.MOVES)))
    rng.shuffle(order)
    plan = shortest_actions(layout, depth, goal, order)
    state = grid.State()
    observations = [observe(layout, depth, state)]
    actions = []
    index = 0
    while index < len(plan) and len(actions) < BUDGET:
        action = plan[index]
        slipped = rng.random() < slip
        state = grid.step(layout, depth, state, grid.WAIT if slipped else action)
        index += not slipped
        actions.append(action)
        observations.append(observe(layout, depth, state))
    return Trajectory(
        np.stack(observations),
        np.asarray(actions, dtype=np.int32),
        state.position == goal,
    )
