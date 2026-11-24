"""A public grid with switchable goals and one or two locked gates per lane."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from random import Random

Position = tuple[int, int]
SIZE = 7
START = (1, 3)
MOVES = ((1, 0), (-1, 0), (0, -1), (0, 1))
WAIT = len(MOVES)
GOALS = ("left", "right")
DEPTHS = (1, 2)
LAYOUT_SPLIT_SEED = 2718
SPLITS = {"train": slice(0, 32), "dev": slice(32, 48), "test": slice(48, 64)}


@dataclass(frozen=True, slots=True)
class Lane:
    column: int
    entry_row: int
    inner_on_route: bool

    @property
    def entry(self) -> Position:
        return self.entry_row, self.column

    @property
    def inner(self) -> Position:
        if self.inner_on_route:
            return 3, self.column
        return 3, self.column + (1 if self.column < START[1] else -1)

    @property
    def goal(self) -> Position:
        return SIZE - 1, self.column


@dataclass(frozen=True, slots=True)
class Layout:
    lanes: tuple[Lane, Lane]

    @property
    def id(self) -> str:
        parts = (
            f"{lane.column}{lane.entry_row}{int(lane.inner_on_route)}"
            for lane in self.lanes
        )
        return "dependency-grid-v1-" + "-".join(parts)

    def walk(self, position: Position, action: int) -> Position:
        if action == WAIT:
            return position
        dy, dx = MOVES[action]
        destination = position[0] + dy, position[1] + dx
        return destination if self.is_free(destination) else position

    def is_free(self, position: Position) -> bool:
        row, column = position
        if not (0 <= row < SIZE and 0 <= column < SIZE):
            return False
        if row in (2, 4) and all(column != lane.column for lane in self.lanes):
            return False
        return row <= 2 or column != START[1]


@dataclass(frozen=True, slots=True)
class State:
    position: Position = START
    keys: frozenset[Position] = frozenset()


def step(layout: Layout, depth: int, state: State, action: int) -> State:
    destination = layout.walk(state.position, action)
    row, column = destination
    for lane in layout.lanes:
        if column == lane.column:
            if row == 2 and depth == 2 and lane.entry not in state.keys:
                return state
            required = lane.inner if depth == 2 else lane.entry
            if row == 4 and required not in state.keys:
                return state
        if destination in (lane.entry, lane.inner):
            return State(destination, state.keys | {destination})
    return State(destination, state.keys)


@lru_cache(maxsize=8192)
def track(layout: Layout, position: Position, target: Position) -> int:
    """Follow a shortest geometric route, treating gates as passable."""
    if position == target:
        return WAIT
    queue = deque([(position, None)])
    seen = {position}
    while queue:
        cell, first_action = queue.popleft()
        for action in range(len(MOVES)):
            destination = layout.walk(cell, action)
            if destination in seen:
                continue
            first = action if first_action is None else first_action
            if destination == target:
                return first
            seen.add(destination)
            queue.append((destination, first))
    raise ValueError(f"Target cell {target} is outside the traversable geometry")


def layouts(split: str) -> list[Layout]:
    params = tuple(product((0, 1), (False, True)))
    left = [Lane(column, row, inner) for column in (0, 1) for row, inner in params]
    right = [Lane(column, row, inner) for column in (5, 6) for row, inner in params]
    catalog = [Layout((a, b)) for a, b in product(left, right)]
    Random(LAYOUT_SPLIT_SEED).shuffle(catalog)
    return catalog[SPLITS[split]]
