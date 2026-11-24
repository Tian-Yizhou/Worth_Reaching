"""Stochastic routes with an optional supply stop and a shared terminal."""

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
from random import Random

START, SHORT, SUPPLY, APPROACH, BYPASS, GATE, EXIT, GOAL, FAILED = range(9)
EDGES = (
    (SHORT, SUPPLY, APPROACH, GATE),
    (GOAL,),
    (BYPASS, GATE),
    (BYPASS,),
    (GOAL,),
    (SUPPLY, EXIT),
    (GOAL,),
    (),
    (),
)
INTERMEDIATE = tuple(range(1, GOAL))
BUDGETS = (2, 3, 4)


@dataclass(frozen=True, slots=True)
class State:
    node: int = START
    supplied: bool = False


@dataclass(frozen=True, slots=True)
class World:
    id: str
    profile: int
    start: State
    chances: tuple[float, ...]


def worlds(split: str) -> list[World]:
    offsets = {"train": (0, 16), "dev": (16, 24), "test": (24, 32)}
    result = []
    for index in range(*offsets[split]):
        profile = index % 2
        rng = Random(f"budget-routes-v1:{index}")
        base = (
            (0.48, 0.64, 0.22, 0.96, 0.86)
            if profile == 0
            else (0.57, 0.72, 0.31, 0.98, 0.90)
        )
        result.append(
            World(
                f"routes-{index:02d}",
                profile,
                State(supplied=bool(index // 2 % 2)),
                tuple(
                    min(0.995, chance + rng.uniform(-0.015, 0.015)) for chance in base
                ),
            )
        )
    return result


def step(world: World, state: State, action: int, draw: float) -> State:
    if action not in EDGES[state.node]:
        raise ValueError("Expected a legal public graph edge")
    if action == GOAL:
        short, bypass, bare, prepared, _ = world.chances
        probability = {
            SHORT: short,
            BYPASS: bypass,
            EXIT: prepared if state.supplied else bare,
        }[state.node]
        return State(GOAL if draw < probability else FAILED, state.supplied)
    supplied = state.supplied or (action == SUPPLY and draw < world.chances[-1])
    return State(action, supplied)


@lru_cache(maxsize=128)
def track(start: int, target: int) -> int | None:
    """Pick the first legal edge of a shortest path using only the public graph."""
    queue = deque([(start, None)])
    seen = {start}
    while queue:
        node, first = queue.popleft()
        if node == target:
            return first
        for neighbor in EDGES[node]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append((neighbor, neighbor if first is None else first))
    return None


def next_action(state: State, route: tuple[int, ...]) -> tuple[int, tuple[int, ...]]:
    while route and state.node == route[0]:
        route = route[1:]
    if route:
        action = track(state.node, route[0])
        if action is not None:
            return action, route
    action = track(state.node, GOAL)
    if action is None:
        raise ValueError("Expected a path from a live state to the terminal")
    return action, ()
