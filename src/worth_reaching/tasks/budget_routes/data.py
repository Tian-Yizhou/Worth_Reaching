"""Collect public demonstration windows and transition observations."""

from collections import Counter, defaultdict
from dataclasses import dataclass
from random import Random

import numpy as np

from worth_reaching.data.completion import (
    CompletionExample,
    CompletionInput,
    CompletionTarget,
)
from worth_reaching.tasks.budget_routes import environment as env

QUERIES = (1, 2)
OBSERVATION_SIZE = len(env.EDGES) + 2
GOAL = "finish"


def observe(profile: int, state: env.State) -> np.ndarray:
    observation = np.zeros(OBSERVATION_SIZE, dtype=np.float32)
    observation[state.node] = 1
    observation[-2:] = profile, state.supplied
    return observation


def public_input(profile: int, state: env.State, budget: int) -> CompletionInput:
    return CompletionInput(
        observe(profile, state)[None],
        np.empty(0, dtype=np.int32),
        GOAL,
        budget,
        np.asarray(QUERIES, dtype=np.int32),
    )


@dataclass(frozen=True, slots=True)
class Demonstration:
    layout_id: str
    episode_id: str
    profile: int
    budget: int
    states: tuple[env.State, ...]
    actions: tuple[int, ...]

    @property
    def success(self) -> bool:
        return self.states[-1].node == env.GOAL

    def window(self) -> CompletionExample:
        queries = public_input(self.profile, self.states[0], self.budget)
        indices = np.minimum(queries.query_steps, len(self.actions))
        return CompletionExample(
            self.layout_id,
            self.episode_id,
            0,
            queries,
            CompletionTarget(
                np.stack([observe(self.profile, self.states[i]) for i in indices]),
                np.asarray([self.actions[i - 1] for i in indices], dtype=np.int32),
            ),
        )


def demonstrations(split: str, count: int, seed: int) -> list[Demonstration]:
    records = []
    for world in env.worlds(split):
        rng = Random(f"demonstrations:{seed}:{world.id}")
        for budget in env.BUDGETS:
            choices = [(env.SHORT, env.GOAL)]
            if budget >= 3:
                choices += [
                    (env.SUPPLY, env.BYPASS, env.GOAL),
                    (env.APPROACH, env.BYPASS, env.GOAL),
                    (env.GATE, env.EXIT, env.GOAL),
                ]
            for index in range(count):
                actions = tuple(rng.choice(choices))
                states = [world.start]
                for action in actions:
                    states.append(env.step(world, states[-1], action, rng.random()))
                records.append(
                    Demonstration(
                        world.id,
                        f"{world.id}:{budget}:{index}",
                        world.profile,
                        budget,
                        tuple(states),
                        actions,
                    )
                )
        # Local continuations supply the gate skill without a full reset-to-goal composition.
        for index in range(count):
            actions = (env.GATE, env.EXIT, env.GOAL)
            states = [env.State(env.SUPPLY, True)]
            for action in actions:
                states.append(env.step(world, states[-1], action, rng.random()))
            records.append(
                Demonstration(
                    world.id,
                    f"{world.id}:fragment:{index}",
                    world.profile,
                    len(actions),
                    tuple(states),
                    actions,
                )
            )
    return records


TransitionCounts = dict[tuple[int, env.State, int], Counter[env.State]]


def exploration(count: int, seed: int) -> tuple[TransitionCounts, int]:
    """Estimate primitive outcomes from complete public training-layout rollouts."""
    counts: TransitionCounts = defaultdict(Counter)
    steps = 0
    for world in env.worlds("train"):
        rng = Random(f"exploration:{seed}:{world.id}")
        for _ in range(count):
            state = world.start
            for _ in range(max(env.BUDGETS)):
                if not env.EDGES[state.node]:
                    break
                action = rng.choice(env.EDGES[state.node])
                new = env.step(world, state, action, rng.random())
                counts[world.profile, state, action][new] += 1
                steps += 1
                state = new
    return counts, steps
