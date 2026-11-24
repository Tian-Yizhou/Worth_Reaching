"""Replay public histories and evaluate controlled continuations in Craftax."""

from dataclasses import dataclass
from typing import Literal, Protocol

import jax
import numpy as np
from numpy.typing import NDArray

from worth_reaching.tasks.craftax.dataset import EpisodeSpec, episode_keys
from worth_reaching.tasks.craftax.environment import (
    CraftaxClassic,
    CraftaxEpisode,
    Outcome,
)


class Continuation(Protocol):
    def initial_memory(self) -> jax.Array: ...
    def observe(
        self, observation: NDArray[np.float32], carry: jax.Array
    ) -> jax.Array: ...
    def sample(
        self, observation: NDArray[np.float32], carry: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, int]: ...


@dataclass(frozen=True, slots=True)
class Attempt:
    outcome: Outcome | Literal["horizon_limit"]
    steps: int
    remaining_actions: int


def replay_prefix(
    env: CraftaxClassic,
    source: EpisodeSpec,
    prefix_steps: int,
    trace: dict,
    policy: Continuation,
) -> tuple[CraftaxEpisode, jax.Array]:
    """Return a live episode and policy memory before its current observation."""
    if not 0 <= prefix_steps <= len(trace["steps"]):
        raise ValueError(f"Invalid replay prefix for {source.episode_id}")
    reset_key, transition_key = episode_keys(source.world_seed)
    episode = env.reset(reset_key)
    expected = trace["initial_observation"]
    memory = policy.initial_memory()
    for index in range(prefix_steps + 1):
        if not np.array_equal(
            episode.observation, np.asarray(expected, dtype=np.float32)
        ):
            raise ValueError(
                f"Replay diverged at observation {index} for {source.episode_id}"
            )
        if index == prefix_steps:
            break
        memory = policy.observe(episode.observation, memory)
        recorded = trace["steps"][index]
        step = episode.step(
            recorded["action"], key=jax.random.fold_in(transition_key, index)
        )
        if step.outcome is not None:
            raise ValueError(
                f"Replay terminated before the branch point for {source.episode_id}"
            )
        expected = recorded["observation"]
    return episode, memory


def run_attempt(
    episode: CraftaxEpisode,
    memory: jax.Array,
    policy: Continuation,
    *,
    horizon: int,
    repeat_key: jax.Array,
    forced_actions: tuple[int, ...] = (),
) -> Attempt:
    """Measure a direct policy or a declared action prefix followed by that policy."""
    if horizon < 1 or len(forced_actions) > horizon:
        raise ValueError("A positive attempt horizon must fit the forced actions")
    transition_key, policy_key = jax.random.split(repeat_key)
    for index in range(horizon):
        if index < len(forced_actions):
            memory = policy.observe(episode.observation, memory)
            action = forced_actions[index]
        else:
            memory, action = policy.sample(
                episode.observation, memory, jax.random.fold_in(policy_key, index)
            )
        step = episode.step(action, key=jax.random.fold_in(transition_key, index))
        if step.outcome is not None:
            return Attempt(step.outcome, index + 1, step.remaining_actions)
    return Attempt("horizon_limit", horizon, step.remaining_actions)
