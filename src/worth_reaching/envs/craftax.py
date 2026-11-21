"""Craftax-Classic symbolic observations and the iron-pickaxe terminal task."""

from dataclasses import dataclass
from typing import Literal

import jax
import numpy as np
from craftax.craftax_classic.constants import Achievement, Action, BlockType
from craftax.craftax_classic.envs.craftax_state import EnvState
from craftax.craftax_classic.envs.craftax_symbolic_env import (
    CraftaxClassicSymbolicEnvNoAutoReset,
)
from numpy.typing import NDArray

TASK = "make_iron_pickaxe"
RULES_VERSION = 1
OBSERVATION_SCHEMA = "craftax-classic-symbolic-v1"
ACTION_NAMES = {action.value: action.name.lower() for action in Action}
Outcome = Literal["success", "death", "timeout"]


@dataclass(frozen=True, slots=True)
class CraftaxStep:
    observation: NDArray[np.float32]
    remaining_actions: int
    outcome: Outcome | None


class CraftaxClassic:
    """Own the native environments and compiled functions for one action budget."""

    def __init__(self, budget: int):
        if not 0 < budget < 2**31 - 1:
            raise ValueError("Craftax budget must be between 1 and 2147483646")
        self.budget = budget
        native = CraftaxClassicSymbolicEnvNoAutoReset()
        params = native.default_params.replace(max_timesteps=budget)
        self.environment_id = native.name
        self._reset = jax.jit(lambda key: native.reset_env(key, params))
        self._observe = jax.jit(native.get_obs)

        @jax.jit
        def advance(key: jax.Array, state: EnvState, action: int):
            observation, next_state, _, _, _ = native.step_env(
                key, state, action, params
            )
            achieved = next_state.achievements[Achievement.MAKE_IRON_PICKAXE.value]
            achievement = (
                achieved & ~state.achievements[Achievement.MAKE_IRON_PICKAXE.value]
            )
            tile = next_state.map[tuple(next_state.player_position)]
            death = (next_state.player_health <= 0) | (tile == BlockType.LAVA.value)
            return observation, next_state, achievement, death

        self._advance = advance

    def reset(self, key: jax.Array) -> "CraftaxEpisode":
        observation, state = self._reset(key)
        return CraftaxEpisode(self, state, np.asarray(observation))

    def restore(self, snapshot: EnvState) -> "CraftaxEpisode":
        """Create a fresh evaluation branch from a live simulator snapshot."""
        return CraftaxEpisode(self, snapshot, np.asarray(self._observe(snapshot)))


class CraftaxEpisode:
    def __init__(
        self, env: CraftaxClassic, state: EnvState, observation: NDArray[np.float32]
    ):
        self._env = env
        self._state = state
        self._remaining = env.budget - int(state.timestep)
        self._finished = False
        self.observation = observation

    def snapshot(self) -> EnvState:
        """Retain the immutable native state on the evaluation side."""
        if self._finished:
            raise RuntimeError("Cannot branch from a finished episode")
        return self._state

    def step(self, action: int, *, key: jax.Array) -> CraftaxStep:
        if self._finished:
            raise RuntimeError("Cannot step a finished episode")
        if action not in ACTION_NAMES:
            raise ValueError(f"Invalid Craftax action: {action}")
        observation, self._state, success, death = self._env._advance(
            key, self._state, action
        )
        self.observation = np.asarray(observation)
        self._remaining -= 1
        outcome: Outcome | None = None
        if success:
            outcome = "success"
        elif death:
            outcome = "death"
        elif self._remaining == 0:
            outcome = "timeout"
        self._finished = outcome is not None
        return CraftaxStep(self.observation, self._remaining, outcome)
