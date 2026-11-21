"""Pinned BabyAI KeyCorridor adapter and its public-observation controller."""

from dataclasses import dataclass
from typing import Any

import gymnasium as gym
from minigrid.core.actions import Actions
from minigrid.core.constants import OBJECT_TO_IDX, STATE_TO_IDX

from worth_reaching.contracts import Action, Observation, PolicyInput

ENVIRONMENT_ID = "BabyAI-KeyCorridorS3R1-v0"
TASK = "pick_up_ball"
CONTINUATION_ID = "keycorridor_front_tile_v1"


@dataclass(frozen=True, slots=True)
class Step:
    action: Action
    observation: Observation
    terminated: bool
    truncated: bool
    success: bool

    @property
    def finished(self) -> bool:
        return self.terminated or self.truncated


def _public_observation(native: dict[str, Any]) -> Observation:
    return Observation(
        image=tuple(
            tuple(tuple(tile) for tile in column) for column in native["image"].tolist()
        ),
        direction=int(native["direction"]),
        mission=native["mission"],
    )


class KeyCorridorEpisode:
    def __init__(self, *, seed: int, budget: int):
        self._env = gym.make(ENVIRONMENT_ID, max_steps=budget)
        native, _ = self._env.reset(seed=seed)
        self.initial_observation = _public_observation(native)
        self._finished = False

    def step(self, action: Action) -> Step:
        if self._finished:
            raise RuntimeError("Cannot step a finished episode")
        observation, reward, terminated, truncated, _ = self._env.step(
            Actions[action.value]
        )
        self._finished = bool(terminated or truncated)
        return Step(
            action=action,
            observation=_public_observation(observation),
            terminated=bool(terminated),
            truncated=bool(truncated),
            success=bool(terminated and reward > 0),
        )

    def close(self) -> None:
        self._env.close()


def continue_toward_ball(policy_input: PolicyInput) -> Action:
    image = policy_input.observations[-1].image
    object_id, _, state = image[len(image) // 2][-2]
    if object_id == OBJECT_TO_IDX["wall"]:
        return Action.RIGHT
    if object_id == OBJECT_TO_IDX["ball"]:
        return Action.PICKUP
    if object_id == OBJECT_TO_IDX["door"] and state != STATE_TO_IDX["open"]:
        return Action.TOGGLE
    return Action.FORWARD
