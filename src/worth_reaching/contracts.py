"""Public inputs for the KeyCorridor diagnostic."""

from dataclasses import dataclass
from enum import StrEnum


class Action(StrEnum):
    LEFT = "left"
    RIGHT = "right"
    FORWARD = "forward"
    PICKUP = "pickup"
    DROP = "drop"
    TOGGLE = "toggle"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class Observation:
    image: tuple[tuple[tuple[int, ...], ...], ...]
    direction: int
    mission: str


@dataclass(frozen=True, slots=True)
class PolicyInput:
    observations: tuple[Observation, ...]
    actions: tuple[Action, ...]
    task: str
    remaining_actions: int
    allowed_actions: tuple[Action, ...] = tuple(Action)
