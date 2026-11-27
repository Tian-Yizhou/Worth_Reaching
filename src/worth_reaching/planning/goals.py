"""Pursue declared intermediate goals in order, then hand control to the terminal goal.

A goal is whatever target the task's executor consumes: a grid cell, a graph node, a
decoded observation, or a latent vector. The arrival rule decides when the current
observation satisfies it. Every task shares the same stage semantics: a stage ends on
arrival, at its deadline, or when the episode terminates, and a miss abandons the
remaining stages so the executor continues toward the terminal goal.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

Observation = TypeVar("Observation")
Target = TypeVar("Target")
Arrival = Callable[[Observation, Target], bool]
Outcome = Literal["arrived", "timeout", "abandoned", "terminated"]


@dataclass(frozen=True, slots=True)
class Stage(Generic[Target]):
    target: Target
    deadline: int | None = None
    """Absolute step by which the target must be reached, or None for the budget."""


@dataclass(frozen=True, slots=True)
class StageEvent:
    index: int
    start_step: int
    end_step: int
    deadline: int | None
    outcome: Outcome


class GoalSequence(Generic[Observation, Target]):
    """Track which declared stage is active and record how each one ended."""

    def __init__(
        self,
        stages: Sequence[Stage[Target]],
        arrived: Arrival[Observation, Target],
    ):
        self.stages = list(stages)
        self.arrived = arrived
        self.stage = 0
        self.stage_start = 0
        self.events: list[StageEvent] = []

    @property
    def active(self) -> Stage[Target] | None:
        return self.stages[self.stage] if self.stage < len(self.stages) else None

    def advance(self, observation: Observation, step: int) -> Stage[Target] | None:
        """Close stages the observation satisfies or whose deadline has passed."""
        while (stage := self.active) is not None:
            if self.arrived(observation, stage.target):
                self._close(step, "arrived")
                continue
            if stage.deadline is not None and step >= stage.deadline:
                self._close(step, "timeout")
                self.stage = len(self.stages)
            break
        return self.active

    def abandon(self, step: int) -> None:
        """Give up the active stage, for example when the executor cannot pursue it."""
        if self.active is not None:
            self._close(step, "abandoned")
            self.stage = len(self.stages)

    def replace_remaining(self, stages: Sequence[Stage[Target]]) -> None:
        """Swap the stages after the active position, as in suffix replanning."""
        self.stages = self.stages[: self.stage] + list(stages)

    def finish(self, observation: Observation, step: int) -> None:
        """Score the final observation and mark any unfinished stage as terminated."""
        self.advance(observation, step)
        if self.active is not None:
            self._close(step, "terminated")
            self.stage = len(self.stages)

    @property
    def arrivals(self) -> list[StageEvent]:
        return [event for event in self.events if event.outcome == "arrived"]

    @property
    def arrived_all(self) -> bool:
        return len(self.arrivals) == len(self.stages)

    @property
    def handoff_step(self) -> int:
        """The step at which control passed to the terminal goal."""
        return self.events[-1].end_step if self.events else 0

    def _close(self, step: int, outcome: Outcome) -> None:
        self.events.append(
            StageEvent(
                self.stage,
                self.stage_start,
                step,
                self.stages[self.stage].deadline,
                outcome,
            )
        )
        self.stage += 1
        self.stage_start = step
