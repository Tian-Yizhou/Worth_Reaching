"""Replay a KeyCorridor prefix and evaluate a forced action."""

from collections.abc import Callable
from dataclasses import dataclass

from worth_reaching.contracts import Action, Observation, PolicyInput
from worth_reaching.envs.minigrid import TASK, KeyCorridorEpisode, Step


@dataclass(frozen=True, slots=True)
class Prefix:
    initial_observation: Observation
    steps: tuple[Step, ...]


@dataclass(frozen=True, slots=True)
class Attempt:
    intervention: Action
    steps: tuple[Step, ...]
    steps_elapsed: int
    remaining_actions: int
    success: bool
    outcome: str


def replay_prefix(episode: KeyCorridorEpisode, actions: tuple[Action, ...]) -> Prefix:
    steps = []
    for action in actions:
        step = episode.step(action)
        if step.finished:
            raise ValueError("The configured prefix ends the episode before branching")
        steps.append(step)
    return Prefix(episode.initial_observation, tuple(steps))


def evaluate_action(
    episode: KeyCorridorEpisode,
    prefix: Prefix,
    intervention: Action,
    *,
    budget: int,
    continuation: Callable[[PolicyInput], Action],
) -> Attempt:
    steps = [episode.step(intervention)]
    while not steps[-1].finished:
        history = prefix.steps + tuple(steps)
        policy_input = PolicyInput(
            observations=(prefix.initial_observation,)
            + tuple(step.observation for step in history),
            actions=tuple(step.action for step in history),
            task=TASK,
            remaining_actions=budget - len(history),
        )
        steps.append(episode.step(continuation(policy_input)))

    last = steps[-1]
    if last.success:
        outcome = "success"
    elif last.terminated:
        outcome = "failure"
    else:
        outcome = "timeout"
    return Attempt(
        intervention=intervention,
        steps=tuple(steps),
        steps_elapsed=len(steps),
        remaining_actions=budget - len(prefix.steps) - len(steps),
        success=last.success,
        outcome=outcome,
    )
