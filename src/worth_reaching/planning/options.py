"""Choose, pursue and interrupt latent options with a budget-conditioned value.

At a decision the world model samples latent futures at each of its query offsets.
Every sample at offset `k` is a candidate option "pursue `z` for at most `k` actions",
and the encoded terminal goal, pursued for the whole remaining budget, is one more
candidate. The option value ranks them, by the mean over its heads plus `risk` times
their spread, and one latent-goal executor pursues the choice. Nothing defines what a
subgoal is or when it has been reached.

Each step the planner re-scores the current option with its remaining duration and
the cached candidates from the current state, and switches when another beats it by
more than `margin` times the spread. It samples new candidates only when the current
option's duration is spent, its value falls by more than `margin` times the spread from
its value at selection, or the encoded observation departs from the latent path the
chosen sample predicted.
"""

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from worth_reaching.models.training import stream_seed, tensor
from worth_reaching.planning.receding_goals import completion_context


@dataclass(frozen=True, slots=True)
class PlannerConfig:
    time_cost: float = 0.001
    samples: int = 8
    """Flow samples per decision; each contributes one candidate per query offset."""
    risk: float = 0.0
    """Rank by mean + risk × spread across heads: negative is cautious."""
    margin: float = 1.0
    """Switch or regenerate only on value changes beyond this many spreads."""
    subgoals: bool = True
    """Sample candidates from the world model; otherwise pursue the goal only."""
    goal_candidate: bool = True
    interrupt: bool = True
    slots: tuple[int, ...] | None = None
    """Restrict durations to these query offsets, for a fixed-duration ablation."""
    surprise: float | None = None
    """Squared latent distance from the predicted path that triggers new candidates."""

    def __post_init__(self) -> None:
        if (
            self.time_cost < 0
            or self.samples < 1
            or self.margin < 0
            or not (self.subgoals or self.goal_candidate)
            or (self.slots is not None and not self.slots)
            or (self.surprise is not None and self.surprise <= 0)
        ):
            raise ValueError(
                "Use a non-negative time cost and margin, positive samples and at "
                "least one source of candidates"
            )


@dataclass(slots=True)
class Option:
    latent: np.ndarray
    duration: int
    """Actions left when the option was last (re)selected."""
    start: int
    goal: bool = False
    path: tuple[tuple[int, np.ndarray], ...] = ()
    """Predicted latents at earlier offsets, as (step, latent)."""


@dataclass(slots=True)
class Selected:
    option: Option
    value: float
    spread: float


@dataclass
class PlannerLog:
    generations: int = 0
    switches: int = 0
    triggers: dict[str, int] = field(default_factory=dict)
    """New decisions by cause: start, duration, value or surprise."""
    decisions: list[dict] = field(default_factory=list)


class OptionPlanner:
    def __init__(
        self,
        completion,
        value,
        act_latent: Callable[[np.ndarray, np.ndarray], np.ndarray],
        key: int,
        limit: int,
        config: PlannerConfig,
        *,
        action_size: int,
    ):
        self.completion, self.value, self.act_latent = completion, value, act_latent
        self.key, self.limit, self.config = key, limit, config
        self.history: deque = deque(maxlen=completion.config.history)
        self.last_action = np.zeros(action_size, dtype=np.float32)
        self.step = 0
        self.current: Selected | None = None
        self.cache: list[Option] = []
        self.log = PlannerLog()

    def encode(self, observation: np.ndarray) -> np.ndarray:
        completion = self.completion
        normalized = (observation - completion.mean) / completion.scale
        return completion.codec.encode(tensor(normalized)[None]).detach().numpy()[0]

    def __call__(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        self.history.append((observation, self.last_action))
        remaining = self.limit - self.step
        state, target = self.encode(observation), self.encode(goal)
        trigger = self._trigger(state, target, remaining)
        if trigger is not None:
            self.log.triggers[trigger] = self.log.triggers.get(trigger, 0) + 1
            self._decide(observation, goal, state, target, remaining)
        action = np.asarray(
            self.act_latent(observation, self.current.option.latent), dtype=np.float32
        )
        self.last_action = action
        self.step += 1
        return action

    def _left(self, option: Option, remaining: int) -> int:
        if option.goal:
            return remaining
        return option.duration - (self.step - option.start)

    def _trigger(self, state, target, remaining) -> str | None:
        """Why to sample new candidates, if at all, after switching among cached ones."""
        if self.current is None:
            return "start"
        option = self.current.option
        if self._left(option, remaining) <= 0:
            return "duration"
        if not self.config.interrupt:
            return None
        config = self.config
        if config.surprise is not None:
            for step, latent in option.path:
                if step == self.step:
                    if float(np.sum(np.square(state - latent))) > config.surprise:
                        return "surprise"
        options = [option] + [other for other in self.cache if other is not option]
        options = [other for other in options if self._left(other, remaining) > 0]
        heads = self._score(state, target, options, remaining)
        utility = heads.mean(axis=0) + config.risk * heads.std(axis=0)
        spread = heads.std(axis=0)
        if utility[0] < self.current.value - config.margin * self.current.spread:
            return "value"
        best = int(np.argmax(utility))
        if best and utility[best] - utility[0] > config.margin * max(
            spread[0], spread[best]
        ):
            chosen = options[best]
            chosen.duration, chosen.start = self._left(chosen, remaining), self.step
            self.current = Selected(chosen, float(utility[best]), float(spread[best]))
            self.log.switches += 1
        return None

    def _score(self, state, target, options: list[Option], remaining: int):
        return self.value.score(
            state,
            target,
            np.stack([option.latent for option in options]),
            np.array(
                [self._left(option, remaining) for option in options], dtype=np.float32
            ),
            remaining,
            self.config.time_cost,
        )

    def _decide(self, observation, goal, state, target, remaining):
        config, options = self.config, []
        if config.subgoals:
            options = self._propose(goal, remaining)
            self.log.generations += 1
        if config.goal_candidate:
            options.append(Option(target, remaining, self.step, goal=True))
        heads = self._score(state, target, options, remaining)
        utility = heads.mean(axis=0) + config.risk * heads.std(axis=0)
        best = int(np.argmax(utility))
        chosen = options[best]
        self.current = Selected(
            chosen, float(utility[best]), float(heads[:, best].std())
        )
        self.cache = options
        self.log.decisions.append(
            {
                "step": self.step,
                "remaining": remaining,
                "duration": chosen.duration,
                "goal": chosen.goal,
                "value": float(utility[best]),
            }
        )

    def _propose(self, goal, remaining) -> list[Option]:
        completion = self.completion
        context = completion_context(
            list(self.history), goal, remaining, completion.config
        )
        routes, _ = completion.propose(
            context, stream_seed(self.key, self.step), candidates=self.config.samples
        )
        offsets = np.maximum(context.offsets[0], 1)
        slots = self.config.slots or range(len(offsets))
        return [
            Option(
                route[slot].astype(np.float32),
                int(offsets[slot]),
                self.step,
                path=tuple(
                    (self.step + int(offsets[earlier]), route[earlier])
                    for earlier in range(slot)
                ),
            )
            for route in routes[0]
            for slot in slots
            if slot < len(offsets)
        ]


def surprise_threshold(completion, batch, seed: int, quantile: float) -> float:
    """A quantile of the squared distance from sampled routes to encoded futures."""
    routes, _ = completion.propose(batch.context, seed, candidates=1)
    normalized = (batch.targets - completion.mean) / completion.scale
    truth = completion.codec.encode(tensor(normalized)).detach().numpy()
    distances = np.sum(np.square(routes[:, 0] - truth), axis=-1)
    return float(np.quantile(distances, quantile))
