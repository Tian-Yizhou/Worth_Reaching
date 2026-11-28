"""Choose, pursue and interrupt latent options with a budget-conditioned value.

At a decision the world model samples latent futures at each of its query offsets,
under the remaining budget and optionally under shorter ones, which adds faster
routes. Every sample at offset `k` is a candidate option "pursue `z` for at most `k`
actions", and the encoded terminal goal, pursued for the whole remaining budget, is
always one more candidate. The option value scores every candidate with the true
remaining budget and ranks them by the mean over its heads plus `risk` times their
spread. One latent-goal executor pursues the choice. Nothing defines what a subgoal is
or when it has been reached.

Each step the planner re-scores the current option with its remaining duration and
the cached candidates, including the goal, and switches when another beats it by more
than `margin` times the spread. It samples new candidates only when the current
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
    proposal_budgets: tuple[float, ...] = (1.0,)
    """Fractions of the remaining budget the world model is conditioned on when it
    samples candidates. Shorter budgets give faster routes; the value still scores every
    candidate with the true remaining budget."""
    explore: float = 0.0
    """Probability of pursuing a random sampled candidate at a decision, for collecting
    the executor's attempts. The draw is uniform over proposal budgets, then uniform
    within the budget. The terminal goal is left out: collection tries it on every
    training reset in an episode of its own."""
    explore_first: bool = False
    """Pursue a sampled candidate drawn as `explore` draws it at the first decision,
    then plan greedily. Its value carries no selection, so comparing it with the
    realized return checks the value's calibration apart from the argmax."""

    def __post_init__(self) -> None:
        if (
            self.time_cost < 0
            or self.samples < 1
            or self.margin < 0
            or not (self.subgoals or self.goal_candidate)
            or (self.slots is not None and not self.slots)
            or (self.surprise is not None and self.surprise <= 0)
            or not self.proposal_budgets
            or not all(0 < fraction <= 1 for fraction in self.proposal_budgets)
            or not 0 <= self.explore <= 1
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
    source: int = -1
    """The proposal budget that produced the candidate; -1 for the terminal goal."""


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
    pursued: list[tuple[int, np.ndarray, int]] = field(default_factory=list)
    """Each option the executor started pursuing, as (step, latent, duration), with the
    duration it was commanded: at most this many actions."""
    decisions: list[dict] = field(default_factory=list)
    """Each new decision: its trigger, the chosen option and the values it compared."""
    switch_log: list[dict] = field(default_factory=list)
    """Each switch among cached candidates, with the values before and after."""


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
            self._decide(observation, goal, state, target, remaining, trigger)
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
        utility = self._utility(heads)
        spread = heads.std(axis=0)
        if utility[0] < self.current.value - config.margin * self.current.spread:
            return "value"
        best = int(np.argmax(utility))
        if best and utility[best] - utility[0] > config.margin * max(
            spread[0], spread[best]
        ):
            chosen = options[best]
            chosen.duration, chosen.start = self._left(chosen, remaining), self.step
            self.log.switch_log.append(
                {
                    "step": self.step,
                    "remaining": remaining,
                    "duration": chosen.duration,
                    "goal": chosen.goal,
                    "source": chosen.source,
                    "value": float(utility[best]),
                    "previous_value": float(utility[0]),
                }
            )
            self.current = Selected(chosen, float(utility[best]), float(spread[best]))
            self.log.switches += 1
            self.log.pursued.append((self.step, chosen.latent, chosen.duration))
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

    def _decide(self, observation, goal, state, target, remaining, trigger):
        config, options = self.config, []
        if config.subgoals:
            options = self._propose(goal, remaining)
            self.log.generations += 1
        if config.goal_candidate:
            options.append(Option(target, remaining, self.step, goal=True))
        heads = self._score(state, target, options, remaining)
        utility = self._utility(heads)
        best = int(np.argmax(utility))
        greedy = best
        sources = sorted({option.source for option in options} - {-1})
        explore = config.explore_first and trigger == "start"
        if (config.explore or explore) and sources:
            rng = np.random.default_rng(stream_seed(self.key, self.step, 2))
            if rng.random() < config.explore or explore:
                source = sources[int(rng.integers(len(sources)))]
                members = [
                    i for i, option in enumerate(options) if option.source == source
                ]
                best = members[int(rng.integers(len(members)))]
        chosen = options[best]
        self.current = Selected(
            chosen, float(utility[best]), float(heads[:, best].std())
        )
        self.cache = options
        self.log.pursued.append((self.step, chosen.latent, chosen.duration))
        sampled = [i for i, option in enumerate(options) if not option.goal]
        self.log.decisions.append(
            {
                "step": self.step,
                "trigger": trigger,
                "remaining": remaining,
                "duration": chosen.duration,
                "goal": chosen.goal,
                "source": chosen.source,
                "explored": best != greedy,
                "value": float(utility[best]),
                "goal_value": float(utility[-1]) if config.goal_candidate else None,
                "best_sampled_value": float(utility[sampled].max())
                if sampled
                else None,
                "candidates": len(options),
                "spread": float(heads[:, best].std()),
                "goal_spread": float(heads[:, -1].std())
                if config.goal_candidate
                else None,
                "sampled_mean": float(utility[sampled].mean()) if sampled else None,
                "sampled_median": float(np.median(utility[sampled]))
                if sampled
                else None,
                "holdout_value": self._holdout(heads),
            }
        )

    def _utility(self, heads: np.ndarray) -> np.ndarray:
        return heads.mean(axis=0) + self.config.risk * heads.std(axis=0)

    def _holdout(self, heads: np.ndarray) -> float:
        """Each head's value of the candidate the other heads rank first, averaged.

        The greedy choice's own utility is biased upward by the noise it was selected
        for; a head that took no part in the selection is not, as far as the heads'
        errors are independent. The gap between the two estimates that bias.
        """
        held = []
        for head in range(len(heads)):
            others = np.delete(heads, head, axis=0)
            held.append(heads[head, int(np.argmax(self._utility(others)))])
        return float(np.mean(held))

    def _propose(self, goal, remaining) -> list[Option]:
        completion, options = self.completion, []
        queries = completion.config.queries
        for index, fraction in enumerate(self.config.proposal_budgets):
            budget = max(min(remaining, queries + 1), round(remaining * fraction))
            context = completion_context(
                list(self.history), goal, budget, completion.config
            )
            routes, _ = completion.propose(
                context,
                stream_seed(self.key, self.step, index),
                candidates=self.config.samples,
            )
            offsets = np.maximum(context.offsets[0], 1)
            slots = self.config.slots or range(len(offsets))
            options += [
                Option(
                    route[slot].astype(np.float32),
                    int(offsets[slot]),
                    self.step,
                    path=tuple(
                        (self.step + int(offsets[earlier]), route[earlier])
                        for earlier in range(slot)
                    ),
                    source=index,
                )
                for route in routes[0]
                for slot in slots
                if slot < len(offsets)
            ]
        return options


def surprise_threshold(completion, batch, seed: int, quantile: float) -> float:
    """A quantile of the squared distance from sampled routes to encoded futures."""
    routes, _ = completion.propose(batch.context, seed, candidates=1)
    normalized = (batch.targets - completion.mean) / completion.scale
    truth = completion.codec.encode(tensor(normalized)).detach().numpy()
    distances = np.sum(np.square(routes[:, 0] - truth), axis=-1)
    return float(np.quantile(distances, quantile))
