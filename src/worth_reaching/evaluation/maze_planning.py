"""Shared policy for pursuing decoded states and replanning from public history."""

from collections import deque
from collections.abc import Callable, Sequence

import numpy as np

from worth_reaching.data.maze_windows import CompletionContext, query_offsets
from worth_reaching.models.maze_completion import CompletionConfig, LearnedCompletion
from worth_reaching.models.training import stream_seed

Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]


def completion_context(
    history: Sequence[tuple[np.ndarray, np.ndarray]]
    | deque[tuple[np.ndarray, np.ndarray]],
    goal: np.ndarray,
    remaining: int,
    config: CompletionConfig,
) -> CompletionContext:
    count = len(history)
    observations = np.zeros((1, config.history, goal.size), dtype=np.float32)
    actions = np.zeros((1, config.history, history[-1][1].size), dtype=np.float32)
    observations[0, -count:], actions[0, -count:] = map(
        np.stack, zip(*history, strict=True)
    )
    present = (np.arange(config.history) >= config.history - count)[None]
    duration = np.array([remaining], dtype=np.int32)
    return CompletionContext(
        observations,
        actions,
        present,
        goal[None],
        duration,
        query_offsets(duration, config.queries),
    )


class RecedingGoals:
    """Condition the frozen actor on one decoded goal until a time-based replan."""

    def __init__(
        self,
        actor: Policy,
        completion: LearnedCompletion,
        key: int,
        limit: int,
        *,
        action_size: int,
        first_route: int = 0,
    ):
        self.actor = actor
        self.completion = completion
        self.key = key
        self.limit = limit
        self.first_route = first_route
        self.step = self.deadline = 0
        self.target = None
        self.history: deque = deque(maxlen=completion.config.history)
        self.last_action = np.zeros(action_size, dtype=np.float32)
        self.plans: list[dict] = []

    def __call__(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        self.history.append((observation, self.last_action))
        if self.step == self.deadline:
            self.target = self._plan(goal)
        action = self.actor(observation, self.target)
        self.last_action = action
        self.step += 1
        return action

    def record_prefix_step(self, observation: np.ndarray, action: np.ndarray) -> None:
        """Ingest an executed action so the next policy call replans at the handoff."""
        self.history.append((observation, self.last_action))
        self.last_action = action
        self.step += 1
        self.deadline = self.step

    def _plan(self, goal: np.ndarray) -> np.ndarray:
        config = self.completion.config
        remaining = min(self.limit - self.step, config.max_horizon)
        if remaining <= config.queries:
            self.deadline = self.limit
            return goal

        context = completion_context(self.history, goal, remaining, config)
        offsets = context.offsets
        routes, decoded = self.completion.propose(
            context, stream_seed(self.key, self.step)
        )
        if not np.isfinite(routes).all() or not np.isfinite(decoded).all():
            raise ValueError("Latent planner produced a non-finite candidate")
        self.deadline = self.step + int(offsets[0, 0])
        selected = self.first_route if self.step == 0 else 0
        self.plans.append(
            {
                "step": self.step,
                "deadline": self.deadline,
                "query_offsets": offsets[0].tolist(),
                "selected_route": selected,
                "selected_query": 0,
                "latent_routes": routes[0].tolist(),
                "decoded_routes": decoded[0].tolist(),
            }
        )
        return decoded[0, selected, 0]
