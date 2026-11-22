"""Mask interior states in public continuous-control trajectory archives."""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray


class CompletionContext(NamedTuple):
    observations: NDArray[np.float32]
    incoming_actions: NDArray[np.float32]
    present: NDArray[np.bool_]
    goal: NDArray[np.float32]
    remaining: NDArray[np.int32]
    offsets: NDArray[np.int32]


class MaskedBatch(NamedTuple):
    context: CompletionContext
    targets: NDArray[np.float32]


def query_offsets(remaining: NDArray[np.int32], queries: int) -> NDArray[np.int32]:
    return (remaining[..., None] * np.arange(1, queries + 1, dtype=np.int32)) // (
        queries + 1
    )


@dataclass(frozen=True, slots=True)
class MazeTrajectories:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    ends: NDArray[np.int64]

    @property
    def episode_lengths(self) -> NDArray[np.int64]:
        return np.diff(self.ends, prepend=-1) - 1

    def sample_windows(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        history: int,
        queries: int,
        max_horizon: int,
        full_span_windows: int = 0,
    ) -> MaskedBatch:
        """Sample a public prefix, held-out middle and hindsight endpoint in one episode."""
        lengths = self.episode_lengths
        begins = self.ends - lengths
        eligible = np.flatnonzero(lengths > queries)
        if not len(eligible):
            raise ValueError("The archive has no episodes long enough for the mask")
        if not 0 <= full_span_windows <= size:
            raise ValueError("Full-span windows must fit within the sample count")
        size -= full_span_windows
        episodes = rng.choice(eligible, size=size)
        first, end = begins[episodes], self.ends[episodes]
        start = rng.integers(first, end - queries)
        start = np.where(rng.random(size) < 0.25, first, start)
        available = np.minimum(end - start, max_horizon)
        # Include long windows alongside random spans to exercise the full action budget.
        remaining = np.where(
            rng.random(size) < 0.25, available, rng.integers(queries + 1, available + 1)
        ).astype(np.int32)
        if full_span_windows:
            supported = np.flatnonzero(lengths >= max_horizon)
            if not len(supported):
                raise ValueError(
                    f"The archive has no episodes spanning {max_horizon} actions"
                )
            reset_episodes = rng.choice(supported, size=full_span_windows)
            start = np.concatenate((start, begins[reset_episodes]))
            first = np.concatenate((first, begins[reset_episodes]))
            remaining = np.concatenate(
                (remaining, np.full(full_span_windows, max_horizon, dtype=np.int32))
            )
        offsets = query_offsets(remaining, queries)
        indexes = start[:, None] + np.arange(1 - history, 1)
        present = indexes >= first[:, None]
        history_rows = np.maximum(indexes, first[:, None])
        has_incoming = indexes > first[:, None]
        action_rows = np.where(has_incoming, indexes - 1, first[:, None])
        return MaskedBatch(
            CompletionContext(
                self.observations[history_rows],
                np.where(has_incoming[..., None], self.actions[action_rows], 0),
                present,
                self.observations[start + remaining],
                remaining,
                offsets,
            ),
            self.observations[start[:, None] + offsets],
        )
