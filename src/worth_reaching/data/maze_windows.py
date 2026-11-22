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

    def sample_windows(
        self,
        rng: np.random.Generator,
        size: int,
        *,
        history: int,
        queries: int,
        max_horizon: int,
    ) -> MaskedBatch:
        """Sample a public prefix, held-out middle and hindsight endpoint in one episode."""
        begins = np.r_[0, self.ends[:-1] + 1]
        eligible = np.flatnonzero(self.ends - begins > queries)
        if not len(eligible):
            raise ValueError("The archive has no episodes long enough for the mask")
        episodes = rng.choice(eligible, size=size)
        first, end = begins[episodes], self.ends[episodes]
        start = rng.integers(first, end - queries)
        start = np.where(rng.random(size) < 0.25, first, start)
        available = np.minimum(end - start, max_horizon)
        # Include long windows alongside random spans to exercise the full action budget.
        remaining = np.where(
            rng.random(size) < 0.25, available, rng.integers(queries + 1, available + 1)
        ).astype(np.int32)
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
