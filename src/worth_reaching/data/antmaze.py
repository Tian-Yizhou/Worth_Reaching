"""Load state-based OGBench trajectories and sample within-episode hindsight goals."""

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path
from typing import NamedTuple

import numpy as np
from numpy.typing import NDArray

DATASETS = ("antmaze-large-navigate-v0", "antmaze-large-stitch-v0")
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
OBSERVATION_SIZE = 29
ACTION_SIZE = 8


@dataclass(frozen=True, slots=True)
class GoalBatch:
    observations: NDArray[np.float32]
    goals: NDArray[np.float32]
    actions: NDArray[np.float32]


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
class AntMazeArchive:
    observations: NDArray[np.float32]
    actions: NDArray[np.float32]
    ends: NDArray[np.int64]
    rows: NDArray[np.int64]
    fingerprint: str

    def sample_indices(
        self, rng: np.random.Generator, size: int, *, max_goal_steps: int | None = None
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        rows = rng.choice(self.rows, size=size)
        latest = self.ends[np.searchsorted(self.ends, rows)]
        if max_goal_steps is not None:
            latest = np.minimum(latest, rows + max_goal_steps)
        goals = rng.integers(rows + 1, latest + 1)
        return rows, goals

    def sample(
        self, rng: np.random.Generator, size: int, *, max_goal_steps: int | None = None
    ) -> GoalBatch:
        rows, goals = self.sample_indices(rng, size, max_goal_steps=max_goal_steps)
        return GoalBatch(
            self.observations[rows], self.observations[goals], self.actions[rows]
        )

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
            raise ValueError(
                "The AntMaze archive has no episodes long enough for the mask"
            )
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


def load_antmaze_archive(
    path: Path, *, max_trajectories: int | None = None
) -> AntMazeArchive:
    """Read original OGBench NPZ rows, including the final observation of each episode."""
    if max_trajectories is not None and max_trajectories < 1:
        raise ValueError("Use a positive trajectory limit")
    with path.open("rb") as stream:
        fingerprint = file_digest(stream, "sha256").hexdigest()
        stream.seek(0)
        with np.load(stream, allow_pickle=False) as data:
            terminal = data["terminals"]
            if (
                terminal.ndim != 1
                or not terminal.size
                or not np.isin(terminal, (0, 1)).all()
            ):
                raise ValueError(f"Expected binary episode boundaries in {path}")
            ends = np.flatnonzero(terminal)
            if not ends.size or ends[-1] != len(terminal) - 1:
                raise ValueError(f"Expected a complete last trajectory in {path}")
            if max_trajectories is not None:
                ends = ends[:max_trajectories]
            stop = int(ends[-1]) + 1
            observations = data["observations"]
            actions = data["actions"]
            if observations.shape != (
                len(terminal),
                OBSERVATION_SIZE,
            ) or actions.shape != (len(terminal), ACTION_SIZE):
                raise ValueError(
                    f"Expected the state AntMaze observation and action shapes in {path}"
                )
            observations = observations[:stop].astype(
                np.float32, copy=stop < len(terminal)
            )
            actions = actions[:stop].astype(np.float32, copy=stop < len(terminal))
    rows = np.flatnonzero(terminal[:stop] == 0)
    if (
        not rows.size
        or not np.isfinite(observations).all()
        or not np.isfinite(actions[rows]).all()
    ):
        raise ValueError(
            f"Expected finite observations and valid transitions in {path}"
        )
    return AntMazeArchive(observations, actions, ends, rows, fingerprint)
