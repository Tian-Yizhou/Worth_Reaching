"""Load state-based OGBench trajectories and sample within-episode hindsight goals."""

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

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
