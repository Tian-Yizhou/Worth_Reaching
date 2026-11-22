"""Load state-based OGBench trajectories and sample within-episode hindsight goals."""

from dataclasses import dataclass
from hashlib import file_digest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.maze_windows import MazeTrajectories


@dataclass(frozen=True, slots=True)
class GoalBatch:
    observations: NDArray[np.float32]
    goals: NDArray[np.float32]
    actions: NDArray[np.float32]


@dataclass(frozen=True, slots=True)
class StateArchive(MazeTrajectories):
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


def load_state_archive(
    path: Path,
    *,
    observation_size: int,
    action_size: int,
    max_trajectories: int | None = None,
) -> StateArchive:
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
                observation_size,
            ) or actions.shape != (len(terminal), action_size):
                raise ValueError(
                    f"Expected {observation_size} observation and {action_size} action features in {path}"
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
    return StateArchive(observations, actions, ends, rows, fingerprint)
