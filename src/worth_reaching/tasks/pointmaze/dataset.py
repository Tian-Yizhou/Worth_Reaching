"""Load the official Minari PointMaze archive and hold out a chronological suffix."""

from hashlib import file_digest
from pathlib import Path

import minari
import numpy as np

from worth_reaching.data.trajectory_windows import ContinuousTrajectories

OBSERVATION_SIZE = 4
ACTION_SIZE = 2


def load_dataset(
    root: Path, dataset_id: str
) -> tuple[minari.MinariDataset, dict[str, str]]:
    path = root / dataset_id / "data"
    fingerprints = {}
    for name in ("main_data.hdf5", "metadata.json"):
        with (path / name).open("rb") as stream:
            fingerprints[name] = file_digest(stream, "sha256").hexdigest()
    dataset = minari.MinariDataset(path)
    if dataset.id != dataset_id:
        raise ValueError(f"Expected Minari dataset {dataset_id}, got {dataset.id}")
    return dataset, fingerprints


def _trajectories(
    dataset: minari.MinariDataset, episode_ids: np.ndarray
) -> ContinuousTrajectories:
    observations, actions = [], []
    for episode in dataset.iterate_episodes(episode_ids):
        states = np.asarray(episode.observations["observation"], dtype=np.float32)
        moves = np.asarray(episode.actions, dtype=np.float32)
        if (
            not len(moves)
            or moves.shape != (len(moves), ACTION_SIZE)
            or states.shape != (len(moves) + 1, OBSERVATION_SIZE)
            or not np.isfinite(states).all()
            or not np.isfinite(moves).all()
        ):
            raise ValueError(
                f"Invalid public observations or actions in episode {episode.id}"
            )
        observations.append(states)
        actions.append(np.pad(moves, ((0, 1), (0, 0))))
    return ContinuousTrajectories(
        np.concatenate(observations),
        np.concatenate(actions),
        np.cumsum([len(states) for states in observations]) - 1,
    )


def split_trajectories(
    dataset: minari.MinariDataset,
) -> tuple[ContinuousTrajectories, ContinuousTrajectories, dict]:
    episode_ids = np.sort(dataset.episode_indices)
    cutoff = len(episode_ids) * 4 // 5
    if cutoff < 1 or cutoff + 1 >= len(episode_ids):
        raise ValueError("Need train and validation episodes with a gap between them")
    train, validation = episode_ids[:cutoff], episode_ids[cutoff + 1 :]
    split = {
        "train_episodes": train.tolist(),
        "validation_episodes": validation.tolist(),
        "gap_episode": int(episode_ids[cutoff]),
    }
    return _trajectories(dataset, train), _trajectories(dataset, validation), split
