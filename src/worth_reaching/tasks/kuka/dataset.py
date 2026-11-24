"""Public state trajectories released with Diffuser's Kuka benchmark."""

from hashlib import sha256
from io import BytesIO
from pathlib import Path

import numpy as np

from worth_reaching.data.trajectory_windows import ContinuousTrajectories, MaskedBatch

JOINTS = 7
BLOCKS = 4
BLOCK_STATE = 8
STATE_SIZE = JOINTS + BLOCKS * BLOCK_STATE
CLIP_STRIDE = 2
CLIP_LIMIT = 300


def block_states(observation: np.ndarray) -> np.ndarray:
    return observation[..., JOINTS:].reshape(
        *observation.shape[:-1], BLOCKS, BLOCK_STATE
    )


def goal_features(observation: np.ndarray) -> np.ndarray:
    """Encode terminal block positions, with free horizontal translation."""
    goal = np.zeros_like(observation)
    positions = block_states(observation)[..., :3]
    destination = block_states(goal)
    destination[..., :3] = positions
    destination[..., :2] -= positions[..., :2].mean(axis=-2, keepdims=True)
    return goal


def stack_goal(
    observation: np.ndarray, order: np.ndarray, block_height: float
) -> np.ndarray:
    target = np.zeros_like(observation)
    floor = block_states(observation)[order[0], 2]
    block_states(target)[order, 2] = floor + block_height * np.arange(BLOCKS)
    return target


def sample_windows(
    data: ContinuousTrajectories,
    rng: np.random.Generator,
    windows: int,
    **settings: int,
) -> MaskedBatch:
    batch = data.sample_windows(rng, windows, **settings)
    return MaskedBatch(
        batch.context._replace(goal=goal_features(batch.context.goal)), batch.targets
    )


def _read_clips(paths: list[Path]) -> tuple[ContinuousTrajectories, str]:
    observations = []
    fingerprint = sha256()
    for path in paths:
        source = path.read_bytes()
        fingerprint.update(path.name.encode())
        fingerprint.update(source)
        clip = np.load(BytesIO(source), allow_pickle=False)
        if clip.ndim != 2 or clip.shape[1] != STATE_SIZE or not np.isfinite(clip).all():
            raise ValueError(f"Invalid Diffuser state clip in {path}")
        observations.append(
            clip[: CLIP_STRIDE * CLIP_LIMIT : CLIP_STRIDE].astype(np.float32)
        )
    states = np.concatenate(observations)
    return ContinuousTrajectories(
        states,
        np.empty((len(states), 0), np.float32),
        np.cumsum([len(clip) for clip in observations]) - 1,
    ), fingerprint.hexdigest()


def load_clips(
    directory: Path,
) -> tuple[ContinuousTrajectories, ContinuousTrajectories, dict]:
    files = sorted(directory.glob("data_*.npy"))
    if len(files) < 5:
        raise ValueError(f"Expected Diffuser demonstration clips in {directory}")
    indexes = np.random.default_rng(7219).permutation(len(files))
    cutoff = len(files) * 4 // 5
    train_files = [files[index] for index in sorted(indexes[:cutoff])]
    validation_files = [files[index] for index in sorted(indexes[cutoff:])]
    train, train_hash = _read_clips(train_files)
    validation, validation_hash = _read_clips(validation_files)
    return (
        train,
        validation,
        {
            "train": {
                "clips": [path.name for path in train_files],
                "sha256": train_hash,
            },
            "validation": {
                "clips": [path.name for path in validation_files],
                "sha256": validation_hash,
            },
        },
    )
