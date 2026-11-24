"""State AntMaze archive names and dimensions."""

from pathlib import Path

from worth_reaching.data.ogbench import StateArchive, load_state_archive

DATASETS = ("antmaze-large-navigate-v0", "antmaze-large-stitch-v0")
DATASET_URL = "https://rail.eecs.berkeley.edu/datasets/ogbench"
OBSERVATION_SIZE = 29
ACTION_SIZE = 8


def load_antmaze_archive(
    path: Path, *, max_trajectories: int | None = None
) -> StateArchive:
    return load_state_archive(
        path,
        observation_size=OBSERVATION_SIZE,
        action_size=ACTION_SIZE,
        max_trajectories=max_trajectories,
    )
