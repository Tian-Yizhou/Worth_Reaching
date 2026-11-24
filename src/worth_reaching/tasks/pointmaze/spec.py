"""Identifiers and official action limits for the supported Maze2D tasks."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PointMazeTask:
    dataset_id: str
    environment_id: str
    action_limit: int


TASKS = {
    "umaze": PointMazeTask("D4RL/pointmaze/umaze-v2", "PointMaze_UMaze-v3", 300),
    "medium": PointMazeTask("D4RL/pointmaze/medium-v2", "PointMaze_Medium-v3", 600),
    "large": PointMazeTask("D4RL/pointmaze/large-v2", "PointMaze_Large-v3", 800),
}
