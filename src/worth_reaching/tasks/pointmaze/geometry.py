"""Privileged maze geometry for evaluation bounds and plan diagnostics.

The layout comes from the evaluation environment and never reaches a model. It gives a
shortest-path oracle that bounds what intermediate goals can achieve under the shared
tracker, and it scores declared targets for wall placement and geodesic progress.
"""

from collections import deque
from functools import cache

import numpy as np

Cell = tuple[int, int]
STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))


class MazeGeometry:
    def __init__(self, maze):
        self.maze = maze
        self.walls = np.array(
            [[str(value) == "1" for value in row] for row in maze.maze_map]
        )

    def cell(self, xy: np.ndarray) -> Cell:
        row, column = self.maze.cell_xy_to_rowcol(np.asarray(xy[:2], dtype=np.float64))
        return int(row), int(column)

    def center(self, cell: Cell) -> np.ndarray:
        return np.asarray(self.maze.cell_rowcol_to_xy(np.array(cell)), dtype=np.float32)

    def free(self, xy: np.ndarray) -> bool:
        row, column = self.cell(xy)
        rows, columns = self.walls.shape
        return 0 <= row < rows and 0 <= column < columns and not self.walls[row, column]

    def distance(self, xy: np.ndarray, goal: np.ndarray) -> int | None:
        """Shortest path length in cells, or None for a point outside free space."""
        if not self.free(xy):
            return None
        return self._distances(self.cell(goal)).get(self.cell(xy))

    def progress(self, state: np.ndarray, target: np.ndarray, goal: np.ndarray) -> bool:
        """Whether the target lies strictly closer to the goal along free cells."""
        here, there = self.distance(state, goal), self.distance(target, goal)
        return here is not None and there is not None and there < here

    def waypoint(
        self, observation: np.ndarray, goal: np.ndarray, cells: int, speed: float
    ) -> np.ndarray:
        """The cell center a few cells along the shortest path, moving at `speed`.

        The target velocity points from the observation toward the waypoint, so the
        position and velocity tracker does not brake at every cell. Within reach of the
        goal cell the waypoint is the goal at rest.
        """
        distances = self._distances(self.cell(goal))
        cell = self.cell(observation)
        if cell not in distances or distances[cell] <= cells:
            return np.concatenate((goal[:2], np.zeros(2))).astype(np.float32)
        for _ in range(cells):
            cell = min(
                (
                    neighbor
                    for neighbor in self._neighbors(cell)
                    if neighbor in distances
                ),
                key=distances.__getitem__,
            )
        # The goal cell is beyond the walk, so the waypoint is a corridor cell.
        center = self.center(cell)
        direction = center - observation[:2]
        velocity = speed * direction / max(float(np.linalg.norm(direction)), 1e-6)
        return np.concatenate((center, velocity)).astype(np.float32)

    def _neighbors(self, cell: Cell):
        rows, columns = self.walls.shape
        for dr, dc in STEPS:
            row, column = cell[0] + dr, cell[1] + dc
            if (
                0 <= row < rows
                and 0 <= column < columns
                and not self.walls[row, column]
            ):
                yield row, column

    @cache  # noqa: B019 - one geometry per evaluation environment
    def _distances(self, goal: Cell) -> dict[Cell, int]:
        distances, frontier = {goal: 0}, deque([goal])
        while frontier:
            cell = frontier.popleft()
            for neighbor in self._neighbors(cell):
                if neighbor not in distances:
                    distances[neighbor] = distances[cell] + 1
                    frontier.append(neighbor)
        return distances
