"""State-based OGBench manipulation task identifiers and dimensions."""

from dataclasses import dataclass

TASKS = (1, 2, 3, 4, 5)
ACTION_SIZE = 5
SEED_OFFSETS = {"dev": 0, "test": 40000}


@dataclass(frozen=True, slots=True)
class ManipulationTask:
    family: str
    name: str
    observation_size: int
    action_limit: int
    seed_base: int

    @property
    def environment(self) -> str:
        return f"{self.name}-v0"

    @property
    def dataset(self) -> str:
        return f"{self.name}-play-v0"


SCENE = ManipulationTask("scene", "scene", 40, 750, 43001)
PUZZLE_BOARDS = {
    "3x3": ManipulationTask("puzzle", "puzzle-3x3", 55, 500, 46001),
    "4x4": ManipulationTask("puzzle", "puzzle-4x4", 83, 500, 46001),
    "4x5": ManipulationTask("puzzle", "puzzle-4x5", 99, 1000, 46001),
    "4x6": ManipulationTask("puzzle", "puzzle-4x6", 115, 1000, 46001),
}
