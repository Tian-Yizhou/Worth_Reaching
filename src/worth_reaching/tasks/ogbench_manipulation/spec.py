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
