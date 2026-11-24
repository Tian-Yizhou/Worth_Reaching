"""Default state-based OGBench Scene environment."""

from worth_reaching.tasks.ogbench_manipulation.environment import Manipulation
from worth_reaching.tasks.ogbench_manipulation.spec import SCENE


class Scene(Manipulation):
    def __init__(self) -> None:
        super().__init__(SCENE)
