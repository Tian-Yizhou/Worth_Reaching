"""Default state-based OGBench Scene environment."""

from worth_reaching.envs.ogbench_manipulation import Manipulation
from worth_reaching.envs.ogbench_tasks import SCENE


class Scene(Manipulation):
    def __init__(self) -> None:
        super().__init__(SCENE)
