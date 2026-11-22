"""Evaluation-side reset and MuJoCo snapshot adapter for state AntMaze-Large."""

from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import ogbench  # noqa: F401 - registers OGBench Gymnasium environments
from numpy.typing import NDArray

from worth_reaching.data.antmaze import OBSERVATION_SIZE

ENVIRONMENT = "antmaze-large-v0"
SEED_BASE = {"dev": 17001, "test": 51001}
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION.value
TASKS = (1, 2, 3, 4, 5)
ACTION_LIMIT = 1000
VERSIONS = tuple(version(name) for name in ("ogbench", "gymnasium", "mujoco"))


@dataclass(frozen=True, slots=True)
class AntMazeStart:
    task: int
    seed: int
    observation: NDArray[np.float32]
    goal: NDArray[np.float32]
    physics: NDArray[np.float64]
    goal_xy: NDArray[np.float64]


def start_ids(split: str, repeats: int) -> tuple[tuple[int, int], ...]:
    return tuple(
        (task, SEED_BASE[split] + index) for task in TASKS for index in range(repeats)
    )


class AntMaze:
    def __init__(self) -> None:
        self.environment = gym.make(ENVIRONMENT)
        native = self.environment.unwrapped
        self.state_size = mujoco.mj_stateSize(native.model, STATE_SPEC)

    def close(self) -> None:
        self.environment.close()

    def _reset(self, task: int, seed: int) -> tuple[np.ndarray, dict]:
        # OGBench uses legacy NumPy position jitter and samples warmup actions.
        legacy_state = np.random.get_state()
        np.random.seed(seed)
        try:
            self.environment.action_space.seed(seed)
            return self.environment.reset(seed=seed, options={"task_id": task})
        finally:
            np.random.set_state(legacy_state)

    def capture(self, task: int, seed: int) -> AntMazeStart:
        observation, info = self._reset(task, seed)
        native = self.environment.unwrapped
        physics = np.empty(self.state_size, dtype=np.float64)
        mujoco.mj_getState(native.model, native.data, physics, STATE_SPEC)
        return AntMazeStart(
            task,
            seed,
            np.array(observation, dtype=np.float32),
            np.array(info["goal"], dtype=np.float32),
            physics,
            np.array(native.cur_goal_xy, dtype=np.float64),
        )

    def restore(self, start: AntMazeStart) -> NDArray[np.float32]:
        self._reset(start.task, start.seed)
        native = self.environment.unwrapped
        native.set_goal(goal_xy=start.goal_xy)
        mujoco.mj_setState(native.model, native.data, start.physics, STATE_SPEC)
        mujoco.mj_forward(native.model, native.data)
        observation = np.asarray(native.get_ob(), dtype=np.float32)
        if not np.array_equal(observation, start.observation):
            raise ValueError(
                "The saved AntMaze state did not restore its public observation"
            )
        return observation

    def step(
        self, action: NDArray[np.float32]
    ) -> tuple[NDArray[np.float32], bool, bool, bool]:
        observation, _, terminated, truncated, info = self.environment.step(action)
        return (
            np.asarray(observation, dtype=np.float32),
            bool(info["success"]),
            bool(terminated),
            bool(truncated),
        )


def save_starts(path: Path, starts: list[AntMazeStart]) -> None:
    np.savez_compressed(
        path,
        environment=ENVIRONMENT,
        versions=VERSIONS,
        state_spec=STATE_SPEC,
        ids=[(start.task, start.seed) for start in starts],
        observation=np.stack([start.observation for start in starts]),
        goal=np.stack([start.goal for start in starts]),
        physics=np.stack([start.physics for start in starts]),
        goal_xy=np.stack([start.goal_xy for start in starts]),
    )


def load_starts(
    path: Path, *, split: str, repeats: int, state_size: int
) -> list[AntMazeStart]:
    expected = start_ids(split, repeats)
    with np.load(path, allow_pickle=False) as values:
        if (
            values["environment"].item() != ENVIRONMENT
            or tuple(values["versions"].tolist()) != VERSIONS
            or values["state_spec"].item() != STATE_SPEC
            or not np.array_equal(values["ids"], expected)
        ):
            raise ValueError(
                "AntMaze start bank disagrees with the environment or requested split"
            )
        observation, goal, physics, goal_xy = (
            values[key] for key in ("observation", "goal", "physics", "goal_xy")
        )
    count = len(expected)
    if (
        observation.shape != (count, OBSERVATION_SIZE)
        or goal.shape != observation.shape
        or physics.shape != (count, state_size)
        or goal_xy.shape != (count, 2)
        or any(
            not np.isfinite(array).all()
            for array in (observation, goal, physics, goal_xy)
        )
    ):
        raise ValueError("Invalid AntMaze start bank arrays")
    return [
        AntMazeStart(*identity, obs, target, state, xy)
        for identity, obs, target, state, xy in zip(
            expected, observation, goal, physics, goal_xy, strict=True
        )
    ]
