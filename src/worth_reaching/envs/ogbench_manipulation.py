"""State-based OGBench manipulation with reproducible official goal resets."""

from dataclasses import dataclass, replace
from functools import cached_property

import gymnasium as gym
import mujoco
import numpy as np
import ogbench  # noqa: F401 - registers OGBench Gymnasium environments
from ogbench.manipspace.envs.puzzle_env import PuzzleEnv
from ogbench.manipspace.envs.scene_env import SceneEnv

from worth_reaching.envs.ogbench_tasks import ManipulationTask

STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION.value


class _PersistentActionSpace:
    @cached_property
    def action_space(self):
        # Upstream samples from this property during goal construction.
        return super().action_space


class _SeededScene(_PersistentActionSpace, SceneEnv):
    pass


class _SeededPuzzle(_PersistentActionSpace, PuzzleEnv):
    pass


_ENV_TYPES = {"scene": _SeededScene, "puzzle": _SeededPuzzle}


@dataclass(frozen=True, slots=True)
class ManipulationStart:
    task: int
    seed: int
    observation: np.ndarray
    goal: np.ndarray
    physics: np.ndarray


class Manipulation:
    def __init__(self, benchmark: ManipulationTask) -> None:
        spec = gym.spec(benchmark.environment)
        if spec.max_episode_steps != benchmark.action_limit:
            raise ValueError(f"Unexpected action limit for {benchmark.environment}")
        self.benchmark = benchmark
        self.environment = gym.make(
            replace(spec, entry_point=_ENV_TYPES[benchmark.family])
        )

    def close(self) -> None:
        self.environment.close()

    def capture(self, task: int, seed: int) -> ManipulationStart:
        self.environment.action_space.seed(seed)
        observation, info = self.environment.reset(seed=seed, options={"task_id": task})
        native = self.environment.unwrapped
        physics = np.empty(
            mujoco.mj_stateSize(native.model, STATE_SPEC), dtype=np.float64
        )
        mujoco.mj_getState(native.model, native.data, physics, STATE_SPEC)
        return ManipulationStart(
            task,
            seed,
            np.asarray(observation, dtype=np.float32),
            np.asarray(info["goal"], dtype=np.float32),
            physics,
        )

    def restore(self, start: ManipulationStart) -> np.ndarray:
        restored = self.capture(start.task, start.seed)
        if not all(
            np.array_equal(a, b)
            for a, b in (
                (restored.observation, start.observation),
                (restored.goal, start.goal),
                (restored.physics, start.physics),
            )
        ):
            raise ValueError(
                f"{self.benchmark.environment} did not reproduce the matched start and goal"
            )
        return restored.observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        observation, _, terminated, truncated, info = self.environment.step(action)
        return (
            np.asarray(observation, dtype=np.float32),
            bool(info["success"]),
            bool(terminated or truncated),
        )
