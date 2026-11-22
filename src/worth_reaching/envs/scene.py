"""State-based OGBench Scene with reproducible official goal resets."""

from dataclasses import dataclass, replace
from functools import cached_property

import gymnasium as gym
import mujoco
import numpy as np
import ogbench  # noqa: F401 - registers OGBench Gymnasium environments
from ogbench.manipspace.envs.scene_env import SceneEnv

ENVIRONMENT = "scene-v0"
DATASET = "scene-play-v0"
OBSERVATION_SIZE = 40
ACTION_SIZE = 5
ACTION_LIMIT = 750
TASKS = (1, 2, 3, 4, 5)
SEED_BASE = {"dev": 43001, "test": 83001}
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION.value


class _SeededScene(SceneEnv):
    @cached_property
    def action_space(self):
        # Upstream samples from this property during goal construction.
        return super().action_space


@dataclass(frozen=True, slots=True)
class SceneStart:
    task: int
    seed: int
    observation: np.ndarray
    goal: np.ndarray
    physics: np.ndarray


class Scene:
    def __init__(self) -> None:
        spec = gym.spec(ENVIRONMENT)
        if spec.max_episode_steps != ACTION_LIMIT:
            raise ValueError("Unexpected Scene action limit")
        self.environment = gym.make(replace(spec, entry_point=_SeededScene))

    def close(self) -> None:
        self.environment.close()

    def capture(self, task: int, seed: int) -> SceneStart:
        self.environment.action_space.seed(seed)
        observation, info = self.environment.reset(seed=seed, options={"task_id": task})
        native = self.environment.unwrapped
        physics = np.empty(
            mujoco.mj_stateSize(native.model, STATE_SPEC), dtype=np.float64
        )
        mujoco.mj_getState(native.model, native.data, physics, STATE_SPEC)
        return SceneStart(
            task,
            seed,
            np.asarray(observation, dtype=np.float32),
            np.asarray(info["goal"], dtype=np.float32),
            physics,
        )

    def restore(self, start: SceneStart) -> np.ndarray:
        restored = self.capture(start.task, start.seed)
        if not all(
            np.array_equal(a, b)
            for a, b in (
                (restored.observation, start.observation),
                (restored.goal, start.goal),
                (restored.physics, start.physics),
            )
        ):
            raise ValueError("Scene did not reproduce the matched start and goal")
        return restored.observation

    def step(self, action: np.ndarray) -> tuple[np.ndarray, bool, bool]:
        observation, _, terminated, truncated, info = self.environment.step(action)
        return (
            np.asarray(observation, dtype=np.float32),
            bool(info["success"]),
            bool(terminated or truncated),
        )
