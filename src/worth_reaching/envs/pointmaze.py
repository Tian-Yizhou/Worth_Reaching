"""Official Farama evaluation environment for the D4RL PointMaze tasks."""

from typing import NamedTuple

import gymnasium as gym
import numpy as np

from worth_reaching.envs.pointmaze_tasks import PointMazeTask

SEED_BASE = {"dev": 29001, "test": 69001}


class Step(NamedTuple):
    observation: np.ndarray
    reward: float
    success: bool
    done: bool


class PointMaze:
    def __init__(self, environment: gym.Env, task: PointMazeTask):
        self.environment = environment
        spec = environment.spec
        if (
            spec is None
            or spec.id != task.environment_id
            or spec.max_episode_steps != task.action_limit
            or spec.kwargs.get("reward_type") != "sparse"
            or spec.kwargs.get("continuing_task") is not True
            or spec.kwargs.get("reset_target") is not False
        ):
            raise ValueError(
                f"Expected the fixed-goal {task.environment_id} evaluation spec "
                f"with {task.action_limit} actions"
            )

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        observation, _ = self.environment.reset(seed=seed)
        state = np.asarray(observation["observation"], dtype=np.float32)
        goal = np.zeros_like(state)
        goal[:2] = observation["desired_goal"]
        return state, goal

    def step(self, action: np.ndarray) -> Step:
        observation, reward, terminated, truncated, info = self.environment.step(action)
        return Step(
            np.asarray(observation["observation"], dtype=np.float32),
            float(reward),
            bool(info["success"]),
            bool(terminated or truncated),
        )

    def arrived(self, observation: np.ndarray, target: np.ndarray) -> bool:
        """Apply the native sparse goal predicate to a public target position."""
        return bool(
            self.environment.unwrapped.compute_reward(observation[:2], target[:2], {})
        )
