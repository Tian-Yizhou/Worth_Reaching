"""Adapter for the author-released PyBullet conditional stacking task."""

import random
import subprocess
import sys
from importlib import import_module
from pathlib import Path

import numpy as np
import pybullet as physics

from worth_reaching.tasks.kuka.dataset import JOINTS, STATE_SIZE, block_states

REVISION = "7c4d4da918c38f30241283d1f70d61a9b2293d09"
ACTION_LIMIT = 384
BLOCK_HEIGHT = 0.062
GRASP_LINK = 8
SUBSTEPS = 10
SEED_BASE = {"dev": 82001, "test": 91001}


class KukaStacking:
    def __init__(self, source_dir: Path):
        source_dir = source_dir.resolve()
        if not (source_dir / "gym_stacking/env.py").is_file():
            raise ValueError(f"Prepare the Diffuser Kuka source at {source_dir}")
        revision = subprocess.run(
            ["git", "-C", str(source_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        if revision.returncode or revision.stdout.strip() != REVISION:
            raise ValueError(
                f"Expected the pinned Diffuser Kuka revision at {source_dir}"
            )
        sys.path.insert(0, str(source_dir))
        try:
            upstream = import_module("gym_stacking.env")
            self.utils = import_module("gym_stacking.utils")
        finally:
            sys.path.pop(0)
        self.native = upstream.StackEnv(conditional=True)

    def close(self) -> None:
        self.utils.disconnect()

    def reset(self, seed: int) -> tuple[np.ndarray, np.ndarray]:
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        try:
            np.random.seed(seed)
            random.seed(seed)
            observation = self.native.reset()
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        self.joints = self.utils.get_movable_joints(self.native.robot)
        self.joint_lower, self.joint_upper = np.asarray(
            [
                self.utils.get_joint_limits(self.native.robot, joint)
                for joint in self.joints
            ]
        ).T
        return observation[:STATE_SIZE].copy(), self.native.goal.copy()

    def track(self, target: np.ndarray) -> tuple[np.ndarray, float]:
        positions = np.clip(target[:JOINTS], self.joint_lower, self.joint_upper)
        physics.setJointMotorControlArray(
            self.native.robot,
            self.joints,
            physics.POSITION_CONTROL,
            targetPositions=positions,
            positionGains=np.ones(JOINTS),
            physicsClientId=self.native.client,
        )
        grip = block_states(target)[:, -1]
        held = int(np.argmax(grip)) if grip.max() > 0.5 else None
        for index, cube in enumerate(self.native.cubes):
            attached = self.native.attachments[index] > 0.5
            if attached and index != held:
                self.utils.remove_fixed_constraint(cube, self.native.robot, GRASP_LINK)
                self.utils.set_velocity(cube, linear=[0, 0, 0], angular=[0, 0, 0])
                self.native.attachments[index] = 0
        if held is not None and self.native.attachments[held] == 0:
            self.utils.add_fixed_constraint(
                self.native.cubes[held], self.native.robot, GRASP_LINK
            )
            self.native.attachments[held] = 1

        for _ in range(SUBSTEPS):
            physics.stepSimulation(physicsClientId=self.native.client)
        observation = self.native.get_state()[:STATE_SIZE]
        return np.asarray(observation, dtype=np.float32), float(
            self.native.compute_reward()
        )

    def stacked_relations(
        self, observation: np.ndarray, order: np.ndarray
    ) -> list[bool]:
        blocks = block_states(observation)
        relations = []
        for bottom, top in zip(order[:-1], order[1:], strict=True):
            touching = physics.getContactPoints(
                self.native.cubes[bottom],
                self.native.cubes[top],
                physicsClientId=self.native.client,
            )
            relations.append(
                bool(
                    blocks[top, 2] > blocks[bottom, 2]
                    and blocks[top, -1] < 0.5
                    and blocks[bottom, -1] < 0.5
                    and bool(touching)
                )
            )
        return relations
