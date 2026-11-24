"""Batch Scene's MuJoCo physics on CUDA while using the OGBench host controller."""

from collections.abc import Sequence

import numpy as np
import warp as wp
from mujoco.mjx.third_party import mujoco_warp
from mujoco.mjx.third_party.mujoco_warp._src import smooth

from worth_reaching.tasks.ogbench_manipulation.environment import Manipulation

# These fields carry the integration state between consecutive policy actions.
_INTEGRATION = ("qpos", "qvel", "act", "qacc_warmstart", "time")
_INPUTS = (
    "ctrl",
    "qfrc_applied",
    "xfrc_applied",
    "mocap_pos",
    "mocap_quat",
    "eq_active",
)
_CONTACTS_PER_WORLD = 64
_CONSTRAINTS_PER_WORLD = 256


class SceneWarpBatch:
    def __init__(self, environments: Sequence[Manipulation], device: str) -> None:
        self.scenes = [env.environment.unwrapped for env in environments]
        self.device = wp.get_device(device)
        if not self.device.is_cuda:
            raise ValueError("Scene Warp evaluation requires a CUDA device")
        first = self.scenes[0]
        self.substeps = first._n_steps
        with wp.ScopedDevice(self.device):
            self.model = mujoco_warp.put_model(first.model)
            self.data = mujoco_warp.put_data(
                first.model,
                first.data,
                nworld=len(self.scenes),
                nconmax=_CONTACTS_PER_WORLD,
                njmax=_CONSTRAINTS_PER_WORLD,
            )
            self.damping = np.stack([s.model.dof_damping for s in self.scenes])
            self.model.dof_damping = wp.array(self.damping, dtype=wp.float32)
            with wp.ScopedCapture() as capture:
                for _ in range(self.substeps):
                    mujoco_warp.step(self.model, self.data)
                smooth.rne_postconstraint(self.model, self.data)
            self.graph = capture.graph

    def _upload(self, names: tuple[str, ...]) -> None:
        for name in names:
            field = getattr(self.data, name)
            if field.size:
                field.assign(
                    np.stack([getattr(scene.data, name) for scene in self.scenes])
                )

    def reset(self) -> None:
        """Upload the native starts before executing the first policy action."""
        with wp.ScopedDevice(self.device):
            self._upload((*_INTEGRATION, *_INPUTS))
        self.host_corrected = False

    def step(
        self, actions: dict[int, np.ndarray]
    ) -> dict[int, tuple[np.ndarray, bool]]:
        for world, action in actions.items():
            native = self.scenes[world]
            native.set_control(action)
            native.pre_step()

        # Every scene has the same compiled model. Locks change the damping per world.
        damping = np.stack([s.model.dof_damping for s in self.scenes])
        with wp.ScopedDevice(self.device):
            if not np.array_equal(damping, self.damping):
                self.model.dof_damping.assign(damping)
                self.damping = damping
            # Keep inactive slots bounded and propagate any upstream stability reset.
            if self.host_corrected or len(actions) != len(self.scenes):
                self._upload(_INTEGRATION)
            self._upload(("ctrl",))
            wp.capture_launch(self.graph)
            state = {name: getattr(self.data, name).numpy() for name in _INTEGRATION}
            contact_forces = self.data.cfrc_ext.numpy()
            if int(self.data.nacon.numpy()[0]) >= self.data.naconmax or np.any(
                self.data.nefc.numpy() >= self.data.njmax
            ):
                raise RuntimeError("Scene Warp contact or constraint capacity exceeded")

        output = {}
        self.host_corrected = False
        for world in actions:
            native = self.scenes[world]
            if (
                not all(np.isfinite(values[world]).all() for values in state.values())
                or not np.isfinite(contact_forces[world]).all()
            ):
                raise RuntimeError(
                    f"Scene Warp returned non-finite state for world {world}"
                )
            native.data.time = float(state["time"][world])
            for name in ("qpos", "qvel", "act", "qacc_warmstart"):
                getattr(native.data, name)[:] = state[name][world]
            native.data.cfrc_ext[:] = contact_forces[world]
            # OGBench updates locks, forwards native kinematics, and scores the task here.
            native.post_step()
            self.host_corrected |= not np.array_equal(
                native.data.qpos, state["qpos"][world]
            )
            output[world] = (
                np.asarray(native.compute_observation(), dtype=np.float32),
                bool(native.terminate_episode()),
            )
        return output
