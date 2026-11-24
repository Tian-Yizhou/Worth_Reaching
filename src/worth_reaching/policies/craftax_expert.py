"""Inference for the published Craftax-Classic PPO-RNN actor."""

import json
from collections.abc import Callable
from hashlib import sha256
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
from flax import linen as nn
from numpy.typing import NDArray

from worth_reaching.tasks.craftax.environment import ACTION_NAMES


class CraftaxExpert:
    """Reset the actor's recurrent state and random stream for every episode."""

    def __init__(self, path: Path):
        path = path.resolve()
        if not path.is_dir():
            raise ValueError(f"Expert checkpoint directory does not exist: {path}")
        steps = [
            child
            for child in path.iterdir()
            if child.name.isdecimal() and (child / "default" / "_METADATA").is_file()
        ]
        if not steps:
            raise ValueError(f"No completed Orbax checkpoint steps found in {path}")
        latest = max(steps, key=lambda child: int(child.name))
        checkpoint = latest / "default"
        with ocp.PyTreeCheckpointer() as checkpointer:
            metadata = checkpointer.metadata(checkpoint).item_metadata.tree["params"][
                "params"
            ]
            restore_args = jax.tree.map(
                lambda _: ocp.RestoreArgs(restore_type=np.ndarray), metadata
            )
            state = checkpointer.restore(
                checkpoint,
                args=ocp.args.PyTreeRestore(
                    item={"params": {"params": metadata}},
                    restore_args={"params": {"params": restore_args}},
                    partial_restore=True,
                ),
            )

        parameters = state["params"]["params"]
        dense = tuple(parameters[f"Dense_{index}"] for index in range(4))
        self.observation_size, width = dense[0]["kernel"].shape
        if dense[-1]["kernel"].shape != (width, len(ACTION_NAMES)):
            raise ValueError(
                "The checkpoint action head does not match Craftax-Classic"
            )
        gates = parameters["ScannedRNN_0"]["GRUCell_1"]
        self._width = width

        # Match the actor in Craftax_Baselines. The critic is unused for collection.
        cell = nn.GRUCell(features=width)

        def predict(observation, carry):
            encoded = jax.nn.relu(observation @ dense[0]["kernel"] + dense[0]["bias"])
            carry, hidden = cell.apply({"params": gates}, carry, encoded)
            for layer in dense[1:3]:
                hidden = jax.nn.relu(hidden @ layer["kernel"] + layer["bias"])
            logits = hidden @ dense[3]["kernel"] + dense[3]["bias"]
            return carry, logits

        @jax.jit
        def sample(observation, carry, key):
            carry, logits = predict(observation, carry)
            return carry, jax.random.categorical(key, logits)

        self._predict = jax.jit(predict)
        self._sample = sample
        digest = sha256()
        for leaf in jax.tree.leaves((dense, gates)):
            digest.update(str((leaf.shape, leaf.dtype.str)).encode())
            digest.update(leaf.tobytes())
        checksum = digest.hexdigest()
        self.policy_id = f"ppo_rnn_expert_v1_{checksum[:12]}"
        self.metadata: dict[str, int | str] = {
            "checkpoint_step": int(latest.name),
            "actor_sha256": checksum,
        }
        source = path / "source.json"
        if source.is_file():
            provenance = json.loads(source.read_text(encoding="utf-8"))
            self.metadata.update(
                checkpoint_repo=provenance["repo"],
                checkpoint_revision=provenance["revision"],
            )

    def initial_memory(self) -> jax.Array:
        return jnp.zeros(self._width, dtype=jnp.float32)

    def _check_observation(self, observation: NDArray[np.float32]) -> None:
        if observation.shape != (self.observation_size,):
            raise ValueError(
                "The checkpoint input does not match the symbolic observation"
            )

    def observe(self, observation: NDArray[np.float32], carry: jax.Array) -> jax.Array:
        """Advance memory from a public observation without sampling an action."""
        self._check_observation(observation)
        return self._predict(observation, carry)[0]

    def sample(
        self, observation: NDArray[np.float32], carry: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, int]:
        self._check_observation(observation)
        memory, action = self._sample(observation, carry, key)
        return memory, int(action)

    def start(self, world_seed: int) -> Callable[[NDArray[np.float32]], int]:
        carry = self.initial_memory()
        stream = jax.random.fold_in(jax.random.key(world_seed), 2)
        index = 0

        def act(observation: NDArray[np.float32]) -> int:
            nonlocal carry, index
            key = jax.random.fold_in(stream, index)
            carry, action = self.sample(observation, carry, key)
            index += 1
            return action

        return act
