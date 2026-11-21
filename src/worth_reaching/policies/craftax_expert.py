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

from worth_reaching.envs.craftax import ACTION_NAMES


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

        @jax.jit
        def sample(observation, carry, key):
            encoded = jax.nn.relu(observation @ dense[0]["kernel"] + dense[0]["bias"])
            carry, hidden = cell.apply({"params": gates}, carry, encoded)
            for layer in dense[1:3]:
                hidden = jax.nn.relu(hidden @ layer["kernel"] + layer["bias"])
            logits = hidden @ dense[3]["kernel"] + dense[3]["bias"]
            return carry, jax.random.categorical(key, logits)

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

    def start(self, world_seed: int) -> Callable[[NDArray[np.float32]], int]:
        carry = jnp.zeros(self._width, dtype=jnp.float32)
        stream = jax.random.fold_in(jax.random.key(world_seed), 2)
        index = 0

        def act(observation: NDArray[np.float32]) -> int:
            nonlocal carry, index
            if observation.shape != (self.observation_size,):
                raise ValueError(
                    "The checkpoint input does not match the symbolic observation"
                )
            key = jax.random.fold_in(stream, index)
            carry, action = self._sample(observation, carry, key)
            index += 1
            return int(action)

        return act
