"""Data-parallel JAX placement for replicated training state and global batches."""

import os
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


@lru_cache(maxsize=1)
def training_mesh() -> Mesh:
    devices = jax.local_devices()
    return jax.make_mesh((len(devices),), ("data",), (AxisType.Auto,), devices=devices)


def shard_batch(tree):
    """Pad standard JAX batches when the global row count does not divide the mesh."""
    mesh = training_mesh()
    if mesh.size == 1:
        return tree, None
    size = jax.tree.leaves(tree)[0].shape[0]
    padding = -size % mesh.size
    placement = NamedSharding(mesh, P("data"))

    def place(leaf):
        if padding:
            xp = jnp if isinstance(leaf, jax.Array) else np
            leaf = xp.concatenate((leaf, xp.repeat(leaf[-1:], padding, axis=0)))
        return jax.device_put(leaf, placement)

    valid = (
        jax.device_put(np.arange(size + padding) < size, placement) if padding else None
    )
    return jax.tree.map(place, tree), valid


def replicate(tree):
    mesh = training_mesh()
    return jax.device_put(tree, NamedSharding(mesh, P())) if mesh.size > 1 else tree


def single_device(tree):
    mesh = training_mesh()
    return jax.device_put(tree, mesh.devices.flat[0]) if mesh.size > 1 else tree


def describe() -> dict[str, object]:
    mesh = training_mesh()
    return {
        "backend": jax.default_backend(),
        "device_count": mesh.size,
        "devices": [
            {"id": device.id, "kind": device.device_kind}
            for device in mesh.devices.flat
        ],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": "data_parallel" if mesh.size > 1 else "single_device",
    }


def mean_squared(error: jax.Array, valid: jax.Array | None) -> jax.Array:
    squared = jnp.square(error)
    if valid is None:
        return squared.mean()
    per_row = squared.mean(axis=tuple(range(1, error.ndim)))
    return jnp.sum(jnp.where(valid, per_row, 0)) / jnp.sum(valid)


def probe(batch_size: int) -> dict[str, object]:
    if batch_size < 1:
        raise ValueError("Device probe batch size must be positive")
    batch, valid = shard_batch(np.arange(batch_size, dtype=np.float32))
    batch = jax.device_put(batch)
    mse = jax.jit(mean_squared)(batch, valid)
    return {
        **describe(),
        "global_batch": batch_size,
        "padded_rows": len(batch) - batch_size,
        "rows_per_device": [len(shard.data) for shard in batch.addressable_shards],
        "mean_squared": float(mse),
    }
