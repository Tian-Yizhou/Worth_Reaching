"""Load AntMaze goal-conditioned behavior cloning checkpoints."""

import json
from pathlib import Path

import numpy as np
from flax import serialization

from worth_reaching.models.codec import legacy_dense
from worth_reaching.models.gcbc import GCBC, policy_from_model
from worth_reaching.models.training import restore_weights
from worth_reaching.tasks.antmaze.dataset import ACTION_SIZE, OBSERVATION_SIZE

CHECKPOINT_SCHEMA = "antmaze_gcbc_v2"
LEGACY_SCHEMA = "antmaze_gcbc_v1"


def load_policy(directory: Path):
    """Read frozen Lightning actors and previously published Flax actors."""
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("schema") not in (CHECKPOINT_SCHEMA, LEGACY_SCHEMA):
        raise ValueError("Expected an AntMaze GCBC checkpoint")
    model = GCBC(manifest["training"]["width"], OBSERVATION_SIZE, ACTION_SIZE)
    values = serialization.msgpack_restore((directory / "weights.msgpack").read_bytes())
    if manifest["schema"] == LEGACY_SCHEMA:
        values = legacy_dense(
            values["params"], "net", *(f"Dense_{index}" for index in range(4))
        )
    restore_weights(model, values)
    with np.load(directory / "normalization.npz", allow_pickle=False) as arrays:
        mean, scale = arrays["mean"], arrays["scale"]
    if (
        mean.shape != (OBSERVATION_SIZE,)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not (np.isfinite(scale) & (scale > 0)).all()
    ):
        raise ValueError("Invalid GCBC normalization")
    return policy_from_model(model, mean, scale)
