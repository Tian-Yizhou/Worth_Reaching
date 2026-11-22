import json
import os
import subprocess
import sys

import pytest

from worth_reaching.compute import ComputeConfig


def test_gpu_allowlist_keeps_the_launchers_visibility(tmp_path, monkeypatch):
    path = tmp_path / "compute.toml"
    path.write_text('[compute]\nplatform = "gpu"\ngpus = [1, 0]\n', encoding="utf-8")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-one,GPU-two")
    monkeypatch.setenv("JAX_PLATFORMS", "cpu")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    monkeypatch.setenv("WR_COMPUTE_CONFIG", str(path))
    ComputeConfig.load(None).apply()
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "GPU-two,GPU-one"
    assert os.environ["JAX_PLATFORMS"] == "cuda"
    with pytest.raises(ValueError, match="visible devices"):
        ComputeConfig("gpu", (2,)).apply()


def test_auto_uses_cpu_when_the_launcher_hides_all_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.delenv("JAX_PLATFORMS", raising=False)
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    ComputeConfig().apply()
    assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_device_cli_shards_a_padded_batch_on_the_cpu_mesh(tmp_path):
    pytest.importorskip("jax")
    config = tmp_path / "cpu.toml"
    config.write_text('[compute]\nplatform = "cpu"\n', encoding="utf-8")
    report = subprocess.run(
        ["wr", "--compute-config", config, "devices", "--batch-size", "5"],
        env={**os.environ, "JAX_NUM_CPU_DEVICES": "3"},
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(report.stdout)
    assert result["backend"] == "cpu"
    assert result["device_count"] == 3
    assert result["rows_per_device"] == [2, 2, 2]
    assert result["padded_rows"] == 1
    assert result["mean_squared"] == 6.0


def test_partitioned_optimizer_update_agrees_with_one_device(tmp_path):
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    import numpy as np

    script = """\
import sys
import jax
import numpy as np
from worth_reaching.models import gcbc, parallel
_, state = gcbc.initialize(8, seed=7, learning_rate=0.0003)
rng = np.random.default_rng(12)
data = tuple(rng.standard_normal(x).astype(np.float32) for x in ((5, 29), (5, 29), (5, 8)))
tensors, valid = parallel.shard_batch(data)
state = gcbc.update(parallel.replicate(state), *tensors, valid)
leaves = jax.tree.leaves(state.params)
assert all(leaf.is_fully_replicated for leaf in leaves)
assert all(len(leaf.addressable_shards) == int(sys.argv[1]) for leaf in leaves)
np.save(sys.argv[2], np.concatenate([np.asarray(leaf).ravel() for leaf in leaves]))
"""
    weights = []
    for count in (1, 3):
        path = tmp_path / f"{count}.npy"
        subprocess.run(
            [sys.executable, "-c", script, str(count), path],
            env={
                **os.environ,
                "JAX_PLATFORMS": "cpu",
                "JAX_NUM_CPU_DEVICES": str(count),
            },
            check=True,
        )
        weights.append(np.load(path))
    np.testing.assert_allclose(*weights, atol=1e-7, rtol=1e-6)
