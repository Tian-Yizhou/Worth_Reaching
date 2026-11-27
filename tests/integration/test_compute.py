import json
import os
import subprocess

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


def test_lightning_batch_padding_preserves_the_sampled_global_batch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("lightning")
    from worth_reaching.models.training import RandomBatches, UpdateBatches

    data = torch.arange(13)
    sample = RandomBatches((data,), 5, (7,))
    (reference,), single_mask = UpdateBatches(sample, 4, 0, 1)[3]
    shards = [UpdateBatches(sample, 4, rank, 3)[3] for rank in range(3)]
    values = torch.stack([batch[0] for batch, _ in shards], dim=1).flatten()
    mask = torch.stack([valid for _, valid in shards], dim=1).flatten()
    assert single_mask.all()
    assert torch.equal(reference, values[mask])
    assert mask.tolist() == [True, True, True, True, True, False]


def test_devices_reports_lightning_by_default(tmp_path):
    pytest.importorskip("lightning")
    config = tmp_path / "cpu.toml"
    config.write_text('[compute]\nplatform = "cpu"\n', encoding="utf-8")
    report = subprocess.run(
        ["wr", "--compute-config", config, "devices"],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads(report.stdout)
    assert result["framework"] == "pytorch-lightning"
    assert result["backend"] == "cpu"
    assert result["device_count"] == 1
