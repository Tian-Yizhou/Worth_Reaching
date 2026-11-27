import json
import shutil

import pytest


def test_predictor_stops_on_masked_dev_error_and_resumes_with_its_best(
    tmp_path, monkeypatch
):
    torch = pytest.importorskip("torch")
    from worth_reaching.models.training import (
        PredictorStopping,
        RandomBatches,
        train_stage,
    )

    monkeypatch.setenv("WR_TRAINING_PLATFORM", "cpu")
    values = torch.ones(2, 1)
    training = RandomBatches(
        (values, values, torch.ones_like(values, dtype=torch.bool)), 2, (7,)
    )
    validation = (
        values,
        torch.tensor([[-1.0], [1000.0]]),
        torch.tensor([[True], [False]]),
    )

    def masked_error(model, batch):
        inputs, targets, mask = batch
        return ((model(inputs) - targets, mask),)

    def run(name, updates, resume=None):
        model = torch.nn.Linear(1, 1, bias=False)
        torch.nn.init.zeros_(model.weight)
        return train_stage(
            model,
            training,
            masked_error,
            steps=updates,
            learning_rate=0.1,
            out=tmp_path / name,
            validation=validation,
            stopping=PredictorStopping(val_every=1, patience=2),
            resume=resume,
        )

    selected = run("uninterrupted", 6)
    report = json.loads((tmp_path / "uninterrupted/selection.json").read_text())
    assert report["best_update"] == 1
    assert report["final_update"] == 3
    assert report["best_val_latent_mse"] == pytest.approx(
        (selected(values[:1]).item() + 1) ** 2
    )
    assert "did not improve" in report["stop_reason"]
    run("partial", 2)
    interrupted = tmp_path / "interrupted"
    interrupted.mkdir()
    for filename in ("last.ckpt", "best.ckpt"):
        shutil.copyfile(tmp_path / "partial" / filename, interrupted / filename)
    resumed = run("resumed", 6, interrupted / "last.ckpt")
    restored = run("completed", 6, tmp_path / "resumed/final.ckpt")
    for name, model in (("resumed", resumed), ("completed", restored)):
        assert json.loads((tmp_path / name / "selection.json").read_text()) == report
        torch.testing.assert_close(model.weight, selected.weight, rtol=0, atol=0)
    final = torch.load(tmp_path / "resumed/final.ckpt", weights_only=True)
    assert not torch.equal(final["state_dict"]["model.weight"], selected.weight)
