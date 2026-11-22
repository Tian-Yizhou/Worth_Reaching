"""Lightning training for step-indexed batches drawn from public trajectories."""

import json
import os
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset
from torchdata.stateful_dataloader import StatefulDataLoader

Batch = tuple[Tensor, ...]
LossTerms = tuple[tuple[Tensor, Tensor | None], ...]


def device_count() -> int:
    platform = os.environ.get("WR_TRAINING_PLATFORM", "auto")
    count = torch.cuda.device_count() if platform != "cpu" else 0
    if platform == "gpu" and count == 0:
        raise ValueError(
            "PyTorch could not initialize CUDA. Install the training extra."
        )
    requested = os.environ.get("WR_TRAINING_GPU_COUNT")
    if requested is not None and count != int(requested):
        raise ValueError("PyTorch could not initialize all requested GPUs")
    return count


def describe() -> dict:
    count = device_count()
    names = [torch.cuda.get_device_name(index) for index in range(count)]
    return {
        "framework": "pytorch-lightning",
        "torch_version": str(torch.__version__),
        "lightning_version": pl.__version__,
        "backend": "gpu" if count else "cpu",
        "device_count": count or 1,
        "devices": names or ["cpu"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "training": "data_parallel" if count > 1 else "single_device",
    }


def tensor(values) -> Tensor:
    if isinstance(values, np.ndarray) and not values.flags.writeable:
        values = values.copy()
    return torch.as_tensor(values, dtype=torch.float32)


def weights(model: nn.Module) -> dict[str, np.ndarray]:
    return {
        name: value.detach().cpu().numpy() for name, value in model.state_dict().items()
    }


def restore_weights(model: nn.Module, values: dict) -> nn.Module:
    model.load_state_dict({name: torch.tensor(value) for name, value in values.items()})
    return model.eval()


def stream_seed(*parts: int) -> int:
    return np.random.SeedSequence(parts).generate_state(1, dtype=np.uint64).item()


@dataclass(frozen=True)
class RandomBatches:
    arrays: Batch
    batch_size: int
    seed: tuple[int, ...]

    def __call__(self, step: int) -> Batch:
        rng = np.random.default_rng([*self.seed, step])
        rows = rng.integers(len(self.arrays[0]), size=self.batch_size)
        return tuple(array[rows] for array in self.arrays)


def regression(model: nn.Module, batch: Batch) -> LossTerms:
    return ((model(*batch[:-1]) - batch[-1], None),)


class UpdateBatches(Dataset):
    """Use the same sampled global batch on every device count."""

    def __init__(
        self, sample: Callable[[int], Batch], steps: int, rank: int, ranks: int
    ):
        self.sample = sample
        self.steps = steps
        self.rank = rank
        self.ranks = ranks

    def __len__(self) -> int:
        return self.steps

    def __getitem__(self, step: int) -> tuple[Batch, Tensor]:
        batch = self.sample(step)
        count = (len(batch[0]) + self.ranks - 1) // self.ranks
        rows = self.rank + self.ranks * torch.arange(count)
        valid = rows < len(batch[0])
        return tuple(value[rows.clamp_max(len(value) - 1)] for value in batch), valid


class TrainStage(pl.LightningModule):
    def __init__(
        self, model, sample, objective, learning_rate, steps, validation_metric
    ):
        super().__init__()
        self.model = model
        self.sample = sample
        self.objective = objective
        self.learning_rate = learning_rate
        self.steps = steps
        self.validation_metric = validation_metric

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def train_dataloader(self):
        # Keep the step-limited run inside the epoch so extending it resumes the
        # saved dataloader cursor instead of starting a fresh epoch.
        batches = UpdateBatches(
            self.sample, self.steps + 1, self.global_rank, self.trainer.world_size
        )
        return StatefulDataLoader(batches, batch_size=None)

    def training_step(self, batch, batch_idx):
        values, valid_rows = batch
        loss = torch.zeros((), device=self.device)
        for error, mask in self.objective(self.model, values):
            valid = valid_rows.reshape(-1, *((1,) * (error.ndim - 1)))
            if mask is not None:
                valid = valid & mask
            count = valid.expand_as(error).sum().to(error.dtype)
            count = self.trainer.strategy.reduce(count, reduce_op="sum")
            loss = loss + (
                error.square().masked_fill(~valid, 0).sum()
                * self.trainer.world_size
                / count.clamp_min(1)
            )
        self.log("train_loss", loss, on_step=True, on_epoch=False, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.log(
            "validation_loss", self.validation_metric(self.model, batch), sync_dist=True
        )
        self.log("validation_update", float(self.global_step), sync_dist=True)


class FinalCheckpoint(Callback):
    def __init__(self, path: Path):
        self.path = path

    def on_train_end(self, trainer, pl_module) -> None:
        trainer.save_checkpoint(self.path)


def resume_checkpoint(directory: Path) -> Path:
    final = directory / "final.ckpt"
    if final.is_file():
        return final
    latest = directory / "last.ckpt"
    if latest.is_file():
        return latest
    raise ValueError(f"No Lightning checkpoint in {directory}")


def validate_resume(out: Path, previous: Path | None) -> Path | None:
    if previous is None:
        return None
    current = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    saved = json.loads((previous / "manifest.json").read_text(encoding="utf-8"))
    ignored = {"code_revision", "compute"}
    if any(
        saved.get(key) != value for key, value in current.items() if key not in ignored
    ):
        raise ValueError(
            "Resume requires the same experiment settings and input artifacts"
        )
    return previous / "checkpoints"


def stage_resume(previous: Path | None, stage: str) -> Path | None:
    if previous is None:
        return None
    directory = previous / stage
    if any((directory / name).is_file() for name in ("final.ckpt", "last.ckpt")):
        return resume_checkpoint(directory)
    return None


def train_stage(
    model: nn.Module,
    sample: Callable[[int], Batch],
    objective: Callable[[nn.Module, Batch], LossTerms],
    *,
    steps: int,
    learning_rate: float,
    out: Path,
    checkpoint_every: int = 1000,
    resume: Path | None = None,
    validation: Batch | None = None,
    validation_metric: Callable[[nn.Module, Batch], Tensor] | None = None,
) -> nn.Module:
    out.mkdir(parents=True, exist_ok=True)
    stage = TrainStage(
        model, sample, objective, learning_rate, steps, validation_metric
    )
    if resume is not None:
        previous = torch.load(resume, map_location="cpu", weights_only=True)
        if previous["global_step"] > steps:
            raise ValueError("Requested updates are fewer than the checkpoint step")
        if previous["global_step"] == steps:
            stage.load_state_dict(previous["state_dict"])
            shutil.copy2(resume, out / "final.ckpt")
            return stage.model.eval()
        del previous

    count = device_count()
    torch.set_num_threads(int(os.environ.get("WR_TORCH_THREADS", "1")))
    trainer = pl.Trainer(
        accelerator="gpu" if count else "cpu",
        devices=count or 1,
        strategy="ddp_spawn" if count > 1 else "auto",
        max_steps=steps,
        max_epochs=-1,
        use_distributed_sampler=False,
        val_check_interval=min(checkpoint_every, steps),
        check_val_every_n_epoch=None,
        num_sanity_val_steps=0,
        enable_model_summary=False,
        enable_progress_bar=False,
        log_every_n_steps=min(50, steps),
        default_root_dir=out,
        logger=CSVLogger(out, name="metrics"),
        callbacks=[
            ModelCheckpoint(
                dirpath=out,
                save_last=True,
                save_top_k=0,
                every_n_train_steps=checkpoint_every,
            ),
            FinalCheckpoint(out / "final.ckpt"),
        ],
    )
    validation_loader = (
        DataLoader([validation], batch_size=None) if validation is not None else None
    )
    trainer.fit(
        stage, ckpt_path=resume, val_dataloaders=validation_loader, weights_only=True
    )
    return stage.model.cpu().eval()
