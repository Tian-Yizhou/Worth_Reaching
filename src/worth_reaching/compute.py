"""Resolve device visibility before loading numerical backends."""

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_CONFIG = Path("configs/compute.toml")


@dataclass(frozen=True, slots=True)
class ComputeConfig:
    platform: str = "auto"
    gpus: tuple[int, ...] | None = None

    @classmethod
    def load(cls, path: Path | None) -> "ComputeConfig":
        override = os.environ.get("WR_COMPUTE_CONFIG")
        if path is None and override is not None:
            path = Path(override)
        if path is None:
            if not DEFAULT_CONFIG.is_file():
                return cls()
            path = DEFAULT_CONFIG
        with path.open("rb") as stream:
            document = tomllib.load(stream)
        settings = document.get("compute")
        if (
            set(document) != {"compute"}
            or not isinstance(settings, dict)
            or (settings.keys() - {"platform", "gpus"})
        ):
            raise ValueError(
                f"Expected a [compute] table with platform and gpus in {path}"
            )
        platform = settings.get("platform", "auto")
        gpus = settings.get("gpus", "all")
        if platform not in ("auto", "cpu", "gpu"):
            raise ValueError("Compute platform must be auto, cpu or gpu")
        if gpus == "all":
            return cls(platform)
        if (
            not isinstance(gpus, list)
            or not gpus
            or any(type(index) is not int or index < 0 for index in gpus)
            or len(set(gpus)) != len(gpus)
            or platform == "cpu"
        ):
            raise ValueError(
                'Use gpus = "all" or distinct nonnegative indices with a GPU platform'
            )
        return cls(platform, tuple(gpus))

    def apply(self) -> None:
        inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["WR_TRAINING_PLATFORM"] = "gpu" if self.gpus else self.platform
        if self.gpus is None:
            os.environ.pop("WR_TRAINING_GPU_COUNT", None)
        else:
            os.environ["WR_TRAINING_GPU_COUNT"] = str(len(self.gpus))
        if self.platform == "cpu":
            os.environ["JAX_PLATFORMS"] = "cpu"
        elif self.platform == "gpu" or self.gpus is not None:
            os.environ["JAX_PLATFORMS"] = "cuda"
        elif inherited is not None and inherited.strip() in ("", "-1"):
            os.environ.setdefault("JAX_PLATFORMS", "cpu")
        os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

        if self.gpus is None:
            return
        selected = tuple(str(index) for index in self.gpus)
        if inherited is not None:
            visible = [token.strip() for token in inherited.split(",")]
            cutoff = next(
                (index for index, token in enumerate(visible) if token in ("", "-1")),
                len(visible),
            )
            if max(self.gpus) >= cutoff:
                raise ValueError(
                    "Requested GPU indices exceed the launcher's visible devices"
                )
            selected = tuple(visible[index] for index in self.gpus)
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected)
