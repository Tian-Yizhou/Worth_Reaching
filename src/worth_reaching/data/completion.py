"""Prepare masked trajectory windows and expose their public model inputs."""

import json
import random
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.craftax_trajectories import SPLIT_VERSION, load_craftax_archive
from worth_reaching.envs.craftax import ACTION_NAMES, SURVIVAL_TASK

SCHEMA = "completion_windows_v1"
CompletionSplit = Literal["train", "dev"]


@dataclass(frozen=True, slots=True)
class WindowConfig:
    seed: int = 0
    windows_per_episode: int = 8
    min_span: int = 1
    max_span: int | None = None

    def __post_init__(self) -> None:
        if self.windows_per_episode < 1:
            raise ValueError("Use at least one window per episode")
        if self.min_span < 1 or (
            self.max_span is not None and self.max_span < self.min_span
        ):
            raise ValueError("Use positive mask spans with max-span >= min-span")

    def sample(self, world_id: str, budget: int) -> list[tuple[int, int]]:
        """One natural reset plus distinct later starts, with an interior mask each."""
        if self.min_span >= budget:
            raise ValueError("min-span must leave room for a separate terminal step")
        rng = random.Random(f"{SCHEMA}:{self.seed}:{world_id}")
        latest_start = budget - self.min_span - 1
        starts = [
            0,
            *rng.sample(
                range(1, latest_start + 1),
                min(latest_start, self.windows_per_episode - 1),
            ),
        ]
        windows = []
        for start in starts:
            span_limit = budget - start - 1
            if self.max_span is not None:
                span_limit = min(span_limit, self.max_span)
            windows.append((start, rng.randint(self.min_span, span_limit)))
        return windows


@dataclass(frozen=True, slots=True)
class CompletionInput:
    observations: NDArray[np.float32]
    actions: NDArray[np.int32]
    goal: str
    remaining_actions: int
    query_steps: NDArray[np.int32]
    terminal_example: NDArray[np.float32] | None = None


@dataclass(frozen=True, slots=True)
class CompletionTarget:
    observations: NDArray[np.float32]
    actions: NDArray[np.int32]


@dataclass(frozen=True, slots=True)
class CompletionExample:
    world_id: str
    episode_id: str
    start_step: int
    inputs: CompletionInput
    target: CompletionTarget


@dataclass(frozen=True, slots=True)
class CompletionCorpus:
    directory: Path
    goal: str
    budget: int

    def iter_examples(
        self, split: CompletionSplit, *, with_terminal_example: bool = False
    ) -> Iterator[CompletionExample]:
        """Yield independent model inputs and hidden targets from one split."""
        if split not in ("train", "dev"):
            raise ValueError("Completion windows are available for train and dev")
        if with_terminal_example and split != "train":
            raise ValueError("Recorded terminal examples are available for train only")
        for path in sorted((self.directory / split).glob("*.npz")):
            with np.load(path, allow_pickle=False) as episode:
                observations = episode["observations"]
                actions = episode["actions"]
                windows = episode["windows"].tolist()
                world_id = episode["world_id"].item()
                episode_id = episode["episode_id"].item()
                endpoint = (
                    episode["terminal_example"] if with_terminal_example else None
                )
            for start, span in windows:
                end = start + span
                yield CompletionExample(
                    world_id=world_id,
                    episode_id=episode_id,
                    start_step=start,
                    inputs=CompletionInput(
                        observations=observations[: start + 1].copy(),
                        actions=actions[:start].copy(),
                        goal=self.goal,
                        remaining_actions=self.budget - start,
                        query_steps=np.arange(1, span + 1, dtype=np.int32),
                        terminal_example=None if endpoint is None else endpoint.copy(),
                    ),
                    target=CompletionTarget(
                        observations=observations[start + 1 : end + 1].copy(),
                        actions=actions[start:end].copy(),
                    ),
                )


def load_completion_corpus(directory: Path) -> CompletionCorpus:
    if not (directory / "summary.json").is_file():
        raise ValueError(
            f"Expected a completed trajectory window export at {directory}"
        )
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema"] != SCHEMA or manifest["task"] != SURVIVAL_TASK:
        raise ValueError(f"Incompatible trajectory window export at {directory}")
    return CompletionCorpus(directory, manifest["task"], manifest["action_budget"])


def _public_arrays(
    trace: dict, budget: int
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    steps = trace["steps"]
    if len(steps) != budget or any(
        type(step["action"]) is not int
        or step["action"] not in ACTION_NAMES
        or step["remaining_actions"] != budget - index
        for index, step in enumerate(steps, start=1)
    ):
        raise ValueError(f"Invalid successful trajectory: {trace['episode_id']}")
    observations = np.asarray(
        [trace["initial_observation"], *(step["observation"] for step in steps)],
        dtype=np.float32,
    )
    if (
        observations.ndim != 2
        or observations.shape[1] == 0
        or not np.isfinite(observations).all()
    ):
        raise ValueError(f"Invalid public observations: {trace['episode_id']}")
    actions = np.asarray([step["action"] for step in steps], dtype=np.int32)
    return observations, actions


def prepare_craftax_windows(
    source: Path, out: Path, config: WindowConfig
) -> dict[str, object]:
    archive = load_craftax_archive(source)
    if archive.task != SURVIVAL_TASK:
        raise ValueError(f"Trajectory windows require the {SURVIVAL_TASK} task")
    budget = archive.manifest["config"]["budget"]
    if config.min_span >= budget:
        raise ValueError("min-span must leave room for a separate terminal step")

    out.mkdir(parents=True, exist_ok=False)
    counts = {}
    for split in ("train", "dev"):
        split_dir = out / split
        split_dir.mkdir()
        stats = {"episodes": 0, "successful_episodes": 0, "windows": 0}
        for spec, trace in archive.iter_episodes(split):
            stats["episodes"] += 1
            if trace["outcome"] != "success":
                continue
            observations, actions = _public_arrays(trace, budget)
            windows = config.sample(spec.world_id, budget)
            stats["successful_episodes"] += 1
            stats["windows"] += len(windows)
            np.savez_compressed(
                split_dir / f"{spec.world_seed:010d}.npz",
                observations=observations[:-1],
                actions=actions[:-1],
                windows=np.asarray(windows, dtype=np.int32),
                world_id=spec.world_id,
                episode_id=spec.episode_id,
                **({"terminal_example": observations[-1]} if split == "train" else {}),
            )
        counts[split] = stats
    if counts["train"]["windows"] == 0:
        raise ValueError("The collection needs a successful training episode")
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "task": archive.task,
            "action_budget": budget,
            "config": asdict(config),
            "split_version": SPLIT_VERSION,
            "source": str(source.resolve()),
            "source_manifest": archive.manifest,
            "source_splits_sha256": sha256(
                (source / "splits.json").read_bytes()
            ).hexdigest(),
            "code_revision": code_revision(),
        },
    )
    summary = {"splits": counts}
    write_json(out / "summary.json", summary)
    return summary
