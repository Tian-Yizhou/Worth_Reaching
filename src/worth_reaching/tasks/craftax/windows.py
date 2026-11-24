"""Export and validate masked windows for the Craftax craft-and-survive task."""

import json
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.completion import SCHEMA, CompletionCorpus, WindowConfig
from worth_reaching.tasks.craftax.dataset import SPLIT_VERSION, load_craftax_archive
from worth_reaching.tasks.craftax.environment import ACTION_NAMES, SURVIVAL_TASK


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
    trace: dict, budget: int, action_names: dict[int, str]
) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    steps = trace["steps"]
    if len(steps) != budget or any(
        type(step["action"]) is not int
        or step["action"] not in action_names
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
            observations, actions = _public_arrays(trace, budget, ACTION_NAMES)
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
