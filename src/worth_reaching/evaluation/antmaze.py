"""Train and evaluate one GCBC policy on the state-based OGBench benchmark."""

import csv
import json
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from flax import serialization

from worth_reaching.data.antmaze import (
    ACTION_SIZE,
    DATASET_URL,
    DATASETS,
    load_antmaze_archive,
)
from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.envs import antmaze
from worth_reaching.models import gcbc
from worth_reaching.models.training import (
    describe,
    regression,
    resume_checkpoint,
    train_stage,
    weights,
)

Policy = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True, slots=True)
class TrainConfig:
    seed: int = 7
    width: int = 256
    batch_size: int = 256
    learning_rate: float = 0.0003
    max_trajectories: int | None = None
    max_goal_steps: int | None = None

    def __post_init__(self) -> None:
        dimensions = [
            self.width,
            self.batch_size,
            self.max_trajectories,
            self.max_goal_steps,
        ]
        if any(size is not None and size < 1 for size in dimensions):
            raise ValueError("GCBC model and data limits must be positive")
        if not 0 <= self.seed < 2**32 or not 0 < self.learning_rate < np.inf:
            raise ValueError(
                "Use an unsigned 32-bit seed and a finite positive learning rate"
            )


@dataclass(frozen=True, slots=True)
class EvalConfig:
    split: str = "dev"
    repeats: int = 4
    action_limit: int = antmaze.ACTION_LIMIT

    def __post_init__(self) -> None:
        if (
            self.split not in antmaze.SEED_BASE
            or not 0
            < self.repeats
            <= (antmaze.SEED_BASE["test"] - antmaze.SEED_BASE["dev"])
            or not 0 < self.action_limit <= antmaze.ACTION_LIMIT
        ):
            raise ValueError(
                "Use disjoint dev/test seeds and an action limit within the benchmark horizon"
            )


def rollout(
    environment: antmaze.AntMaze,
    start: antmaze.AntMazeStart,
    policy: Policy,
    limit: int,
    *,
    after_step: Callable[[np.ndarray, int], None] | None = None,
) -> dict[str, object]:
    observation = environment.restore(start)
    goal = start.goal.copy()
    goal.flags.writeable = False
    initial_distance = closest_distance = distance = float(
        np.linalg.norm(observation[:2] - goal[:2])
    )
    for step in range(1, limit + 1):
        observation, success, terminated, truncated = environment.step(
            policy(observation, goal)
        )
        distance = float(np.linalg.norm(observation[:2] - goal[:2]))
        closest_distance = min(closest_distance, distance)
        if after_step is not None:
            after_step(observation, step)
        if success or terminated or truncated:
            break
    return {
        "task": start.task,
        "seed": start.seed,
        "success": success,
        "steps": step,
        "outcome": "success" if success else "terminated" if terminated else "timeout",
        "initial_distance": initial_distance,
        "closest_distance": closest_distance,
        "final_distance": distance,
    }


def summarize_episodes(results: list[dict]) -> dict:
    return {
        "episodes": len(results),
        "successes": sum(result["success"] for result in results),
        "success_rate": float(np.mean([result["success"] for result in results])),
        "steps": sum(result["steps"] for result in results),
        "mean_closest_distance": float(
            np.mean([result["closest_distance"] for result in results])
        ),
        "per_task": {
            str(task): float(
                np.mean(
                    [result["success"] for result in results if result["task"] == task]
                )
            )
            for task in antmaze.TASKS
        },
    }


def paired_metrics(direct: list[dict], guided: list[dict]) -> dict:
    pairs = list(zip(direct, guided, strict=True))
    if any((a["task"], a["seed"]) != (b["task"], b["seed"]) for a, b in pairs):
        raise ValueError("Paired AntMaze runs must share the same task and reset IDs")
    outcome_pairs = Counter((a["success"], b["success"]) for a, b in pairs)
    task_deltas = np.array(
        [
            np.mean(
                [
                    int(b["success"]) - int(a["success"])
                    for a, b in pairs
                    if a["task"] == task
                ]
            )
            for task in antmaze.TASKS
        ]
    )
    samples = np.random.default_rng(31).integers(
        len(task_deltas), size=(2000, len(task_deltas))
    )
    return {
        "both_succeeded": outcome_pairs[True, True],
        "only_direct": outcome_pairs[True, False],
        "only_guided": outcome_pairs[False, True],
        "neither_succeeded": outcome_pairs[False, False],
        "terminal_delta": float(task_deltas.mean()),
        "terminal_delta_task_bootstrap_95": np.quantile(
            task_deltas[samples].mean(axis=1), [0.025, 0.975]
        ).tolist(),
    }


def evaluation_starts(
    environment: antmaze.AntMaze,
    evaluation: EvalConfig,
    start_bank: Path | None,
) -> list[antmaze.AntMazeStart]:
    if start_bank is not None:
        return antmaze.load_starts(
            start_bank,
            split=evaluation.split,
            repeats=evaluation.repeats,
            state_size=environment.state_size,
        )
    return [
        environment.capture(task, seed)
        for task, seed in antmaze.start_ids(evaluation.split, evaluation.repeats)
    ]


def score(
    environment: antmaze.AntMaze,
    starts: list[antmaze.AntMazeStart],
    policies: dict[str, Policy],
    limit: int,
) -> tuple[list[dict], dict]:
    episodes = []
    metrics = {}
    for name, policy in policies.items():
        results = [rollout(environment, start, policy, limit) for start in starts]
        episodes.extend({"policy": name, **row} for row in results)
        metrics[name] = summarize_episodes(results)
    return episodes, metrics


def run(
    dataset: str,
    data_dir: Path,
    out: Path,
    train: TrainConfig,
    evaluation: EvalConfig,
    *,
    updates: int,
    checkpoint_every: int,
    resume: Path | None = None,
    start_bank: Path | None = None,
) -> dict:
    if dataset not in DATASETS or updates < 1 or checkpoint_every < 1:
        raise ValueError(
            "Use an official state AntMaze dataset and positive update counts"
        )
    train_path = data_dir / f"{dataset}.npz"
    validation_path = data_dir / f"{dataset}-val.npz"
    archive = load_antmaze_archive(train_path, max_trajectories=train.max_trajectories)
    validation = load_antmaze_archive(validation_path)
    if archive.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation must use distinct archives")
    torch.manual_seed(train.seed)
    model = gcbc.GCBC(train.width)
    mean = archive.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(archive.observations.std(axis=0, dtype=np.float64), 1e-3).astype(
        np.float32
    )
    provenance = {
        "schema": gcbc.CHECKPOINT_SCHEMA,
        "dataset": dataset,
        "dataset_url": DATASET_URL,
        "train_sha256": archive.fingerprint,
        "validation_sha256": validation.fingerprint,
        "training": asdict(train),
    }
    checkpoint = None
    first_step = 0
    if resume is not None:
        previous = json.loads((resume / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("schema") == gcbc.LEGACY_SCHEMA:
            raise ValueError(
                "Flax GCBC checkpoints can be used as frozen policies. Resume a Lightning run with --resume."
            )
        if any(previous.get(key) != value for key, value in provenance.items()):
            raise ValueError(
                "Resume checkpoint disagrees with the training recipe or dataset files"
            )
        checkpoint = resume_checkpoint(resume / "checkpoints" / "executor")
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        first_step = state["global_step"]
        model.load_state_dict(
            {
                name.removeprefix("model."): value
                for name, value in state["state_dict"].items()
            }
        )
    if updates < first_step:
        raise ValueError(
            "The requested total updates must be at least the checkpoint step"
        )

    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            **provenance,
            "environment": antmaze.ENVIRONMENT,
            "observation_schema": "ogbench-antmaze-state-v0",
            "versions": {
                key: version(key)
                for key in (
                    "ogbench",
                    "gymnasium",
                    "mujoco",
                    "torch",
                    "lightning",
                    "numpy",
                )
            },
            "compute": describe(),
            "evaluation": asdict(evaluation),
            "starts": str(start_bank.resolve()) if start_bank else None,
            "updates": updates,
            "checkpoint_every": checkpoint_every,
            "resumed_at": first_step,
            "resume": str(resume.resolve()) if resume else None,
            "code_revision": code_revision(),
            "data_dir": str(data_dir.resolve()),
        },
    )
    np.savez(out / "normalization.npz", mean=mean, scale=scale)
    validation_batch = validation.sample(
        np.random.default_rng([train.seed, 202]),
        4096,
        max_goal_steps=train.max_goal_steps,
    )

    heldout = gcbc.normalized_batch(validation_batch, mean, scale)
    with torch.no_grad():
        initial_error = gcbc.validation_error(model, heldout).item()
    stage = out / "checkpoints" / "executor"
    model = train_stage(
        model,
        gcbc.BehaviorBatches(
            archive,
            mean,
            scale,
            (train.seed, 101),
            train.batch_size,
            train.max_goal_steps,
        ),
        regression,
        steps=updates,
        learning_rate=train.learning_rate,
        out=stage,
        checkpoint_every=checkpoint_every,
        resume=checkpoint,
        validation=heldout,
        validation_metric=gcbc.validation_error,
    )
    history = [{"step": first_step, "validation_action_mse": initial_error}]
    metrics_file = stage / "metrics" / "version_0" / "metrics.csv"
    if metrics_file.is_file():
        with metrics_file.open(newline="", encoding="utf-8") as stream:
            for row in csv.DictReader(stream):
                if row.get("validation_loss"):
                    history.append(
                        {
                            "step": int(float(row["validation_update"])),
                            "validation_action_mse": float(row["validation_loss"]),
                        }
                    )
    with torch.no_grad():
        final_error = gcbc.validation_error(model, heldout).item()
    if history[-1]["step"] != updates:
        history.append({"step": updates, "validation_action_mse": final_error})
    write_json(out / "history.json", history)
    (out / "weights.msgpack").write_bytes(serialization.to_bytes(weights(model)))
    clone = gcbc.policy_from_model(model, mean, scale)

    def zero(observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        return np.zeros(ACTION_SIZE, dtype=np.float32)

    environment = antmaze.AntMaze()
    try:
        starts = evaluation_starts(environment, evaluation, start_bank)
        antmaze.save_starts(out / "starts.npz", starts)
        episodes, metrics = score(
            environment, starts, {"gcbc": clone, "zero": zero}, evaluation.action_limit
        )
    finally:
        environment.close()
    write_json(out / "episodes.json", episodes)
    summary = {
        "train_trajectories": len(archive.ends),
        "train_transitions": len(archive.rows),
        "validation_trajectories": len(validation.ends),
        "validation_initial_action_mse": initial_error,
        "validation_action_mse": history[-1]["validation_action_mse"],
        "updates": updates,
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(result["steps"] for result in episodes),
        "policies": metrics,
    }
    write_json(out / "summary.json", summary)
    return summary
