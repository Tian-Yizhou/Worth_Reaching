"""Test whether frozen completion energy predicts value in PointMaze."""

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.models.continuous_completion import (
    BudgetedCompletionConfig,
    CompletionConfig,
    LearnedCompletion,
)
from worth_reaching.models.latent_energy import prefix_energy
from worth_reaching.models.training import (
    WEIGHTS_FILE,
    PredictorStopping,
    load_weights,
    stream_seed,
)
from worth_reaching.planning.goals import GoalSequence, Stage
from worth_reaching.planning.receding_goals import completion_context
from worth_reaching.tasks.pointmaze.dataset import ACTION_SIZE, load_dataset
from worth_reaching.tasks.pointmaze.environment import SEED_BASE, PointMaze
from worth_reaching.tasks.pointmaze.experiment import (
    ALIGNED_SCHEMA,
    BUDGET_SCHEMA,
    rollout,
    track,
)
from worth_reaching.tasks.pointmaze.spec import TASKS


@dataclass(frozen=True, slots=True)
class FrozenRun:
    directory: Path
    manifest: dict
    model: LearnedCompletion
    weights_sha256: str
    episodes: dict

    @classmethod
    def load(cls, directory: Path) -> "FrozenRun":
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema") not in {
            "pointmaze_latent_completion_v2",
            ALIGNED_SCHEMA,
            BUDGET_SCHEMA,
        }:
            raise ValueError(f"Expected a saved PyTorch PointMaze run in {directory}")
        if "action_limit" not in manifest["evaluation"]:
            raise ValueError(f"Choose a budget-<steps> subdirectory within {directory}")
        settings = dict(manifest["training"])
        if "predictor_stopping" in settings:
            settings["predictor_stopping"] = PredictorStopping(
                **settings["predictor_stopping"]
            )
        config_type = (
            BudgetedCompletionConfig
            if manifest["schema"] == BUDGET_SCHEMA
            else CompletionConfig
        )
        config = config_type(**settings)
        path = directory / WEIGHTS_FILE
        model = LearnedCompletion.from_weights(config, load_weights(path))
        raw = path.read_bytes()
        episodes = json.loads((directory / "episodes.json").read_text())
        return cls(directory, manifest, model, sha256(raw).hexdigest(), episodes)


def _declare(
    frozen: FrozenRun,
    seed: int,
    observation: np.ndarray,
    goal: np.ndarray,
    limit: int,
    prefix_length: int,
    candidates: int,
    density_samples: int,
) -> dict:
    model = frozen.model
    context = completion_context(
        [(observation, np.zeros(ACTION_SIZE, dtype=np.float32))],
        goal,
        limit,
        model.config,
    )
    _, forecast = model.forecast(context)
    key = stream_seed(model.config.seed, seed)
    candidate_seed = stream_seed(key, 0)
    reference_seed = stream_seed(key, 1)
    latent, decoded = model.propose(context, candidate_seed, candidates=candidates)
    reference, _ = model.propose(context, reference_seed, candidates=density_samples)
    latent = latent[0, :, :prefix_length]
    decoded = decoded[0, :, :prefix_length]
    forecast = forecast.numpy()[0, :prefix_length]
    if not np.isfinite(decoded).all() or not np.isfinite(forecast).all():
        raise ValueError("Frozen completion produced a non-finite forecast or target")
    energies = prefix_energy(
        latent, reference[0, :, :prefix_length], model.flow_scale.numpy()
    )
    errors = np.mean(np.square(latent - forecast), axis=(1, 2))
    return {
        "seed": seed,
        "start": observation.tolist(),
        "goal": goal.tolist(),
        "prediction_span": min(limit, model.config.max_horizon),
        "remaining_budget": limit,
        "deadlines": context.offsets[0, :prefix_length].tolist(),
        "candidate_seed": candidate_seed,
        "reference_seed": reference_seed,
        "forecast_latent_prefix": forecast.tolist(),
        "selections": {
            "lowest_energy": int(energies.argmin()),
            "highest_energy": int(energies.argmax()),
            "lowest_forecast_mse": int(errors.argmin()),
            "first_draw": 0,
        },
        "candidates": [
            {
                "index": index,
                "latent_prefix": points.tolist(),
                "targets": targets.tolist(),
                "energy": float(energy),
                "forecast_mse": float(error),
            }
            for index, (points, targets, energy, error) in enumerate(
                zip(latent, decoded, energies, errors, strict=True)
            )
        ],
    }


def attempt_prefix(
    env: PointMaze, seed: int, limit: int, targets: np.ndarray, deadlines: list[int]
) -> tuple[dict, np.ndarray]:
    """Track each declared target in order, then continue toward the terminal goal."""
    goals = GoalSequence(
        [
            Stage(target, deadline)
            for target, deadline in zip(targets, deadlines, strict=True)
        ],
        env.arrived,
    )
    step = 0

    def policy(observation, goal):
        nonlocal step
        stage = goals.advance(observation, step)
        action = track(observation, goal if stage is None else stage.target)
        step += 1
        return action

    episode, path = rollout(env, seed, limit, policy)
    goals.finish(path[-1], step)
    handoff = goals.handoff_step
    terminal = episode["first_success_step"]
    return {
        **episode,
        "stage_events": [asdict(event) for event in goals.events],
        "arrived_all": goals.arrived_all,
        "attempt_steps": handoff,
        "remaining_at_handoff": limit - handoff,
        "success_without_full_arrival": episode["success"] and not goals.arrived_all,
        "success_after_handoff": terminal is not None and terminal > handoff,
    }, path


def _summarize(rows: list[dict], direct: dict[int, dict]) -> dict:
    summary = {}
    for arm in rows[0]["selections"]:
        selected = [row["attempts"][row["selections"][arm]] for row in rows]
        summary[arm] = {
            "successes": sum(attempt["success"] for attempt in selected),
            "arrived_all": sum(attempt["arrived_all"] for attempt in selected),
            "success_without_full_arrival": sum(
                attempt["success_without_full_arrival"] for attempt in selected
            ),
            "mean_attempt_steps": float(
                np.mean([attempt["attempt_steps"] for attempt in selected])
            ),
            "only_selected": sum(
                attempt["success"] and not direct[attempt["seed"]]["success"]
                for attempt in selected
            ),
            "only_direct": sum(
                direct[attempt["seed"]]["success"] and not attempt["success"]
                for attempt in selected
            ),
        }
    low_high = [
        (
            row["attempts"][row["selections"]["lowest_energy"]]["success"],
            row["attempts"][row["selections"]["highest_energy"]]["success"],
        )
        for row in rows
    ]
    summary["lowest_vs_highest_energy"] = {
        "only_lowest": sum(low and not high for low, high in low_high),
        "only_highest": sum(high and not low for low, high in low_high),
    }
    summary["uniform_candidate_success_rate"] = float(
        np.mean(
            [
                np.mean([attempt["success"] for attempt in row["attempts"]])
                for row in rows
            ]
        )
    )
    return summary


def run(
    data_dir: Path,
    run_paths: list[Path],
    out: Path,
    *,
    split: str,
    repeats: int,
    prefix_length: int,
    candidates: int,
    density_samples: int,
) -> dict:
    if (
        not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"]
        or min(candidates, density_samples) < 2
    ):
        raise ValueError(
            "Use disjoint reset banks and at least two candidate and reference draws"
        )
    torch.set_num_threads(1)
    runs = [FrozenRun.load(path) for path in run_paths]
    base = runs[0].manifest
    limit = base["evaluation"]["action_limit"]
    config = runs[0].model.config
    if len({frozen.model.config.seed for frozen in runs}) != len(runs):
        raise ValueError("Supply distinct training seeds")
    if not 1 <= prefix_length <= config.queries:
        raise ValueError("The prefix must fit within the frozen model's query schedule")
    for frozen in runs:
        if (
            frozen.manifest["schema"] != base["schema"]
            or frozen.manifest["schema"] not in {ALIGNED_SCHEMA, BUDGET_SCHEMA}
            or frozen.manifest["evaluation"]["action_limit"] != limit
            or frozen.model.config.max_action_budget != config.max_action_budget
            or frozen.model.config.max_horizon != config.max_horizon
            or frozen.model.config.queries != config.queries
            or any(
                frozen.manifest[key] != base[key]
                for key in ("task", "dataset_files_sha256", "versions")
            )
        ):
            raise ValueError(
                "Use PointMaze runs with matched training protocols, budgets, archives, versions and query schedules"
            )
    task = TASKS[base["task"]]
    dataset, hashes = load_dataset(data_dir, task.dataset_id)
    if hashes != base["dataset_files_sha256"]:
        raise ValueError("Dataset files disagree with the frozen runs")
    for name, expected in base["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Expected {name} {expected} for frozen evaluation")

    seeds = range(SEED_BASE[split], SEED_BASE[split] + repeats)
    bank = {}
    native = dataset.recover_environment(eval_env=True)
    try:
        env = PointMaze(native, task)
        starts = {seed: env.reset(seed) for seed in seeds}
        for frozen in runs:
            bank[str(frozen.model.config.seed)] = [
                _declare(
                    frozen,
                    seed,
                    start,
                    goal,
                    limit,
                    prefix_length,
                    candidates,
                    density_samples,
                )
                for seed, (start, goal) in starts.items()
            ]

        out.mkdir(parents=True, exist_ok=False)
        bank_path = out / "candidate-bank.json"
        write_json(bank_path, bank)
        with bank_path.open("rb") as source:
            bank_hash = file_digest(source, "sha256").hexdigest()
        write_json(
            out / "manifest.json",
            {
                "schema": "pointmaze_latent_energy_v1",
                "task": base["task"],
                "dataset": task.dataset_id,
                "dataset_files_sha256": hashes,
                "versions": base["versions"],
                "action_limit": limit,
                "split": split,
                "seeds": list(seeds),
                "prefix_length": prefix_length,
                "candidates": candidates,
                "density_samples": density_samples,
                "energy": "negative log Gaussian-smoothed marginal flow density per latent coordinate; Scott bandwidth in normalized coordinates",
                "arrival": "native sparse public XY rule at each declared target",
                "execution": "track ordered targets until arrival or absolute query deadline; on first miss or final arrival track terminal goal",
                "candidate_bank_sha256": bank_hash,
                "source_runs": [
                    {
                        "path": str(frozen.directory),
                        "seed": frozen.model.config.seed,
                        "weights_sha256": frozen.weights_sha256,
                    }
                    for frozen in runs
                ],
                "code_revision": code_revision(),
            },
        )

        direct, trajectories = {}, {}
        for seed, (start, goal) in starts.items():
            row, path = rollout(env, seed, limit, track)
            if (row["start"], row["goal"]) != (start.tolist(), goal.tolist()):
                raise ValueError(f"Direct rollout disagrees with declared reset {seed}")
            direct[seed] = row
            trajectories[f"direct_{seed}"] = path

        records = defaultdict(list)
        for trained, declarations in bank.items():
            for declaration in declarations:
                seed = declaration["seed"]
                attempts = []
                for candidate in declaration["candidates"]:
                    index = candidate["index"]
                    targets = np.array(candidate["targets"], dtype=np.float32)
                    row, path = attempt_prefix(
                        env, seed, limit, targets, declaration["deadlines"]
                    )
                    if (row["start"], row["goal"]) != (
                        declaration["start"],
                        declaration["goal"],
                    ):
                        raise ValueError(
                            f"Candidate disagrees with declared reset {seed}"
                        )
                    trajectories[f"{trained}_{seed}_{index}"] = path
                    attempts.append({"index": index, **row})
                records[trained].append(
                    {
                        "seed": seed,
                        "selections": declaration["selections"],
                        "attempts": attempts,
                    }
                )
    finally:
        native.close()

    write_json(
        out / "episodes.json", {"direct": list(direct.values()), "models": records}
    )
    np.savez_compressed(out / "trajectories.npz", **trajectories)
    summary = {
        "starts": repeats,
        "direct_successes": sum(row["success"] for row in direct.values()),
        "models": {seed: _summarize(rows, direct) for seed, rows in records.items()},
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(
            len(path) - 1 for path in trajectories.values()
        ),
    }
    write_json(out / "summary.json", summary)
    return summary
