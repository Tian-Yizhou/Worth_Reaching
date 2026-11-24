"""Evaluate the first decoded target from frozen PointMaze models."""

import json
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from hashlib import file_digest, sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.models.continuous_completion import (
    BudgetedCompletionConfig,
    CompletionConfig,
    LearnedCompletion,
)
from worth_reaching.models.training import PredictorStopping, stream_seed
from worth_reaching.planning.receding_goals import RecedingGoals, completion_context
from worth_reaching.tasks.pointmaze.dataset import ACTION_SIZE, load_dataset
from worth_reaching.tasks.pointmaze.environment import PointMaze
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
        raw = (directory / "weights.msgpack").read_bytes()
        model = LearnedCompletion.from_weights(
            config, serialization.msgpack_restore(raw)
        )
        episodes = json.loads((directory / "episodes.json").read_text())
        return cls(directory, manifest, model, sha256(raw).hexdigest(), episodes)


def attempt(
    env: PointMaze,
    seed: int,
    limit: int,
    target: np.ndarray,
    deadline: int,
    planner: RecedingGoals | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Execute a first target and return the rollout and its prefix actions."""
    prefix_actions = []
    continuing = False

    def policy(observation, goal):
        nonlocal continuing
        if not continuing:
            continuing = len(prefix_actions) == deadline or env.arrived(
                observation, target
            )
        if continuing:
            return (
                track(observation, goal)
                if planner is None
                else planner(observation, goal)
            )
        action = track(observation, target)
        prefix_actions.append(action)
        if planner is not None:
            planner.record_prefix_step(observation, action)
        return action

    episode, path = rollout(env, seed, limit, policy)
    cutoff = min(deadline, episode["steps"])
    arrival = next(
        (
            step
            for step, state in enumerate(path[: cutoff + 1])
            if env.arrived(state, target)
        ),
        None,
    )
    end = cutoff if arrival is None else arrival
    hit = episode["first_success_step"]
    return (
        {
            **episode,
            "target": target.tolist(),
            "deadline": deadline,
            "arrival_step": arrival,
            "attempt_end_step": end,
            "success_without_arrival": bool(episode["success"] and arrival is None),
            "success_after_timeout": bool(
                arrival is None and hit is not None and hit > end
            ),
        },
        path,
        np.asarray(prefix_actions, dtype=np.float32).reshape(-1, ACTION_SIZE),
    )


def continuation_summary(pairs: list[dict]) -> dict:
    def counts(rows):
        return {
            "pairs": len(rows),
            "direct_successes": sum(row["direct"]["success"] for row in rows),
            "replanning_successes": sum(row["replanning"]["success"] for row in rows),
            "replanning_only": sum(
                row["replanning"]["success"] and not row["direct"]["success"]
                for row in rows
            ),
            "direct_only": sum(
                row["direct"]["success"] and not row["replanning"]["success"]
                for row in rows
            ),
        }

    return {
        **counts(pairs),
        "arrived": counts(
            [row for row in pairs if row["direct"]["arrival_step"] is not None]
        ),
        "missed": counts(
            [row for row in pairs if row["direct"]["arrival_step"] is None]
        ),
        "arrivals_at_reset": sum(row["direct"]["arrival_step"] == 0 for row in pairs),
        "successes_in_shared_prefix": sum(row["success_in_prefix"] for row in pairs),
        "continuation_plans": sum(len(row["plans"]) for row in pairs),
        "rollout_seconds": {
            arm: sum(row["rollout_seconds"][arm] for row in pairs)
            for arm in ("direct", "replanning")
        },
    }


def compare_after_prefix(
    env: PointMaze,
    seed: int,
    limit: int,
    target: np.ndarray,
    deadline: int,
    planner: RecedingGoals,
) -> tuple[dict, dict[str, np.ndarray]]:
    started = perf_counter()
    direct, direct_path, prefix_actions = attempt(env, seed, limit, target, deadline)
    direct_seconds = perf_counter() - started
    started = perf_counter()
    replanning, replanned_path, replayed_actions = attempt(
        env, seed, limit, target, deadline, planner
    )
    replanning_seconds = perf_counter() - started
    handoff = direct["attempt_end_step"]
    if (
        len(prefix_actions) != handoff
        or (
            replanning["attempt_end_step"],
            replanning["arrival_step"],
            replanning["goal"],
        )
        != (handoff, direct["arrival_step"], direct["goal"])
        or not np.array_equal(prefix_actions, replayed_actions)
        or not np.array_equal(direct_path[: handoff + 1], replanned_path[: handoff + 1])
    ):
        raise ValueError(
            f"Continuations disagree on the executed prefix at reset {seed}"
        )

    def prefix_hit(episode):
        hit = episode["first_success_step"]
        return hit if hit is not None and hit <= handoff else None

    if prefix_hit(direct) != prefix_hit(replanning):
        raise ValueError(
            f"Continuations disagree on terminal events before handoff at reset {seed}"
        )
    row = {
        "seed": seed,
        "handoff_step": handoff,
        "success_in_prefix": prefix_hit(direct) is not None,
        "direct": direct,
        "replanning": replanning,
        "plans": [
            {
                "step": plan["step"],
                "deadline": plan["deadline"],
                "target": plan["decoded_routes"][plan["selected_route"]][
                    plan["selected_query"]
                ],
            }
            for plan in planner.plans
        ],
        "rollout_seconds": {"direct": direct_seconds, "replanning": replanning_seconds},
    }
    return row, {
        "direct": direct_path,
        "replanning": replanned_path,
        "prefix_actions": prefix_actions,
    }


def reference_targets(runs: list[FrozenRun], seeds: list[int], deadline: int) -> dict:
    """Use the earliest successful archived trajectory at each reset as a diagnostic."""
    eligible = defaultdict(list)
    starts = {row["seed"]: row["start"] for row in runs[0].episodes["direct"]}
    for index, run in enumerate(runs):
        for arm in ("direct", "latent"):
            for row in run.episodes[arm]:
                hit = row["first_success_step"]
                if row["seed"] in seeds and hit is not None and hit > deadline:
                    eligible[row["seed"]].append((hit, index, arm))
    targets = {}
    with ExitStack() as stack:
        archives = [
            stack.enter_context(
                np.load(item.directory / "trajectories.npz", allow_pickle=False)
            )
            for item in runs
        ]
        for seed, candidates in eligible.items():
            hit, index, arm = min(candidates)
            path = archives[index][f"{arm}_{seed}"]
            if (
                path.ndim != 2
                or len(path) <= hit
                or not np.array_equal(path[0], starts[seed])
                or not np.isfinite(path[deadline]).all()
            ):
                raise ValueError(f"Invalid archived reference at reset {seed}")
            targets[seed] = (
                path[deadline],
                {
                    "training_seed": runs[index].model.config.seed,
                    "arm": arm,
                    "source_first_success": hit,
                },
            )
    return targets


def paired_summary(rows: list[dict], direct: dict[int, dict]) -> dict:
    arrived = [row for row in rows if row["arrival_step"] is not None]
    return {
        "attempts": len(rows),
        "arrivals": len(arrived),
        "arrivals_at_reset": sum(row["arrival_step"] == 0 for row in arrived),
        "mean_arrival_step": float(np.mean([row["arrival_step"] for row in arrived]))
        if arrived
        else None,
        "successes": sum(row["success"] for row in rows),
        "success_with_arrival": sum(row["success"] for row in arrived),
        "success_without_arrival": sum(row["success_without_arrival"] for row in rows),
        "success_after_timeout": sum(row["success_after_timeout"] for row in rows),
        "only_target": sum(
            row["success"] and not direct[row["seed"]]["success"] for row in rows
        ),
        "only_direct": sum(
            not row["success"] and direct[row["seed"]]["success"] for row in rows
        ),
    }


def run(
    data_dir: Path,
    run_paths: list[Path],
    out: Path,
    max_starts: int | None = None,
    *,
    compare_continuations: bool = False,
) -> dict:
    if max_starts is not None and max_starts < 1:
        raise ValueError("Use a positive number of starts")
    torch.set_num_threads(1)
    runs = [FrozenRun.load(path) for path in run_paths]
    base = runs[0].manifest
    if len({item.model.config.seed for item in runs}) != len(runs):
        raise ValueError("Supply runs with distinct training seeds")
    for item in runs[1:]:
        if any(
            item.manifest[key] != base[key]
            for key in ("task", "evaluation", "dataset_files_sha256", "versions")
        ):
            raise ValueError(
                "Frozen runs must share the task, archives, versions and matched start protocol"
            )
        if item.episodes["direct"] != runs[0].episodes["direct"]:
            raise ValueError("Frozen runs disagree on archived direct rollouts")
    task = TASKS[base["task"]]
    dataset, hashes = load_dataset(data_dir, task.dataset_id)
    if hashes != base["dataset_files_sha256"]:
        raise ValueError("Dataset files disagree with the frozen runs")
    for name, expected in base["versions"].items():
        if version(name) != expected:
            raise ValueError(f"Expected {name} {expected} for frozen evaluation")
    evaluation = base["evaluation"]
    seeds = evaluation["seeds"][:max_starts]
    limit = evaluation["action_limit"]
    config = runs[0].model.config
    if (
        any(
            item.model.config.queries != config.queries
            or item.model.config.max_horizon != config.max_horizon
            or item.model.config.max_action_budget != config.max_action_budget
            for item in runs
        )
        or min(limit, config.max_horizon) <= config.queries
    ):
        raise ValueError(
            "Frozen runs need a common schedule with at least one interior query"
        )
    deadline = min(limit, config.max_horizon) // (config.queries + 1)
    references = (
        {} if compare_continuations else reference_targets(runs, seeds, deadline)
    )
    archived = {row["seed"]: row for row in runs[0].episodes["direct"]}

    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "pointmaze_continuations_v1"
            if compare_continuations
            else "pointmaze_first_targets_v1",
            "task": base["task"],
            "dataset": task.dataset_id,
            "dataset_files_sha256": hashes,
            "versions": base["versions"],
            "evaluation": {**evaluation, "seeds": seeds},
            "attempt_deadline": deadline,
            "arrival_rule": "native sparse goal predicate on public XY",
            **(
                {
                    "first_target": "first sampled route, earliest queried state",
                    "handoff": "arrival or first query deadline",
                    "continuations": [
                        "direct terminal tracking",
                        "replan with the executed history",
                    ],
                }
                if compare_continuations
                else {
                    "continuation": "track terminal goal after arrival or deadline",
                    "reference_rule": "same-reset archived successful public trajectory, sampled at the declared deadline",
                }
            ),
            "source_runs": [
                {
                    "path": str(item.directory.resolve()),
                    "training_seed": item.model.config.seed,
                    "weights_sha256": item.weights_sha256,
                    "episodes_sha256": _digest(item.directory / "episodes.json"),
                    "trajectories_sha256": _digest(item.directory / "trajectories.npz"),
                }
                for item in runs
            ],
            "code_revision": code_revision(),
        },
    )
    native = dataset.recover_environment(eval_env=True)
    direct, recorded = {}, {}
    by_model = {str(item.model.config.seed): defaultdict(list) for item in runs}
    pairs_by_model = defaultdict(list)
    reference_rows = []
    try:
        env = PointMaze(native, task)
        for seed in seeds:
            control, path = rollout(env, seed, limit, track)
            if control != archived[seed]:
                raise ValueError(
                    f"Direct rollout did not reproduce archived reset {seed}"
                )
            direct[seed], recorded[f"{seed}_direct"] = control, path
            start, goal = (
                np.array(control["start"], dtype=np.float32),
                np.array(control["goal"], dtype=np.float32),
            )

            def execute(name, target):
                if target.shape != goal.shape or not np.isfinite(target).all():
                    raise ValueError(f"Invalid decoded target in {name}")
                row, trajectory, _ = attempt(env, seed, limit, target, deadline)
                if (row["start"], row["goal"]) != (control["start"], control["goal"]):
                    raise ValueError(f"Branch did not reproduce reset {seed}")
                recorded[f"{seed}_{name}"] = trajectory
                return row

            for item in runs:
                model = item.model
                model_seed = str(model.config.seed)
                context = completion_context(
                    [(start, np.zeros(ACTION_SIZE, dtype=np.float32))],
                    goal,
                    limit,
                    model.config,
                )
                key = stream_seed(model.config.seed, seed)
                _, sampled = model.propose(context, stream_seed(key, 0))
                if compare_continuations:
                    target = sampled[0, 0, 0]
                    if target.shape != goal.shape or not np.isfinite(target).all():
                        raise ValueError(
                            f"Invalid decoded first target at reset {seed}"
                        )
                    planner = RecedingGoals(
                        track, model, key, limit, action_size=ACTION_SIZE
                    )
                    pair, traces = compare_after_prefix(
                        env, seed, limit, target, deadline, planner
                    )
                    if (pair["direct"]["start"], pair["direct"]["goal"]) != (
                        control["start"],
                        control["goal"],
                    ):
                        raise ValueError(f"Pair did not reproduce reset {seed}")
                    pairs_by_model[model_seed].append(pair)
                    for name, trace in traces.items():
                        recorded[f"{seed}_{model_seed}_{name}"] = trace
                    continue
                _, forecast = model.forecast(context)
                targets = {"forecast": model.decode(forecast)[0, 0]}
                targets.update(
                    {
                        f"sample_{index}": target
                        for index, target in enumerate(sampled[0, :, 0])
                    }
                )
                for name, target in targets.items():
                    by_model[model_seed][name].append(
                        execute(f"{model_seed}_{name}", target)
                    )
            if seed in references:
                target, source = references[seed]
                reference_rows.append(
                    {**execute("archived_reference", target), "source": source}
                )
    finally:
        native.close()

    if compare_continuations:
        write_json(
            out / "episodes.json",
            {"direct": list(direct.values()), "models": pairs_by_model},
        )
        np.savez_compressed(out / "trajectories.npz", **recorded)
        summary = {
            "starts": len(seeds),
            "attempt_deadline": deadline,
            "direct_from_reset_successes": sum(
                row["success"] for row in direct.values()
            ),
            "models": {
                seed: continuation_summary(pairs)
                for seed, pairs in pairs_by_model.items()
            },
            "training_environment_steps": 0,
            "evaluation_environment_steps": sum(
                len(path) - 1
                for name, path in recorded.items()
                if not name.endswith("_prefix_actions")
            ),
        }
        write_json(out / "summary.json", summary)
        return summary

    write_json(
        out / "episodes.json",
        {
            "direct": list(direct.values()),
            "models": by_model,
            "archived_reference": reference_rows,
        },
    )
    np.savez_compressed(out / "trajectories.npz", **recorded)
    models = {}
    for seed, branches in by_model.items():
        samples = [
            rows for name, rows in branches.items() if name.startswith("sample_")
        ]
        groups = list(zip(*samples, strict=True))
        models[seed] = {
            "arms": {
                name: paired_summary(rows, direct) for name, rows in branches.items()
            },
            "starts_with_any_sample_success": sum(
                any(row["success"] for row in group) for group in groups
            ),
            "starts_where_alternative_succeeds_and_first_fails": sum(
                not group[0]["success"] and any(row["success"] for row in group[1:])
                for group in groups
            ),
            "starts_where_sample_succeeds_and_direct_fails": sum(
                not direct[group[0]["seed"]]["success"]
                and any(row["success"] for row in group)
                for group in groups
            ),
            "starts_where_reached_sample_succeeds_and_direct_fails": sum(
                not direct[group[0]["seed"]]["success"]
                and any(
                    row["success"] and row["arrival_step"] is not None for row in group
                )
                for group in groups
            ),
        }
    summary = {
        "starts": len(seeds),
        "attempt_deadline": deadline,
        "direct_successes": sum(row["success"] for row in direct.values()),
        "models": models,
        "archived_reference": paired_summary(reference_rows, direct),
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(
            len(path) - 1 for path in recorded.values()
        ),
    }
    write_json(out / "summary.json", summary)
    return summary


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return file_digest(source, "sha256").hexdigest()
