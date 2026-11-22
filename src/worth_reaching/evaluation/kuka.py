"""Paired closed-loop completion on the conditional Kuka stacking task."""

import gzip
import json
from collections import deque
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import numpy as np
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.kuka import STATE_SIZE, load_clips, sample_windows, stack_goal
from worth_reaching.data.maze_windows import CompletionContext, query_offsets
from worth_reaching.envs.kuka import (
    ACTION_LIMIT,
    BLOCK_HEIGHT,
    REVISION,
    SEED_BASE,
    KukaStacking,
)
from worth_reaching.models.maze_completion import (
    CompletionConfig,
    LearnedCompletion,
    fit,
)
from worth_reaching.models.training import describe, stream_seed, validate_resume


def _context(
    history: deque, goal: np.ndarray, remaining: int, config: CompletionConfig
) -> CompletionContext:
    observations = np.zeros((1, config.history, STATE_SIZE), dtype=np.float32)
    observations[0, -len(history) :] = np.stack(history)
    present = (np.arange(config.history) >= config.history - len(history))[None]
    duration = np.array([remaining], dtype=np.int32)
    return CompletionContext(
        observations,
        np.empty((1, config.history, 0), dtype=np.float32),
        present,
        goal[None],
        duration,
        query_offsets(duration, config.queries),
    )


def rollout(
    env: KukaStacking,
    model: LearnedCompletion,
    seed: int,
    limit: int,
    *,
    sampled: bool,
) -> tuple[dict, np.ndarray, list[dict]]:
    state, order = env.reset(seed)
    goal = stack_goal(state, order, BLOCK_HEIGHT)
    config = model.config
    plan_key = stream_seed(config.seed, seed)
    history = deque([state], maxlen=config.history)
    recorded, plans = [state], []
    target, deadline, placement_score = state, 0, 0.0
    for step in range(limit):
        remaining = min(limit - step, config.max_horizon)
        if step == deadline and remaining > config.queries:
            context = _context(history, goal, remaining, config)
            if sampled:
                latents, decoded = model.propose(context, stream_seed(plan_key, step))
            else:
                _, forecast = model.forecast(context)
                latents, decoded = (
                    np.asarray(forecast)[:, None],
                    model.decode(forecast)[:, None],
                )
            if not np.isfinite(decoded).all() or not np.isfinite(latents).all():
                raise ValueError("The Kuka planner produced a non-finite plan")
            deadline = step + int(context.offsets[0, 0])
            target = decoded[0, 0, 0]
            plans.append(
                {
                    "step": step,
                    "deadline": deadline,
                    "offsets": context.offsets[0].tolist(),
                    "latent_routes": latents[0].tolist(),
                    "decoded_routes": decoded[0].tolist(),
                    "selected_route": 0,
                    "selected_query": 0,
                }
            )
        state, reward = env.track(target)
        history.append(state)
        recorded.append(state)
        placement_score += reward
    relations = env.stacked_relations(state, order)
    return (
        {
            "seed": seed,
            "order_bottom_to_top": order.tolist(),
            "steps": limit,
            "success": all(relations),
            "final_ordered_contacts": sum(relations),
            "upstream_placement_score": placement_score,
            "plans": len(plans),
        },
        np.stack(recorded),
        plans,
    )


def _summarize(episodes: list[dict]) -> dict:
    return {
        "episodes": len(episodes),
        "successes": sum(episode["success"] for episode in episodes),
        "mean_upstream_placement_score": float(
            np.mean([episode["upstream_placement_score"] for episode in episodes])
        ),
        "mean_final_ordered_contacts": float(
            np.mean([episode["final_ordered_contacts"] for episode in episodes])
        ),
    }


def run(
    source_dir: Path,
    data_dir: Path,
    out: Path,
    config: CompletionConfig,
    *,
    split: str,
    repeats: int,
    limit: int,
    resume: Path | None = None,
) -> dict:
    if (
        not config.queries < limit <= ACTION_LIMIT
        or not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"]
    ):
        raise ValueError("Choose a valid Kuka action budget and disjoint reset seeds")
    env = KukaStacking(source_dir)
    try:
        train, validation, demos = load_clips(data_dir)
        out.mkdir(parents=True, exist_ok=False)
        seeds = list(range(SEED_BASE[split], SEED_BASE[split] + repeats))
        write_json(
            out / "manifest.json",
            {
                "schema": "diffuser_kuka_conditional_v2",
                "upstream_revision": REVISION,
                "versions": {name: version(name) for name in ("gymnasium", "pybullet")},
                "training": asdict(config),
                "evaluation": {"split": split, "seeds": seeds, "action_limit": limit},
                "compute": describe(),
                "code_revision": code_revision(),
                "demonstration_clips": {
                    key: len(group["clips"]) for key, group in demos.items()
                },
                "demonstrations_sha256": {
                    key: group["sha256"] for key, group in demos.items()
                },
            },
        )
        write_json(out / "split.json", demos)
        settings = dict(
            history=config.history,
            queries=config.queries,
            max_horizon=config.max_horizon,
        )
        batch = sample_windows(
            train, np.random.default_rng([config.seed, 800]), config.windows, **settings
        )
        mean = train.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = np.maximum(
            train.observations.std(axis=0, dtype=np.float64), 1e-3
        ).astype(np.float32)
        model = fit(
            batch,
            mean,
            scale,
            config,
            out / "checkpoints",
            validate_resume(out, resume),
        )
        heldout = sample_windows(
            validation, np.random.default_rng(801), 512, **settings
        )
        metrics = model.validate(heldout, compare_xy=False)
        write_json(out / "validation.json", metrics)
        (out / "weights.msgpack").write_bytes(
            serialization.to_bytes(
                {
                    **model.weights,
                    "mean": mean,
                    "scale": scale,
                    "flow_scale": model.flow_scale.numpy(),
                }
            )
        )
        episodes = {"forecast": [], "sampled": []}
        recorded = {}
        with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as stream:
            for seed in seeds:
                for arm in episodes:
                    result, path, plans = rollout(
                        env, model, seed, limit, sampled=arm == "sampled"
                    )
                    episodes[arm].append(result)
                    recorded[f"{arm}_{seed}"] = path
                    for plan in plans:
                        stream.write(
                            json.dumps(
                                {"arm": arm, "seed": seed, **plan},
                                separators=(",", ":"),
                            )
                            + "\n"
                        )
                if episodes["forecast"][-1]["order_bottom_to_top"] != episodes[
                    "sampled"
                ][-1]["order_bottom_to_top"] or not np.array_equal(
                    recorded[f"forecast_{seed}"][0], recorded[f"sampled_{seed}"][0]
                ):
                    raise ValueError(
                        "Kuka reset did not reproduce its start and terminal order"
                    )
    finally:
        env.close()
    np.savez_compressed(out / "trajectories.npz", **recorded)
    write_json(out / "episodes.json", episodes)
    summary = {
        "validation": metrics,
        **{arm: _summarize(results) for arm, results in episodes.items()},
        "training_environment_steps": 0,
        "evaluation_environment_steps": 2 * repeats * limit,
    }
    write_json(out / "summary.json", summary)
    return summary
