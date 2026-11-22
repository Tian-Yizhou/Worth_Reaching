"""Fit masked completion and try decoded goals in matched AntMaze rollouts."""

import gzip
import json
from collections import deque
from dataclasses import asdict
from hashlib import file_digest
from pathlib import Path

import jax
import numpy as np
from flax import serialization

from worth_reaching.data.antmaze import (
    ACTION_SIZE,
    DATASETS,
    OBSERVATION_SIZE,
    CompletionContext,
    load_antmaze_archive,
    query_offsets,
)
from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.envs import antmaze
from worth_reaching.evaluation.antmaze import (
    EvalConfig,
    Policy,
    evaluation_starts,
    paired_metrics,
    rollout,
    summarize_episodes,
)
from worth_reaching.models.antmaze_completion import (
    CompletionConfig,
    LearnedCompletion,
    fit,
)
from worth_reaching.models.gcbc import load_policy
from worth_reaching.models.parallel import describe


class RecedingGoals:
    """Condition the frozen actor on one decoded goal until a time-based replan."""

    def __init__(
        self, actor: Policy, completion: LearnedCompletion, key: jax.Array, limit: int
    ):
        self.actor = actor
        self.completion = completion
        self.key = key
        self.limit = limit
        self.step = self.deadline = 0
        self.target = None
        self.history: deque = deque(maxlen=completion.config.history)
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)
        self.plans: list[dict] = []

    def __call__(self, observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
        self.history.append((observation, self.last_action))
        if self.step == self.deadline:
            self.target = self._plan(goal)
        action = self.actor(observation, self.target)
        self.last_action = action
        self.step += 1
        return action

    def _plan(self, goal: np.ndarray) -> np.ndarray:
        config = self.completion.config
        remaining = min(self.limit - self.step, config.max_horizon)
        if remaining <= config.queries:
            self.deadline = self.limit
            return goal

        count = len(self.history)
        observations = np.zeros((1, config.history, OBSERVATION_SIZE), dtype=np.float32)
        actions = np.zeros((1, config.history, ACTION_SIZE), dtype=np.float32)
        observations[0, -count:], actions[0, -count:] = map(
            np.stack, zip(*self.history, strict=True)
        )
        present = (np.arange(config.history) >= config.history - count)[None]
        duration = np.array([remaining], dtype=np.int32)
        offsets = query_offsets(duration, config.queries)
        context = CompletionContext(
            observations, actions, present, goal[None], duration, offsets
        )
        routes, decoded = self.completion.propose(
            context, jax.random.fold_in(self.key, self.step)
        )
        if not np.isfinite(routes).all() or not np.isfinite(decoded).all():
            raise ValueError("AntMaze latent planner produced a non-finite candidate")
        self.deadline = self.step + int(offsets[0, 0])
        self.plans.append(
            {
                "step": self.step,
                "deadline": self.deadline,
                "query_offsets": offsets[0].tolist(),
                "selected_route": 0,
                "selected_query": 0,
                "latent_routes": routes[0].tolist(),
                "decoded_routes": decoded[0].tolist(),
            }
        )
        return decoded[0, 0, 0]


def _fingerprint(path: Path) -> str:
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def run(
    dataset: str,
    data_dir: Path,
    checkpoint: Path,
    out: Path,
    config: CompletionConfig,
    evaluation: EvalConfig,
    start_bank: Path | None,
) -> dict:
    if dataset not in DATASETS or evaluation.action_limit <= config.queries:
        raise ValueError(
            "Use an AntMaze dataset and an evaluation budget longer than the query count"
        )
    archive = load_antmaze_archive(data_dir / f"{dataset}.npz")
    validation = load_antmaze_archive(data_dir / f"{dataset}-val.npz")
    if archive.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation must use distinct AntMaze archives")
    actor = load_policy(checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "antmaze_latent_completion_v1",
            "environment": antmaze.ENVIRONMENT,
            "environment_versions": dict(
                zip(("ogbench", "gymnasium", "mujoco"), antmaze.VERSIONS, strict=True)
            ),
            "dataset": dataset,
            "data_dir": str(data_dir.resolve()),
            "train_sha256": archive.fingerprint,
            "validation_sha256": validation.fingerprint,
            "executor": str(checkpoint.resolve()),
            "executor_weights_sha256": _fingerprint(checkpoint / "weights.msgpack"),
            "executor_normalization_sha256": _fingerprint(
                checkpoint / "normalization.npz"
            ),
            "training": asdict(config),
            "evaluation": asdict(evaluation),
            "compute": describe(),
            "code_revision": code_revision(),
            "start_source": str(start_bank.resolve()) if start_bank else None,
            "selection": "first sampled route, earliest predicted state",
        },
    )
    settings = {
        "history": config.history,
        "queries": config.queries,
        "max_horizon": config.max_horizon,
    }
    windows = archive.sample_windows(
        np.random.default_rng([config.seed, 800]), config.windows, **settings
    )
    mean = archive.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(archive.observations.std(axis=0, dtype=np.float64), 1e-3).astype(
        np.float32
    )
    model = fit(windows, mean, scale, config)
    eval_windows = validation.sample_windows(
        np.random.default_rng([config.seed, 801]), 512, **settings
    )
    metrics = model.validate(eval_windows)
    (out / "weights.msgpack").write_bytes(
        serialization.to_bytes(
            {
                **model.weights,
                "mean": mean,
                "scale": scale,
                "flow_scale": model.flow_scale,
            }
        )
    )

    env = antmaze.AntMaze()
    direct, guided = [], []
    try:
        starts = evaluation_starts(env, evaluation, start_bank)
        antmaze.save_starts(out / "starts.npz", starts)
        with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as stream:
            for start in starts:
                direct.append(rollout(env, start, actor, evaluation.action_limit))
                key = jax.random.fold_in(
                    jax.random.fold_in(jax.random.key(config.seed), start.task),
                    start.seed,
                )
                controller = RecedingGoals(actor, model, key, evaluation.action_limit)
                episode = rollout(env, start, controller, evaluation.action_limit)
                guided.append({**episode, "plans": len(controller.plans)})
                for plan in controller.plans:
                    stream.write(
                        json.dumps(
                            {
                                "task": start.task,
                                "seed": start.seed,
                                **plan,
                                "executed_actions": min(
                                    plan["deadline"], episode["steps"]
                                )
                                - plan["step"],
                            },
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
    finally:
        env.close()
    write_json(out / "episodes.json", {"direct": direct, "latent": guided})
    summary = {
        "validation": metrics,
        "direct": summarize_episodes(direct),
        "latent": summarize_episodes(guided),
        "paired": paired_metrics(direct, guided),
        "plans": sum(row["plans"] for row in guided),
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(row["steps"] for row in (*direct, *guided)),
    }
    write_json(out / "summary.json", summary)
    return summary
