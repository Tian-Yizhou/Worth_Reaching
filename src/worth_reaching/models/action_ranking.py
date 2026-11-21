"""Matched action-ranking baselines for paired Craftax outcomes."""

import json
from hashlib import sha256
from math import comb
from pathlib import Path
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ranking import FEATURE_SCHEMA, RankingData, load_ranking_data

LEARNING_RATE = 0.5
L2_PENALTY = 0.01
REFERENCE_ALPHA = 0.05
BOOTSTRAP_SAMPLES = 1000


def _scores(weights: NDArray, features: NDArray, actions: NDArray) -> NDArray:
    return np.einsum("dcf,df->dc", weights[actions], features)


def _sigmoid(scores: NDArray) -> NDArray:
    return np.exp(-np.logaddexp(0.0, -scores))


def fit_head(
    features: NDArray,
    actions: NDArray,
    successes: NDArray,
    num_actions: int,
    iterations: int,
    loss: Literal["return", "pairwise"],
) -> NDArray:
    weights = np.zeros((num_actions, features.shape[1]))
    frequencies = successes.mean(axis=1)
    recorded_only = ((successes[:, :, 0] == 1) & (successes[:, :, 1] == 0)).sum(axis=1)
    discordant = np.count_nonzero(successes[:, :, 0] != successes[:, :, 1], axis=1)
    for _ in range(iterations):
        scores = _scores(weights, features, actions)
        if loss == "return":
            residuals = (_sigmoid(scores) - frequencies) / scores.size
        else:
            residual = (
                _sigmoid(scores[:, 0] - scores[:, 1]) * discordant - recorded_only
            )
            residual = residual / max(1, discordant.sum())
            residuals = np.stack((residual, -residual), axis=1)
        gradient = L2_PENALTY * weights
        np.add.at(gradient, actions, residuals[:, :, None] * features[:, None, :])
        weights -= LEARNING_RATE * gradient
    return weights


def _first_action_weight(difference: NDArray) -> NDArray:
    return (np.sign(difference) + 1) / 2


def _resolved_reference(selection_pairs: NDArray) -> NDArray[np.bool_]:
    differences = selection_pairs[:, :, 0] - selection_pairs[:, :, 1]
    wins = (differences > 0).sum(axis=1)
    losses = (differences < 0).sum(axis=1)
    probabilities = []
    for win, loss in zip(map(int, wins), map(int, losses), strict=True):
        count = win + loss
        tail = sum(comb(count, index) for index in range(min(win, loss) + 1))
        probabilities.append(min(1.0, 2 * tail / 2**count))
    return np.asarray(probabilities) <= REFERENCE_ALPHA


def evaluate_scores(
    evaluation: RankingData, scores: dict[str, NDArray], seed: int
) -> dict[str, object]:
    selection_pairs = evaluation.successes[:, ::2]
    estimate_pairs = evaluation.successes[:, 1::2]
    differences = (selection_pairs[:, :, 0] - selection_pairs[:, :, 1]).mean(axis=1)
    reference_first = _first_action_weight(differences)
    resolved = _resolved_reference(selection_pairs)
    held_out_rates = estimate_pairs.mean(axis=1)
    reference_value = (
        reference_first * held_out_rates[:, 0]
        + (1 - reference_first) * held_out_rates[:, 1]
    )
    bootstrap = np.random.default_rng(seed).integers(
        len(reference_first), size=(BOOTSTRAP_SAMPLES, len(reference_first))
    )
    metrics = {}
    for name, logits in scores.items():
        first = _first_action_weight(logits[:, 0] - logits[:, 1])
        selected = first * held_out_rates[:, 0] + (1 - first) * held_out_rates[:, 1]
        regret = reference_value - selected
        interval = np.quantile(regret[bootstrap].mean(axis=1), [0.025, 0.975])
        accuracy = np.where(reference_first == 1, first, 1 - first)
        metrics[name] = {
            "estimated_selected_success": float(selected.mean()),
            "mean_reference_minus_selected_success": float(regret.mean()),
            "world_bootstrap_interval_95": interval.tolist(),
            "agreement_on_resolved_reference": float(accuracy[resolved].mean())
            if resolved.any()
            else None,
        }
    return {
        "reference_selection_repeats": selection_pairs.shape[1],
        "reference_estimation_repeats": estimate_pairs.shape[1],
        "reference_ties": int(np.count_nonzero(differences == 0)),
        "resolved_reference_worlds": int(resolved.sum()),
        "estimated_reference_success": float(reference_value.mean()),
        "models": metrics,
    }


def run_ranking(
    dataset: Path,
    train_branches: Path,
    eval_branches: Path,
    out: Path,
    *,
    deadline: int,
    iterations: int,
    seed: int,
) -> dict[str, object]:
    if min(deadline, iterations) < 1 or seed < 0:
        raise ValueError(
            "Use a positive deadline and iteration count and a nonnegative seed"
        )
    train = load_ranking_data(dataset, train_branches, deadline)
    evaluation = load_ranking_data(dataset, eval_branches, deadline)
    if train.split != "train" or evaluation.split not in {"dev", "test"}:
        raise ValueError("Fit on train and evaluate on dev or test")
    if set(train.worlds) & set(evaluation.worlds):
        raise ValueError("Training and evaluation worlds overlap")
    if evaluation.successes.shape[1] < 2:
        raise ValueError(
            "Evaluation requires separate selection and estimation repeats"
        )

    source = json.loads((dataset / "manifest.json").read_bytes())
    num_actions = len(source["action_names"])
    action_train = np.ones((len(train.worlds), 1))
    action_eval = np.ones((len(evaluation.worlds), 1))
    weights = {
        "action_only": fit_head(
            action_train,
            train.actions,
            train.successes,
            num_actions,
            iterations,
            "return",
        ),
        "public_return": fit_head(
            train.features,
            train.actions,
            train.successes,
            num_actions,
            iterations,
            "return",
        ),
        "public_pairwise": fit_head(
            train.features,
            train.actions,
            train.successes,
            num_actions,
            iterations,
            "pairwise",
        ),
    }
    scores = {
        "uniform": np.zeros_like(evaluation.actions, dtype=float),
        "action_only": _scores(weights["action_only"], action_eval, evaluation.actions),
        "public_return": _scores(
            weights["public_return"], evaluation.features, evaluation.actions
        ),
        "public_pairwise": _scores(
            weights["public_pairwise"], evaluation.features, evaluation.actions
        ),
    }
    metrics = evaluate_scores(evaluation, scores, seed)
    summary = {
        "train_worlds": len(train.worlds),
        "evaluation_worlds": len(evaluation.worlds),
        "evaluation_split": evaluation.split,
        "deadline": deadline,
        "discordant_training_pairs": int(
            np.count_nonzero(train.successes[:, :, 0] != train.successes[:, :, 1])
        ),
        **metrics,
    }
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "feature_schema": FEATURE_SCHEMA,
            "code_revision": code_revision(),
            "source_dataset": str(dataset),
            "source_run_id": source["run_id"],
            "branches": {
                label: {
                    "path": str(directory),
                    "manifest_sha256": sha256(
                        (directory / "manifest.json").read_bytes()
                    ).hexdigest(),
                }
                for label, directory in (
                    ("train", train_branches),
                    ("evaluation", eval_branches),
                )
            },
            "deadline": deadline,
            "iterations": iterations,
            "learning_rate": LEARNING_RATE,
            "l2_penalty": L2_PENALTY,
            "bootstrap_seed": seed,
        },
    )
    np.savez_compressed(out / "weights.npz", **weights)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as stream:
        for index, decision in enumerate(evaluation.decision_ids):
            stream.write(
                json.dumps(
                    {
                        "decision_id": decision,
                        "world_id": evaluation.worlds[index],
                        "actions": evaluation.actions[index].tolist(),
                        "scores": {
                            name: logits[index].tolist()
                            for name, logits in scores.items()
                        },
                    }
                )
                + "\n"
            )
    write_json(out / "summary.json", summary)
    return summary
