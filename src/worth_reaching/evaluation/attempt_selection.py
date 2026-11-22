"""Fit and compare selectors on shared banks of matched prospective attempts."""

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json

Array = NDArray[np.float64]
THRESHOLDS = (0.5, 0.25, 0.0, 0.75)


def _sigmoid(logits: Array) -> Array:
    return 1 / (1 + np.exp(-np.clip(logits, -40, 40)))


def _logistic(design: Array, numerator: Array, trials: Array) -> Array:
    """Fit a regularized binomial model to candidate-level aggregate labels."""
    scale = max(1, trials.sum())
    penalty = np.full(design.shape[1], 0.005)
    penalty[-1] = 0.001
    coefficients = np.zeros(design.shape[1])

    def loss(theta: Array) -> float:
        logits = design @ theta
        return float(
            np.sum(trials * np.logaddexp(0, logits) - numerator * logits) / scale
            + 0.5 * np.dot(penalty, theta**2)
        )

    for _ in range(30):
        probabilities = _sigmoid(design @ coefficients)
        gradient = (
            design.T @ (trials * probabilities - numerator) / scale
            + penalty * coefficients
        )
        if np.max(np.abs(gradient)) < 1e-8:
            break
        curvature = trials * probabilities * (1 - probabilities) / scale
        hessian = design.T @ (curvature[:, None] * design) + np.diag(penalty)
        direction = np.linalg.solve(hessian, gradient)
        current = loss(coefficients)
        step = 1.0
        while loss(coefficients - step * direction) > current and step > 1e-6:
            step *= 0.5
        coefficients -= step * direction
    return coefficients


@dataclass(frozen=True, slots=True)
class ScoreHeads:
    mean: Array
    scale: Array
    attempt: Array
    reach: Array
    continuation: Array
    miss_continuation: Array
    waypoint_success: Array
    arrived_success: Array

    def predict(
        self, features: Array
    ) -> tuple[Array, Array, Array, Array, Array, Array]:
        design = _design(features, self.mean, self.scale)
        return tuple(
            _sigmoid(design @ coefficients)
            for coefficients in (
                self.attempt,
                self.reach,
                self.continuation,
                self.miss_continuation,
                self.waypoint_success,
                self.arrived_success,
            )
        )


def _design(features: Array, mean: Array, scale: Array) -> Array:
    return np.column_stack(((features - mean) / scale, np.ones(len(features))))


def fit_attempt_heads(decisions: list[dict]) -> ScoreHeads:
    features, counts = [], []
    for decision in decisions:
        for candidate in decision["candidates"]:
            attempts = candidate["trials"]
            reached = [row for row in attempts if row["arrived"] is True]
            features.append(candidate["features"])
            counts.append(
                [
                    len(attempts),
                    sum(row["success"] for row in attempts),
                    len(reached),
                    sum(row["success"] for row in reached),
                    candidate["id"] != "direct",
                ]
            )
    x = np.asarray(features, dtype=float)
    mean, scale = x.mean(axis=0), np.maximum(x.std(axis=0), 1e-6)
    design = _design(x, mean, scale)
    total, successes, arrived, after_arrival, is_waypoint = np.asarray(
        counts, dtype=float
    ).T
    return ScoreHeads(
        mean,
        scale,
        _logistic(design, successes, total),
        _logistic(design, arrived, total * is_waypoint),
        _logistic(design, after_arrival, arrived),
        _logistic(
            design,
            (successes - after_arrival) * is_waypoint,
            (total - arrived) * is_waypoint,
        ),
        _logistic(design, successes * is_waypoint, total * is_waypoint),
        _logistic(design, after_arrival, total * is_waypoint),
    )


def choose_candidates(
    heads: ScoreHeads, decision: dict, threshold: float, seed: int
) -> dict[str, int]:
    candidates = decision["candidates"]
    names = [row["id"] for row in candidates]
    direct = names.index("direct")
    waypoints = [i for i in range(len(names)) if i != direct]
    order = [direct, *waypoints]
    scores = heads.predict(np.array([row["features"] for row in candidates]))
    (
        attempt,
        reach,
        continuation,
        miss_continuation,
        waypoint_success,
        arrived_success,
    ) = scores
    value = continuation.copy()
    value[direct] = attempt[direct]
    joint = reach * continuation
    both_branches = joint + (1 - reach) * miss_continuation
    joint[direct] = attempt[direct]
    both_branches[direct] = attempt[direct]
    hash_seed = sha256(f"{seed}:{decision['world_id']}".encode()).digest()
    random = np.random.default_rng(int.from_bytes(hash_seed[:8], "big"))
    result = {
        "direct_only": direct,
        "arrival_only": max(waypoints, key=reach.__getitem__, default=direct),
        "terminal_waypoint": max(
            waypoints, key=waypoint_success.__getitem__, default=direct
        ),
        "terminal_arrived_waypoint": max(
            waypoints, key=arrived_success.__getitem__, default=direct
        ),
        "direct_attempt": max(order, key=attempt.__getitem__),
        "factorized_filter": max(
            (i for i in order if i == direct or reach[i] >= threshold),
            key=value.__getitem__,
        ),
        "value_without_filter": max(order, key=value.__getitem__),
        "joint_arrival_success": max(order, key=joint.__getitem__),
        "both_branches": max(order, key=both_branches.__getitem__),
        "random_waypoint": int(random.choice(waypoints)) if waypoints else direct,
    }
    if "native_choice" in decision:
        result["native"] = names.index(decision["native_choice"])
    if waypoints and all("support" in candidates[i] for i in waypoints):
        result["demonstration_support"] = max(
            waypoints, key=lambda i: candidates[i]["support"]
        )
    return result


def read_banks(path: Path, split: str) -> tuple[dict, list[dict]]:
    if not (path / "summary.json").is_file():
        raise ValueError(f"Expected completed attempt bank at {path}")
    manifest = json.loads((path / "manifest.json").read_text())
    config = manifest["config"]
    if manifest["schema"] != "shared_attempts_v1" or config["split"] != split:
        raise ValueError(f"Incompatible schema or expected {split} split in {path}")
    with (path / "decisions.jsonl").open(encoding="utf-8") as stream:
        decisions = [json.loads(line) for line in stream]
    seen = set()
    for row in decisions:
        if row["world_id"] in seen:
            raise ValueError(f"Duplicate world in {path}: {row['world_id']}")
        seen.add(row["world_id"])
        candidates = row["candidates"]
        ids = [candidate["id"] for candidate in candidates]
        if len(set(ids)) != len(ids) or "direct" not in ids:
            raise ValueError(f"Expected distinct candidates and a direct arm in {path}")
        if "native_choice" in row and row["native_choice"] not in ids:
            raise ValueError(
                f"Native choice is outside the common candidate bank in {path}"
            )
        for candidate in candidates:
            features = np.asarray(candidate["features"], dtype=float)
            attempts = candidate["trials"]
            if (
                features.shape != (len(manifest["features"]),)
                or not np.isfinite(features).all()
            ):
                raise ValueError(f"Invalid public features in {path}")
            if len(attempts) != config["repeats"] or not attempts:
                raise ValueError(f"Invalid paired repeat count in {path}")
            for attempt in attempts:
                arrival = attempt["arrived"]
                valid_arrival = (
                    arrival is None
                    if candidate["id"] == "direct"
                    else type(arrival) is bool
                )
                if not valid_arrival or type(attempt["success"]) is not bool:
                    raise ValueError(f"Invalid paired attempt labels in {path}")
    if not decisions:
        raise ValueError(f"No decisions in {path}")
    return manifest, decisions


def _rates(values: dict[str, list[int]], repeats: int) -> dict:
    return {
        key: {
            "successes": sum(successes),
            "attempts": len(successes) * repeats,
            "rate": sum(successes) / (len(successes) * repeats),
        }
        for key, successes in values.items()
    }


def public_bank(decision: dict) -> dict:
    return {
        "world_id": decision["world_id"],
        **(
            {"native_choice": decision["native_choice"]}
            if "native_choice" in decision
            else {}
        ),
        "candidates": [
            {
                key: candidate[key]
                for key in ("id", "features", "support")
                if key in candidate
            }
            for candidate in decision["candidates"]
        ],
    }


def compare_attempt_banks(
    train: Path, dev: Path, test: Path, out: Path, seed: int
) -> dict:
    bundles = [
        read_banks(path, split)
        for path, split in zip(
            (train, dev, test), ("train", "dev", "test"), strict=True
        )
    ]
    source, training = bundles[0]
    worlds = [row["world_id"] for _, card in bundles for row in card]
    if len(set(worlds)) != len(worlds):
        raise ValueError("Attempt banks must use disjoint train, dev and test worlds")
    for manifest, _ in bundles[1:]:
        for key in ("protocol", "features", "source_revision", "versions"):
            if manifest.get(key) != source.get(key):
                raise ValueError(f"Attempt bank mismatch for {key}")
        for key in ("horizon", "approach"):
            if manifest["config"].get(key) != source["config"].get(key):
                raise ValueError(f"Attempt bank mismatch for {key}")
    heads = fit_attempt_heads(training)
    development = bundles[1][1]
    threshold_scores = {
        value: sum(
            sum(
                a["success"]
                for a in row["candidates"][
                    choose_candidates(heads, public_bank(row), value, seed)[
                        "factorized_filter"
                    ]
                ]["trials"]
            )
            for row in development
        )
        for value in THRESHOLDS
    }
    threshold = max(THRESHOLDS, key=threshold_scores.__getitem__)
    testing = bundles[2][1]
    repeats = bundles[2][0]["config"]["repeats"]
    values, by_task = defaultdict(list), defaultdict(lambda: defaultdict(list))
    misses, arrivals, waypoint_attempts = Counter(), Counter(), Counter()
    disagreements = []
    out.mkdir(parents=True, exist_ok=False)
    with (out / "choices.jsonl").open("w", encoding="utf-8") as stream:
        for row in testing:
            picked = choose_candidates(heads, public_bank(row), threshold, seed)
            named = {
                method: row["candidates"][index]["id"]
                for method, index in picked.items()
            }
            disagreements.append(named["terminal_waypoint"] != named["arrival_only"])
            for method, index in picked.items():
                trials = row["candidates"][index]["trials"]
                successes = sum(t["success"] for t in trials)
                values[method].append(successes)
                by_task[str(row.get("task_id", "all"))][method].append(successes)
                waypoint_attempts[method] += (
                    len(trials) if named[method] != "direct" else 0
                )
                arrivals[method] += sum(t["arrived"] is True for t in trials)
                misses[method] += sum(
                    t["success"] and t["arrived"] is False for t in trials
                )
            stream.write(
                json.dumps({"world_id": row["world_id"], "choices": named}) + "\n"
            )
    if any(len(rows) != len(testing) for rows in values.values()):
        raise ValueError("Optional baselines must be present at every test world")
    indices = np.random.default_rng(seed).integers(
        len(testing), size=(2000, len(testing))
    )

    def contrast(left: str, right: str) -> dict:
        deltas = (np.array(values[left]) - values[right]) / repeats
        return {
            "difference": float(deltas.mean()),
            "world_bootstrap_95": np.quantile(
                deltas[indices].mean(axis=-1), [0.025, 0.975]
            ).tolist(),
        }

    result = {
        "worlds": {
            "train": len(training),
            "dev": len(development),
            "test": len(testing),
        },
        "reachability_threshold": threshold,
        "development_threshold_successes": threshold_scores,
        "test": _rates(values, repeats),
        "test_by_task": {
            task: _rates(values, repeats) for task, values in by_task.items()
        },
        "miss_then_success_by_selector": dict(misses),
        "arrivals_by_selector": dict(arrivals),
        "waypoint_attempts_by_selector": dict(waypoint_attempts),
        "terminal_vs_arrival_disagreements": {
            "worlds": sum(disagreements),
            "attempts_per_selector": sum(disagreements) * repeats,
            "terminal_waypoint_successes": sum(
                s
                for s, different in zip(
                    values["terminal_waypoint"], disagreements, strict=True
                )
                if different
            ),
            "arrival_only_successes": sum(
                s
                for s, different in zip(
                    values["arrival_only"], disagreements, strict=True
                )
                if different
            ),
        },
        "paired_contrasts": {
            "terminal_waypoint_vs_arrival_only": contrast(
                "terminal_waypoint", "arrival_only"
            ),
            "include_missed_successes": contrast(
                "terminal_waypoint", "terminal_arrived_waypoint"
            ),
            "direct_attempt_vs_direct_only": contrast("direct_attempt", "direct_only"),
            "direct_attempt_vs_filter": contrast("direct_attempt", "factorized_filter"),
            "both_branches_vs_arrival_only": contrast(
                "both_branches", "joint_arrival_success"
            ),
        },
    }
    np.savez(
        out / "heads.npz",
        mean=heads.mean,
        scale=heads.scale,
        attempt=heads.attempt,
        reach=heads.reach,
        continuation=heads.continuation,
        miss_continuation=heads.miss_continuation,
        waypoint_success=heads.waypoint_success,
        arrived_success=heads.arrived_success,
    )
    write_json(
        out / "manifest.json",
        {
            "schema": "shared_attempt_selection_v1",
            "code_revision": code_revision(),
            "protocol": source["protocol"],
            "features": source["features"],
            "train": str(train),
            "dev": str(dev),
            "test": str(test),
            "seed": seed,
        },
    )
    write_json(out / "summary.json", result)
    return result
