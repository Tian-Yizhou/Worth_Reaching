"""Estimate waypoint scoring effects from paired calibration and evaluation logs."""

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

EXPERT = "expert"
COMPARISONS = (
    ("direct", "factorized"),
    ("direct", "joint_arrival_success"),
    ("direct", "direct_with_filter"),
    ("conditional", "factorized"),
    ("direct", EXPERT),
    ("factorized", EXPERT),
)


@dataclass(frozen=True)
class Attempt:
    arrival_step: int | None
    executor_steps: int
    total_steps: int
    remaining_actions: int
    outcome: str

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    def at_horizon(self, horizon: int) -> "Attempt":
        if horizon < self.executor_steps:
            raise ValueError(
                "Censoring requires the shorter horizon to fit the executor"
            )
        if self.total_steps <= horizon:
            return self
        return replace(
            self,
            total_steps=horizon,
            remaining_actions=self.remaining_actions + self.total_steps - horizon,
            outcome="horizon_limit",
        )


Pools = dict[str, dict[str, list[Attempt]]]


def json_lines(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def load_run(run: Path) -> tuple[dict, dict[str, Pools]]:
    manifest = json.loads((run / "manifest.json").read_text())
    config = manifest["config"]
    if not config["offer_expert"]:
        raise ValueError(f"{run} did not calibrate the shared expert")
    worlds = {
        row["source"]["world_id"]: {
            pool: {name: [] for name in (EXPERT, *row["options"])}
            for pool in ("calibration", "evaluation")
        }
        for row in json_lines(run / "worlds.jsonl")
    }
    if len(worlds) != config["worlds"]:
        raise ValueError(f"{run} has an incomplete world set")
    for row in json_lines(run / "attempts.jsonl"):
        attempts = worlds[row.pop("world_id")][row.pop("pool")][row.pop("waypoint")]
        if row.pop("repeat") != len(attempts):
            raise ValueError(f"{run} has a missing or duplicated repeat")
        attempts.append(Attempt(**row))
    if any(
        len(attempts) != config["repeats"]
        for pools in worlds.values()
        for options in pools.values()
        for attempts in options.values()
    ):
        raise ValueError(f"{run} has incomplete paired attempts")
    return config, worlds


def pick(calibration: dict[str, list[Attempt]]) -> dict[str, str]:
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for name, attempts in calibration.items():
        successes = sum(a.success for a in attempts)
        reached = [a for a in attempts if a.arrival_step is not None]
        reached_successes = sum(a.success for a in reached)
        rate = successes / len(attempts)
        scores["direct"][name] = rate
        if name == EXPERT:
            for method in (
                "joint_arrival_success",
                "conditional",
                "factorized",
                "direct_with_filter",
            ):
                scores[method][name] = rate
            continue
        scores["joint_arrival_success"][name] = reached_successes / len(attempts)
        if reached:
            scores["conditional"][name] = reached_successes / len(reached)
        if 2 * len(reached) >= len(attempts):
            scores["factorized"][name] = scores["conditional"][name]
            scores["direct_with_filter"][name] = rate
    return {
        **{
            method: max(values, key=values.__getitem__)
            for method, values in scores.items()
        },
        EXPERT: EXPERT,
    }


def paired_interval(deltas: list[float]) -> dict:
    values = np.asarray(deltas)
    indices = np.random.default_rng(1729).integers(
        len(values), size=(5000, len(values))
    )
    return {
        "difference": float(values.mean()),
        "world_bootstrap_95": np.quantile(
            values[indices].mean(axis=1), [0.025, 0.975]
        ).tolist(),
        "wins_ties_losses_by_world": [
            int(sum(test)) for test in (values > 0, values == 0, values < 0)
        ],
    }


def at_horizon(worlds: dict[str, Pools], horizon: int) -> dict[str, Pools]:
    return {
        world: {
            pool: {
                option: [attempt.at_horizon(horizon) for attempt in attempts]
                for option, attempts in options.items()
            }
            for pool, options in pools.items()
        }
        for world, pools in worlds.items()
    }


def summarize(worlds: dict[str, Pools]) -> dict:
    by_world = {}
    counts: dict[str, Counter] = defaultdict(Counter)
    candidates: dict[str, Counter] = defaultdict(Counter)
    disagreements: dict[str, Counter] = defaultdict(Counter)
    outcomes = {}
    for world, pools in worlds.items():
        calibration = pools["calibration"]
        evaluation = pools["evaluation"]
        choices = pick(calibration)
        half = len(calibration[EXPERT]) // 2
        first = pick({name: rows[:half] for name, rows in calibration.items()})
        last = pick({name: rows[half:] for name, rows in calibration.items()})
        wins = {
            name: sum(a.success for a in evaluation[choice])
            for name, choice in choices.items()
        }
        repeats = len(evaluation[EXPERT])
        outcomes[world] = {name: count / repeats for name, count in wins.items()}
        by_world[world] = {
            "choices": choices,
            "wins": wins,
            "half_pool_choices": [first, last],
        }
        for name, choice in choices.items():
            counts[name].update(
                attempts=repeats,
                successes=wins[name],
                expert_worlds=choice == EXPERT,
                half_pool_agreement_worlds=first[name] == last[name],
                first_half_successes=sum(a.success for a in evaluation[first[name]]),
                second_half_successes=sum(a.success for a in evaluation[last[name]]),
            )
        for option, attempts in evaluation.items():
            if option == EXPERT:
                continue
            for trial, expert in zip(attempts, evaluation[EXPERT], strict=True):
                arrived = trial.arrival_step is not None
                candidates[option].update(
                    attempts=1,
                    arrivals=arrived,
                    arrived_successes=arrived and trial.success,
                    missed_successes=not arrived and trial.success,
                    recovered=not arrived
                    and trial.success
                    and trial.total_steps > trial.executor_steps,
                    missed_attempt_only=not arrived
                    and trial.success
                    and not expert.success,
                    missed_expert_only=not arrived
                    and expert.success
                    and not trial.success,
                )
        direct, factorized = choices["direct"], choices["factorized"]
        if direct != factorized:
            kind = "both_waypoint"
            if direct == EXPERT:
                kind = "direct_expert"
            if factorized == EXPERT:
                kind = "factorized_expert"
            disagreements[kind].update(
                worlds=1,
                direct_successes=wins["direct"],
                factorized_successes=wins["factorized"],
            )
    return {
        "worlds": len(worlds),
        "selectors": counts,
        "candidate_outcomes": candidates,
        "disagreements": disagreements,
        "comparisons": {
            f"{left}_minus_{right}": paired_interval(
                [row[left] - row[right] for row in outcomes.values()]
            )
            for left, right in COMPARISONS
        },
        "by_world": by_world,
    }


def calibration_depth(worlds: dict[str, Pools]) -> dict:
    levels = {}
    for depth in (4, 8, 12, 24):
        wins = defaultdict(list)
        low_arrival_selections = Counter()
        blocks = 24 // depth
        per_world_attempts = 24 * blocks
        for pools in worlds.values():
            calibration, evaluation = pools["calibration"], pools["evaluation"]
            subsets = [
                {
                    option: rows[start : start + depth]
                    for option, rows in calibration.items()
                }
                for start in range(0, 24, depth)
            ]
            labels = [pick(subset) for subset in subsets]
            for method in labels[0]:
                options = [choices[method] for choices in labels]
                wins[method].append(
                    sum(a.success for option in options for a in evaluation[option])
                )
                low_arrival_selections[method] += sum(
                    option != EXPERT
                    and 2 * sum(a.arrival_step is not None for a in subset[option])
                    < depth
                    for option, subset in zip(options, subsets, strict=True)
                )
        levels[depth] = {
            "disjoint_calibration_blocks_per_world": blocks,
            "success_rates": {
                method: sum(values) / (per_world_attempts * len(worlds))
                for method, values in wins.items()
            },
            "below_half_arrival_selections": low_arrival_selections,
            "comparisons": {
                f"{left}_minus_{right}": paired_interval(
                    [
                        (first - second) / per_world_attempts
                        for first, second in zip(wins[left], wins[right], strict=True)
                    ]
                )
                for left, right in COMPARISONS
            },
        }
    return levels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("raw", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    groups: dict[tuple, dict[str, Pools]] = defaultdict(dict)
    runs = {}
    for manifest in sorted(args.raw.glob("*/manifest.json")):
        run = manifest.parent
        if not (run / "summary.json").exists():
            raise ValueError(f"Unfinished simulation run {run}")
        config, worlds = load_run(run)
        runs[run.name] = worlds
        prefix = config["prefix_steps"]
        key = (
            config["source_outcome"],
            prefix,
            config["slip_probability"],
            config["region_radius"],
            config["horizon"],
        )
        if not run.name.startswith("near-calibration-"):
            if groups[key].keys() & worlds.keys():
                raise ValueError(f"Duplicate world in {key}")
            groups[key].update(worlds)
        for row in json_lines(run / "worlds.jsonl"):
            choices = pick(worlds[row["source"]["world_id"]]["calibration"])
            expected = row["selected"]
            if (
                choices["direct"] != expected["direct_attempt"]
                or choices["factorized"] != expected["factorized_filter"]
            ):
                raise ValueError(f"Recorded and rescored choices differ in {run}")
    near = runs["near-0.25-12"]
    if runs["near-short-check"] != at_horizon(near, 16):
        raise ValueError("Shorter-horizon replay and offline censoring disagree")

    deep = {}
    for skip in (12, 18, 24, 30):
        worlds = runs[f"near-calibration-{skip}"]
        shallow = at_horizon(runs[f"near-0.25-{skip}"], 16)
        if shallow.keys() != worlds.keys() or any(
            attempts[:8] != shallow[world][pool][option]
            for world, pools in worlds.items()
            for pool, options in pools.items()
            for option, attempts in options.items()
        ):
            raise ValueError(f"Calibration-depth records drifted at offset {skip}")
        deep.update(worlds)

    report = {
        "primary_condition": "near/slip=0.25/h=16",
        "conditions": {},
        "matched_slip_contrasts": {},
        "radius_control": {},
        "calibration_depth": calibration_depth(deep),
    }
    for (outcome, prefix, slip, radius, longest), worlds in groups.items():
        if radius != 0.002 or longest == 16:
            continue
        group = "near" if prefix is None else f"early-{outcome}"
        expected_worlds = {"near": 24, "early-success": 8, "early-death": 6}
        if len(worlds) != expected_worlds[group]:
            raise ValueError(
                f"Incomplete world set for {group} at slip probability {slip}"
            )
        nominal = groups[(outcome, prefix, 0.0, radius, longest)]
        if nominal.keys() != worlds.keys() or any(
            pools[pool][EXPERT] != nominal[world][pool][EXPERT]
            for world, pools in worlds.items()
            for pool in ("calibration", "evaluation")
        ):
            raise ValueError(
                f"Shared expert records differ for {group} at slip probability {slip}"
            )
        for horizon in (longest // 2, longest):
            report["conditions"][f"{group}/slip={slip}/h={horizon}"] = summarize(
                at_horizon(worlds, horizon)
            )
    for label, trial in report["conditions"].items():
        group, slip, horizon = label.split("/")
        if slip == "slip=0.0":
            continue
        nominal = report["conditions"][f"{group}/slip=0.0/{horizon}"]
        deltas = []
        repeats = trial["selectors"][EXPERT]["attempts"] / trial["worlds"]
        for world, traced in trial["by_world"].items():
            baseline = nominal["by_world"][world]["wins"]
            wins = traced["wins"]
            deltas.append(
                (
                    wins["direct"]
                    - wins["factorized"]
                    - baseline["direct"]
                    + baseline["factorized"]
                )
                / repeats
            )
        report["matched_slip_contrasts"][label] = paired_interval(deltas)
    for radius, worlds in (
        (0.001, runs["radius-0.001"]),
        (0.002, near),
        (0.004, runs["radius-0.004"]),
    ):
        if near.keys() != worlds.keys() or any(
            pools[pool][EXPERT] != near[world][pool][EXPERT]
            for world, pools in worlds.items()
            for pool in ("calibration", "evaluation")
        ):
            raise ValueError(f"Radius control is unmatched at {radius}")
        for horizon in (16, 32):
            report["radius_control"][f"radius={radius}/h={horizon}"] = summarize(
                at_horizon(worlds, horizon)
            )
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
