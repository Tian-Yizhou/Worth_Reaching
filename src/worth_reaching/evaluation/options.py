"""Configure, correct, record and summarize option-planner runs in any task.

A task supplies the environment, the executor and the bounds. This module holds what
the Maze2D and Scene runs share: the online-correction rounds that continue fitting the
option value on the executor's attempts, the arms that compare the offline and
corrected values with and without shorter-budget proposals, and the per-arm summary.
"""

import argparse
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np

from worth_reaching.models.option_value import (
    NEVER,
    HindsightOptions,
    OptionAttempts,
    OptionValue,
    OptionValueConfig,
    fit_option_value,
    value_range,
)
from worth_reaching.models.training import restore_weights, weights
from worth_reaching.planning.options import OptionPlanner, PlannerConfig

DIRECT, ORACLE, DIRECT_LATENT = "direct", "oracle", "direct_latent"
OFFLINE, ONLINE = "offline", "online"
COLLECT, COLLECT_GOAL = "collect", "collect_goal"
ATTEMPTS_FILE = "executor-attempts.npz"


@dataclass(frozen=True, slots=True)
class OnlineConfig:
    rounds: int = 3
    episodes: int = 256
    """Training resets per round."""
    explore: float = 0.2
    """Probability of pursuing a uniformly drawn sampled candidate at a decision."""
    goal_attempts: bool = True
    """Also play every training reset pursuing the encoded terminal goal for the whole
    budget without interruption, so the value learns the goal option's own outcome
    rather than a bootstrap after an early switch."""
    updates: int = 50_000
    """Value updates per round, continuing from the previous value."""
    online_share: float = 0.5
    """Share of each batch drawn from the executor's attempts."""

    def __post_init__(self) -> None:
        if (
            min(self.episodes, self.updates) < 1
            or self.rounds < 0
            or not 0 <= self.explore <= 1
            or not 0 < self.online_share <= 1
        ):
            raise ValueError(
                "Use non-negative rounds, positive episodes and updates, and shares "
                "in (0, 1]"
            )


@dataclass(frozen=True, slots=True)
class OptionRunConfig:
    value: OptionValueConfig = field(default_factory=OptionValueConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    fast_budgets: tuple[float, ...] = (1.0, 0.5, 0.25)
    """Budget fractions for the arms that also propose faster routes."""
    eval_costs: tuple[float, ...] = (0.0, 0.001)
    """Time costs evaluated with the full planner; collection and ablations use the
    last."""
    surprise_quantile: float = 0.95

    def __post_init__(self) -> None:
        if (
            not self.eval_costs
            or not set(self.eval_costs) <= set(self.value.time_costs)
            or not 0 < self.surprise_quantile < 1
        ):
            raise ValueError(
                "Evaluate time costs the value was trained on, with a surprise "
                "quantile in (0, 1)"
            )

    @property
    def recipe(self) -> dict:
        """The settings that determine the trained values."""
        settings = {
            "value": asdict(self.value),
            "online": asdict(self.online),
            "collection": asdict(self.collection),
        }
        return json.loads(json.dumps(settings))

    @property
    def collection(self) -> PlannerConfig:
        return replace(
            self.planner,
            time_cost=self.eval_costs[-1],
            proposal_budgets=self.fast_budgets,
            explore=self.online.explore,
        )

    def collection_arms(self, surprise: float) -> dict[str, "Arm"]:
        """The planner that collects attempts and, if enabled, uninterrupted goal
        pursuit on the same training resets."""
        result: dict[str, Arm] = {
            COLLECT: (ONLINE, replace(self.collection, surprise=surprise))
        }
        if self.online.goal_attempts:
            result[COLLECT_GOAL] = (
                ONLINE,
                replace(
                    self.planner,
                    time_cost=self.eval_costs[-1],
                    subgoals=False,
                    interrupt=False,
                ),
            )
        return result


Arm = tuple[str, PlannerConfig | str]
"""The value an arm plans with, and its planner settings or a bound's name."""


def option_arms(config: OptionRunConfig, bounds: tuple[str, ...]) -> dict[str, Arm]:
    """Bounds, direct latent pursuit, both values with and without faster routes, and
    ablations of the corrected value with faster routes. The random-first ablation
    measures the value's calibration without selection at the first decision."""
    planner, cost = config.planner, config.eval_costs[-1]
    fast = replace(planner, time_cost=cost, proposal_budgets=config.fast_budgets)
    result: dict[str, Arm] = {bound: (ONLINE, bound) for bound in bounds}
    result[DIRECT_LATENT] = (ONLINE, replace(planner, time_cost=cost, subgoals=False))
    for value in (OFFLINE, ONLINE):
        for setting in config.eval_costs:
            result[f"{value}/options@{setting:g}"] = (
                value,
                replace(planner, time_cost=setting),
            )
        result[f"{value}/fast@{cost:g}"] = (value, fast)
    result[f"online/fast_no_goal@{cost:g}"] = (
        ONLINE,
        replace(fast, goal_candidate=False),
    )
    result[f"online/fast_no_interrupt@{cost:g}"] = (
        ONLINE,
        replace(fast, interrupt=False),
    )
    result[f"online/fast_fixed_duration@{cost:g}"] = (ONLINE, replace(fast, slots=(0,)))
    result[f"online/fast_random_first@{cost:g}"] = (
        ONLINE,
        replace(fast, explore_first=True),
    )
    return result


Collect = Callable[[OptionValue, int], tuple[list[dict], list[OptionAttempts]]]
"""Play one round of training resets with a value; return rows and attempts."""


def correct(
    collect: Collect,
    offline: OptionValue,
    options: HindsightOptions,
    latents: np.ndarray,
    limit: int,
    config: OptionRunConfig,
    *,
    seed: Callable[[int], int],
    out: Path,
) -> tuple[OptionValue, list[dict]]:
    """Collect the executor's attempts in rounds and continue fitting the value."""
    online = config.online
    value = restore_weights(
        OptionValue(latents.shape[-1], limit, config.value), weights(offline)
    )
    gathered: list[OptionAttempts] = []
    rounds = []
    for index in range(online.rounds):
        rows, collected = collect(value.eval(), index)
        gathered += collected
        attempts = OptionAttempts.concatenate(gathered)
        rounds.append(_round(index, rows, attempts))
        value = fit_option_value(
            options,
            latents,
            limit,
            config.value,
            seed=seed(index),
            out=out / "checkpoints" / f"value-round-{index}",
            attempts=attempts,
            online_share=online.online_share,
            model=value,
            updates=online.updates,
        )
        rounds[-1]["value_range"] = value_range(
            value.eval(), options, latents, config.eval_costs[-1]
        )
    if gathered:
        attempts = OptionAttempts.concatenate(gathered)
        np.savez_compressed(
            out / ATTEMPTS_FILE,
            **{name: getattr(attempts, name) for name in OptionAttempts.__slots__},
        )
    return value.eval(), rounds


def _round(index: int, rows: list[dict], attempts: OptionAttempts) -> dict:
    """Collection statistics; success and choices are those of the planner episodes."""
    goal_rows = [row for row in rows if row["arm"] == COLLECT_GOAL]
    environment_steps = sum(row["steps"] for row in rows)
    episodes = len(rows)
    rows = [row for row in rows if row["arm"] != COLLECT_GOAL]
    successes = [row["first_success_step"] for row in rows if row["success"]]
    decisions = sum(row["decisions"] for row in rows)
    return {
        "round": index,
        "episodes": episodes,
        "environment_steps": environment_steps,
        "goal_pursuit_success": float(np.mean([row["success"] for row in goal_rows]))
        if goal_rows
        else None,
        "success": float(np.mean([row["success"] for row in rows])),
        "mean_first_success_step": float(np.mean(successes)) if successes else None,
        "goal_choice_share": sum(row["goal_choices"] for row in rows) / decisions
        if decisions
        else None,
        "attempts": len(attempts),
        "attempt_arrival_share": float(np.mean(attempts.first < NEVER)),
    }


def settings(args: argparse.Namespace) -> dict:
    """`OptionRunConfig` fields from the options of `recipes.add_option_arguments`."""
    return {
        "value": OptionValueConfig(
            updates=args.value_updates,
            options=args.value_options,
            heads=args.value_heads,
            time_costs=tuple(args.time_costs),
        ),
        "online": OnlineConfig(
            rounds=args.online_rounds,
            episodes=args.online_episodes,
            explore=args.explore,
            goal_attempts=args.goal_attempts,
            updates=args.online_updates,
        ),
        "planner": PlannerConfig(
            samples=args.samples, risk=args.risk, margin=args.margin
        ),
        "fast_budgets": tuple(args.fast_budgets),
        "eval_costs": tuple(args.eval_costs),
    }


def record(arm: str, seed: int, task, policy, success: bool, steps: int, first):
    row = {
        "arm": arm,
        "seed": seed,
        "task": task,
        "success": bool(success),
        "first_success_step": first,
        "steps": steps,
    }
    if isinstance(policy, OptionPlanner):
        decisions = policy.log.decisions
        # Count only what happened before the first success.
        before = [d for d in decisions if first is None or d["step"] < first]
        row |= {
            "generations": policy.log.generations,
            "switches": policy.log.switches,
            "triggers": policy.log.triggers,
            "decisions": len(before),
            "goal_choices": sum(d["goal"] for d in before),
            "mean_duration": float(np.mean([d["duration"] for d in before]))
            if before
            else None,
            "mean_value": float(np.mean([d["value"] for d in before]))
            if before
            else None,
            "mean_goal_gap": float(
                np.mean([d["value"] - d["goal_value"] for d in gaps])
            )
            if (gaps := [d for d in before if d["goal_value"] is not None])
            else None,
            "first_decision": decisions[0] if decisions else None,
            "decision_log": decisions,
            "switch_log": policy.log.switch_log,
        }
    return row


def attempts(
    policy: OptionPlanner, trajectory: np.ndarray, goal: np.ndarray, first, limit: int
) -> OptionAttempts | None:
    """The options an episode pursued until its first success, with their outcomes.

    `trajectory[t]` is the observation after `t` actions and `first` the action count at
    the first success. An option started at step `s` and replaced at step `e` covers
    observations `s + 1` to `e`; the one that contains the success stops there.

    An interrupted option is the option with the duration it ran. The option that
    reached the goal keeps the duration it was commanded: "pursue `z` for at most `k`"
    succeeded after `d ≤ k` actions. Recording `d` instead would teach the value that an
    option commanded for the whole budget always fails, since only failures run that
    long.
    """
    pursued = policy.log.pursued
    if not pursued:
        return None
    starts = np.array([step for step, _, _ in pursued])
    commanded = np.array([duration for _, _, duration in pursued])
    stop = NEVER if first is None else first
    ends = np.minimum(np.append(starts[1:], len(trajectory) - 1), stop)
    keep = np.flatnonzero((starts < stop) & (ends > starts))
    if not len(keep):
        return None
    starts, ends, commanded = starts[keep], ends[keep], commanded[keep]
    latents = np.stack([latent for _, latent, _ in pursued])[keep]
    encode = policy.completion.encode_observations
    arrived = (starts < stop) & (stop <= ends)
    return OptionAttempts(
        encode(trajectory[starts]),
        np.repeat(encode(goal[None]), len(starts), axis=0),
        latents.astype(np.float32),
        encode(trajectory[ends]),
        np.where(arrived, np.maximum(commanded, stop - starts), ends - starts).astype(
            np.int32
        ),
        (limit - starts).astype(np.int32),
        np.where(arrived, stop - starts, NEVER).astype(np.int32),
    )


def _mean(values) -> float | None:
    """Mean over the episodes that have a value."""
    present = [value for value in values if value is not None]
    return float(np.mean(present)) if present else None


def summarize(rows: list[dict], *, bootstrap_seed: int = 20) -> dict:
    """Per-arm success and timing, with start-paired differences from direct."""
    by_arm: dict[str, dict[tuple, dict]] = {}
    for row in rows:
        by_arm.setdefault(row["arm"], {})[(row["task"] or 0, row["seed"])] = row
    starts = sorted(by_arm[DIRECT])
    draws = np.random.default_rng(bootstrap_seed).integers(
        len(starts), size=(2000, len(starts))
    )

    def success(name):
        return np.array([by_arm[name][start]["success"] for start in starts], float)

    direct = success(DIRECT)
    oracle = success(ORACLE) if ORACLE in by_arm else None
    arms = {}
    for name, grouped in sorted(by_arm.items()):
        trials = [grouped[start] for start in starts]
        value = success(name)
        firsts = [t["first_success_step"] for t in trials if t["success"]]
        summary = {
            "success": float(value.mean()),
            "mean_first_success_step": float(np.mean(firsts)) if firsts else None,
        }
        if "generations" in trials[0]:
            decisions = sum(t["decisions"] for t in trials)
            durations = [t["mean_duration"] for t in trials if t["mean_duration"]]
            summary |= {
                "generations_per_episode": float(
                    np.mean([t["generations"] for t in trials])
                ),
                "switches_per_episode": float(np.mean([t["switches"] for t in trials])),
                "triggers_per_episode": {
                    cause: float(np.mean([t["triggers"].get(cause, 0) for t in trials]))
                    for cause in sorted({c for t in trials for c in t["triggers"]})
                },
                "goal_choice_share": sum(t["goal_choices"] for t in trials) / decisions
                if decisions
                else None,
                "mean_chosen_duration": float(np.mean(durations))
                if durations
                else None,
                "mean_chosen_value": _mean(t["mean_value"] for t in trials),
                "mean_gap_over_goal": _mean(t["mean_goal_gap"] for t in trials),
            }
        for reference in (DIRECT, DIRECT_LATENT):
            if name == reference or reference not in by_arm:
                continue
            delta = value - success(reference)
            summary[f"difference_from_{reference}"] = float(delta.mean())
            summary[f"difference_from_{reference}_bootstrap_95"] = np.quantile(
                delta[draws].mean(axis=1), (0.025, 0.975)
            ).tolist()
        if oracle is not None and name not in (DIRECT, ORACLE):
            gap = oracle.mean() - direct.mean()
            summary["share_of_oracle_gap"] = (
                float((value.mean() - direct.mean()) / gap) if gap > 0 else None
            )
        arms[name] = summary
    return {"starts": len(starts), "arms": arms}
