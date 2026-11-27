"""Record and summarize option-planner episodes against direct execution."""

import numpy as np

from worth_reaching.planning.options import OptionPlanner

DIRECT, ORACLE, DIRECT_LATENT = "direct", "oracle", "direct_latent"


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
            "first_decision": decisions[0] if decisions else None,
        }
    return row


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
