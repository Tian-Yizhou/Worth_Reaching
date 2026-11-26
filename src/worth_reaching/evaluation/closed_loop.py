"""Closed-loop pursuit of latent intermediate goals, shared by the continuous tasks.

From the public history, an arm asks the world model for one completion and declares
its first queries as stages, each with its query offset as the deadline. The arm pursues
the stages with its executor and arrival rule and replans when they close. When the
remaining budget no longer covers the queries, the task's terminal policy takes over.
The task supplies the executors, the native arrival predicate, an optional privileged
oracle and optional target checks through `Controls`.
"""

from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np

from worth_reaching.data.trajectory_windows import query_offsets
from worth_reaching.evaluation.regions import Arm
from worth_reaching.models.training import stream_seed, tensor
from worth_reaching.planning.arrival import LatentGoal
from worth_reaching.planning.goals import GoalSequence, Stage
from worth_reaching.planning.receding_goals import completion_context

MODES = ("forecast", "sample")
DIRECT, ORACLE = "direct", "oracle"
Action = np.ndarray


@dataclass
class RegionModels:
    """The frozen world model, its region decoder and their shared normalization."""

    completion: object
    decoder: object

    def normalize(self, observation):
        return (observation - self.completion.mean) / self.completion.scale

    def denormalize(self, values):
        return values * self.completion.scale + self.completion.mean

    def encode(self, normalized: np.ndarray) -> np.ndarray:
        return (
            self.completion.codec.encode(tensor(normalized)[None]).detach().numpy()[0]
        )


@dataclass(frozen=True)
class Controls:
    act: Callable[[np.ndarray, np.ndarray], Action]
    """Observation executor toward a target observation."""
    act_latent: Callable[[np.ndarray, np.ndarray], Action]
    """Latent executor from the raw observation toward z."""
    terminal: Callable[[np.ndarray, np.ndarray], Action]
    arrived: Callable[[np.ndarray, np.ndarray], bool]
    """Native arrival at a target observation."""
    oracle: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None
    """A privileged target for the ceiling arm."""
    check: Callable[[np.ndarray, np.ndarray, np.ndarray], dict[str, bool]] | None = None
    """Privileged checks of a declared target, from the observation and the goal."""


@dataclass(frozen=True, slots=True)
class Episode:
    """One closed-loop attempt: an arm, a proposal mode, a start and a repeat."""

    arm: str
    mode: str
    seed: int
    repeat: int = 0
    task: int | None = None


class ClosedLoop:
    """Plan, pursue declared stages and replan, then hand over to the terminal policy.

    `arm` is a region arm, ORACLE for the privileged ceiling or DIRECT for the terminal
    policy alone.
    """

    def __init__(
        self,
        models: RegionModels,
        controls: Controls,
        arm: Arm | str,
        mode: str,
        key: int,
        limit: int,
        *,
        stages: int,
        region_samples: int,
        action_size: int,
    ):
        self.models, self.controls = models, controls
        self.arm, self.mode, self.key, self.limit = arm, mode, key, limit
        self.stages, self.region_samples = stages, region_samples
        self.config = models.completion.config
        self.history: deque = deque(maxlen=self.config.history)
        self.last_action = np.zeros(action_size, dtype=np.float32)
        self.goals: GoalSequence | None = None
        self.step = 0
        self.plans: list[dict] = []

    def __call__(self, observation: np.ndarray, goal: np.ndarray) -> Action:
        self.history.append((observation, self.last_action))
        stage = None
        if self.arm != DIRECT:
            if self.goals is not None:
                stage = self.goals.advance(observation, self.step)
            if stage is None and self.limit - self.step > self.config.queries:
                stage = self._replan(observation, goal)
        if stage is None:
            action = self.controls.terminal(observation, goal)
        elif isinstance(self.arm, Arm) and self.arm.executor == "latent":
            action = self.controls.act_latent(observation, stage.target.latent)
        else:
            action = self.controls.act(observation, stage.target.realization)
        self.last_action = np.asarray(action, dtype=np.float32)
        self.step += 1
        return action

    def _replan(self, observation, goal):
        if self.arm == ORACLE:
            stages, targets = self._oracle(observation, goal)
        else:
            stages, targets = self._propose(observation, goal)
        record = {"step": self.step, "targets": np.asarray(targets).tolist()}
        if self.controls.check is not None:
            record["checks"] = [
                self.controls.check(observation, target, goal) for target in targets
            ]
        self.plans.append(record)
        if self.goals is None:
            self.goals = GoalSequence(stages, self._arrived)
        else:
            self.goals.replace_remaining(stages)
        return self.goals.active

    def _oracle(self, observation, goal):
        target = self.controls.oracle(observation, goal)
        span = min(self.limit - self.step, self.config.max_horizon)
        offset = int(query_offsets(np.array([span]), self.config.queries)[0, 0])
        stage = Stage(
            LatentGoal(np.zeros(0, np.float32), realization=target), self.step + offset
        )
        return [stage], [target]

    def _propose(self, observation, goal):
        models, completion = self.models, self.models.completion
        context = completion_context(
            list(self.history), goal, self.limit - self.step, self.config
        )
        count = min(self.stages, self.config.queries)
        if self.mode == "forecast":
            _, forecast = completion.forecast(context)
            latents = forecast.numpy()[0, :count]
        else:
            routes, _ = completion.propose(
                context, stream_seed(self.key, self.step), candidates=1
            )
            latents = routes[0, 0, :count]
        decoded = completion.decode(tensor(latents))
        draws = models.decoder.sample(
            latents,
            models.normalize(observation),
            self.region_samples + 1,
            stream_seed(self.key, self.step, 1),
        )
        realizations = models.denormalize(draws[:, -1])
        chosen = decoded if self.arm.target == "decoded" else realizations
        offsets = context.offsets[0, :count]
        stages = [
            Stage(
                LatentGoal(
                    latents[slot].astype(np.float32),
                    samples=draws[slot, :-1],
                    realization=chosen[slot].astype(np.float32),
                ),
                self.step + int(offsets[slot]),
            )
            for slot in range(count)
        ]
        # Latent arms are checked at the deterministic decoding of the goal they pursue.
        return stages, decoded if self.arm.executor == "latent" else chosen

    def _arrived(self, observation, goal: LatentGoal) -> bool:
        if self.arm == ORACLE or self.arm.arrival is None:
            return self.controls.arrived(observation, goal.realization)
        return self.arm.arrival(self.models.normalize(observation), goal)


def episodes(
    arms: Sequence[str],
    seeds: Sequence[int],
    *,
    sample_repeats: int,
    bounds: Sequence[str] = (DIRECT,),
    tasks: Sequence[int | None] = (None,),
) -> list[Episode]:
    """Bounds once per start; each arm once with the forecast and repeatedly sampled."""
    result = []
    for task in tasks:
        for seed in seeds:
            result += [Episode(bound, "forecast", seed, task=task) for bound in bounds]
            for name in arms:
                result.append(Episode(name, "forecast", seed, task=task))
                result += [
                    Episode(name, "sample", seed, repeat, task=task)
                    for repeat in range(sample_repeats)
                ]
    return result


def episode_key(models: RegionModels, episode: Episode) -> int:
    return stream_seed(
        models.completion.config.seed,
        episode.seed,
        MODES.index(episode.mode),
        episode.repeat,
        0 if episode.task is None else episode.task + 1,
    )


def record(episode: Episode, policy: ClosedLoop, success: bool, steps: int, first):
    """Summarize one attempt, counting how its declared targets fared on each check."""
    events = policy.goals.events if policy.goals is not None else []
    checks = [check for plan in policy.plans for check in plan.get("checks", [])]
    return {
        **asdict(episode),
        "success": bool(success),
        "first_success_step": first,
        "steps": steps,
        "plans": len(policy.plans),
        "stages": len(events),
        "arrivals": sum(event.outcome == "arrived" for event in events),
        "targets": len(checks),
        "checks": {
            name: sum(bool(check[name]) for check in checks)
            for name in (checks[0] if checks else {})
        },
        "first_plan": policy.plans[0] if policy.plans else None,
    }


def label(row: dict) -> str:
    return (
        row["arm"] if row["arm"] in (DIRECT, ORACLE) else f"{row['arm']}/{row['mode']}"
    )


def summarize(rows: list[dict], *, bootstrap_seed: int = 20) -> dict:
    """Average repeats within each start, then compare arms paired by start.

    Differences from the direct floor and from `point_native` under the same mode carry
    start-bootstrap intervals. With an oracle, each arm also reports its share of the
    gap between the floor and the ceiling.
    """
    by_arm: dict[str, dict[tuple, list[dict]]] = {}
    for row in rows:
        start = (row.get("task"), row["seed"])
        by_arm.setdefault(label(row), {}).setdefault(start, []).append(row)
    starts = sorted(by_arm[DIRECT], key=lambda start: (start[0] or 0, start[1]))
    draws = np.random.default_rng(bootstrap_seed).integers(
        len(starts), size=(2000, len(starts))
    )

    def per_start(name):
        return np.array(
            [np.mean([r["success"] for r in by_arm[name][start]]) for start in starts]
        )

    def ratio(trials, count):
        total = sum(t["targets"] for t in trials)
        return sum(count(t) for t in trials) / total if total else None

    direct = per_start(DIRECT)
    oracle = per_start(ORACLE) if ORACLE in by_arm else None
    arms = {}
    for name, grouped in sorted(by_arm.items()):
        trials = [trial for start in starts for trial in grouped[start]]
        success = per_start(name)
        arrival_steps = [t["first_success_step"] for t in trials if t["success"]]
        stages = sum(t["stages"] for t in trials)
        checks = sorted({key for t in trials for key in t["checks"]})
        summary = {
            "success": float(success.mean()),
            "mean_first_success_step": float(np.mean(arrival_steps))
            if arrival_steps
            else None,
            "plans_per_episode": float(np.mean([t["plans"] for t in trials])),
            "stage_arrival": sum(t["arrivals"] for t in trials) / stages
            if stages
            else None,
            "target_checks": {
                key: ratio(trials, lambda t, key=key: t["checks"].get(key, 0))
                for key in checks
            },
        }
        comparisons = {DIRECT: direct} if name != DIRECT else {}
        if "/" in name and not name.startswith("point_native/"):
            comparisons["point_native"] = per_start(
                "point_native/" + name.split("/")[-1]
            )
        for reference, baseline in comparisons.items():
            delta = success - baseline
            summary[f"difference_from_{reference}"] = float(delta.mean())
            summary[f"difference_from_{reference}_bootstrap_95"] = np.quantile(
                delta[draws].mean(axis=1), (0.025, 0.975)
            ).tolist()
        if oracle is not None and name not in (DIRECT, ORACLE):
            gap = oracle.mean() - direct.mean()
            summary["share_of_oracle_gap"] = (
                float((success.mean() - direct.mean()) / gap) if gap > 0 else None
            )
        arms[name] = summary
    return {"starts": len(starts), "arms": arms}
