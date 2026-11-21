"""Compare waypoint selectors using a shared replay candidate bank."""

import gzip
import json
import re
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import jax
import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.collection import EpisodeSpec
from worth_reaching.envs.craftax import CraftaxEpisode, Outcome
from worth_reaching.evaluation.craftax_branches import (
    ACTION_IDS,
    Continuation,
    load_expert_archive,
    replay_prefix,
)

LEAD = 16
DEFAULT_HORIZON = 24
WAYPOINT_OFFSETS = (2, 5, 8)
RADIUS = 0.002
ARRIVAL_THRESHOLD = 0.5
FALLBACK = "expert"
EPISODE_ID_HEADER = re.compile(r'\s*\{\s*"episode_id"\s*:\s*')


@dataclass(frozen=True, slots=True)
class WaypointConfig:
    worlds: int
    repeats: int
    slip_probability: float
    seed: int
    split: Literal["train", "dev"]
    horizon: int = DEFAULT_HORIZON
    skip_worlds: int = 0
    offer_expert: bool = False
    prefix_steps: int | None = None
    source_outcome: Outcome | Literal["any"] = "success"
    region_radius: float = RADIUS

    def __post_init__(self) -> None:
        if self.worlds < 1 or self.repeats < 1:
            raise ValueError("worlds and repeats must be positive")
        if self.horizon < max(WAYPOINT_OFFSETS) or self.skip_worlds < 0:
            raise ValueError(
                "Use a horizon of at least eight actions and a nonnegative world offset"
            )
        if not 0 <= self.slip_probability <= 1:
            raise ValueError("slip probability must be between 0 and 1")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.split not in {"train", "dev"}:
            raise ValueError("The scoring pilot uses train or dev worlds")
        if self.prefix_steps is None and self.source_outcome != "success":
            raise ValueError("Other source outcomes require a fixed prefix")
        if self.prefix_steps is not None and self.prefix_steps < 0:
            raise ValueError("Prefix steps must be nonnegative")
        if not 0 <= self.region_radius <= 1:
            raise ValueError("Mean absolute observation radius must be between 0 and 1")


@dataclass(frozen=True, slots=True)
class Waypoint:
    center: NDArray[np.float32]
    actions: tuple[int, ...]
    radius: float = RADIUS

    def arrived(self, observation: NDArray[np.float32]) -> bool:
        return bool(np.abs(self.center - observation).mean() <= self.radius)


@dataclass(frozen=True, slots=True)
class WaypointAttempt:
    arrival_step: int | None
    executor_steps: int
    total_steps: int
    remaining_actions: int
    outcome: Outcome | Literal["horizon_limit"]

    @property
    def success(self) -> bool:
        return self.outcome == "success"

    @property
    def arrived(self) -> bool:
        return self.arrival_step is not None

    @property
    def recovered(self) -> bool:
        return (
            self.success and not self.arrived and self.total_steps > self.executor_steps
        )


def _eligible_traces(
    dataset: Path, episodes: dict[str, EpisodeSpec], config: WaypointConfig
) -> Iterator[tuple[EpisodeSpec, dict, int]]:
    decoder = json.JSONDecoder()
    with gzip.open(dataset / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            # The collector writes the identifier first. Skip other splits without
            # decoding their observation arrays, and accept reordered records below.
            header = EPISODE_ID_HEADER.match(line)
            if header:
                episode_id, _ = decoder.raw_decode(line, header.end())
                if episodes[episode_id].split != config.split:
                    continue
            trace = json.loads(line)
            spec = episodes[trace["episode_id"]]
            if spec.split != config.split:
                continue
            if config.source_outcome not in ("any", trace["outcome"]):
                continue
            length = len(trace["steps"])
            prefix = config.prefix_steps
            if prefix is None:
                prefix = length - LEAD
            if 0 <= prefix and prefix + max(WAYPOINT_OFFSETS) < length:
                yield spec, trace, prefix


def _bank(trace: dict, prefix: int, radius: float) -> dict[str, Waypoint]:
    suffix = trace["steps"][prefix : prefix + max(WAYPOINT_OFFSETS)]
    return {
        f"replay_{offset}": Waypoint(
            np.asarray(suffix[offset - 1]["observation"], dtype=np.float32),
            tuple(step["action"] for step in suffix[:offset]),
            radius,
        )
        for offset in WAYPOINT_OFFSETS
    }


def attempt_waypoint(
    episode: CraftaxEpisode,
    memory: jax.Array,
    policy: Continuation,
    waypoint: Waypoint | None,
    slipped: NDArray[np.bool_],
    transition_key: jax.Array,
    policy_key: jax.Array,
    *,
    horizon: int = DEFAULT_HORIZON,
) -> WaypointAttempt:
    arrival = None
    executor_steps = 0
    for index in range(horizon):
        if waypoint is not None and arrival is None and index < len(waypoint.actions):
            memory = policy.observe(episode.observation, memory)
            action = ACTION_IDS["noop"] if slipped[index] else waypoint.actions[index]
            executor_steps += 1
            executing = True
        else:
            memory, action = policy.sample(
                episode.observation, memory, jax.random.fold_in(policy_key, index)
            )
            executing = False
        step = episode.step(action, key=jax.random.fold_in(transition_key, index))
        if executing and waypoint.arrived(step.observation):
            arrival = index + 1
        if step.outcome is not None:
            return WaypointAttempt(
                arrival, executor_steps, index + 1, step.remaining_actions, step.outcome
            )
    return WaypointAttempt(
        arrival, executor_steps, horizon, step.remaining_actions, "horizon_limit"
    )


def score_bank(calibration: dict[str, list[WaypointAttempt]]) -> dict[str, str]:
    def success_rate(name: str) -> float:
        attempts = calibration[name]
        return sum(attempt.success for attempt in attempts) / len(attempts)

    def arrival_rate(name: str) -> float:
        attempts = calibration[name]
        return sum(attempt.arrived for attempt in attempts) / len(attempts)

    def factorized_value(name: str) -> float:
        if name == FALLBACK:
            return success_rate(name)
        arrived = [attempt for attempt in calibration[name] if attempt.arrived]
        return sum(attempt.success for attempt in arrived) / len(arrived)

    # Give the unmodified policy the first position in either selector's tie break.
    names = [name for name in calibration if name != FALLBACK]
    if FALLBACK in calibration:
        names.insert(0, FALLBACK)
    available = [
        name
        for name in names
        if name == FALLBACK or arrival_rate(name) >= ARRIVAL_THRESHOLD
    ]
    return {
        "direct_attempt": max(names, key=success_rate),
        "factorized_filter": max(available, key=factorized_value, default=FALLBACK),
        "expert_fallback": FALLBACK,
    }


def _summary(
    selections: list[dict[str, str]],
    values: dict[str, list[list[bool]]],
    observations: Counter,
    replay_steps: int,
    rollout_steps: int,
    config: WaypointConfig,
) -> dict[str, object]:
    rates = {name: np.mean(worlds, axis=1) for name, worlds in values.items()}
    direct = values["direct_attempt"]
    factorized = values["factorized_filter"]
    paired = Counter(
        {
            "both_success": 0,
            "direct_only": 0,
            "factorized_only": 0,
            "neither_success": 0,
        }
    )
    keys = {
        (True, True): "both_success",
        (True, False): "direct_only",
        (False, True): "factorized_only",
        (False, False): "neither_success",
    }
    for first, second in zip(direct, factorized, strict=True):
        paired.update(keys[pair] for pair in zip(first, second, strict=True))
    delta = rates["direct_attempt"] - rates["factorized_filter"]
    indices = np.random.default_rng(config.seed).integers(
        len(delta), size=(1000, len(delta))
    )
    interval = np.quantile(delta[indices].mean(axis=1), (0.025, 0.975))
    return {
        "worlds": len(selections),
        "evaluation_repeats_per_world": config.repeats,
        "replay_steps": replay_steps,
        "rollout_steps": rollout_steps,
        "rollouts": observations["rollouts"],
        "selection_disagreements": sum(
            row["direct_attempt"] != row["factorized_filter"] for row in selections
        ),
        "direct_expert_worlds": sum(
            row["direct_attempt"] == FALLBACK for row in selections
        ),
        "factorized_fallback_worlds": sum(
            row["factorized_filter"] == FALLBACK for row in selections
        ),
        "evaluation_candidate_attempts": observations["candidate_attempts"],
        "evaluation_candidate_arrivals": observations["arrivals"],
        "evaluation_candidate_missed_successes": observations["missed_successes"],
        "evaluation_candidate_recoveries": observations["recoveries"],
        "selectors": {
            name: {
                "successes": sum(map(sum, worlds)),
                "attempts": len(worlds) * config.repeats,
                "success_rate": float(rates[name].mean()),
            }
            for name, worlds in values.items()
        },
        "paired_selector_outcomes": dict(paired),
        "direct_minus_factorized": float(delta.mean()),
        "world_bootstrap_interval_95": interval.tolist(),
    }


def run_waypoint_pilot(
    dataset: Path, checkpoint: Path, out: Path, config: WaypointConfig
) -> dict[str, object]:
    archive = load_expert_archive(dataset, checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            **archive.provenance,
            "pilot": "craftax_shared_replay_waypoints_v1",
            "run_id": out.name,
            "code_revision": code_revision(),
            "config": asdict(config),
            "lead": LEAD if config.prefix_steps is None else None,
            "terminal_horizon": config.horizon,
            "waypoint_offsets": WAYPOINT_OFFSETS,
            "mean_absolute_observation_radius": config.region_radius,
            "arrival_threshold": ARRIVAL_THRESHOLD,
            "bootstrap_samples": 1000,
        },
    )
    values: dict[str, list[list[bool]]] = {
        name: [] for name in ("direct_attempt", "factorized_filter", "expert_fallback")
    }
    selections = []
    observations = Counter()
    replay_steps = rollout_steps = 0
    random_key = jax.random.key(config.seed)
    with (
        (out / "worlds.jsonl").open("w", encoding="utf-8") as worlds,
        (out / "attempts.jsonl").open("w", encoding="utf-8") as attempts,
    ):
        for position, (spec, trace, prefix) in enumerate(
            _eligible_traces(dataset, archive.episodes, config)
        ):
            if position < config.skip_worlds:
                continue
            bank = _bank(trace, prefix, config.region_radius)
            options = {**bank, FALLBACK: None}
            calibration_options = options if config.offer_expert else bank
            episode, memory = replay_prefix(
                archive.env, spec, prefix, trace, archive.policy
            )
            snapshot = episode.snapshot()
            replay_steps += prefix
            world_key = jax.random.fold_in(random_key, spec.world_seed)
            calibration = {name: [] for name in calibration_options}
            evaluation = {name: [] for name in options}
            for pool_index, pool in enumerate(("calibration", "evaluation")):
                pool_key = jax.random.fold_in(world_key, pool_index)
                if pool == "evaluation":
                    selected = score_bank(calibration)
                    selections.append(selected)
                    worlds.write(
                        json.dumps(
                            {
                                "source": asdict(spec),
                                "source_outcome": trace["outcome"],
                                "prefix_steps": prefix,
                                "options": {
                                    name: goal.actions for name, goal in bank.items()
                                },
                                "selected": selected,
                            }
                        )
                        + "\n"
                    )
                active = calibration_options if pool == "calibration" else options
                for repeat in range(config.repeats):
                    env_key, policy_key, slip_key = jax.random.split(
                        jax.random.fold_in(pool_key, repeat), 3
                    )
                    slipped = np.asarray(
                        jax.random.uniform(slip_key, (max(WAYPOINT_OFFSETS),))
                        < config.slip_probability
                    )
                    for name, waypoint in active.items():
                        result = attempt_waypoint(
                            archive.env.restore(snapshot),
                            memory,
                            archive.policy,
                            waypoint,
                            slipped,
                            env_key,
                            policy_key,
                            horizon=config.horizon,
                        )
                        rollout_steps += result.total_steps
                        observations["rollouts"] += 1
                        if pool == "calibration":
                            calibration[name].append(result)
                        else:
                            evaluation[name].append(result.success)
                            if waypoint is not None:
                                observations["candidate_attempts"] += 1
                                observations["arrivals"] += result.arrived
                                observations["missed_successes"] += (
                                    result.success and not result.arrived
                                )
                                observations["recoveries"] += result.recovered
                        attempts.write(
                            json.dumps(
                                {
                                    "world_id": spec.world_id,
                                    "waypoint": name,
                                    "pool": pool,
                                    "repeat": repeat,
                                    **asdict(result),
                                }
                            )
                            + "\n"
                        )
            for selector, name in selected.items():
                values[selector].append(evaluation[name])
            if len(selections) == config.worlds:
                break
    if len(selections) != config.worlds:
        raise ValueError(f"Only {len(selections)} eligible histories in {config.split}")
    summary = _summary(
        selections, values, observations, replay_steps, rollout_steps, config
    )
    write_json(out / "summary.json", summary)
    return summary
