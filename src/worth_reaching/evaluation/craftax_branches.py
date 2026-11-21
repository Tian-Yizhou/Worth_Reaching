"""Prospective action branches from replayed Craftax expert histories."""

import gzip
import json
import sys
from collections import Counter
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import jax
import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.collection import SPLIT_VERSION, EpisodeSpec, episode_keys
from worth_reaching.envs.craftax import (
    ACTION_NAMES,
    OBSERVATION_SCHEMA,
    RULES_VERSION,
    TASK,
    CraftaxClassic,
    CraftaxEpisode,
    Outcome,
)

if TYPE_CHECKING:
    from worth_reaching.policies.craftax_expert import CraftaxExpert

ACTION_IDS = {name: code for code, name in ACTION_NAMES.items()}
SELECTION = "successful_expert_trace_fixed_lead_v1"


class Continuation(Protocol):
    def initial_memory(self) -> jax.Array: ...
    def observe(
        self, observation: NDArray[np.float32], carry: jax.Array
    ) -> jax.Array: ...
    def sample(
        self, observation: NDArray[np.float32], carry: jax.Array, key: jax.Array
    ) -> tuple[jax.Array, int]: ...


@dataclass(frozen=True, slots=True)
class BranchConfig:
    split: str
    decisions: int
    repeats: int
    lead: int
    horizon: int
    seed: int

    def __post_init__(self) -> None:
        if self.split not in {"train", "dev", "test"}:
            raise ValueError("split must be train, dev or test")
        if min(self.decisions, self.repeats, self.lead, self.horizon) < 1:
            raise ValueError("decisions, repeats, lead and horizon must be positive")
        if not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")


@dataclass(frozen=True, slots=True)
class Decision:
    source: EpisodeSpec
    prefix_steps: int
    remaining_actions: int
    horizon: int
    recorded_action: int
    alternative_action: int

    @property
    def id(self) -> str:
        return f"{self.source.episode_id}:{self.prefix_steps}"


@dataclass(frozen=True, slots=True)
class Attempt:
    outcome: Outcome | Literal["horizon_limit"]
    steps: int
    remaining_actions: int


def _select_decisions(
    dataset: Path, config: BranchConfig, budget: int, episodes: dict[str, EpisodeSpec]
) -> Iterator[tuple[Decision, dict]]:
    with gzip.open(dataset / "trajectories.jsonl.gz", "rt", encoding="utf-8") as rows:
        for line in rows:
            trace = json.loads(line)
            spec = episodes[trace["episode_id"]]
            steps = trace["steps"]
            if (
                spec.split != config.split
                or trace["outcome"] != "success"
                or len(steps) < config.lead
            ):
                continue
            if len(steps) > budget:
                raise ValueError(
                    f"Source episode exceeds the recorded budget: {spec.episode_id}"
                )
            prefix_steps = len(steps) - config.lead
            remaining = budget - prefix_steps
            recorded = steps[prefix_steps]["action"]
            alternative = (
                ACTION_IDS["noop"]
                if recorded != ACTION_IDS["noop"]
                else ACTION_IDS["left"]
            )
            yield (
                Decision(
                    spec,
                    prefix_steps,
                    remaining,
                    min(config.horizon, remaining),
                    recorded,
                    alternative,
                ),
                trace,
            )


def replay_decision(
    env: CraftaxClassic, decision: Decision, trace: dict, policy: Continuation
) -> tuple[CraftaxEpisode, jax.Array]:
    episode, memory = replay_prefix(
        env, decision.source, decision.prefix_steps, trace, policy
    )
    return episode, policy.observe(episode.observation, memory)


def replay_prefix(
    env: CraftaxClassic,
    source: EpisodeSpec,
    prefix_steps: int,
    trace: dict,
    policy: Continuation,
) -> tuple[CraftaxEpisode, jax.Array]:
    reset_key, transition_key = episode_keys(source.world_seed)
    episode = env.reset(reset_key)
    expected = trace["initial_observation"]
    memory = policy.initial_memory()
    for index in range(prefix_steps + 1):
        if not np.array_equal(
            episode.observation, np.asarray(expected, dtype=np.float32)
        ):
            raise ValueError(
                f"Replay diverged at observation {index} for {source.episode_id}"
            )
        if index == prefix_steps:
            break
        memory = policy.observe(episode.observation, memory)
        recorded = trace["steps"][index]
        step = episode.step(
            recorded["action"], key=jax.random.fold_in(transition_key, index)
        )
        if step.outcome is not None:
            raise ValueError(
                f"Replay terminated before the branch point for {source.episode_id}"
            )
        expected = recorded["observation"]
    return episode, memory


def run_attempt(
    episode: CraftaxEpisode,
    memory: jax.Array,
    policy: Continuation,
    action: int,
    horizon: int,
    repeat_key: jax.Array,
) -> Attempt:
    if horizon < 1:
        raise ValueError("Attempt horizon must be positive")
    transition_key, policy_key = jax.random.split(repeat_key)
    for index in range(horizon):
        if index > 0:
            memory, action = policy.sample(
                episode.observation, memory, jax.random.fold_in(policy_key, index)
            )
        step = episode.step(action, key=jax.random.fold_in(transition_key, index))
        if step.outcome is not None:
            return Attempt(step.outcome, index + 1, step.remaining_actions)
    return Attempt("horizon_limit", horizon, step.remaining_actions)


@dataclass(frozen=True, slots=True)
class ExpertArchive:
    env: CraftaxClassic
    policy: "CraftaxExpert"
    episodes: dict[str, EpisodeSpec]
    provenance: dict[str, object]


def load_expert_archive(dataset: Path, checkpoint: Path) -> ExpertArchive:
    from worth_reaching.policies.craftax_expert import CraftaxExpert

    required = ("summary.json", "manifest.json", "splits.json", "trajectories.jsonl.gz")
    if any(not (dataset / name).is_file() for name in required):
        raise ValueError(f"Expected a completed collection at {dataset}")
    manifest_bytes = (dataset / "manifest.json").read_bytes()
    split_bytes = (dataset / "splits.json").read_bytes()
    source = json.loads(manifest_bytes)
    budget = source["config"]["budget"]
    env = CraftaxClassic(budget=budget)
    compatibility = {
        "schema_version": 1,
        "environment": env.environment_id,
        "environment_version": version("craftax"),
        "rules_version": RULES_VERSION,
        "observation_schema": OBSERVATION_SCHEMA,
        "jax_version": jax.__version__,
        "task": TASK,
    }
    for key, expected in compatibility.items():
        if source.get(key) != expected:
            raise ValueError(
                f"Dataset {key} is {source.get(key)!r}, expected {expected!r}"
            )
    policy = CraftaxExpert(checkpoint)
    if source["policy_metadata"].get("actor_sha256") != policy.metadata["actor_sha256"]:
        raise ValueError(
            "The continuation checkpoint must match the dataset's expert actor"
        )
    source_splits = json.loads(split_bytes)
    if source_splits["version"] != SPLIT_VERSION:
        raise ValueError(
            f"Unrecognized split manifest version: {source_splits['version']}"
        )
    episodes = {
        row["episode_id"]: EpisodeSpec(**row) for row in source_splits["episodes"]
    }
    if len(episodes) != len(source_splits["episodes"]):
        raise ValueError("Duplicate episode IDs in the split manifest")
    if len({episode.world_id for episode in episodes.values()}) != len(episodes):
        raise ValueError("Expected one source episode per world")
    return ExpertArchive(
        env,
        policy,
        episodes,
        {
            **compatibility,
            "source_dataset": str(dataset),
            "source_run_id": source["run_id"],
            "source_manifest_sha256": sha256(manifest_bytes).hexdigest(),
            "source_splits_sha256": sha256(split_bytes).hexdigest(),
            "continuation_policy": policy.policy_id,
            "policy_metadata": policy.metadata,
        },
    )


def label_branches(
    dataset: Path, out: Path, checkpoint: Path, config: BranchConfig
) -> dict[str, object]:
    archive = load_expert_archive(dataset, checkpoint)
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            **archive.provenance,
            "run_id": out.name,
            "code_revision": code_revision(),
            "selection": SELECTION,
            "alternative": "noop_or_left",
            "sampling": "fresh_paired_continuations",
            "config": asdict(config),
        },
    )
    summary = _run_decisions(
        archive.env, dataset, out, archive.episodes, archive.policy, config
    )
    write_json(out / "summary.json", summary)
    return summary


def _run_decisions(
    env: CraftaxClassic,
    dataset: Path,
    out: Path,
    episodes: dict[str, EpisodeSpec],
    policy: Continuation,
    config: BranchConfig,
) -> dict[str, object]:
    totals = {
        arm: Counter(success=0, death=0, timeout=0, horizon_limit=0)
        for arm in ("recorded", "alternative")
    }
    paired = Counter(
        both_success=0, recorded_only=0, alternative_only=0, neither_success=0
    )
    decisions = replay_steps = branch_steps = 0
    seed = jax.random.key(config.seed)
    with (
        (out / "decisions.jsonl").open("w", encoding="utf-8") as records,
        (out / "attempts.jsonl").open("w", encoding="utf-8") as attempts,
    ):
        for decision, trace in _select_decisions(dataset, config, env.budget, episodes):
            episode, memory = replay_decision(env, decision, trace, policy)
            snapshot = episode.snapshot()
            records.write(
                json.dumps({"decision_id": decision.id, **asdict(decision)}) + "\n"
            )
            records.flush()
            replay_steps += decision.prefix_steps
            world_key = jax.random.fold_in(seed, decision.source.world_seed)
            decision_key = jax.random.fold_in(world_key, decision.prefix_steps)
            for repeat in range(config.repeats):
                repeat_key = jax.random.fold_in(decision_key, repeat)
                successes = []
                for arm, action in (
                    ("recorded", decision.recorded_action),
                    ("alternative", decision.alternative_action),
                ):
                    result = run_attempt(
                        env.restore(snapshot),
                        memory,
                        policy,
                        action,
                        decision.horizon,
                        repeat_key,
                    )
                    totals[arm][result.outcome] += 1
                    branch_steps += result.steps
                    successes.append(result.outcome == "success")
                    attempts.write(
                        json.dumps(
                            {
                                "decision_id": decision.id,
                                "repeat": repeat,
                                "arm": arm,
                                "action": action,
                                **asdict(result),
                            }
                        )
                        + "\n"
                    )
                paired[
                    {
                        (True, True): "both_success",
                        (True, False): "recorded_only",
                        (False, True): "alternative_only",
                        (False, False): "neither_success",
                    }[tuple(successes)]
                ] += 1
            decisions += 1
            if decisions % 10 == 0:
                print(
                    f"{decisions}/{config.decisions} decisions | {branch_steps} branch steps",
                    file=sys.stderr,
                    flush=True,
                )
            if decisions == config.decisions:
                break
    if decisions != config.decisions:
        raise ValueError(
            f"Only {decisions} eligible decisions in split {config.split}, requested {config.decisions}"
        )
    return {
        "split": config.split,
        "decisions": decisions,
        "pairs": decisions * config.repeats,
        "attempts": 2 * decisions * config.repeats,
        "prefix_replay_steps": replay_steps,
        "branch_steps": branch_steps,
        "outcomes": {arm: dict(counts) for arm, counts in totals.items()},
        "paired_outcomes": dict(paired),
    }
