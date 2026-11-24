"""Collect complete Craftax trajectories in a fixed world split."""

import gzip
import json
import sys
import tomllib
from collections import Counter
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

import jax
import numpy as np
from numpy.typing import NDArray

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.craftax_trajectories import (
    RANDOM_POLICY,
    SPLIT_VERSION,
    EpisodeSpec,
    episode_keys,
    episode_spec,
)
from worth_reaching.tasks.craftax.environment import (
    ACTION_NAMES,
    OBSERVATION_SCHEMA,
    SURVIVAL_TASK,
    TASK,
    CraftaxClassic,
)


@dataclass(frozen=True, slots=True)
class CollectionConfig:
    seed_start: int
    episodes: int
    budget: int

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        episodes: int | None = None,
        budget: int | None = None,
        seed_start: int | None = None,
    ) -> "CollectionConfig":
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        if values.keys() != {"seed_start", "episodes", "budget"}:
            raise ValueError("Config must contain seed_start, episodes and budget")
        if any(type(value) is not int for value in values.values()):
            raise ValueError("seed_start, episodes and budget must be integers")
        return cls(
            values["seed_start"] if seed_start is None else seed_start,
            values["episodes"] if episodes is None else episodes,
            values["budget"] if budget is None else budget,
        )

    def __post_init__(self) -> None:
        if self.episodes < 1 or not 0 <= self.seed_start <= 2**32 - self.episodes:
            raise ValueError(
                "Use a positive episode count within the 32-bit world seed range"
            )
        if not 0 < self.budget < 2**31 - 1:
            raise ValueError("Craftax budget must be between 1 and 2147483646")


def _random_policy(seed: int) -> Callable[[NDArray[np.float32]], int]:
    rng = np.random.default_rng([seed, 2])
    action_ids = tuple(ACTION_NAMES)

    def act(observation: NDArray[np.float32]) -> int:
        return int(rng.choice(action_ids))

    return act


def _collect_episode(
    env: CraftaxClassic,
    spec: EpisodeSpec,
    act: Callable[[NDArray[np.float32]], int],
) -> tuple[list, list, str, int | None]:
    reset_key, transitions_key = episode_keys(spec.world_seed)
    episode = env.reset(reset_key)
    initial = episode.observation.tolist()
    steps = []
    first_craft = None
    for index in range(env.budget):
        action = act(episode.observation)
        step = episode.step(action, key=jax.random.fold_in(transitions_key, index))
        if step.crafted_iron_pickaxe:
            first_craft = index + 1
        steps.append(
            {
                "action": action,
                "observation": step.observation.tolist(),
                "remaining_actions": step.remaining_actions,
            }
        )
        if step.outcome is not None:
            return initial, steps, step.outcome, first_craft
    raise RuntimeError(
        "Craftax episode exhausted the budget without a terminal outcome"
    )


def collect(
    config: CollectionConfig,
    out: Path,
    expert_checkpoint: Path | None = None,
    *,
    task: str = TASK,
) -> dict[str, object]:
    policy_id = RANDOM_POLICY
    policy_factory = _random_policy
    policy_metadata: dict[str, int | str] = {}
    if expert_checkpoint is not None:
        from worth_reaching.policies.craftax_expert import CraftaxExpert

        expert = CraftaxExpert(expert_checkpoint)
        policy_id = expert.policy_id
        policy_factory = expert.start
        policy_metadata = expert.metadata

    env = CraftaxClassic(config.budget, task)
    revision = code_revision()
    out.mkdir(parents=True, exist_ok=False)
    specs = [
        episode_spec(seed, policy_id)
        for seed in range(config.seed_start, config.seed_start + config.episodes)
    ]
    write_json(
        out / "splits.json",
        {
            "version": SPLIT_VERSION,
            "episodes": [asdict(spec) for spec in specs],
        },
    )
    write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "run_id": out.name,
            "environment": env.environment_id,
            "environment_version": version("craftax"),
            "rules_version": env.rules_version,
            "observation_schema": OBSERVATION_SCHEMA,
            "task": env.task,
            "policy": policy_id,
            "policy_metadata": policy_metadata,
            "action_names": ACTION_NAMES,
            "jax_version": jax.__version__,
            "jax_backend": jax.default_backend(),
            "code_revision": revision,
            "config": asdict(config),
        },
    )
    outcomes = Counter({"success": 0, "death": 0, "timeout": 0})
    crafted_then = Counter()
    environment_steps = 0
    with gzip.open(
        out / "trajectories.jsonl.gz", "wt", encoding="utf-8", compresslevel=1
    ) as stream:
        for completed, spec in enumerate(specs, start=1):
            initial, steps, outcome, first_craft = _collect_episode(
                env, spec, policy_factory(spec.world_seed)
            )
            if first_craft is not None:
                crafted_then[outcome] += 1
            stream.write(
                json.dumps(
                    {
                        "episode_id": spec.episode_id,
                        "initial_observation": initial,
                        "steps": steps,
                        "outcome": outcome,
                        **(
                            {"first_craft_step": first_craft}
                            if task == SURVIVAL_TASK
                            else {}
                        ),
                    },
                    separators=(",", ":"),
                )
                + "\n"
            )
            environment_steps += len(steps)
            outcomes[outcome] += 1
            if completed % 100 == 0:
                print(
                    f"{completed}/{len(specs)} episodes | {environment_steps} steps | "
                    f"{outcomes['success']} successes",
                    file=sys.stderr,
                    flush=True,
                )
    summary = {
        "episodes": len(specs),
        "environment_steps": environment_steps,
        "policy": policy_id,
        "outcomes": dict(outcomes),
        "split_counts": dict(Counter(spec.split for spec in specs)),
    }
    if task == SURVIVAL_TASK:
        summary["crafted_by_deadline"] = sum(crafted_then.values())
        summary["crafted_then_died"] = crafted_then["death"]
    write_json(out / "summary.json", summary)
    return summary
