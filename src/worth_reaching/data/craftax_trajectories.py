"""World splits and public trajectory records shared by collection and training."""

import gzip
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal

import jax

from worth_reaching.envs.craftax import (
    OBSERVATION_SCHEMA,
    RULES_VERSION,
    TASK,
    TASK_RULES,
)

RANDOM_POLICY = "uniform_random_v1"
SPLIT_VERSION = "craftax-classic-world-hash-v1"
EPISODE_ID_HEADER = re.compile(r'\s*\{\s*"episode_id"\s*:\s*')
Split = Literal["train", "dev", "test"]


@dataclass(frozen=True, slots=True)
class EpisodeSpec:
    world_seed: int
    world_id: str
    episode_id: str
    split: Split


def episode_spec(seed: int, policy_id: str = RANDOM_POLICY) -> EpisodeSpec:
    world_id = f"craftax-classic-v{RULES_VERSION}-{seed}"
    digest = sha256(f"{SPLIT_VERSION}:{world_id}".encode()).digest()
    bucket = int.from_bytes(digest[:8], "big") % 100
    split = "train" if bucket < 80 else "dev" if bucket < 90 else "test"
    return EpisodeSpec(seed, world_id, f"{world_id}-{policy_id}", split)


def episode_keys(world_seed: int) -> tuple[jax.Array, jax.Array]:
    root_key = jax.random.key(world_seed)
    return jax.random.fold_in(root_key, 0), jax.random.fold_in(root_key, 1)


@dataclass(frozen=True, slots=True)
class CraftaxArchive:
    directory: Path
    task: str
    manifest: dict
    episodes: dict[str, EpisodeSpec]

    def iter_episodes(self, split: Split) -> Iterator[tuple[EpisodeSpec, dict]]:
        """Stream one split without decoding the other splits' observation arrays."""
        if split not in ("train", "dev", "test"):
            raise ValueError(f"Unknown world split: {split}")
        expected = {
            episode.episode_id
            for episode in self.episodes.values()
            if episode.split == split
        }
        seen = set()
        decoder = json.JSONDecoder()
        with gzip.open(
            self.directory / "trajectories.jsonl.gz", "rt", encoding="utf-8"
        ) as stream:
            for line in stream:
                header = EPISODE_ID_HEADER.match(line)
                if header:
                    episode_id, _ = decoder.raw_decode(line, header.end())
                    if episode_id in self.episodes and episode_id not in expected:
                        continue
                row = json.loads(line)
                episode_id = row["episode_id"]
                if episode_id not in self.episodes:
                    raise ValueError(
                        f"Unknown episode in {self.directory}: {episode_id}"
                    )
                if episode_id not in expected:
                    continue
                if episode_id in seen:
                    raise ValueError(
                        f"Duplicate trajectory in {self.directory}: {episode_id}"
                    )
                seen.add(episode_id)
                yield self.episodes[episode_id], row
        if seen != expected:
            raise ValueError(
                f"Missing {len(expected - seen)} {split} trajectories in {self.directory}"
            )


def load_craftax_archive(directory: Path) -> CraftaxArchive:
    """Read a completed collection without restoring a policy or simulator."""
    required = ("manifest.json", "splits.json", "trajectories.jsonl.gz", "summary.json")
    if any(not (directory / name).is_file() for name in required):
        raise ValueError(f"Expected a completed Craftax collection at {directory}")
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    task = manifest.get("task", TASK)
    if (
        manifest["schema_version"] != 1
        or manifest["observation_schema"] != OBSERVATION_SCHEMA
        or task not in TASK_RULES
        or manifest["rules_version"] != TASK_RULES[task]
    ):
        raise ValueError(f"Incompatible Craftax collection at {directory}")
    splits = json.loads((directory / "splits.json").read_text(encoding="utf-8"))
    if splits["version"] != SPLIT_VERSION:
        raise ValueError(f"Unsupported world split version at {directory}")
    episodes = [EpisodeSpec(**row) for row in splits["episodes"]]
    if (
        len(episodes) != manifest["config"]["episodes"]
        or len({row.world_id for row in episodes}) != len(episodes)
        or any(
            row != episode_spec(row.world_seed, manifest["policy"]) for row in episodes
        )
    ):
        raise ValueError(f"Invalid world split manifest at {directory}")
    return CraftaxArchive(
        directory, task, manifest, {row.episode_id: row for row in episodes}
    )
