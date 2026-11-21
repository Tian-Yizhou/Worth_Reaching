"""Join paired action labels to the public prefix in a Craftax collection."""

import gzip
import json
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

ARMS = ("recorded", "alternative")
FEATURE_SCHEMA = "craftax_public_last_four_v1"


@dataclass(frozen=True, slots=True)
class RankingData:
    decision_ids: tuple[str, ...]
    worlds: tuple[str, ...]
    features: NDArray[np.float64]
    actions: NDArray[np.int64]
    successes: NDArray[np.int8]
    split: str


def _public_features(
    trace: dict, prefix: int, budget: int, num_actions: int
) -> NDArray:
    steps = trace["steps"]
    if not 0 <= prefix < len(steps) or prefix >= budget:
        raise ValueError("Invalid decision prefix")
    start = max(0, prefix - 3)
    recent = [
        trace["initial_observation"] if index == 0 else steps[index - 1]["observation"]
        for index in range(start, prefix + 1)
    ]
    recent = np.asarray(recent, dtype=np.float64)
    counts = np.bincount(
        [step["action"] for step in steps[max(0, prefix - 4) : prefix]],
        minlength=num_actions,
    )
    context = np.concatenate(
        (
            recent[-1],
            recent.mean(axis=0),
            counts / max(1, counts.sum()),
            [(budget - prefix) / budget],
        )
    )
    context /= max(1, np.linalg.norm(context))
    return np.append(context, 1.0)


def _read_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def load_ranking_data(dataset: Path, branches: Path, deadline: int) -> RankingData:
    if not (branches / "summary.json").is_file():
        raise ValueError(f"Expected completed branch labels at {branches}")
    parent = (dataset / "manifest.json").read_bytes()
    splits = (dataset / "splits.json").read_bytes()
    manifest = json.loads((branches / "manifest.json").read_bytes())
    if (
        manifest.get("source_manifest_sha256") != sha256(parent).hexdigest()
        or manifest.get("source_splits_sha256") != sha256(splits).hexdigest()
    ):
        raise ValueError(
            f"Branch labels do not identify the specified collection: {branches}"
        )
    collection = json.loads(parent)
    specs = {row["episode_id"]: row for row in json.loads(splits)["episodes"]}
    split = manifest["config"]["split"]
    repeats = manifest["config"]["repeats"]
    decisions = list(_read_jsonl(branches / "decisions.jsonl"))
    if len(decisions) != manifest["config"]["decisions"] or not decisions:
        raise ValueError(f"Incomplete decision list in {branches}")
    by_decision = {row["decision_id"]: index for index, row in enumerate(decisions)}
    by_episode = {
        row["source"]["episode_id"]: index for index, row in enumerate(decisions)
    }
    worlds = tuple(row["source"]["world_id"] for row in decisions)
    if not (len(by_decision) == len(by_episode) == len(set(worlds)) == len(decisions)):
        raise ValueError("Expected distinct decision, episode and world IDs")
    for row in decisions:
        source = row["source"]
        if specs.get(source["episode_id"]) != source or source["split"] != split:
            raise ValueError("Decision does not match the collection split manifest")
        if not 1 <= deadline <= row["horizon"]:
            raise ValueError(
                f"Deadline exceeds the available horizon for {row['decision_id']}"
            )
    actions = np.array(
        [[row["recorded_action"], row["alternative_action"]] for row in decisions]
    )
    num_actions = len(collection["action_names"])
    if (
        np.any(actions < 0)
        or np.any(actions >= num_actions)
        or np.any(actions[:, 0] == actions[:, 1])
    ):
        raise ValueError("Invalid action candidates in the decision list")

    successes = np.full((len(decisions), repeats, len(ARMS)), -1, dtype=np.int8)
    for row in _read_jsonl(branches / "attempts.jsonl"):
        index = by_decision[row["decision_id"]]
        repeat, arm = row["repeat"], ARMS.index(row["arm"])
        if (
            not 0 <= repeat < repeats
            or successes[index, repeat, arm] != -1
            or row["action"] != actions[index, arm]
            or not 0 < row["steps"] <= decisions[index]["horizon"]
            or row["outcome"] not in {"success", "death", "timeout", "horizon_limit"}
            or (
                row["outcome"] == "horizon_limit"
                and row["steps"] != decisions[index]["horizon"]
            )
        ):
            raise ValueError(f"Invalid or duplicate attempt for {row['decision_id']}")
        successes[index, repeat, arm] = (
            row["outcome"] == "success" and row["steps"] <= deadline
        )
    if np.any(successes == -1):
        raise ValueError(f"Incomplete paired attempts in {branches}")

    features = {}
    with gzip.open(dataset / "trajectories.jsonl.gz", "rt", encoding="utf-8") as stream:
        for line in stream:
            trace = json.loads(line)
            index = by_episode.get(trace["episode_id"])
            if index is None:
                continue
            prefix = decisions[index]["prefix_steps"]
            features[index] = _public_features(
                trace, prefix, collection["config"]["budget"], num_actions
            )
            if len(features) == len(decisions):
                break
    if len(features) != len(decisions):
        raise ValueError("Some decision prefixes are missing from the collection")
    return RankingData(
        tuple(by_decision),
        worlds,
        np.stack([features[i] for i in range(len(decisions))]),
        actions,
        successes,
        split,
    )
