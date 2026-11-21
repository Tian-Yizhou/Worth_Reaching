import gzip
import json

import jax
import numpy as np

from worth_reaching.data.collection import EpisodeSpec
from worth_reaching.envs.craftax import CraftaxStep
from worth_reaching.evaluation.craftax_waypoints import (
    ACTION_IDS,
    Waypoint,
    WaypointAttempt,
    WaypointConfig,
    _eligible_traces,
    attempt_waypoint,
    score_bank,
)


def test_fixed_prefix_selects_preterminal_death_histories_in_archive_order(tmp_path):
    episodes = {
        name: EpisodeSpec(seed, name, name, split)
        for seed, (name, split) in enumerate(
            [
                ("train", "train"),
                ("first", "dev"),
                ("short", "dev"),
                ("reordered", "dev"),
            ]
        )
    }
    rows = [
        {"episode_id": "train", "outcome": "death", "steps": [{}] * 11},
        {"episode_id": "first", "outcome": "death", "steps": [{}] * 11},
        {"episode_id": "short", "outcome": "death", "steps": [{}] * 10},
        {"outcome": "death", "steps": [{}] * 11, "episode_id": "reordered"},
    ]
    with gzip.open(
        tmp_path / "trajectories.jsonl.gz", "wt", encoding="utf-8"
    ) as stream:
        stream.writelines(json.dumps(row) + "\n" for row in rows)
    config = WaypointConfig(2, 1, 0, 0, "dev", prefix_steps=2, source_outcome="death")
    selected = list(_eligible_traces(tmp_path, episodes, config))
    assert [(spec.episode_id, prefix) for spec, _, prefix in selected] == [
        ("first", 2),
        ("reordered", 2),
    ]


def test_direct_scoring_counts_recoveries_below_arrival_threshold():
    reached_success = WaypointAttempt(1, 1, 2, 8, "success")
    missed_success = WaypointAttempt(None, 1, 2, 8, "success")
    reached_failure = WaypointAttempt(1, 1, 2, 8, "horizon_limit")
    bank = {
        "recoverable": [
            reached_success,
            missed_success,
            missed_success,
            missed_success,
        ],
        "frequent_arrival": [
            reached_success,
            reached_success,
            reached_failure,
            reached_failure,
        ],
    }
    assert score_bank(bank) == {
        "direct_attempt": "recoverable",
        "factorized_filter": "frequent_arrival",
        "expert_fallback": "expert",
    }
    assert (
        score_bank({"recoverable": bank["recoverable"]})["factorized_filter"]
        == "expert"
    )


def test_both_selectors_can_choose_expert_from_shared_calibration():
    reached_success = WaypointAttempt(1, 1, 2, 8, "success")
    missed_success = WaypointAttempt(None, 1, 2, 8, "success")
    expert_success = WaypointAttempt(None, 0, 2, 8, "success")
    expert_failure = WaypointAttempt(None, 0, 2, 8, "horizon_limit")
    bank = {
        "recoverable": [
            reached_success,
            missed_success,
            missed_success,
            missed_success,
        ],
        "expert": [expert_success, expert_success, expert_failure, expert_failure],
    }
    assert score_bank(bank)["direct_attempt"] == "recoverable"
    assert score_bank(bank)["factorized_filter"] == "expert"

    bank["reliable"] = [reached_success] * 4
    bank["expert"] = [expert_success] * 4
    assert score_bank(bank)["direct_attempt"] == "expert"
    assert score_bank(bank)["factorized_filter"] == "expert"


def test_executor_miss_continues_with_public_memory():
    observation = np.zeros(1, dtype=np.float32)

    class Episode:
        def __init__(self):
            self.observation = observation
            self.actions = []

        def step(self, action, *, key):
            self.actions.append(action)
            done = len(self.actions) == 2
            return CraftaxStep(
                observation, 10 - len(self.actions), "success" if done else None
            )

    class Expert:
        def observe(self, observation, carry):
            return carry + 1

        def sample(self, observation, carry, key):
            assert int(carry) == 1
            return carry + 1, ACTION_IDS["make_iron_pickaxe"]

    episode = Episode()
    waypoint = Waypoint(np.ones(1, dtype=np.float32), (ACTION_IDS["right"],))
    attempt = attempt_waypoint(
        episode,
        jax.numpy.array(0),
        Expert(),
        waypoint,
        np.array([True]),
        jax.random.key(0),
        jax.random.key(1),
    )
    assert episode.actions == [ACTION_IDS["noop"], ACTION_IDS["make_iron_pickaxe"]]
    assert attempt.success and attempt.recovered
    assert attempt.arrival_step is None
    assert attempt.executor_steps == 1
    assert attempt.total_steps == 2
    assert attempt.remaining_actions == 8
