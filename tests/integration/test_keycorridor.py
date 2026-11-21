import json
import subprocess
from contextlib import closing
from dataclasses import asdict
from pathlib import Path

from minigrid.core.world_object import Ball

from worth_reaching.contracts import Action, PolicyInput
from worth_reaching.envs.minigrid import KeyCorridorEpisode, continue_toward_ball
from worth_reaching.evaluation.branches import evaluate_action, replay_prefix
from worth_reaching.runner import PilotConfig

CONFIG = Path(__file__).resolve().parents[2] / "configs/minigrid/keycorridor.toml"


def test_pilot_cli_replays_and_scores_the_last_step(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    for out in (first, second):
        subprocess.run(["wr", "pilot", "--config", CONFIG, "--out", out], check=True)

    for filename in ("prefix.json", "attempts.jsonl", "summary.json"):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()

    prefix = json.loads((first / "prefix.json").read_text())
    manifest = json.loads((first / "manifest.json").read_text())
    assert len(prefix["steps"]) == 12
    assert manifest["environment_resets"] == 3
    assert manifest["prefix_recording_steps"] == 12
    assert manifest["prefix_replay_steps"] == 24
    assert manifest["branch_steps"] == 4
    good, wasteful = map(
        json.loads, (first / "attempts.jsonl").read_text().splitlines()
    )
    assert [step["action"] for step in good["steps"]] == ["right", "pickup"]
    assert [step["action"] for step in wasteful["steps"]] == ["forward", "right"]
    assert good["outcome"] == "success"
    assert good["steps"][-1]["terminated"]
    assert good["steps"][-1]["truncated"]
    assert wasteful["outcome"] == "timeout"
    assert wasteful["steps"][-1]["truncated"]
    assert not wasteful["steps"][-1]["terminated"]
    assert good["remaining_actions"] == wasteful["remaining_actions"] == 0
    summary = json.loads((first / "summary.json").read_text())
    assert summary == {
        "attempts": 2,
        "successes": sum(row["steps"][-1]["success"] for row in (good, wasteful)),
        "outcomes": {"right": "success", "forward": "timeout"},
    }


def test_hidden_tile_and_branch_isolation():
    config = PilotConfig.load(CONFIG)
    with (
        closing(
            KeyCorridorEpisode(seed=config.seed, budget=config.budget)
        ) as reference,
        closing(KeyCorridorEpisode(seed=config.seed, budget=config.budget)) as branch,
    ):
        reference_prefix = replay_prefix(reference, config.prefix)
        branch_prefix = replay_prefix(branch, config.prefix)
        assert branch_prefix == reference_prefix

        # Change an unseen tile behind the agent, using simulator access only in the evaluator test.
        native = branch._env.unwrapped
        assert native.grid.get(1, 1) is None
        native.grid.set(1, 1, Ball("red"))
        policy_calls = []

        def observe(policy_input: PolicyInput) -> Action:
            policy_calls.append(
                json.dumps(asdict(policy_input), sort_keys=True).encode()
            )
            return continue_toward_ball(policy_input)

        attempts = [
            evaluate_action(
                episode,
                prefix,
                Action.RIGHT,
                budget=config.budget,
                continuation=observe,
            )
            for episode, prefix in (
                (branch, branch_prefix),
                (reference, reference_prefix),
            )
        ]
        assert len(policy_calls) == 2
        assert policy_calls[0] == policy_calls[1]
        assert attempts[0] == attempts[1]
        assert attempts[1].success
