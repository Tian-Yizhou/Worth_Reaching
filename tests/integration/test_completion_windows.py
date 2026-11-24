import gzip
import json
import subprocess
from dataclasses import asdict

import pytest


def test_window_cli_preserves_timing_and_hides_future(tmp_path):
    pytest.importorskip("craftax")
    import numpy as np

    from worth_reaching.tasks.craftax.dataset import SPLIT_VERSION, episode_spec
    from worth_reaching.tasks.craftax.environment import (
        OBSERVATION_SCHEMA,
        SURVIVAL_TASK,
        TASK_RULES,
    )
    from worth_reaching.tasks.craftax.windows import load_completion_corpus

    specs = [episode_spec(seed) for seed in range(100)]
    train, failure = [spec for spec in specs if spec.split == "train"][:2]
    dev = next(spec for spec in specs if spec.split == "dev")
    test = next(spec for spec in specs if spec.split == "test")
    selected = [train, failure, dev, test]
    budget = 8
    collection = tmp_path / "collection"
    collection.mkdir()
    for filename, data in {
        "manifest.json": {
            "schema_version": 1,
            "task": SURVIVAL_TASK,
            "rules_version": TASK_RULES[SURVIVAL_TASK],
            "observation_schema": OBSERVATION_SCHEMA,
            "policy": "uniform_random_v1",
            "config": {"episodes": len(selected), "budget": budget},
        },
        "splits.json": {
            "version": SPLIT_VERSION,
            "episodes": [asdict(spec) for spec in selected],
        },
        "summary.json": {},
    }.items():
        (collection / filename).write_text(json.dumps(data), encoding="utf-8")
    with gzip.open(collection / "trajectories.jsonl.gz", "wt") as stream:
        for spec in selected:
            trace = {"episode_id": spec.episode_id}
            if spec == test:
                stream.write(json.dumps(trace) + "\n")
                continue
            trace.update(
                initial_observation=[spec.world_seed, 0],
                steps=[
                    {
                        "action": step,
                        "observation": [spec.world_seed, step],
                        "remaining_actions": budget - step,
                        "snapshot": f"private-{step}",
                    }
                    for step in range(1, budget + 1)
                ],
                outcome="timeout" if spec == failure else "success",
                first_craft_step=4,
            )
            stream.write(json.dumps(trace) + "\n")

    out = tmp_path / "windows"
    result = subprocess.run(
        [
            "wr",
            "prepare-windows",
            "--collection",
            collection,
            "--out",
            out,
            "--seed",
            "7",
            "--windows-per-episode",
            "4",
            "--min-span",
            "2",
            "--max-span",
            "3",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout)["splits"] == {
        "train": {"episodes": 2, "successful_episodes": 1, "windows": 4},
        "dev": {"episodes": 1, "successful_episodes": 1, "windows": 4},
    }
    assert not (out / "test").exists()
    corpus = load_completion_corpus(out)
    for spec in (train, dev):
        examples = list(corpus.iter_examples(spec.split))
        assert len({example.start_step for example in examples}) == 4
        assert examples[0].start_step == 0
        for example in examples:
            assert example.world_id == spec.world_id
            assert example.episode_id == spec.episode_id
            inputs, target = example.inputs, example.target
            assert set(asdict(inputs)) == {
                "observations",
                "actions",
                "goal",
                "remaining_actions",
                "query_steps",
                "terminal_example",
            }
            start = example.start_step
            span = len(inputs.query_steps)
            assert 2 <= span <= 3
            assert start + span < budget
            assert inputs.goal == SURVIVAL_TASK
            assert inputs.remaining_actions == budget - start
            assert inputs.terminal_example is None
            assert inputs.observations.base is None and inputs.actions.base is None
            np.testing.assert_array_equal(inputs.observations[:, 1], range(start + 1))
            np.testing.assert_array_equal(inputs.actions, range(1, start + 1))
            np.testing.assert_array_equal(inputs.query_steps, range(1, span + 1))
            hidden_steps = range(start + 1, start + span + 1)
            np.testing.assert_array_equal(target.observations[:, 1], hidden_steps)
            np.testing.assert_array_equal(target.actions, hidden_steps)
        with np.load(next((out / spec.split).glob("*.npz"))) as shard:
            allowed = {"observations", "actions", "windows", "world_id", "episode_id"}
            if spec == train:
                allowed.add("terminal_example")
            assert set(shard.files) == allowed
            assert shard["observations"][-1, 1] == budget - 1
            assert shard["actions"][-1] == budget - 1

    with_endpoint = next(corpus.iter_examples("train", with_terminal_example=True))
    np.testing.assert_array_equal(
        with_endpoint.inputs.terminal_example, [train.world_seed, budget]
    )
    with pytest.raises(ValueError, match="train only"):
        next(corpus.iter_examples("dev", with_terminal_example=True))
    with pytest.raises(ValueError, match="train and dev"):
        next(corpus.iter_examples("test"))


def test_windows_are_reproducible_and_fit_the_smallest_interior():
    pytest.importorskip("craftax")
    from worth_reaching.data.completion import WindowConfig

    config = WindowConfig(seed=7)
    assert config.sample("world-a", 2) == [(0, 1)]
    assert config.sample("world-a", 256) == WindowConfig(seed=7).sample("world-a", 256)
    assert config.sample("world-a", 256) != WindowConfig(seed=8).sample("world-a", 256)
    with pytest.raises(ValueError, match="terminal step"):
        config.sample("world-a", 1)
