import gzip
import json
import subprocess
from pathlib import Path

import pytest

CONFIG = Path(__file__).resolve().parents[2] / "configs/craftax/iron_pickaxe.toml"


def test_collect_cli_records_a_frozen_split(tmp_path):
    pytest.importorskip("craftax")
    out = tmp_path / "craftax"
    subprocess.run(
        [
            "wr",
            "collect",
            "--config",
            CONFIG,
            "--out",
            out,
            "--episodes",
            "2",
            "--budget",
            "2",
        ],
        check=True,
    )
    splits = json.loads((out / "splits.json").read_text())
    manifest = json.loads((out / "manifest.json").read_text())
    summary = json.loads((out / "summary.json").read_text())
    with gzip.open(out / "trajectories.jsonl.gz", "rt") as stream:
        trajectories = [json.loads(line) for line in stream]
    assert manifest["config"] == {"seed_start": 1000, "episodes": 2, "budget": 2}
    assert [row["world_seed"] for row in splits["episodes"]] == [1000, 1001]
    assert [row["episode_id"] for row in splits["episodes"]] == [
        row["episode_id"] for row in trajectories
    ]
    assert summary["outcomes"] == {"success": 0, "death": 0, "timeout": 2}
    assert summary["environment_steps"] == 4
    for row in trajectories:
        assert set(row) == {"episode_id", "initial_observation", "steps", "outcome"}
        assert row["outcome"] == "timeout"
        assert [step["remaining_actions"] for step in row["steps"]] == [1, 0]
        assert len(row["initial_observation"]) == len(row["steps"][0]["observation"])


def test_craftax_branch_replay_and_terminal_labels():
    pytest.importorskip("craftax")
    import jax
    import numpy as np
    from craftax.craftax_classic.constants import Action, BlockType

    from worth_reaching.envs.craftax import CraftaxClassic

    env = CraftaxClassic(budget=2)
    ordinary = env.reset(jax.random.key(0))
    prefix = ordinary.step(Action.NOOP.value, key=jax.random.key(1))
    snapshot = ordinary.snapshot()
    restored = env.restore(snapshot)
    np.testing.assert_array_equal(restored.observation, prefix.observation)
    key = jax.random.key(2)
    branch_step = restored.step(Action.RIGHT.value, key=key)
    assert ordinary.snapshot() is snapshot
    ordinary_step = ordinary.step(Action.RIGHT.value, key=key)
    np.testing.assert_array_equal(branch_step.observation, ordinary_step.observation)
    assert branch_step.outcome == ordinary_step.outcome == "timeout"

    # Arrange the last action of the crafting sequence in a private fixture.
    row, col = map(int, snapshot.player_position)
    ready = snapshot.replace(
        map=snapshot.map.at[row - 1, col]
        .set(BlockType.CRAFTING_TABLE.value)
        .at[row + 1, col]
        .set(BlockType.FURNACE.value),
        inventory=snapshot.inventory.replace(iron=1, wood=1, coal=1, stone=1),
    )
    successful = env.restore(ready)
    step = successful.step(Action.MAKE_IRON_PICKAXE.value, key=key)
    assert step.remaining_actions == 0
    assert step.outcome == "success"
    with pytest.raises(RuntimeError, match="finished episode"):
        successful.step(Action.NOOP.value, key=key)

    lava = snapshot.replace(map=snapshot.map.at[row, col + 1].set(BlockType.LAVA.value))
    step = env.restore(lava).step(Action.RIGHT.value, key=key)
    assert step.remaining_actions == 0
    assert step.outcome == "death"

    initial = env.reset(jax.random.key(0)).snapshot()
    mob_patterns = set()
    for seed in range(32):
        branch = env.restore(initial)
        branch.step(Action.NOOP.value, key=jax.random.key(seed))
        mobs = branch.snapshot().cows
        mob_patterns.add(np.asarray(mobs.mask).tobytes())
    assert len(mob_patterns) > 1


def test_replay_natural_iron_pickaxe_trajectory():
    pytest.importorskip("craftax")
    import jax

    from worth_reaching.data.collection import episode_spec
    from worth_reaching.envs.craftax import CraftaxClassic
    from worth_reaching.evaluation.craftax_branches import (
        ACTION_IDS,
        Decision,
        replay_decision,
        run_attempt,
    )

    fixture = CONFIG.parents[2] / "tests/fixtures/craftax_iron_pickaxe.json"
    trace = json.loads(fixture.read_text(encoding="utf-8"))
    env = CraftaxClassic(budget=trace["budget"])
    world_key = jax.random.key(trace["world_seed"])
    episode = env.reset(jax.random.fold_in(world_key, 0))
    transition_key = jax.random.fold_in(world_key, 1)
    recorded = {"initial_observation": episode.observation.tolist(), "steps": []}
    outcomes = []
    for index, action in enumerate(trace["actions"]):
        step = episode.step(action, key=jax.random.fold_in(transition_key, index))
        outcomes.append(step.outcome)
        recorded["steps"].append(
            {"action": action, "observation": step.observation.tolist()}
        )
    assert outcomes[:-1] == [None] * (len(outcomes) - 1)
    assert len(outcomes) == 141
    assert outcomes[-1] == "success"

    iron, noop = ACTION_IDS["make_iron_pickaxe"], ACTION_IDS["noop"]
    decision = Decision(
        episode_spec(trace["world_seed"], trace["source_policy"]),
        prefix_steps=140,
        remaining_actions=860,
        horizon=2,
        recorded_action=iron,
        alternative_action=noop,
    )

    class CraftWhenCalled:
        def initial_memory(self):
            return jax.numpy.array(0)

        def observe(self, observation, carry):
            assert observation.shape == (1345,)
            return carry + 1

        def sample(self, observation, carry, key):
            assert int(carry) == 141
            return self.observe(observation, carry), iron

    policy = CraftWhenCalled()
    replayed, memory = replay_decision(env, decision, recorded, policy)
    assert int(memory) == 141
    snapshot = replayed.snapshot()
    repeat_key = jax.random.key(1234)
    for action, expected_steps in ((iron, 1), (noop, 2)):
        result = run_attempt(
            env.restore(snapshot), memory, policy, action, 2, repeat_key
        )
        assert result.outcome == "success"
        assert result.steps == expected_steps
    limited = run_attempt(env.restore(snapshot), memory, policy, noop, 1, repeat_key)
    assert limited.outcome == "horizon_limit"
    assert limited.remaining_actions == 859
    assert replayed.snapshot() is snapshot
