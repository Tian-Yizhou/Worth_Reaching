import json
import subprocess

import pytest


def _archive(np, path, seed):
    rng = np.random.default_rng(seed)
    terminals = np.arange(30) % 15 == 14
    actions = rng.uniform(-1, 1, (30, 5)).astype(np.float32)
    actions[terminals] = np.nan
    np.savez(
        path,
        observations=rng.normal(size=(30, 40)).astype(np.float32),
        actions=actions,
        terminals=terminals,
        qpos=np.full((30, 15), np.nan),
        qvel=np.full((30, 14), np.nan),
        button_states=np.full((30, 2), np.nan),
    )


def test_gcivl_batches_label_goals_within_trajectories(tmp_path):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from worth_reaching.data.ogbench import load_state_archive
    from worth_reaching.models.gcivl import GCIVL, GCIVLBatches, GCIVLConfig, gcivl_loss

    _archive(np, tmp_path / "play.npz", 1)
    archive = load_state_archive(
        tmp_path / "play.npz", observation_size=40, action_size=5
    )
    archive.observations[:] = np.arange(30, dtype=np.float32)[:, None]
    config = GCIVLConfig(batch_size=512, width=8, depth=1)
    batch = GCIVLBatches(archive, config, (3,))(0)
    observation, following, action, value_goal, actor_goal, reward, mask = batch
    rows = observation[:, 0].long()
    assert torch.equal(following[:, 0].long(), rows + 1)
    # Rewards are zero exactly when the value goal is the current state.
    at_goal = value_goal[:, 0].long() == rows
    assert torch.equal(reward == 0, at_goal) and torch.equal(mask == 0, at_goal)
    assert 0.1 < at_goal.float().mean() < 0.3
    # Actor goals are later states of the same trajectory.
    final = torch.where(rows < 14, torch.tensor(14), torch.tensor(29))
    goals = actor_goal[:, 0].long()
    assert bool(((goals > rows) & (goals <= final)).all())

    model = GCIVL(40, 40, 5, config)
    target = [value.clone() for value in model.target.parameters()]
    terms = gcivl_loss(model, batch)
    assert [error.shape for error, _ in terms] == [(512,), (512,), (512, 5)]
    assert all(
        torch.equal(before, after)
        for before, after in zip(target, model.target.parameters(), strict=True)
    )


def test_scene_executor_cli_trains_and_scores_official_tasks(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    pytest.importorskip("torch")
    for suffix, seed in (("", 5), ("-val", 6)):
        _archive(np, tmp_path / f"scene-play-v0{suffix}.npz", seed)
    out = tmp_path / "executor"
    subprocess.run(
        [
            "wr",
            "scene-executor",
            "--data-dir",
            tmp_path,
            "--out",
            out,
            "--updates",
            "2",
            "--batch-size",
            "16",
            "--task",
            "2",
            "--eval-repeats",
            "1",
            "--eval-workers",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["executor"]["kind"] == "gcivl"
    assert manifest["evaluation"]["seeds"] == [43001]
    (episode,) = json.loads((out / "episodes.json").read_text())
    assert (episode["task"], episode["seed"]) == (2, 43001)
    summary = json.loads((out / "summary.json").read_text())
    assert set(summary["per_task"]) == {"2"}
