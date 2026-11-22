import json
import subprocess

import pytest


def _archive(path, offset=0):
    np = pytest.importorskip("numpy")
    observation = np.zeros((7, 29), dtype=np.float32)
    observation[:, 0] = np.arange(7) + offset
    actions = np.full((7, 8), 0.3, dtype=np.float32)
    actions[[3, 6]] = np.nan
    np.savez(
        path,
        observations=observation,
        actions=actions,
        terminals=[0, 0, 0, 1, 0, 0, 1],
        qpos=np.full((7, 15), np.nan),
    )


def test_hindsight_sampling_respects_episode_endpoints(tmp_path):
    np = pytest.importorskip("numpy")
    from worth_reaching.data.antmaze import load_antmaze_archive

    path = tmp_path / "archive.npz"
    _archive(path)
    archive = load_antmaze_archive(path)
    rng = np.random.default_rng(7)
    rows, goals = archive.sample_indices(rng, 1000)
    assert np.isin(rows, [0, 1, 2, 4, 5]).all()
    assert np.all(goals > rows)
    assert np.array_equal(rows <= 3, goals <= 3)
    assert {3, 6}.issubset(set(goals))
    rows, goals = archive.sample_indices(rng, 100, max_goal_steps=1)
    assert np.all(goals == rows + 1)
    limited = load_antmaze_archive(path, max_trajectories=1)
    assert limited.observations.shape == (4, 29)
    assert limited.sample(rng, 8).actions.shape == (8, 8)


def test_antmaze_cli_resumes_and_shares_saved_starts(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    pytest.importorskip("flax")
    from worth_reaching.envs import antmaze

    dataset = "antmaze-large-navigate-v0"
    _archive(tmp_path / f"{dataset}.npz")
    _archive(tmp_path / f"{dataset}-val.npz", offset=10)
    base_args = [
        "wr",
        "antmaze-gcbc",
        "--data-dir",
        str(tmp_path),
        "--width",
        "8",
        "--batch-size",
        "4",
        "--eval-repeats",
        "1",
        "--eval-steps",
        "2",
        "--checkpoint-every",
        "2",
    ]
    first, continued, uninterrupted = (
        tmp_path / name for name in ("first", "resume", "full")
    )

    def train(out, updates, *settings):
        subprocess.run(
            [*base_args, "--out", str(out), "--updates", str(updates), *settings],
            check=True,
            capture_output=True,
            text=True,
        )

    train(first, 2)
    starts_path = str(first / "starts.npz")
    train(continued, 4, "--resume", str(first), "--starts", starts_path)
    train(uninterrupted, 4, "--starts", starts_path)
    assert (continued / "weights.msgpack").read_bytes() == (
        uninterrupted / "weights.msgpack"
    ).read_bytes()
    assert (continued / "episodes.json").read_bytes() == (
        uninterrupted / "episodes.json"
    ).read_bytes()
    assert json.loads((continued / "manifest.json").read_text())["resumed_at"] == 2
    summary = json.loads((continued / "summary.json").read_text())
    assert summary["training_environment_steps"] == 0
    assert (
        summary["policies"]["gcbc"]["episodes"]
        == summary["policies"]["zero"]["episodes"]
        == 5
    )

    environment = antmaze.AntMaze()
    try:
        bank = antmaze.load_starts(
            first / "starts.npz",
            split="dev",
            repeats=1,
            state_size=environment.state_size,
        )
        saved = bank[0]
        action = np.full(8, 0.25, dtype=np.float32)
        environment.restore(saved)
        first_step = environment.step(action)[0]
        environment.step(-action)
        np.testing.assert_array_equal(environment.restore(saved), saved.observation)
        np.testing.assert_array_equal(environment.step(action)[0], first_step)
    finally:
        environment.close()


def test_official_success_on_last_action_overrides_timeout():
    pytest.importorskip("ogbench")
    np = pytest.importorskip("numpy")
    from worth_reaching.envs.antmaze import AntMazeStart
    from worth_reaching.evaluation.antmaze import rollout

    start = AntMazeStart(1, 7, np.zeros(29), np.ones(29), np.zeros(10), np.zeros(2))

    class Episode:
        def restore(self, saved):
            assert saved is start
            return saved.observation

        def step(self, action):
            return start.observation, True, True, True

    def policy(observation, goal):
        assert not goal.flags.writeable
        np.testing.assert_array_equal(goal, start.goal)
        return np.zeros(8)

    assert rollout(Episode(), start, policy, 1)["outcome"] == "success"
