import gzip
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
    from worth_reaching.tasks.antmaze.dataset import load_antmaze_archive

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
    windows = archive.sample_windows(rng, 30, history=4, queries=1, max_horizon=3)
    context = windows.context
    initial = context.observations[:, -1, 0]
    assert np.array_equal(initial <= 3, context.goal[:, 0] <= 3)
    np.testing.assert_array_equal(
        windows.targets[:, :, 0], initial[:, None] + context.offsets
    )
    np.testing.assert_array_equal(context.goal[:, 0], initial + context.remaining)
    assert (context.offsets > 0).all()
    assert (context.offsets < context.remaining[:, None]).all()
    begin = np.where(initial <= 3, 0, 4)
    expected = initial[:, None] + np.arange(-3, 1)
    np.testing.assert_array_equal(context.present, expected >= begin[:, None])
    assert (context.incoming_actions[expected <= begin[:, None]] == 0).all()


def test_antmaze_cli_resumes_and_shares_saved_starts(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    pytest.importorskip("flax")
    from worth_reaching.tasks.antmaze import environment as antmaze

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

    waypoints = tmp_path / "waypoints"
    subprocess.run(
        [
            "wr",
            "antmaze-waypoints",
            "--checkpoint",
            continued,
            "--starts",
            starts_path,
            "--eval-repeats",
            "1",
            "--eval-steps",
            "2",
            "--waypoint-steps",
            "1",
            "--out",
            waypoints,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    paired = json.loads((waypoints / "episodes.json").read_text())
    baseline = json.loads((continued / "episodes.json").read_text())
    assert paired["direct"] == [
        {key: value for key, value in row.items() if key != "policy"}
        for row in baseline
        if row["policy"] == "gcbc"
    ]
    assert all(
        row["attempts"][0]["outcome"] == "timeout" for row in paired["waypoints"]
    )

    latent = tmp_path / "latent"
    subprocess.run(
        [
            "wr",
            "antmaze-latent",
            "--checkpoint",
            continued,
            "--starts",
            starts_path,
            "--data-dir",
            tmp_path,
            "--eval-repeats",
            "1",
            "--eval-steps",
            "6",
            "--windows",
            "12",
            "--queries",
            "2",
            "--candidates",
            "2",
            "--probe-first-candidates",
            "--codec-updates",
            "1",
            "--predictor-updates",
            "1",
            "--flow-updates",
            "1",
            "--out",
            latent,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    episodes = json.loads((latent / "episodes.json").read_text())
    assert len(episodes["direct"]) == len(episodes["latent"]) == 5
    assert all(
        (a["task"], a["seed"]) == (b["task"], b["seed"])
        for a, b in zip(episodes["direct"], episodes["latent"], strict=True)
    )
    with gzip.open(latent / "plans.jsonl.gz", "rt", encoding="utf-8") as stream:
        plans = [json.loads(line) for line in stream]
    assert len(plans) == sum(episode["plans"] for episode in episodes["latent"])
    assert plans[0]["step"] == 0
    assert plans[0]["executed_actions"] == plans[0]["query_offsets"][0] == 2
    assert np.asarray(plans[0]["decoded_routes"]).shape == (2, 2, 29)
    assert np.asarray(plans[0]["latent_routes"]).shape == (2, 2, 16)
    candidate_episodes = json.loads((latent / "first_candidates.json").read_text())
    assert candidate_episodes[0] == episodes["latent"]
    assert [(row["task"], row["seed"]) for row in candidate_episodes[1]] == [
        (row["task"], row["seed"]) for row in episodes["direct"]
    ]
    summary = json.loads((latent / "summary.json").read_text())
    assert summary["evaluation_environment_steps"] == sum(
        row["steps"] for arm in [episodes["direct"], *candidate_episodes] for row in arm
    )
    assert summary["first_candidate_probe"]["starts"] == 5
    validation = json.loads((latent / "validation.json").read_text())
    assert validation["reconstructed_xy_mean_error"] >= 0
    assert (
        validation["best_sampled_route_xy_mean_error"]
        <= validation["sampled_xy_mean_error"]
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
    from worth_reaching.tasks.antmaze.baseline import rollout
    from worth_reaching.tasks.antmaze.environment import AntMazeStart

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


def test_candidate_probe_changes_only_first_attempt():
    pytest.importorskip("ogbench")
    np = pytest.importorskip("numpy")
    from worth_reaching.models.continuous_completion import CompletionConfig
    from worth_reaching.planning.receding_goals import RecedingGoals

    class Completion:
        config = CompletionConfig(queries=2, candidates=2)

        def propose(self, context, key):
            latents = np.zeros((1, 2, 2, 16))
            targets = np.broadcast_to(
                np.array([1.0, 2.0])[None, :, None, None], (1, 2, 2, 29)
            )
            return latents, targets

    def actor(observation, goal):
        return np.full(8, goal[0])

    observation = np.zeros(29)
    goal = np.full(29, 3.0)
    primary = RecedingGoals(actor, Completion(), 7, 6, action_size=8)
    alternate = RecedingGoals(actor, Completion(), 7, 6, action_size=8, first_route=1)
    assert [primary(observation, goal)[0] for _ in range(6)] == [1, 1, 1, 1, 3, 3]
    assert [alternate(observation, goal)[0] for _ in range(6)] == [2, 2, 1, 1, 3, 3]
    assert [row["selected_route"] for row in alternate.plans] == [1, 0, 0]


def test_waypoint_plan_keeps_corners_and_excludes_terminal_goal():
    pytest.importorskip("ogbench")
    np = pytest.importorskip("numpy")
    from worth_reaching.tasks.antmaze.waypoints import (
        shortest_cells,
        waypoint_cells,
    )

    distances = np.array([[3, 2, 1, 0], [4, 3, 2, 1]])
    path = shortest_cells(distances, (1, 0), (0, 3))
    assert path == [(1, 0), (0, 0), (0, 1), (0, 2), (0, 3)]
    assert waypoint_cells(path, 2) == [(0, 0), (0, 2)]
    assert waypoint_cells([(0, 0), (0, 1)], 1) == []


def test_waypoint_controller_advances_on_arrival_and_falls_back_on_timeout():
    pytest.importorskip("ogbench")
    np = pytest.importorskip("numpy")
    from worth_reaching.tasks.antmaze.waypoints import (
        FollowWaypoints,
        WaypointConfig,
    )

    first, second, terminal = np.array([(4, 0), (8, 0), (12, 0)])
    follower = FollowWaypoints(
        lambda observation, goal: goal, [first, second], WaypointConfig(1, 2, 1)
    )
    current = np.array([0, 0])
    np.testing.assert_array_equal(follower(current, terminal), first)
    follower.observe(first, 1)
    np.testing.assert_array_equal(follower(first, terminal), second)
    follower.observe(first, 2)
    np.testing.assert_array_equal(follower(first, terminal), second)
    follower.observe(first, 3)
    np.testing.assert_array_equal(follower(first, terminal), terminal)
    follower.finish(3)
    assert [
        (x["outcome"], x["start_step"], x["end_step"]) for x in follower.attempts
    ] == [
        ("arrived", 0, 1),
        ("timeout", 1, 3),
    ]
