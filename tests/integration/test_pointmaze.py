import gzip
import json
import subprocess
from hashlib import sha256

import pytest


def test_pointmaze_conditions_on_budget_without_extending_queries():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.models.codec import ObservationCodec
    from worth_reaching.models.continuous_completion import (
        BudgetedCompletionConfig,
        public_features,
    )
    from worth_reaching.planning.receding_goals import completion_context

    config = BudgetedCompletionConfig(max_horizon=391, max_action_budget=600)
    state, goal = np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32)
    history = [(state, np.zeros(2, dtype=np.float32))]
    short, long = (
        completion_context(history, goal, budget, config) for budget in (400, 600)
    )
    np.testing.assert_array_equal(short.offsets, long.offsets)
    assert short.offsets[0, 0] == 43
    assert (short.remaining[0], long.remaining[0]) == (400, 600)
    codec = ObservationCodec(config.width, config.latent_size, state.size)
    short_features = public_features(codec, short, state, goal, config)
    long_features = public_features(codec, long, state, goal, config)
    assert short_features[0, -9].item() == pytest.approx(400 / 600)
    assert long_features[0, -9].item() == pytest.approx(1)


@pytest.mark.parametrize(
    ("task", "environment", "action_limit"),
    [
        ("umaze", "PointMaze_UMaze-v3", 300),
        ("medium", "PointMaze_Medium-v3", 600),
        ("large", "PointMaze_Large-v3", 800),
    ],
)
def test_pointmaze_cli_trains_and_pairs_rollouts(
    tmp_path, monkeypatch, task, environment, action_limit
):
    gym = pytest.importorskip("gymnasium")
    robotics = pytest.importorskip("gymnasium_robotics")
    minari = pytest.importorskip("minari")
    np = pytest.importorskip("numpy")
    gym.register_envs(robotics)
    data_dir = tmp_path / "datasets"
    dataset_id = f"D4RL/pointmaze/{task}-v2"
    monkeypatch.setenv("MINARI_DATASETS_PATH", str(data_dir))
    settings = {"continuing_task": True, "reset_target": False, "reward_type": "sparse"}
    evaluation = gym.make(environment, max_episode_steps=action_limit, **settings)
    collector = minari.DataCollector(
        gym.make(environment, max_episode_steps=18, **settings), data_format="hdf5"
    )
    try:
        rng = np.random.default_rng(7)
        for seed in range(7):
            collector.reset(seed=seed)
            for _ in range(18):
                collector.step(rng.uniform(-1, 1, 2).astype(np.float32))
        collector.create_dataset(dataset_id, eval_env=evaluation)
    finally:
        evaluation.close()
        collector.close()

    root = tmp_path / "run"
    selection = ["--eval-budgets", "12", "30"] if task == "medium" else []
    if selection:
        recipe = tmp_path / "completion.toml"
        recipe.write_text(
            "[model]\nwidth = 48\nlatent_size = 10\nwindows = 128\nlearning_rate = 0.002\n"
        )
        selection.extend(["--config", recipe])
    steps = 30 if selection else action_limit
    span = 18
    subprocess.run(
        [
            "wr",
            "pointmaze-latent",
            "--data-dir",
            data_dir,
            "--task",
            task,
            "--out",
            root,
            "--windows",
            "32",
            "--codec-updates",
            "1",
            "--predictor-updates",
            "1",
            "--flow-updates",
            "1",
            "--eval-repeats",
            "2",
            *selection,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    out = root / "budget-30" if selection else root
    if selection:
        sweep = json.loads((root / "summary.json").read_text())
        assert set(sweep["budgets"]) == {"12", "30"}
        assert sweep["evaluation_environment_steps"] == 4 * (12 + 30)
        assert (root / "budget-12/weights.pt").samefile(root / "weights.pt")
        assert (out / "weights.pt").samefile(root / "weights.pt")
        with gzip.open(root / "budget-12/plans.jsonl.gz", "rt") as stream:
            shorter = json.loads(next(stream))
        assert shorter["remaining_budget"] == shorter["prediction_span"] == 12
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["task"] == task
    assert manifest["dataset"] == dataset_id
    assert manifest["environment"] == environment
    assert manifest["training"]["max_horizon"] == span
    assert manifest["training"]["max_action_budget"] == action_limit
    if selection:
        assert manifest["training"]["width"] == 48
        assert manifest["training"]["learning_rate"] == 0.002
        assert manifest["training"]["windows"] == 32
    assert manifest["evaluation"]["action_limit"] == steps
    assert manifest["window_sampling"]["span_source"] == "training_episode_p90"
    for name, minimum, episodes in (("train", 8, 5), ("validation", 128, 1)):
        coverage = manifest["window_sampling"][name]
        assert coverage["max_span"] == span
        assert coverage["max_remaining_budget"] == action_limit
        assert coverage["windows_with_budget_above_span"] > 0
        assert coverage["full_span_reset_windows"] >= minimum
        assert coverage["episodes_long_enough"] == episodes
    split = json.loads((root / "split.json").read_text())
    assert split == {
        "train_episodes": [0, 1, 2, 3, 4],
        "validation_episodes": [6],
        "gap_episode": 5,
    }
    episodes = json.loads((out / "episodes.json").read_text())
    if selection:
        short_episodes = json.loads((root / "budget-12/episodes.json").read_text())
        assert [(row["start"], row["goal"]) for row in short_episodes["direct"]] == [
            (row["start"], row["goal"]) for row in episodes["direct"]
        ]
    with np.load(out / "trajectories.npz", allow_pickle=False) as paths:
        for direct, latent in zip(episodes["direct"], episodes["latent"], strict=True):
            assert (direct["seed"], direct["start"], direct["goal"]) == (
                latent["seed"],
                latent["start"],
                latent["goal"],
            )
            assert direct["steps"] == latent["steps"] == steps
            assert paths[f"latent_{latent['seed']}"].shape == (steps + 1, 4)
    with gzip.open(out / "plans.jsonl.gz", "rt") as stream:
        plan = json.loads(next(stream))
        assert plan["remaining_budget"] == steps
        assert plan["prediction_span"] == span
        assert all(
            row["remaining_budget"] == steps - row["step"]
            and row["prediction_span"] == min(row["remaining_budget"], span)
            for row in map(json.loads, stream)
        )
    assert np.asarray(plan["latent_routes"]).shape == (4, 8, 10 if selection else 16)
    assert np.asarray(plan["decoded_routes"]).shape == (4, 8, 4)
    assert (
        json.loads((out / "summary.json").read_text())["evaluation_environment_steps"]
        == 4 * steps
    )
    if task == "medium":
        unsupported = tmp_path / "unsupported"
        rejected = subprocess.run(
            [
                "wr",
                "pointmaze-latent",
                "--data-dir",
                data_dir,
                "--out",
                unsupported,
                "--prediction-span",
                "19",
            ],
            capture_output=True,
            text=True,
        )
        assert rejected.returncode == 2
        assert "The 19-step prediction span is unsupported" in rejected.stderr
        assert not unsupported.exists()
        energy_out = tmp_path / "energy"
        subprocess.run(
            [
                "wr",
                "pointmaze-latent-energy",
                "--runs",
                out,
                "--data-dir",
                data_dir,
                "--out",
                energy_out,
                "--eval-repeats",
                "1",
                "--prefix-length",
                "2",
                "--candidates",
                "3",
                "--density-samples",
                "8",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        bank_bytes = (energy_out / "candidate-bank.json").read_bytes()
        energy_manifest = json.loads((energy_out / "manifest.json").read_text())
        assert (
            energy_manifest["candidate_bank_sha256"] == sha256(bank_bytes).hexdigest()
        )
        (declared,) = json.loads(bank_bytes)["7"]
        assert declared["seed"] == 69001
        assert declared["prediction_span"] == span
        assert declared["remaining_budget"] == steps
        assert declared["deadlines"] == [2, 4]
        assert declared["candidate_seed"] != declared["reference_seed"]
        assert (
            declared["selections"]["lowest_energy"]
            == min(declared["candidates"], key=lambda row: row["energy"])["index"]
        )
        outcomes = json.loads((energy_out / "episodes.json").read_text())
        (observed,) = outcomes["models"]["7"]
        assert observed["selections"] == declared["selections"]
        for candidate, outcome in zip(
            declared["candidates"], observed["attempts"], strict=True
        ):
            assert "success" not in candidate
            assert len(candidate["latent_prefix"]) == len(candidate["targets"]) == 2
            assert outcome["index"] == candidate["index"]
            assert (outcome["start"], outcome["goal"], outcome["steps"]) == (
                declared["start"],
                declared["goal"],
                steps,
            )
        assert (
            json.loads((energy_out / "summary.json").read_text())[
                "evaluation_environment_steps"
            ]
            == 4 * steps
        )

        regions_out = tmp_path / "regions"
        subprocess.run(
            [
                "wr",
                "pointmaze-regions",
                "--checkpoint",
                out,
                "--data-dir",
                data_dir,
                "--out",
                regions_out,
                "--eval-repeats",
                "1",
                "--sample-repeats",
                "1",
                "--region-updates",
                "2",
                "--executor-updates",
                "2",
                "--executor-pairs",
                "64",
                "--recall-levels",
                "0.75",
                "--region-samples",
                "4",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = json.loads((regions_out / "manifest.json").read_text())
        assert manifest["source_weights_sha256"]
        assert {"point_native", "sample_support@0.75", "region_flow@0.75"} <= set(
            manifest["arms"]
        )
        regions_summary = json.loads((regions_out / "summary.json").read_text())
        assert regions_summary["starts"] == 1
        arms = regions_summary["arms"]
        assert {"direct", "oracle", "point_native/forecast"} <= set(arms)
        assert arms["direct"]["plans_per_episode"] == 0
        assert arms["oracle"]["plans_per_episode"] >= 1
        flow = arms["region_flow@0.75/sample"]
        assert "difference_from_point_native" in flow
        assert 0 <= flow["target_checks"]["in_wall"] <= 1
        with gzip.open(regions_out / "attempts.jsonl.gz", "rt") as stream:
            attempts = [json.loads(line) for line in stream]
        assert len(attempts) == 2 + 2 * len(manifest["arms"])
        assert {row["seed"] for row in attempts} == {29001}


def test_maze_geometry_scores_walls_and_walks_the_shortest_path():
    gym = pytest.importorskip("gymnasium")
    robotics = pytest.importorskip("gymnasium_robotics")
    np = pytest.importorskip("numpy")
    from worth_reaching.tasks.pointmaze.geometry import MazeGeometry

    gym.register_envs(robotics)
    env = gym.make("PointMaze_Medium-v3")
    try:
        geometry = MazeGeometry(env.unwrapped.maze)
    finally:
        env.close()
    start = np.array([2.5, 1.5, 0, 0], np.float32)
    goal = np.array([2.5, -2.5, 0, 0], np.float32)
    # The cell below the start is a wall, so the path detours through the middle.
    assert not geometry.free(np.array([2.5, 0.5]))
    assert not geometry.free(np.array([9.0, 0.0]))
    assert geometry.distance(start, goal) == 8
    waypoint = geometry.waypoint(start, goal, cells=1, speed=2.0)
    np.testing.assert_allclose(waypoint[:2], [1.5, 1.5])
    np.testing.assert_allclose(waypoint[2:], [-2.0, 0.0], atol=1e-6)
    assert geometry.progress(start, waypoint, goal)
    near = np.array([2.5, -1.5, 0, 0], np.float32)
    np.testing.assert_array_equal(
        geometry.waypoint(near, goal, cells=1, speed=2.0), goal
    )
