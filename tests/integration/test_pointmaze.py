import gzip
import json
import subprocess

import pytest


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

    out = tmp_path / "run"
    selection = ["--eval-steps", "18"] if task == "medium" else ["--task", task]
    steps = 18 if task == "medium" else action_limit
    subprocess.run(
        [
            "wr",
            "pointmaze-latent",
            "--data-dir",
            data_dir,
            "--out",
            out,
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
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["task"] == task
    assert manifest["dataset"] == dataset_id
    assert manifest["environment"] == environment
    assert manifest["training"]["max_horizon"] == action_limit
    assert manifest["evaluation"]["action_limit"] == steps
    split = json.loads((out / "split.json").read_text())
    assert split == {
        "train_episodes": [0, 1, 2, 3, 4],
        "validation_episodes": [6],
        "gap_episode": 5,
    }
    episodes = json.loads((out / "episodes.json").read_text())
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
    assert np.asarray(plan["latent_routes"]).shape == (4, 8, 16)
    assert np.asarray(plan["decoded_routes"]).shape == (4, 8, 4)
    assert (
        json.loads((out / "summary.json").read_text())["evaluation_environment_steps"]
        == 4 * steps
    )
