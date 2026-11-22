import gzip
import json
import subprocess

import pytest


def test_scene_restores_goal_physics_and_next_observation():
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    from worth_reaching.envs.scene import Scene

    env = Scene()
    try:
        start = env.capture(4, 43001)
        action = np.full(5, 0.2, dtype=np.float32)
        first, success, done = env.step(action)
        env.capture(2, 81000)
        np.testing.assert_array_equal(env.restore(start), start.observation)
        replay, replay_success, replay_done = env.step(action)
        np.testing.assert_array_equal(first, replay)
        assert (replay_success, replay_done) == (success, done)
    finally:
        env.close()


def test_scene_cli_trains_and_reuses_start_and_controller(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    pytest.importorskip("jax")
    for suffix, seed in (("", 5), ("-val", 6)):
        rng = np.random.default_rng(seed)
        terminals = np.arange(30) % 15 == 14
        actions = rng.uniform(-1, 1, (30, 5)).astype(np.float32)
        actions[terminals] = np.nan
        np.savez(
            tmp_path / f"scene-play-v0{suffix}.npz",
            observations=rng.normal(size=(30, 40)).astype(np.float32),
            actions=actions,
            terminals=terminals,
            qpos=np.full((30, 15), np.nan),
            qvel=np.full((30, 14), np.nan),
            button_states=np.full((30, 2), np.nan),
        )

    out = tmp_path / "scene"
    subprocess.run(
        [
            "wr",
            "scene-latent",
            "--data-dir",
            tmp_path,
            "--out",
            out,
            "--task",
            "4",
            "--eval-repeats",
            "1",
            "--eval-steps",
            "12",
            "--windows",
            "16",
            "--executor-updates",
            "1",
            "--codec-updates",
            "1",
            "--predictor-updates",
            "1",
            "--flow-updates",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["evaluation"]["tasks"] == [4]
    assert manifest["observation"] == "official_state"
    assert manifest["train_sha256"] != manifest["validation_sha256"]
    episodes = json.loads((out / "episodes.json").read_text())
    direct, latent = episodes["direct"][0], episodes["latent"][0]
    assert direct["seed"] == latent["seed"] == 43001
    assert direct["task"] == latent["task"] == 4
    with np.load(out / "trajectories.npz", allow_pickle=False) as paths:
        np.testing.assert_array_equal(
            paths["direct_4_43001_observations"][0],
            paths["latent_4_43001_observations"][0],
        )
        assert paths["latent_4_43001_actions"].shape == (latent["steps"], 5)
    with gzip.open(out / "plans.jsonl.gz", "rt") as stream:
        plan = json.loads(next(stream))
    assert np.asarray(plan["latent_routes"]).shape == (4, 8, 16)
    assert np.asarray(plan["decoded_routes"]).shape == (4, 8, 40)
    assert (
        json.loads((out / "summary.json").read_text())["evaluation_environment_steps"]
        == direct["steps"] + latent["steps"]
    )
