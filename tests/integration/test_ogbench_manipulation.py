import gzip
import json
import subprocess

import pytest


def test_drawer_waypoints_require_a_stable_placement_within_one_episode():
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    pytest.importorskip("torch")
    from worth_reaching.data.ogbench import StateArchive
    from worth_reaching.evaluation.scene_reachability import (
        _arrival,
        _drawer_placed,
        _drawer_waypoints,
    )

    observations = np.zeros((130, 40), dtype=np.float32)
    observations[:, 36] = -0.16 * 18
    observations[:, 19:22] = [-0.75, 0.5, 0.2]
    put_in_drawer = [-0.95, -2.4, 0.66]
    observations[30:34, 19:22] = put_in_drawer
    observations[70:75, 19:22] = put_in_drawer
    observations[97:103, 19:22] = put_in_drawer
    observations[110:115, 19:22] = put_in_drawer
    archive = StateArchive(
        observations,
        np.zeros((130, 5), dtype=np.float32),
        np.array([99, 129]),
        np.flatnonzero(~np.isin(np.arange(130), [99, 129])),
        "fixture",
    )
    assert list(_drawer_waypoints(archive, [64], 20, 803)) == [
        ("drawer_insertion", 64, 0, 10)
    ]
    assert _arrival(observations[:74], observations[74], _drawer_placed, 5) == (
        None,
        False,
    )
    assert _arrival(observations[:75], observations[74], _drawer_placed, 5) == (
        74,
        True,
    )


def test_scene_restores_goal_physics_and_next_observation():
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    from worth_reaching.tasks.ogbench_manipulation.scene import Scene

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


@pytest.mark.parametrize(
    (
        "command",
        "options",
        "tasks",
        "environment",
        "observation_size",
        "limit",
        "reset_seed",
    ),
    [
        (
            "scene-latent",
            ["--task", "4", "--task", "5", "--eval-workers", "2"],
            [4, 5],
            "scene",
            40,
            750,
            43001,
        ),
    ],
)
def test_manipulation_cli_trains_and_reuses_start_and_controller(
    tmp_path, command, options, tasks, environment, observation_size, limit, reset_seed
):
    np = pytest.importorskip("numpy")
    pytest.importorskip("ogbench")
    pytest.importorskip("torch")
    for suffix, seed in (("", 5), ("-val", 6)):
        rng = np.random.default_rng(seed)
        terminals = np.arange(30) % 15 == 14
        actions = rng.uniform(-1, 1, (30, 5)).astype(np.float32)
        actions[terminals] = np.nan
        np.savez(
            tmp_path / f"{environment}-play-v0{suffix}.npz",
            observations=rng.normal(size=(30, observation_size)).astype(np.float32),
            actions=actions,
            terminals=terminals,
            qpos=np.full((30, 15), np.nan),
            qvel=np.full((30, 14), np.nan),
            button_states=np.full((30, 2), np.nan),
        )

    out = tmp_path / environment
    subprocess.run(
        [
            "wr",
            command,
            *options,
            "--data-dir",
            tmp_path,
            "--out",
            out,
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
    assert manifest["environment"] == f"{environment}-v0"
    assert manifest["dataset"] == f"{environment}-play-v0"
    assert manifest["training"]["max_horizon"] == limit
    assert manifest["evaluation"]["tasks"] == tasks
    assert manifest["compute"]["eval_workers"] == len(tasks)
    assert manifest["observation"] == "official_state"
    assert manifest["train_sha256"] != manifest["validation_sha256"]
    episodes = json.loads((out / "episodes.json").read_text())
    for arm in ("direct", "latent"):
        assert [(row["task"], row["seed"]) for row in episodes[arm]] == [
            (task, reset_seed) for task in tasks
        ]
    direct, latent = episodes["direct"][0], episodes["latent"][0]
    assert direct["seed"] == latent["seed"] == reset_seed
    assert direct["task"] == latent["task"] == 4
    with np.load(out / "trajectories.npz", allow_pickle=False) as paths:
        np.testing.assert_array_equal(
            paths[f"direct_4_{reset_seed}_observations"][0],
            paths[f"latent_4_{reset_seed}_observations"][0],
        )
        assert paths[f"latent_4_{reset_seed}_actions"].shape == (latent["steps"], 5)
    with gzip.open(out / "plans.jsonl.gz", "rt") as stream:
        plan = json.loads(next(stream))
    assert np.asarray(plan["latent_routes"]).shape == (4, 8, 16)
    assert np.asarray(plan["decoded_routes"]).shape == (4, 8, observation_size)
    assert json.loads((out / "summary.json").read_text())[
        "evaluation_environment_steps"
    ] == sum(row["steps"] for rows in episodes.values() for row in rows)
    if command == "scene-latent":
        frozen = tmp_path / "frozen"
        subprocess.run(
            [
                "wr",
                "scene-evaluate",
                "--checkpoint",
                out,
                "--out",
                frozen,
                "--task",
                "4",
                "--task",
                "5",
                "--eval-repeats",
                "1",
                "--eval-steps",
                "12",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        assert json.loads((frozen / "episodes.json").read_text()) == episodes
        with (
            np.load(out / "trajectories.npz", allow_pickle=False) as parallel,
            np.load(frozen / "trajectories.npz", allow_pickle=False) as serial,
        ):
            assert parallel.files == serial.files
            for key in parallel:
                np.testing.assert_array_equal(parallel[key], serial[key])


@pytest.mark.parametrize("singular_cube", [False, True])
def test_scene_reachability_cli_pairs_recorded_actions_with_frozen_actor(
    tmp_path, singular_cube
):
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    pytest.importorskip("ogbench")
    from ogbench.manipspace.envs.scene_env import SceneEnv

    from worth_reaching.models.gcbc import GCBC
    from worth_reaching.models.training import save_weights, weights

    env = SceneEnv(env_type="scene", mode="data_collection", terminate_at_goal=False)
    try:
        observation, info = env.reset(seed=17)
        if singular_cube:
            pose = env.data.qpos.copy()
            cube_quat = int(env.model.joint("object_joint_0").qposadr[0]) + 3
            pose[cube_quat : cube_quat + 4] = np.sqrt(0.5) * np.array([1, 0, 1, 0])
            env.set_state(pose, env.data.qvel.copy(), info["button_states"])
            observation = env.compute_observation()
        action = np.array([0.4, 0, 0, 0, 0], dtype=np.float32)
        observations, qpos, qvel, buttons = [], [], [], []
        for step in range(17):
            observations.append(observation)
            qpos.append(env.data.qpos.copy())
            qvel.append(env.data.qvel.copy())
            buttons.append(info["button_states"])
            if step < 16:
                observation, _, _, _, info = env.step(action)
    finally:
        env.close()
    if singular_cube:
        # The cube quaternion matches the reset while its undefined yaw is archived differently.
        observations[0][26:28] *= -1
    np.savez(
        tmp_path / "scene-play-v0-val.npz",
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.tile(action, (17, 1)),
        terminals=np.arange(17) == 16,
        qpos=qpos,
        qvel=qvel,
        button_states=buttons,
    )
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "manifest.json").write_text(
        json.dumps(
            {
                "schema": "ogbench_scene_latent_v2",
                "environment": "scene-v0",
                "train_sha256": "fixture",
                "executor": {"width": 8},
            }
        )
    )
    model = GCBC(8, 40, 5, relative_coordinates=0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    save_weights(
        checkpoint / "weights.pt",
        {
            "executor": weights(model),
            "mean": np.zeros(40, dtype=np.float32),
            "scale": np.ones(40, dtype=np.float32),
        },
    )
    out = tmp_path / "probe"
    subprocess.run(
        [
            "wr",
            "scene-reachability",
            "--checkpoint",
            checkpoint,
            "--data-dir",
            tmp_path,
            "--out",
            out,
            "--seed",
            "17",
            "--offsets",
            "16",
            "--budgets",
            "8",
            "16",
            "20",
            "--cases-per-group",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    [case] = json.loads((out / "attempts.json").read_text())
    assert case["target_row"] - case["start_row"] == 16
    assert case["replay_matches_at_offset"]
    assert case["actor_arrival"] is None
    assert case["initial_cube_yaw_singular"] is singular_cube
    if singular_cube:
        assert case["initial_cube_yaw_error"] > 1
    with np.load(out / "trajectories.npz", allow_pickle=False) as paths:
        np.testing.assert_array_equal(
            paths["0_actor_observations"][0], paths["0_replay_observations"][0]
        )
        np.testing.assert_array_equal(
            paths["0_replay_actions"], np.tile(action, (16, 1))
        )
        np.testing.assert_array_equal(paths["0_actor_actions"], np.zeros((20, 5)))
    summary = json.loads((out / "summary.json").read_text())
    assert summary["groups"]["pose_only_16"]["actor_arrivals_by_budget"] == {
        "8": 0,
        "16": 0,
        "20": 0,
    }
    assert summary["evaluation_environment_steps"] == 36
