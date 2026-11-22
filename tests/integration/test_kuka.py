from pathlib import Path

import pytest


def test_kuka_windows_preserve_clips_and_only_condition_on_block_positions(tmp_path):
    np = pytest.importorskip("numpy")
    from worth_reaching.data.kuka import (
        block_states,
        goal_features,
        load_clips,
        sample_windows,
        stack_goal,
    )

    for index in range(5):
        observation = np.full((40, 39), index, dtype=np.float32)
        block_states(observation)[..., :2] += np.arange(4)[None, :, None]
        np.save(tmp_path / f"data_{index}.npy", observation)
    train, validation, splits = load_clips(tmp_path)
    assert len(train.ends) == 4 and len(validation.ends) == 1
    assert set(splits["train"]["clips"]).isdisjoint(splits["validation"]["clips"])
    windows = sample_windows(
        train, np.random.default_rng(7), 32, history=4, queries=8, max_horizon=30
    )
    assert windows.context.incoming_actions.shape == (32, 4, 0)
    assert np.all(
        windows.targets[..., 0] == windows.context.observations[:, -1, 0, None]
    )
    goal = windows.context.goal
    assert not goal[:, :7].any()
    assert not block_states(goal)[..., 3:].any()
    np.testing.assert_allclose(block_states(goal)[..., :2].mean(axis=-2), 0)
    initial = train.observations[0]
    desired = stack_goal(initial, np.array([2, 1, 3, 0]), 0.062)
    np.testing.assert_allclose(
        block_states(desired)[[2, 1, 3, 0], 2],
        block_states(initial)[2, 2] + 0.062 * np.arange(4),
    )
    np.testing.assert_array_equal(goal_features(desired), desired)


def test_kuka_native_labels_require_ordered_physical_contacts():
    np = pytest.importorskip("numpy")
    physics = pytest.importorskip("pybullet")
    source = Path("results/external/diffuser-kuka")
    if not (source / "gym_stacking/env.py").is_file():
        pytest.skip("Prepare the optional author-released Kuka simulator")
    from worth_reaching.data.kuka import block_states
    from worth_reaching.envs.kuka import BLOCK_HEIGHT, KukaStacking

    env = KukaStacking(source)
    try:
        initial, order = env.reset(82001)
        replay, replay_order = env.reset(82001)
        np.testing.assert_array_equal(initial, replay)
        np.testing.assert_array_equal(order, replay_order)
        after_step, reward = env.track(initial)
        assert reward == 0 and not any(env.stacked_relations(after_step, order))

        base = block_states(after_step)[order[0], :3]
        for level, block in enumerate(order):
            physics.resetBasePositionAndOrientation(
                env.native.cubes[block],
                [*base[:2], float(base[2] + level * BLOCK_HEIGHT)],
                [0, 0, 0, 1],
                physicsClientId=env.native.client,
            )
        for _ in range(120):
            physics.stepSimulation(physicsClientId=env.native.client)
        state = env.native.get_state()[:39]
        assert env.stacked_relations(state, order) == [True, True, True]
        assert not any(env.stacked_relations(state, order[::-1]))
    finally:
        env.close()
