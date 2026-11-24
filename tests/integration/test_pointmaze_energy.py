import pytest


def test_latent_energy_scores_multiple_modes():
    np = pytest.importorskip("numpy")
    from worth_reaching.models.latent_energy import prefix_energy

    reference = np.array([[[-4.0]], [[-4.1]], [[4.0]], [[4.1]]])
    proposals = np.array([[[-4.0]], [[0.0]], [[4.0]]])
    scale = np.ones(1)
    low_left, middle, low_right = prefix_energy(proposals, reference, scale)
    assert low_left == pytest.approx(low_right)
    assert max(low_left, low_right) < middle
    np.testing.assert_allclose(
        prefix_energy(proposals * 7, reference * 7, scale * 7),
        [low_left, middle, low_right],
    )


@pytest.mark.parametrize("miss_first", [False, True])
def test_latent_prefix_handoff_and_terminal_labels(miss_first):
    np = pytest.importorskip("numpy")
    pytest.importorskip("gymnasium")
    pytest.importorskip("torch")
    pytest.importorskip("flax")
    pytest.importorskip("minari")
    from worth_reaching.tasks.pointmaze.energy import attempt_prefix
    from worth_reaching.tasks.pointmaze.environment import Step

    def state(x):
        return np.array([x, 0, 0, 0], dtype=np.float32)

    class Replay:
        def reset(self, seed):
            self.actions = []
            return state(0), state(-1)

        def step(self, action):
            self.actions.append(action)
            step = len(self.actions)
            return Step(state(step), 0.0, step == 4, step == 4)

        def arrived(self, observation, target):
            return observation[0] == target[0]

    env = Replay()
    targets = np.stack((state(5 if miss_first else 1), state(2)))
    row, path = attempt_prefix(env, 0, 4, targets, [1, 2])
    assert len(path) == 5
    if miss_first:
        assert [event["outcome"] for event in row["stage_events"]] == ["timeout"]
        assert row["attempt_steps"] == 1
        assert env.actions[1][0] < 0
    else:
        assert [event["end_step"] for event in row["stage_events"]] == [1, 2]
        assert row["attempt_steps"] == 2
        assert env.actions[1][0] > 0
    assert row["remaining_at_handoff"] == 4 - row["attempt_steps"]
    assert row["arrived_all"] is not miss_first
    assert row["success_without_full_arrival"] is miss_first
    assert row["success_after_handoff"] and row["first_success_step"] == 4
