import pytest


def _line_archive(np):
    """Two straight trajectories along x, one step per row."""
    from worth_reaching.data.trajectory_windows import ContinuousTrajectories

    xs = np.concatenate((np.arange(40), np.arange(40)[::-1])).astype(np.float32)
    observations = np.stack((xs, np.zeros_like(xs)), axis=1)
    actions = np.zeros_like(observations)
    return ContinuousTrajectories(observations, actions, np.array([39, 79]))


def test_hindsight_options_label_first_arrival_within_the_segment():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.models.option_value import (
        NEVER,
        OptionValueConfig,
        label_options,
    )

    archive = _line_archive(np)

    def reached(states, goals):
        return np.abs(states[:, 0] - goals[:, 0]) < 0.5

    options = label_options(
        archive,
        reached,
        2000,
        max_budget=20,
        max_horizon=20,
        queries=4,
        config=OptionValueConfig(random_goal_share=0.0),
        rng=np.random.default_rng(3),
    )
    ends = archive.ends[np.searchsorted(archive.ends, options.starts)]
    assert np.all(options.starts + options.durations <= ends)
    assert np.all(options.durations <= options.budgets)
    distance = options.goals - options.starts
    # On a line the goal is first reached exactly at its hindsight offset.
    inside = distance <= options.durations
    assert np.array_equal(options.first[inside], distance[inside])
    assert np.all(options.first[~inside] == NEVER)
    goal_options = options.targets == options.goals
    assert goal_options.any() and (~goal_options).any()
    assert np.array_equal(
        options.targets[~goal_options],
        (options.starts + options.durations)[~goal_options],
    )


def test_option_value_trains_and_scores_candidates_per_head(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.models.option_value import (
        OptionValueConfig,
        fit_option_value,
        label_options,
    )

    archive = _line_archive(np)
    config = OptionValueConfig(
        updates=3, batch_size=32, width=16, depth=1, heads=3, options=256
    )
    options = label_options(
        archive,
        lambda states, goals: np.abs(states[:, 0] - goals[:, 0]) < 0.5,
        config.options,
        max_budget=20,
        max_horizon=20,
        queries=4,
        config=config,
        rng=np.random.default_rng(4),
    )
    value = fit_option_value(
        options, archive.observations, 20, config, seed=1, out=tmp_path
    )
    scores = value.score(
        archive.observations[0],
        archive.observations[10],
        archive.observations[[3, 6, 10]],
        np.array([3, 6, 10], dtype=np.float32),
        20,
        0.001,
    )
    assert scores.shape == (3, 3) and np.isfinite(scores).all()


def test_planner_switches_to_a_cached_candidate_and_falls_back_to_the_goal():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.planning.options import Option, OptionPlanner, PlannerConfig

    class Codec:
        def encode(self, values):
            return values

    class Completion:
        config = type("C", (), {"history": 2})()
        codec = Codec()
        mean, scale = np.zeros(2, np.float32), np.ones(2, np.float32)

    class Value:
        """Values each candidate by its first latent coordinate, with fixed spread."""

        def score(self, state, goal, latents, durations, budget, cost):
            base = latents[:, 0].astype(np.float64)
            return np.stack((base - 0.01, base + 0.01))

    acted = []
    planner = OptionPlanner(
        Completion(),
        Value(),
        lambda observation, latent: acted.append(latent.copy()) or np.zeros(2),
        key=0,
        limit=50,
        config=PlannerConfig(subgoals=False),
        action_size=2,
    )
    goal = np.array([0.5, 0.0], np.float32)
    planner(np.zeros(2, np.float32), goal)
    assert planner.current.option.goal and planner.log.triggers == {"start": 1}
    better = Option(np.array([2.0, 0.0], np.float32), 10, planner.step)
    planner.cache.append(better)
    planner(np.zeros(2, np.float32), goal)
    assert planner.current.option is better and planner.log.switches == 1
    np.testing.assert_array_equal(acted[-1], better.latent)
    for _ in range(10):
        planner(np.zeros(2, np.float32), goal)
    # The switched option's duration ran out, so the planner decided again.
    assert planner.log.triggers.get("duration") == 1
