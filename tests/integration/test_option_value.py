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
        value_range,
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
    ranges = value_range(value, options, archive.observations, 0.001, size=64)
    assert len(ranges["all"]) == len(ranges["quantiles"])


def test_backup_clips_the_following_value_to_reachable_returns():
    pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    from worth_reaching.models.option_value import (
        NEVER,
        OptionValue,
        OptionValueConfig,
        option_value_loss,
    )

    model = OptionValue(2, 20, OptionValueConfig(width=8, depth=1, heads=2))
    latent = torch.zeros(4, 2)
    duration = torch.tensor([5, 5, 5, 5])
    budget = torch.tensor([20, 20, 20, 3])
    first = torch.tensor([NEVER, NEVER, 4, NEVER])
    cost = torch.full((4,), 0.01)
    mask = torch.ones(4, 2, dtype=torch.bool)
    for following, expected in (
        (50.0, [-0.05 + 0.99, -0.05 + 0.99, 0.96, -0.03]),
        (-50.0, [-0.2, -0.2, 0.96, -0.03]),
    ):
        model.state = lambda *_, value=following: torch.full((2, 4), value)
        (error, _), _ = option_value_loss(
            model, (latent, latent, latent, latent, duration, budget, first, cost, mask)
        )
        q = model.option(latent, latent, latent, duration, budget, cost).T
        assert torch.allclose(q - error, torch.tensor(expected)[:, None], atol=1e-5)


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
    (switch,) = planner.log.switch_log
    assert switch["value"] == pytest.approx(2.0)
    assert switch["previous_value"] == pytest.approx(0.5)
    np.testing.assert_array_equal(acted[-1], better.latent)
    for _ in range(10):
        planner(np.zeros(2, np.float32), goal)
    # The switched option's duration ran out, so the planner decided again.
    assert planner.log.triggers.get("duration") == 1
    assert [d["trigger"] for d in planner.log.decisions] == ["start", "duration"]

    # Exploration draws among sampled candidates and never the terminal goal, even
    # when the goal has the highest value.
    explorer = OptionPlanner(
        Completion(),
        Value(),
        lambda observation, latent: np.zeros(2),
        key=0,
        limit=50,
        config=PlannerConfig(explore=1.0, interrupt=False),
        action_size=2,
    )
    sampled = [
        Option(np.array([0.1 * i, 0.0], np.float32), 5, 0, source=i % 2)
        for i in range(4)
    ]
    explorer._propose = lambda goal, remaining: list(sampled)
    for _ in range(20):
        explorer(np.zeros(2, np.float32), np.array([3.0, 0.0], np.float32))
    assert explorer.log.decisions and not any(d["goal"] for d in explorer.log.decisions)
    assert all(d["explored"] for d in explorer.log.decisions)
    assert all(d["goal_value"] == pytest.approx(3.0) for d in explorer.log.decisions)
    first = explorer.log.decisions[0]
    assert first["candidates"] == 5 and first["spread"] == pytest.approx(0.01)
    assert first["sampled_median"] == pytest.approx(0.15)
    # Each head values the goal, the other head's first choice, at 3 -/+ 0.01.
    assert first["holdout_value"] == pytest.approx(3.0)

    # Random-first draws a sampled candidate at the start only, then plans greedily.
    first_only = OptionPlanner(
        Completion(),
        Value(),
        lambda observation, latent: np.zeros(2),
        key=0,
        limit=50,
        config=PlannerConfig(explore_first=True, interrupt=False),
        action_size=2,
    )
    first_only._propose = lambda goal, remaining: list(sampled)
    for _ in range(20):
        first_only(np.zeros(2, np.float32), np.array([3.0, 0.0], np.float32))
    start, *later = first_only.log.decisions
    assert start["explored"] and not start["goal"]
    assert later and all(d["goal"] and not d["explored"] for d in later)


def test_attempts_split_an_episode_at_each_pursued_option_and_stop_at_success():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from types import SimpleNamespace

    from worth_reaching.evaluation.options import attempts
    from worth_reaching.models.option_value import NEVER

    def encode(values):
        return np.asarray(values, np.float32)[:, :1] * 10

    trajectory = np.arange(11, dtype=np.float32)[:, None].repeat(2, axis=1)
    latents = [np.full(1, value, np.float32) for value in (1, 2, 3)]
    policy = SimpleNamespace(
        completion=SimpleNamespace(encode_observations=encode),
        log=SimpleNamespace(
            pursued=list(zip((0, 4, 9), latents, (4, 10, 11), strict=True))
        ),
    )
    goal = np.full(2, 7, np.float32)
    found = attempts(policy, trajectory, goal, 6, limit=20)
    # The option started at step 9 began after the success at step 6. The option that
    # succeeded keeps its commanded duration (10), not the two actions it took.
    np.testing.assert_array_equal(found.durations, [4, 10])
    np.testing.assert_array_equal(found.budgets, [20, 16])
    np.testing.assert_array_equal(found.first, [NEVER, 2])
    np.testing.assert_array_equal(found.states[:, 0], [0, 40])
    np.testing.assert_array_equal(found.landings[:, 0], [40, 60])
    np.testing.assert_array_equal(found.targets[:, 0], [1, 2])
    np.testing.assert_array_equal(found.goals[:, 0], [70, 70])
    unfinished = attempts(policy, trajectory, goal, None, limit=20)
    np.testing.assert_array_equal(unfinished.durations, [4, 5, 1])
    assert np.all(unfinished.first == NEVER)
