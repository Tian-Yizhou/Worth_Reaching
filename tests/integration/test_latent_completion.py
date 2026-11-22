import json
import subprocess

import pytest


def test_grid_evaluation_uses_matched_public_starts_and_masks():
    np = pytest.importorskip("numpy")
    pytest.importorskip("flax")
    from worth_reaching.envs.grid_routes import (
        BUDGET,
        GOALS,
        LAYOUT_COUNTS,
        STARTS,
        grid_demonstrations,
        layout_id,
    )
    from worth_reaching.evaluation.completion import score_paired_routes

    datasets = grid_demonstrations(include_test=True)
    world_ids = [{example.world_id for example in rows} for rows in datasets.values()]
    assert tuple(map(len, world_ids)) == tuple(
        count * len(STARTS) for count in LAYOUT_COUNTS
    )
    maps = [{layout_id(world_id) for world_id in worlds} for worlds in world_ids]
    assert tuple(map(len, maps)) == LAYOUT_COUNTS
    assert len(set.union(*maps)) == sum(LAYOUT_COUNTS)

    rows = datasets["test"]
    for first, second in zip(rows[::2], rows[1::2], strict=True):
        assert (
            first.world_id == second.world_id
            and first.start_step == second.start_step == 0
        )
        assert (first.inputs.goal, second.inputs.goal) == GOALS
        assert (
            first.inputs.remaining_actions == second.inputs.remaining_actions == BUDGET
        )
        np.testing.assert_array_equal(
            first.inputs.observations, second.inputs.observations
        )
        np.testing.assert_array_equal(first.inputs.actions, second.inputs.actions)
        np.testing.assert_array_equal(
            first.inputs.query_steps, second.inputs.query_steps
        )
        assert first.inputs.terminal_example is second.inputs.terminal_example is None

    targets = np.stack([example.target.observations for example in rows])
    valid = np.ones(targets.shape[:2], dtype=bool)
    layouts = [layout_id(example.world_id) for example in rows[::2]]
    own = score_paired_routes(targets, targets, valid, layouts)
    swapped = score_paired_routes(
        targets.reshape(-1, 2, *targets.shape[1:])[:, ::-1].reshape(targets.shape),
        targets,
        valid,
        layouts,
    )
    assert 0 < own["worlds"] < len(layouts)
    assert own["worlds"] == swapped["worlds"]
    assert own["layouts"] == LAYOUT_COUNTS[-1]
    assert own["matched_route_win_rate"] == own["exact_cell_rate"] == 1.0
    assert (
        own["matched_route_error_cells"] == swapped["swapped_route_error_cells"] == 0.0
    )
    assert swapped["matched_route_win_rate"] == 0.0

    # Once a route reaches its destination the demonstration stays there.
    positions = targets[..., :2]
    finished = (positions == positions[:, -1:, :]).all(axis=-1)
    either_finished = finished[::2] | finished[1::2]
    perturbed = targets.copy()
    perturbed[np.repeat(either_finished, 2, axis=0)] = -1
    assert score_paired_routes(perturbed, targets, valid, layouts) == own


def test_training_cli_saves_public_input_predictors(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("flax")
    from flax import serialization

    from worth_reaching.envs.grid_routes import grid_demonstrations
    from worth_reaching.models.completion import (
        LatentPredictor,
        ModelSpec,
        ObservationCodec,
        TrainingConfig,
        pack_inputs,
        predict,
    )

    out = tmp_path / "model"
    result = subprocess.run(
        [
            "wr",
            "train-completion",
            "--grid",
            "--out",
            out,
            "--encoder-updates",
            "2",
            "--updates",
            "2",
            "--history-length",
            "4",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    assert json.loads(result.stdout)["evaluation"] == summary["evaluation"]
    assert set(summary["evaluation"]) == {"dev"}
    assert summary["evaluation"]["dev"]["target_variance"] > 0
    assert summary["evaluation"]["dev"]["goal_blind"]["matched_route_win_rate"] == 0.5

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    spec_data = manifest["model"]
    spec_data["goals"] = tuple(spec_data["goals"])
    spec = ModelSpec(**spec_data)
    config = TrainingConfig(**manifest["config"])
    codec = ObservationCodec(config.width, config.latent_size, spec.observation_size)
    predictor = LatentPredictor(spec, config.width, config.latent_size)
    weights = serialization.msgpack_restore((out / "weights.msgpack").read_bytes())
    source = grid_demonstrations()
    recent = max(source["train"], key=lambda window: window.start_step).inputs
    packed = pack_inputs([recent], spec, config)
    np.testing.assert_array_equal(packed.observations[0], recent.observations[-4:])
    np.testing.assert_array_equal(packed.actions[0], recent.actions[-4:])

    public = pack_inputs(
        [example.inputs for example in source["dev"][:2]], spec, config
    )
    latent, decoded = predict(
        codec, predictor, weights["codec"], weights["goal_blind"], public.without_goal()
    )
    assert latent.shape == (2, spec.budget - 1, config.latent_size)
    np.testing.assert_array_equal(decoded[0], decoded[1])

    routes = tmp_path / "routes"
    subprocess.run(
        [
            "wr",
            "grid-routes",
            "--checkpoint",
            out,
            "--out",
            routes,
            "--updates",
            "2",
            "--layouts",
            "1",
            "--candidates",
            "2",
            "--repeats",
            "2",
            "--slip-probability",
            "0",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads((routes / "summary.json").read_text(encoding="utf-8"))
    assert report["cases"] == 18
    assert report["arms"]["direct"]["terminal_success_rate"] == 1.0
    assert report["arms"]["flow_feedback"]["attempts"] == 72
    assert report["feedback_minus_replay_layout_bootstrap_95"] == [0.0, 0.0]
    assert set(
        serialization.msgpack_restore((routes / "flow.msgpack").read_bytes())
    ) == {"params", "scale"}
    with (routes / "decisions.jsonl").open(encoding="utf-8") as stream:
        row = json.loads(next(stream))
    assert [item["id"] for item in row["candidates"]] == [
        "direct",
        "deterministic",
        "flow_0",
        "flow_1",
    ]
    assert row["candidates"][2]["trials"] == row["candidates"][2]["open_loop"]

    feedback = tmp_path / "feedback"
    subprocess.run(
        [
            "wr",
            "grid-feedback",
            "--checkpoint",
            out,
            "--flow-run",
            routes,
            "--out",
            feedback,
            "--updates",
            "2",
            "--layouts",
            "1",
            "--candidates",
            "2",
            "--repeats",
            "2",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = json.loads((feedback / "summary.json").read_text())
    settings = json.loads((feedback / "manifest.json").read_text())
    sets = [set(group) for group in settings["layouts"].values()]
    assert len(set.union(*sets)) == sum(map(len, sets))
    assert result["evaluation_cases"] == 18 and result["training_cases"] == 576
    assert set(result["arms"]) == {"frozen", "demonstrations", "feedback"}
    with np.load(feedback / "feedback_targets.npz") as labels:
        assert len(labels["mask"]) == result["credited_waypoints"]
        assert (labels["mask"].sum(axis=1) == config.latent_size).all()
        assert not (labels["targets"] * (1 - labels["mask"])).any()
    with np.load(feedback / "repeat_masks.npz") as masks:
        assert masks["training"].shape == (576, 2, spec.budget, 2)
        assert masks["evaluation"].shape == (18, 2, spec.budget, 2)


def test_rollout_retains_terminal_success_after_waypoint_miss():
    np = pytest.importorskip("numpy")
    pytest.importorskip("flax")
    from worth_reaching.data.completion import CompletionInput
    from worth_reaching.envs import grid_routes as grid
    from worth_reaching.evaluation.grid_execution import Waypoint, rollout
    from worth_reaching.evaluation.grid_feedback import _optimal_policy

    start, markers = (2, 2), ((5, 0), (5, 6))
    observation = grid.grid_observation(start, (1, 5), markers)
    public = CompletionInput(
        observation[None],
        np.empty(0, dtype=np.int32),
        grid.GOALS[0],
        grid.BUDGET,
        np.arange(1, grid.BUDGET, dtype=np.int32),
    )
    waypoint = Waypoint((2, 1), 1)
    slips = np.zeros(grid.BUDGET, dtype=bool)
    nominal, replay = rollout(public, waypoint, slips)
    assert nominal.arrived_step == 1 and nominal.success and nominal.passage == "near"
    slips[0] = True
    feedback, _ = rollout(public, waypoint, slips)
    open_loop, _ = rollout(public, waypoint, slips, replay=replay)
    assert feedback.waypoint_attempted and feedback.arrived_step is None
    assert feedback.success and feedback.passage == "near"
    assert open_loop.arrived_step is None and not open_loop.success
    with pytest.raises(ValueError, match="unachieved"):
        rollout(public, Waypoint(start, 1), slips)

    passages = (1, 6)
    threatened = CompletionInput(
        grid.grid_observation(start, passages, ((5, 1), (5, 6)))[None],
        public.actions,
        public.goal,
        public.remaining_actions,
        public.query_steps,
    )
    slips[:] = False
    gate_slips = np.ones(grid.BUDGET, dtype=bool)
    direct, _ = rollout(threatened, None, slips, passage_slips=gate_slips)
    risk_aware, _ = rollout(
        threatened,
        None,
        slips,
        passage_slips=gate_slips,
        terminal_policy=_optimal_policy((5, 1), passages, 1.0),
    )
    assert not direct.success
    assert risk_aware.success and risk_aware.passage == "far"
