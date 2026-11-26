import json
import subprocess

import pytest


def test_interpolated_pairs_span_prediction_to_encoding():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.models.region import interpolated_pairs

    predicted = np.zeros((3, 2), dtype=np.float32)
    encoded = np.ones((3, 2), dtype=np.float32)
    states = np.arange(6, dtype=np.float32).reshape(3, 2)
    latents, realized = interpolated_pairs(predicted, encoded, states, 4, seed=1)
    assert latents.shape == (12, 2) and realized.shape == (12, 2)
    np.testing.assert_array_equal(realized[3:6], states)
    # Each latent lies on the segment from the prediction to the encoding.
    assert np.all((latents >= 0) & (latents <= 1))
    np.testing.assert_allclose(latents[:, 0], latents[:, 1])


def test_arrival_rules_follow_encoding_distance_and_region_support():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.planning.arrival import (
        EncoderConsistency,
        GaussianSupport,
        LatentGoal,
        RegionSupport,
    )

    near, far = np.array([0.0, 0.0]), np.array([1.0, 0.0])
    goal = LatentGoal(
        np.array([0.1, 0.0]),
        samples=np.array([[0.0, 0.0], [0.0, 0.05], [2.0, 2.0]]),
    )
    consistency = EncoderConsistency(lambda observation: observation, epsilon=0.2)
    assert consistency(near, goal) and not consistency(far, goal)
    support = RegionSupport(bandwidth=0.1, threshold=0.5)
    assert support(near, goal) and not support(far, goal)
    assert not RegionSupport(bandwidth=0.1, threshold=0.9)(near, goal)
    tight = LatentGoal(np.zeros(2), samples=np.array([[0.0, 0.0], [0.0, 0.1]]))
    gaussian = GaussianSupport(floor=0.1, threshold=0.5)
    assert gaussian(near, tight) and not gaussian(far, tight)
    with pytest.raises(ValueError, match="decoder samples"):
        support(near, LatentGoal(np.zeros(2)))


def test_hindsight_pairs_label_each_action_with_later_states():
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from worth_reaching.models.executor import hindsight_pairs

    states = np.arange(3, dtype=np.float32)[:, None]
    observations, goals, actions = hindsight_pairs([(states, np.array([4, 2]))], 2)
    np.testing.assert_array_equal(observations[:, 0], [0, 0, 1, 1])
    np.testing.assert_array_equal(goals[:, 0], [1, 2, 2, 2])
    np.testing.assert_array_equal(actions, [4, 4, 2, 2])


def test_regions_cli_declares_regions_before_matched_attempts(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    out = tmp_path / "regions"
    subprocess.run(
        [
            "wr",
            "dependency-grid-regions",
            "--out",
            out,
            "--layouts",
            "1",
            "--demonstrations",
            "1",
            "--encoder-updates",
            "2",
            "--updates",
            "2",
            "--flow-updates",
            "2",
            "--region-updates",
            "2",
            "--executor-updates",
            "2",
            "--repeats",
            "2",
            "--candidates",
            "2",
            "--region-samples",
            "3",
            "--recall-levels",
            "0.5",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    assert set(manifest["arms"]) == {
        "point_bfs",
        "point_learned",
        "latent_point@0.5",
        "region_gaussian@0.5",
        "region_flow@0.5",
        "region_decoded@0.5",
    }
    bank = json.loads((out / "candidate-bank.json").read_text())
    assert len(bank) == 4
    for row in bank:
        assert [candidate["id"] for candidate in row["candidates"]] == [
            "forecast",
            "flow_0",
            "flow_1",
        ]
        assert all(
            len(candidate["latents"]) == row["depth"] for candidate in row["candidates"]
        )
    with np.load(out / "candidate-regions.npz") as regions:
        key = f"{bank[0]['layout_id']}:{bank[0]['depth']}:{bank[0]['goal']}"
        assert regions[f"{key}:samples"].shape == (3, bank[0]["depth"], 3, 15)
        assert regions[f"{key}:realizations"].shape == (3, bank[0]["depth"], 2, 15)
    summary = json.loads((out / "summary.json").read_text())
    for depth in ("1", "2"):
        arms = summary["by_depth"][depth]["arms"]
        assert {
            "direct",
            "point_bfs/uniform",
            "region_flow@0.5/reference_best",
        } <= set(arms)
        assert "paired_difference_from_point_learned" in arms["region_flow@0.5/uniform"]
    rows = [json.loads(line) for line in (out / "attempts.jsonl").open()]
    assert all(row["arms"]["direct"]["arrival"] == 0 for row in rows)
