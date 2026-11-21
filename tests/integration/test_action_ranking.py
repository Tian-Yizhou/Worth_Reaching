from copy import deepcopy

import numpy as np

from worth_reaching.data.ranking import RankingData, _public_features
from worth_reaching.models.action_ranking import evaluate_scores, fit_head


def test_rank_features_ignore_future_actions_observations_and_outcome():
    trace = {
        "initial_observation": [0.0, 1.0],
        "steps": [
            {"action": 0, "observation": [0.1, 0.9]},
            {"action": 1, "observation": [0.2, 0.8]},
            {"action": 2, "observation": [0.3, 0.7]},
        ],
        "outcome": "success",
    }
    altered = deepcopy(trace)
    altered["outcome"] = "death"
    altered["steps"][2] = {"action": 0, "observation": [1.0, 1.0]}
    np.testing.assert_array_equal(
        _public_features(trace, 2, 10, 3), _public_features(altered, 2, 10, 3)
    )


def test_public_heads_learn_opposite_rankings_for_the_same_actions():
    features = np.array([[-1.0, 1.0], [1.0, 1.0]])
    actions = np.array([[0, 1], [0, 1]])
    labels = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]], dtype=np.int8)
    for loss in ("return", "pairwise"):
        weights = fit_head(features, actions, labels, 2, 60, loss=loss)
        predictions = (weights[actions] * features[:, None, :]).sum(axis=-1)
        np.testing.assert_array_equal(predictions.argmax(axis=1), [0, 1])


def test_reference_action_is_scored_on_independent_repeats():
    repeats = np.array([[[1, 0], [0, 1]] * 6], dtype=np.int8)
    data = RankingData(
        ("decision",), ("world",), np.ones((1, 1)), np.array([[0, 1]]), repeats, "dev"
    )
    report = evaluate_scores(data, {"model": np.array([[0.0, 1.0]])}, 0)
    assert report["reference_selection_repeats"] == 6
    assert report["reference_estimation_repeats"] == 6
    assert report["resolved_reference_worlds"] == 1
    assert report["estimated_reference_success"] == 0.0
    assert report["models"]["model"]["estimated_selected_success"] == 1.0
    assert report["models"]["model"]["mean_reference_minus_selected_success"] == -1.0
    assert report["models"]["model"]["agreement_on_resolved_reference"] == 0.0
