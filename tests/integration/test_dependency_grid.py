import json
import subprocess
from hashlib import sha256

from worth_reaching.tasks.dependency_grid import environment as grid
from worth_reaching.tasks.dependency_grid.experiment import BUDGET, DEADLINES, attempt


def test_dependency_grid_cli_pairs_goal_switches_and_order(tmp_path):
    out = tmp_path / "dependency-grid"
    subprocess.run(
        [
            "wr",
            "dependency-grid",
            "--out",
            str(out),
            "--repeats",
            "2",
            "--slip-probability",
            "0",
        ],
        check=True,
        capture_output=True,
    )
    summary = json.loads((out / "summary.json").read_text())
    assert all(
        count > 0
        for count in summary["layouts_with_goal_dependent_rank_reversal"].values()
    )
    decisions = [
        json.loads(line) for line in (out / "decisions.jsonl").read_text().splitlines()
    ]
    assert len(decisions) == 4 * summary["layouts"]
    for block in range(0, len(decisions), 4):
        first = decisions[block]
        for row in decisions[block + 1 : block + 4]:
            for field in ("layout_id", "start", "slips", "candidates"):
                assert row[field] == first[field]
    depth_two = summary["by_depth_and_goal"]["2"]
    for goal in grid.GOALS:
        assert (
            depth_two[goal][f"{goal}_entry_then_inner"]["successes"]
            == 2 * summary["layouts"]
        )
    assert summary["evaluation_environment_steps"] == sum(
        trial["environment_steps"]
        for row in decisions
        for trials in row["trials"].values()
        for trial in trials
    )


def test_dependency_grid_respects_target_order_and_last_action():
    left = grid.Lane(1, 0, False)
    layout = grid.Layout((left, grid.Lane(5, 0, False)))
    slips = (False,) * BUDGET
    ordered = (left.entry, left.inner)
    trial = attempt(layout, 2, left.goal, ordered, slips)
    assert trial.arrived and trial.success
    assert not attempt(layout, 2, left.goal, ordered[::-1], slips).success
    assert not attempt(layout, 2, left.goal, ordered[:1], slips).success
    assert attempt(
        layout, 2, left.goal, ordered, slips[: trial.environment_steps]
    ).success
    assert not attempt(
        layout, 2, left.goal, ordered, slips[: trial.environment_steps - 1]
    ).success


def test_dependency_grid_continues_after_timeout_and_splits_by_layout():
    left = grid.Lane(1, 1, False)
    layout = grid.Layout((left, grid.Lane(5, 0, False)))
    deadline = DEADLINES[0]
    slips = (True,) * deadline + (False,) * (BUDGET - deadline)
    trial = attempt(layout, 1, left.goal, (left.inner,), slips)
    assert trial.success and trial.arrived is False
    assert trial.attempt_steps == deadline
    splits = [{layout.id for layout in grid.layouts(split)} for split in grid.SPLITS]
    assert sum(map(len, splits)) == len(set.union(*splits)) == 64


def test_dependency_grid_latent_cli_uses_public_windows_and_frozen_choices(tmp_path):
    out = tmp_path / "latent"
    subprocess.run(
        [
            "wr",
            "dependency-grid-latent",
            "--out",
            str(out),
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
            "--repeats",
            "2",
            "--candidates",
            "2",
            "--density-samples",
            "4",
            "--selector-repeats",
            "2",
        ],
        check=True,
        capture_output=True,
    )
    manifest = json.loads((out / "manifest.json").read_text())
    train, dev = (set(manifest["layouts"][split]) for split in ("train", "dev"))
    assert train.isdisjoint(dev)
    assert set(manifest["layouts"]["evaluation"]) <= dev
    assert manifest["query_steps"] == [4, 8]
    assert manifest["collection_seed"] == 7
    assert manifest["outcome_selector"]["repeats"] == 2
    weights = (out / "selector.npz").read_bytes()
    assert manifest["outcome_selector"]["weights_sha256"] == sha256(weights).hexdigest()
    selector_data = [
        json.loads(line)
        for line in (out / "selector-training-attempts.jsonl").read_text().splitlines()
    ]
    assert {row["layout_id"] for row in selector_data} == train
    for row in selector_data:
        assert all(
            episode.split(":")[0] != row["layout_id"]
            for episode in row["replay_episode_ids"]
        )
        candidates = row["candidates"]
        assert candidates[0]["id"] == "direct"
        assert len({tuple(map(tuple, c["targets"])) for c in candidates}) == len(
            candidates
        )
        assert all(len(c["trials"]) == 2 for c in candidates)
    bank_path = out / "candidate-bank.json"
    assert (
        manifest["candidate_bank_sha256"] == sha256(bank_path.read_bytes()).hexdigest()
    )
    bank = json.loads(bank_path.read_text())
    traces = [
        json.loads(line)
        for line in (out / "trajectories.jsonl").read_text().splitlines()
    ]
    train_episodes = {row["episode_id"] for row in traces if row["split"] == "train"}
    for row in traces:
        assert len(row["observations"]) == len(row["actions"]) + 1
        assert "slips" not in row
    evaluated = [
        json.loads(line) for line in (out / "attempts.jsonl").read_text().splitlines()
    ]
    for declaration, result in zip(bank, evaluated, strict=True):
        assert declaration["layout_id"] == result["layout_id"]
        assert "outcomes" not in declaration and "slips" not in declaration
        assert len(declaration["flow"]) == len(declaration["replay"]) == 2
        assert {row["episode_id"] for row in declaration["replay"]} <= train_episodes
        assert declaration["choices"]["lowest_energy"] == min(
            range(2), key=lambda i: declaration["flow"][i]["energy"]
        )
        best = result["reference_choices"]["flow"]
        assert best["index"] == max(
            range(2), key=best["reference_successes"].__getitem__
        )
        picked = result["outcomes"]["flow"][declaration["choices"]["lowest_energy"]]
        assert (
            result["arms"]["lowest_energy"]["success"]
            == sum(trial["success"] for trial in picked) / 2
        )
        scores = declaration["outcome_scores"]
        for source in ("flow", "replay"):
            source_scores = scores[source]
            probabilities = source_scores["probabilities"]
            chosen = max(range(2), key=probabilities.__getitem__)
            if probabilities[chosen] <= scores["direct_probability"]:
                assert source_scores["chosen_index"] is None
                trials = result["outcomes"]["direct"]
            else:
                assert source_scores["chosen_index"] == chosen
                trials = result["outcomes"][source][chosen]
            assert result["arms"][f"{source}_outcome"]["success"] == sum(
                trial["success"] for trial in trials
            ) / len(trials)
