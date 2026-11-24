import json
import subprocess

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
