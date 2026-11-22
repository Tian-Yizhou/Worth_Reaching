import json
import subprocess


def _write_bank(path, split, world, flip_success=False):
    path.mkdir()
    manifest = {
        "schema": "shared_attempts_v1",
        "protocol": "fixture",
        "features": ["direct", "fragile", "recoverable"],
        "config": {"split": split, "repeats": 8},
    }
    (path / "manifest.json").write_text(json.dumps(manifest))
    (path / "summary.json").write_text("{}")
    arm_outcomes = {
        "direct": [(None, True)] * 2 + [(None, False)] * 6,
        "fragile": [(True, True)] * 4 + [(True, False)] * 2 + [(False, False)] * 2,
        "recoverable": [(True, True)] * 2 + [(True, False)] * 2 + [(False, True)] * 4,
    }
    decision = {
        "world_id": world,
        "native_choice": "fragile",
        "candidates": [
            {
                "id": name,
                "features": [int(i == j) for j in range(3)],
                "trials": [
                    {"arrived": arrived, "success": success != flip_success}
                    for arrived, success in outcomes
                ],
            }
            for i, (name, outcomes) in enumerate(arm_outcomes.items())
        ],
    }
    (path / "decisions.jsonl").write_text(json.dumps(decision) + "\n")
    return path


def test_shared_bank_cli_selects_before_opening_test_outcomes(tmp_path):
    train = _write_bank(tmp_path / "train", "train", "world-train")
    dev = _write_bank(tmp_path / "dev", "dev", "world-dev")
    original = _write_bank(tmp_path / "original", "test", "world-test")
    changed = _write_bank(tmp_path / "changed", "test", "world-test", True)
    before, after = tmp_path / "before", tmp_path / "after"
    for test, out in ((original, before), (changed, after)):
        subprocess.run(
            [
                "wr",
                "attempt-compare",
                "--train",
                str(train),
                "--dev",
                str(dev),
                "--test",
                str(test),
                "--out",
                str(out),
            ],
            check=True,
            capture_output=True,
        )

    choices = (before / "choices.jsonl").read_bytes()
    assert choices == (after / "choices.jsonl").read_bytes()
    selected = json.loads(choices)["choices"]
    assert selected["native"] == selected["joint_arrival_success"] == "fragile"
    assert selected["direct_attempt"] == selected["both_branches"] == "recoverable"
    assert selected["terminal_waypoint"] == "recoverable"
    assert selected["terminal_arrived_waypoint"] == "fragile"
    assert selected["arrival_only"] == "fragile"
    original_summary = json.loads((before / "summary.json").read_text())
    changed_summary = json.loads((after / "summary.json").read_text())
    assert original_summary["test"]["both_branches"]["successes"] == 6
    assert changed_summary["test"]["both_branches"]["successes"] == 2
    assert original_summary["miss_then_success_by_selector"]["terminal_waypoint"] == 4
    assert original_summary["terminal_vs_arrival_disagreements"]["worlds"] == 1
