import json
import subprocess
from hashlib import sha256

from worth_reaching.tasks.budget_routes import data, planner
from worth_reaching.tasks.budget_routes import environment as env
from worth_reaching.tasks.budget_routes.experiment import _support, rollout


def test_budget_changes_preferred_route_and_replanning_recovers():
    counts, _ = data.exploration(192, 7)
    value = planner.terminal_value(counts)
    world = env.worlds("dev")[0]
    pair = (env.SUPPLY, env.GATE)
    reversed_pair = pair[::-1]
    routes = planner.compose([(env.SUPPLY, env.BYPASS), (env.GATE,)], 24)
    assert pair in routes and reversed_pair in routes
    assert planner.rank(value, world.profile, world.start, 2, routes)[0] == ()
    assert planner.rank(value, world.profile, world.start, 3, routes)[0] == (
        env.SUPPLY,
    )
    choice, scores = planner.rank(value, world.profile, world.start, 4, routes)
    assert choice == pair
    assert scores[pair] > scores[reversed_pair]

    draws = (0.999, 0.5, 0.5, 0.5)
    assert not rollout(world, 4, pair, draws)["success"]
    closed_loop = rollout(world, 4, pair, draws, value=value, suffixes=[(env.BYPASS,)])
    assert closed_loop["success"] and closed_loop["switches"] == 1
    assert closed_loop["actions"] == [env.SUPPLY, env.BYPASS, env.GOAL]


def test_route_rarity_counts_once_per_episode_and_preserves_order():
    rare = _support((env.SUPPLY, env.GATE), [(env.SUPPLY,) * 2, (env.GATE, env.SUPPLY)])
    assert rare == {
        "eligible_episodes": 2,
        "component_support": [2, 1],
        "ordered_support": 0,
        "rare_pair": True,
    }
    assert not _support((env.SUPPLY, env.GATE), [])["rare_pair"]


def test_budget_routes_cli_declares_compositions_and_keeps_sources_separate(tmp_path):
    output = tmp_path / "routes"
    subprocess.run(
        [
            "wr",
            "budget-routes",
            "--out",
            str(output),
            "--layouts",
            "1",
            "--demonstrations",
            "4",
            "--exploration",
            "24",
            "--encoder-updates",
            "2",
            "--updates",
            "2",
            "--flow-updates",
            "2",
            "--repeats",
            "4",
            "--candidates",
            "4",
        ],
        capture_output=True,
        check=True,
    )
    bank_path = output / "candidate-bank.json"
    bank = json.loads(bank_path.read_text())
    manifest = json.loads((output / "manifest.json").read_text())
    assert (
        manifest["candidate_bank_sha256"] == sha256(bank_path.read_bytes()).hexdigest()
    )
    assert set(manifest["train_layouts"]).isdisjoint(manifest["evaluation_layouts"])
    assert {row["budget"] for row in bank} == set(env.BUDGETS)
    assert all(row["fixed_pool"] == bank[-1]["fixed_pool"] for row in bank)
    for row in bank:
        assert "attempts" not in row
        prototypes = set(map(int, row["prototype_slots"]))
        assert all(
            set(candidate["route"]) <= prototypes
            for candidate in row["banks"]["composed"]
        )
        assert all(
            len(candidates) <= manifest["candidate_cap"]
            for candidates in row["banks"].values()
        )
        assert all(
            len(trial["actions"]) <= row["budget"]
            for result in json.loads((output / "attempts.json").read_text())
            if result["budget"] == row["budget"]
            for trials in result["attempts"].values()
            for trial in trials
        )
