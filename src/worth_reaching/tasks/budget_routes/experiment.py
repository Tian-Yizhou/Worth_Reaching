"""Compare budgeted latent route planners with paired execution and replay."""

from collections import defaultdict
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from random import Random

import numpy as np

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.models.completion import TrainingConfig
from worth_reaching.models.training import WEIGHTS_FILE
from worth_reaching.tasks.budget_routes import data, planner
from worth_reaching.tasks.budget_routes import environment as env

SCHEMA = "budget_routes_v1"
RARE_SUPPORT = 0.05
RELIABLE = 0.50
MINIMUM_GAIN = 0.10


def _demonstrated(
    train: list[data.Demonstration], profile: int, start: env.State, budget: int
) -> list[tuple[int, ...]]:
    return [
        tuple(state.node for state in row.states[1:-1])
        for row in train
        if row.profile == profile
        and row.states[0] == start
        and row.budget == budget
        and row.success
    ]


def _support(route: planner.Route, demonstrations: list[planner.Route]) -> dict:
    def contains(order):
        remaining = iter(order)
        return all(any(node == target for node in remaining) for target in route)

    total = len(demonstrations)
    count = sum(contains(order) for order in demonstrations)
    components = [sum(node in order for order in demonstrations) for node in route]
    return {
        "eligible_episodes": total,
        "ordered_support": count,
        "component_support": components,
        "rare_pair": len(route) == 2
        and total > 0
        and all(components)
        and count / total <= RARE_SUPPORT,
    }


def _trace(start: env.State, route: planner.Route, budget: int) -> tuple[int, ...]:
    nodes = []
    for _ in range(budget):
        action, route = env.next_action(start, route)
        if action == env.GOAL:
            break
        nodes.append(action)
        start = env.State(action)
    return tuple(nodes)


def _declare(fitted, value, world, budget, train, count, seed) -> dict:
    rng = Random(f"proposal:{seed}:{world.id}:{budget}")
    joint, energies, latents = planner.latent_candidates(
        fitted, world.profile, world.start, budget, rng.randrange(2**32), count
    )
    observed = _demonstrated(train, world.profile, world.start, budget)
    replay = [route[:2] for route in rng.sample(observed, min(len(observed), count))]
    composed = planner.compose(joint, count)
    prototypes = {}
    for index, route in enumerate(joint):
        for slot, node in enumerate(route):
            prototypes.setdefault(node, (index, slot))
    banks = {"joint": joint, "composed": composed, "replay": replay}
    choices = {
        "direct": (),
        "joint_density": joint[int(np.argmin(energies))],
    }
    singles = [route for route in composed if len(route) == 1]
    choices["single"], _ = planner.rank(
        value, world.profile, world.start, budget, singles
    )
    declarations = {}
    for source, routes in banks.items():
        best, probabilities = planner.rank(
            value, world.profile, world.start, budget, routes
        )
        choices[source] = best
        declarations[source] = [
            {
                "route": route,
                "terminal_probability": probabilities[route],
                "planning_energy": float(-np.log(max(1e-8, probabilities[route]))),
                "trace": _trace(world.start, route, budget),
                **_support(route, observed),
            }
            for route in routes
        ]
    selected = choices["composed"]
    order_controls = {}
    if len(selected) == 2:
        order_controls = {
            "first_only": selected[:1],
            "second_only": selected[1:],
            "reversed": selected[::-1],
        }
    return {
        "layout_id": world.id,
        "profile": world.profile,
        "start": asdict(world.start),
        "budget": budget,
        "joint_density": energies,
        "latent_draws": latents,
        "prototype_slots": prototypes,
        "choices": choices,
        "order_controls": order_controls,
        "banks": declarations,
        "ranking_queries": {name: len(routes) + 1 for name, routes in banks.items()},
    }


def rollout(world, budget, route, draws, *, value=None, suffixes=()) -> dict:
    state = world.start
    pending = route
    reached, actions = [], []
    switches = suffix_queries = 0
    for index, draw in enumerate(draws):
        action, pending = env.next_action(state, pending)
        state = env.step(world, state, action, draw)
        actions.append(action)
        if state.node in (env.GOAL, env.FAILED):
            break
        if pending and state.node == pending[0]:
            reached.append(pending[0])
            pending = pending[1:]
            remaining = budget - index - 1
            if value is not None and len(reached) == 1 and remaining:
                available = [
                    route
                    for route in suffixes
                    if env.track(state.node, route[0]) is not None
                ]
                new, _ = planner.rank(
                    value,
                    world.profile,
                    state,
                    remaining,
                    [*available, *([pending] if pending else [])],
                )
                suffix_queries += len(available) + bool(pending) + 1
                switches += new != pending
                pending = new
    return {
        "success": state.node == env.GOAL,
        "reached": reached,
        "actions": actions,
        "switches": switches,
        "suffix_queries": suffix_queries,
    }


def _evaluate(world, row, value, repeats, seed) -> dict:
    budget = row["budget"]

    def streams(name):
        rng = Random(f"{SCHEMA}:{name}:{seed}:{world.id}")
        return [
            tuple(rng.random() for _ in range(max(env.BUDGETS)))[:budget]
            for _ in range(repeats)
        ]

    evaluation, reference = streams("evaluation"), streams("reference")
    main = {
        name: [rollout(world, budget, route, draws) for draws in evaluation]
        for name, route in row["choices"].items()
    }
    single = [
        entry["route"] for entry in row["banks"]["composed"] if len(entry["route"]) == 1
    ]
    main["replan"] = [
        rollout(
            world,
            budget,
            row["choices"]["composed"],
            draws,
            value=value,
            suffixes=single,
        )
        for draws in evaluation
    ]
    order_controls = {
        name: [rollout(world, budget, route, draws) for draws in evaluation]
        for name, route in row["order_controls"].items()
    }
    steps = sum(
        len(trial["actions"])
        for trials in (*main.values(), *order_controls.values())
        for trial in trials
    )
    reference_steps = 0

    def reliability(trials, route):
        return (
            sum(
                trial["success"] and tuple(trial["reached"]) == route
                for trial in trials
            )
            / repeats
        )

    discovery = {}
    direct = sum(trial["success"] for trial in main["direct"]) / repeats
    for source, entries in row["banks"].items():
        rare = {}
        for entry in entries:
            if entry["rare_pair"]:
                rare.setdefault(entry["trace"], entry["route"])
        verified = []
        for trace, pair in rare.items():
            trials = [rollout(world, budget, pair, draws) for draws in reference]
            reference_steps += sum(len(trial["actions"]) for trial in trials)
            support = reliability(trials, pair)
            if support < RELIABLE:
                continue
            attempts = [rollout(world, budget, pair, draws) for draws in evaluation]
            steps += sum(len(trial["actions"]) for trial in attempts)
            gain = sum(trial["success"] for trial in attempts) / repeats - direct
            reliable = reliability(attempts, pair)
            verified.append(
                {
                    "route": pair,
                    "trace": trace,
                    "reference_reliability": support,
                    "evaluation_reliability": reliable,
                    "gain_over_direct": gain,
                    "confirmed": reliable >= RELIABLE and gain >= MINIMUM_GAIN,
                }
            )
        discovery[source] = {"rare_traces": len(rare), "reference_selected": verified}
    return {
        "layout_id": world.id,
        "budget": budget,
        "initially_supplied": world.start.supplied,
        "attempts": main,
        "order_controls": order_controls,
        "rare_discovery": discovery,
        "evaluation_environment_steps": steps,
        "reference_environment_steps": reference_steps,
    }


def _summarize(outcomes, declarations) -> dict:
    by_world = defaultdict(dict)
    budgets = {}
    for budget in env.BUDGETS:
        rows = [row for row in outcomes if row["budget"] == budget]
        rates = {
            name: np.array(
                [np.mean([t["success"] for t in row["attempts"][name]]) for row in rows]
            )
            for name in rows[0]["attempts"]
        }
        for row, declaration in zip(
            rows, (d for d in declarations if d["budget"] == budget), strict=True
        ):
            by_world[row["layout_id"]][budget] = declaration["choices"]
        draws = np.random.default_rng(20).integers(len(rows), size=(2000, len(rows)))
        contrasts = {}
        for left, right in (
            ("composed", "joint"),
            ("composed", "single"),
            ("composed", "replay"),
            ("composed", "direct"),
            ("replan", "composed"),
        ):
            delta = rates[left] - rates[right]
            contrasts[f"{left}_vs_{right}"] = {
                "difference": float(delta.mean()),
                "layout_bootstrap_95": np.quantile(
                    delta[draws].mean(axis=1), [0.025, 0.975]
                ).tolist(),
            }
        budgets[str(budget)] = {
            "terminal_success": {
                name: float(rate.mean()) for name, rate in rates.items()
            },
            "terminal_success_by_start_inventory": {
                label: {
                    name: float(
                        np.mean(
                            [
                                rate
                                for row, rate in zip(rows, values, strict=True)
                                if row["initially_supplied"] == supplied
                            ]
                        )
                    )
                    for name, values in rates.items()
                }
                for label, supplied in (("supplied", True), ("unsupplied", False))
                if any(row["initially_supplied"] == supplied for row in rows)
            },
            "paired_contrasts": contrasts,
            "closed_loop_suffix_changes": sum(
                t["switches"] for row in rows for t in row["attempts"]["replan"]
            ),
            "closed_loop_suffix_queries": sum(
                t["suffix_queries"] for row in rows for t in row["attempts"]["replan"]
            ),
            "rare_discovery": {
                source: {
                    "proposed_layout_trace_pairs": sum(
                        row["rare_discovery"][source]["rare_traces"] for row in rows
                    ),
                    "confirmed_layout_trace_pairs": sum(
                        item["confirmed"]
                        for row in rows
                        for item in row["rare_discovery"][source]["reference_selected"]
                    ),
                }
                for source in ("joint", "composed", "replay")
            },
        }
        order_rows = [
            {**row["attempts"], **row["order_controls"]}
            for row in rows
            if row["order_controls"]
        ]
        if order_rows:
            budgets[str(budget)]["selected_pair_order"] = {
                "worlds": len(order_rows),
                "terminal_success": {
                    name: float(
                        np.mean(
                            [
                                trial["success"]
                                for arms in order_rows
                                for trial in arms[name]
                            ]
                        )
                    )
                    for name in (
                        "composed",
                        "direct",
                        "first_only",
                        "second_only",
                        "reversed",
                    )
                },
            }
    return {
        "by_budget": budgets,
        "layouts_changing_route_across_budgets": {
            arm: sum(
                len({choices[arm] for choices in budgets.values()}) > 1
                for budgets in by_world.values()
            )
            for arm in ("composed", "fixed_pool")
        },
    }


def run(
    out: Path,
    training: TrainingConfig,
    *,
    split: str,
    layouts: int | None,
    repeats: int,
    candidates: int,
    demonstrations: int,
    exploration: int,
    flow_updates: int,
) -> dict:
    worlds = env.worlds(split)
    if layouts is not None and not 1 <= layouts <= len(worlds):
        raise ValueError("Expected a positive layout count within the chosen split")
    if min(repeats, demonstrations, exploration, flow_updates) < 1 or candidates < 2:
        raise ValueError(
            "Expected positive experiment counts and at least two proposals"
        )
    worlds = worlds[:layouts]
    out.mkdir(parents=True, exist_ok=False)
    train, dev = (
        data.demonstrations(group, demonstrations, 7) for group in ("train", "dev")
    )
    write_json(
        out / "demonstrations.json",
        {
            "train": [asdict(row) for row in train],
            "dev": [asdict(row) for row in dev],
        },
    )
    counts, exploration_steps = data.exploration(exploration, 7)
    write_json(
        out / "transition-counts.json",
        [
            {
                "profile": profile,
                "state": asdict(state),
                "action": action,
                "outcomes": [
                    {"state": asdict(new), "count": n} for new, n in outcomes.items()
                ],
            }
            for (profile, state, action), outcomes in counts.items()
        ],
    )
    value = planner.terminal_value(counts)
    fitted = planner.fit_proposer(train, dev, training, flow_updates, out)
    declarations = [
        _declare(fitted, value, world, budget, train, candidates, 7)
        for world in worlds
        for budget in env.BUDGETS
    ]
    for index, world in enumerate(worlds):
        cases = declarations[index * len(env.BUDGETS) : (index + 1) * len(env.BUDGETS)]
        fixed_pool = [entry["route"] for entry in cases[-1]["banks"]["composed"]]
        for row in cases:
            choice, _ = planner.rank(
                value, world.profile, world.start, row["budget"], fixed_pool
            )
            row["fixed_pool"] = fixed_pool
            row["choices"]["fixed_pool"] = choice
    bank = out / "candidate-bank.json"
    write_json(bank, declarations)
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "code_revision": code_revision(),
            "training": asdict(training),
            "budgets": env.BUDGETS,
            "repeats": repeats,
            "candidate_cap": candidates,
            "train_layouts": [world.id for world in env.worlds("train")],
            "dev_layouts": [world.id for world in env.worlds("dev")],
            "evaluation_layouts": [world.id for world in worlds],
            "evaluation_split": split,
            "collection_seed": 7,
            "proposal_seed": 7,
            "evaluation_seed": 43301,
            "demonstrations_per_case": demonstrations,
            "exploration_episodes_per_layout": exploration,
            "flow_updates": flow_updates,
            "flow_samples_per_case": [candidates, 64],
            "rare_support": RARE_SUPPORT,
            "min_reliability": RELIABLE,
            "min_gain": MINIMUM_GAIN,
            "candidate_bank_sha256": sha256(bank.read_bytes()).hexdigest(),
            "weights_sha256": sha256((out / WEIGHTS_FILE).read_bytes()).hexdigest(),
        },
    )
    layouts_by_id = {world.id: world for world in worlds}
    outcomes = [
        _evaluate(layouts_by_id[row["layout_id"]], row, value, repeats, 43301)
        for row in declarations
    ]
    write_json(out / "attempts.json", outcomes)
    summary = {
        "layouts": len(worlds),
        "demonstration_environment_steps": {
            "train": sum(len(row.actions) for row in train),
            "dev": sum(len(row.actions) for row in dev),
        },
        "exploration_environment_steps": exploration_steps,
        "evaluation_environment_steps": sum(
            row["evaluation_environment_steps"] for row in outcomes
        ),
        "reference_environment_steps": sum(
            row["reference_environment_steps"] for row in outcomes
        ),
        **_summarize(outcomes, declarations),
    }
    write_json(out / "summary.json", summary)
    return summary
