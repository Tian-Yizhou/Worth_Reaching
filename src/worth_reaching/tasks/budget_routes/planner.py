"""Assemble latent candidates and predict budget-limited terminal success."""

from functools import lru_cache
from itertools import permutations
from pathlib import Path

from flax import serialization

from worth_reaching.models import flow
from worth_reaching.models.completion import (
    FrozenCompletion,
    ModelSpec,
    TrainingConfig,
    fit,
    pack_inputs,
    pack_windows,
)
from worth_reaching.models.latent_energy import prefix_energy
from worth_reaching.models.training import stream_seed, weights
from worth_reaching.tasks.budget_routes import data
from worth_reaching.tasks.budget_routes import environment as env

SPEC = ModelSpec(data.OBSERVATION_SIZE, len(env.EDGES), (data.GOAL,), max(env.BUDGETS))
FLOW_STEPS = 16
Route = tuple[int, ...]


def fit_proposer(
    train: list[data.Demonstration],
    dev: list[data.Demonstration],
    config: TrainingConfig,
    flow_updates: int,
    out: Path,
):
    training = [record.window() for record in train if record.success]
    validation = [record.window() for record in dev if record.success]
    codec, predictor, _ = fit(training, validation, SPEC, config, out / "checkpoints")
    model = FrozenCompletion(SPEC, config, {"kind": "budget_routes"}, codec, predictor)
    inputs, targets = pack_windows(training, SPEC, config)
    forecast = model.forecast(inputs)
    fitted_flow, scale = flow.fit(
        flow.route_conditions(inputs, forecast, len(SPEC.goals)),
        model.encode(targets) - forecast,
        seed=stream_seed(config.seed, 4801),
        updates=flow_updates,
        out=out / "checkpoints" / "flow",
    )
    (out / "weights.msgpack").write_bytes(
        serialization.to_bytes(
            {
                "codec": weights(codec),
                "conditioned": weights(predictor),
                "flow": weights(fitted_flow),
                "scale": scale.numpy(),
            }
        )
    )
    return model, fitted_flow, scale


def latent_candidates(
    fitted, profile: int, state: env.State, budget: int, seed: int, count: int
) -> tuple[list[Route], list[float], list]:
    model, fitted_flow, scale = fitted
    inputs = pack_inputs(
        [data.public_input(profile, state, budget)], SPEC, model.config
    )
    forecast = model.forecast(inputs)
    conditions = flow.route_conditions(inputs, forecast, len(SPEC.goals))
    proposals, reference = (
        flow.sample(
            fitted_flow,
            forecast,
            conditions,
            scale,
            stream_seed(seed, stream),
            candidates=samples,
            steps=FLOW_STEPS,
        )[0]
        for stream, samples in ((0, count), (1, 64))
    )
    decoded = model.decode(proposals).numpy()
    routes = []
    for states in decoded:
        nodes = states[:, : len(env.EDGES)].argmax(axis=-1).tolist()
        route = []
        for node in nodes:
            if node not in env.INTERMEDIATE or node in route:
                break
            route.append(node)
        routes.append(tuple(route))
    scores = prefix_energy(proposals.numpy(), reference.numpy(), scale.numpy())
    return routes, scores.tolist(), proposals.tolist()


def compose(routes: list[Route], limit: int) -> list[Route]:
    """Preserve distinct singles before expanding ordered pairs of proposed targets."""
    nodes = list(dict.fromkeys(node for route in routes for node in route))
    pairs = [
        (first, second)
        for first, second in permutations(nodes, 2)
        if env.track(first, second) is not None
    ]
    return ([(node,) for node in nodes] + pairs)[:limit]


def terminal_value(counts: data.TransitionCounts):
    @lru_cache(maxsize=8192)
    def value(profile: int, state: env.State, budget: int, route: Route) -> float:
        if state.node == env.GOAL:
            return 1.0
        if state.node == env.FAILED or budget == 0:
            return 0.0
        action, pending = env.next_action(state, route)
        outcomes = counts.get((profile, state, action))
        if outcomes is None:
            return 0.0
        total = outcomes.total()
        return sum(
            (count / total) * value(profile, future, budget - 1, pending)
            for future, count in outcomes.items()
        )

    return value


def rank(value, profile: int, state: env.State, budget: int, routes: list[Route]):
    probabilities = {
        route: value(profile, state, budget, route) for route in [(), *routes]
    }
    best = max(probabilities, key=probabilities.__getitem__)
    return best, probabilities
