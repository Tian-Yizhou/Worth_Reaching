"""Budget-conditioned option value learned from hindsight options in an archive.

An option pursues a latent `z` for at most `k` actions. Every archive segment from
`s_t` to `s_{t+k}` is an option that pursued `z = enc(s_{t+k})`, and the encoded terminal
goal is the option `z = enc(g)` pursued for the whole remaining budget. Terminal goals
are relabeled from the same trajectory or drawn from other trajectories.

`Q(s, g, z, k, b, λ)` and `V(s, g, b, λ)` are fitted on frozen latent features with
in-sample semi-Markov backups, as in implicit Q-learning. If the goal is first reached
after `d` actions within the segment and the budget, the target is `1 − λd`. If the
budget is spent first, it is `−λb`. Otherwise it is `−λk + V(s_{t+k}, g, b − k, λ)`, with
`V` clipped to the range any return can take, `[−λ(b − k), 1 − λ]`. The backups have no
discount, so without the clip each backup adds the upper expectile's bias to the last
and values grow with the number of backups a budget spans. `V`
is an upper expectile of the target network's `Q` over the options the archive pursued
from the same state, which avoids maximizing over candidates nobody tried. The time
cost `λ` is an input sampled during training, so one value serves every setting.

Several heads, each trained on a bootstrap mask of every batch, give a spread that the
planner uses as its switching margin and to rank candidates by a quantile.
"""

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from worth_reaching.data.trajectory_windows import (
    ContinuousTrajectories,
    query_offsets,
)
from worth_reaching.models.gcivl import layer_norm_mlp
from worth_reaching.models.training import tensor, train_stage

Reached = Callable[[np.ndarray, np.ndarray], np.ndarray]
"""Vectorized terminal success predicate on observations and goals."""

NEVER = np.iinfo(np.int32).max


@dataclass(frozen=True, slots=True)
class OptionValueConfig:
    updates: int = 200_000
    batch_size: int = 1024
    width: int = 512
    depth: int = 3
    heads: int = 4
    learning_rate: float = 3e-4
    expectile: float = 0.7
    tau: float = 0.005
    time_costs: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002)
    """Per-action costs sampled during training; the planner picks one."""
    options: int = 1_000_000
    """Hindsight options labeled before training."""
    goal_option_share: float = 0.25
    """Share of options that pursue the relabeled terminal goal itself."""
    random_goal_share: float = 0.2
    """Share of terminal goals drawn from another trajectory."""
    bootstrap: float = 0.8
    """Probability that a head trains on a given row."""

    def __post_init__(self) -> None:
        if (
            min(self.updates, self.batch_size, self.width, self.depth, self.options) < 1
            or self.heads < 2
            or self.learning_rate <= 0
            or not 0 < self.expectile < 1
            or not 0 < self.tau <= 1
            or not self.time_costs
            or min(self.time_costs) < 0
            or not 0 <= self.goal_option_share < 1
            or not 0 <= self.random_goal_share < 1
            or not 0 < self.bootstrap <= 1
        ):
            raise ValueError(
                "Use positive sizes, at least two heads, rates in (0, 1) and "
                "non-negative time costs"
            )


@dataclass(frozen=True, slots=True)
class HindsightOptions:
    """Labeled options: archive rows, duration, budget, goal row and first arrival."""

    starts: np.ndarray
    targets: np.ndarray
    """The row whose encoding is the pursued latent: the landing or the goal."""
    durations: np.ndarray
    budgets: np.ndarray
    goals: np.ndarray
    first: np.ndarray
    """Actions until the goal was first reached within the segment, or NEVER."""


def label_options(
    archive: ContinuousTrajectories,
    reached: Reached,
    size: int,
    *,
    max_budget: int,
    max_horizon: int,
    queries: int,
    config: OptionValueConfig,
    rng: np.random.Generator,
    chunk: int = 512,
) -> HindsightOptions:
    """Sample options whose durations follow the world model's query offsets."""
    lengths = archive.episode_lengths
    begins = archive.ends - lengths
    eligible = np.flatnonzero(lengths >= 1)
    episodes = rng.choice(eligible, size=size)
    first_row, end = begins[episodes], archive.ends[episodes]
    starts = rng.integers(first_row, end)
    available = end - starts
    budgets = rng.integers(1, max_budget + 1, size=size)
    # Hindsight goals up to twice the budget ahead give both successes and failures.
    reach = np.minimum(available, 2 * budgets)
    goals = starts + rng.integers(1, reach + 1)
    random = rng.random(size) < config.random_goal_share
    goals = np.where(random, rng.integers(len(archive.observations), size=size), goals)
    span = np.minimum(budgets, max_horizon)
    offsets = query_offsets(span, queries)
    slot = rng.integers(queries, size=size)
    durations = np.maximum(offsets[np.arange(size), slot], 1)
    goal_option = (rng.random(size) < config.goal_option_share) & ~random
    durations = np.where(goal_option, budgets, durations)
    durations = np.minimum(durations, available).astype(np.int32)
    first = np.full(size, NEVER, dtype=np.int32)
    observations = archive.observations
    for begin in range(0, size, chunk):
        part = slice(begin, begin + chunk)
        steps = int(durations[part].max())
        rows = starts[part, None] + np.arange(1, steps + 1)
        inside = np.arange(1, steps + 1) <= durations[part, None]
        rows = np.where(inside, rows, starts[part, None])
        goal = observations[goals[part]]
        hit = reached(
            observations[rows].reshape(-1, observations.shape[-1]),
            np.repeat(goal, steps, axis=0),
        ).reshape(len(rows), steps)
        hit &= inside
        any_hit = hit.any(axis=1)
        first[part] = np.where(any_hit, hit.argmax(axis=1) + 1, NEVER)
    return HindsightOptions(
        starts.astype(np.int64),
        np.where(goal_option, goals, starts + durations).astype(np.int64),
        durations,
        budgets.astype(np.int32),
        goals.astype(np.int64),
        first,
    )


@dataclass(frozen=True, slots=True)
class OptionAttempts:
    """Options the executor actually pursued, as frozen latents and outcomes.

    An option interrupted after `duration` actions is the option with that duration,
    since the executor acts on the latent alone.
    """

    states: np.ndarray
    goals: np.ndarray
    targets: np.ndarray
    landings: np.ndarray
    durations: np.ndarray
    budgets: np.ndarray
    first: np.ndarray

    def __len__(self) -> int:
        return len(self.durations)

    @classmethod
    def concatenate(cls, parts: list["OptionAttempts"]) -> "OptionAttempts":
        return cls(
            *(
                np.concatenate([getattr(part, name) for part in parts])
                for name in cls.__slots__
            )
        )


class OptionValue(nn.Module):
    def __init__(self, latent_size: int, max_budget: int, config: OptionValueConfig):
        super().__init__()
        self.config, self.max_budget = config, max_budget
        q_size, v_size = 3 * latent_size + 3, 2 * latent_size + 2
        self.q = nn.ModuleList(
            layer_norm_mlp(q_size, config.width, 1, config.depth)
            for _ in range(config.heads)
        )
        self.v = nn.ModuleList(
            layer_norm_mlp(v_size, config.width, 1, config.depth)
            for _ in range(config.heads)
        )
        self.q_target = deepcopy(self.q).requires_grad_(False)

    def _scalars(self, *values: Tensor, cost: Tensor) -> Tensor:
        scaled = [value[..., None] / self.max_budget for value in values]
        return torch.cat((*scaled, cost[..., None] * self.max_budget), dim=-1)

    def option(self, state, goal, latent, duration, budget, cost, *, target=False):
        """Q for every head, shape (heads, batch)."""
        features = torch.cat(
            (state, goal, latent, self._scalars(duration, budget, cost=cost)), dim=-1
        )
        heads = self.q_target if target else self.q
        return torch.stack([head(features)[..., 0] for head in heads])

    def state(self, state, goal, budget, cost):
        """V for every head, shape (heads, batch)."""
        features = torch.cat((state, goal, self._scalars(budget, cost=cost)), dim=-1)
        return torch.stack([head(features)[..., 0] for head in self.v])

    @torch.no_grad()
    def score(
        self,
        state: np.ndarray,
        goal: np.ndarray,
        latents: np.ndarray,
        durations: np.ndarray,
        budget: int,
        cost: float,
    ) -> np.ndarray:
        """Q of each candidate option from one state, shape (heads, candidates)."""
        count = len(latents)
        return (
            self.option(
                tensor(np.broadcast_to(state, (count, state.size))),
                tensor(np.broadcast_to(goal, (count, goal.size))),
                tensor(latents),
                tensor(durations),
                tensor(np.full(count, budget)),
                tensor(np.full(count, cost)),
            )
            .numpy()
            .astype(np.float64)
        )

    @torch.no_grad()
    def update_target(self) -> None:
        for target, online in zip(
            self.q_target.parameters(), self.q.parameters(), strict=True
        ):
            target.lerp_(online, self.config.tau)


@dataclass(frozen=True)
class OptionBatches:
    options: HindsightOptions
    latents: np.ndarray
    """Frozen encodings aligned with the archive rows."""
    config: OptionValueConfig
    seed: tuple[int, ...]
    attempts: OptionAttempts | None = None
    online_share: float = 0.0
    """Share of each batch drawn from the executor's attempts."""

    def __call__(self, step: int):
        rng = np.random.default_rng([*self.seed, step])
        options, config = self.options, self.config
        size = config.batch_size
        online = round(size * self.online_share) if self.attempts else 0
        rows = rng.integers(len(options.starts), size=size - online)
        start, duration = options.starts[rows], options.durations[rows]
        columns = [
            self.latents[start],
            self.latents[options.goals[rows]],
            self.latents[options.targets[rows]],
            self.latents[start + duration],
            duration,
            options.budgets[rows],
            options.first[rows],
        ]
        if online:
            attempts = self.attempts
            picked = rng.integers(len(attempts), size=online)
            extra = [
                attempts.states,
                attempts.goals,
                attempts.targets,
                attempts.landings,
                attempts.durations,
                attempts.budgets,
                attempts.first,
            ]
            columns = [
                np.concatenate((column, values[picked]))
                for column, values in zip(columns, extra, strict=True)
            ]
        costs = np.asarray(config.time_costs, dtype=np.float32)
        cost = costs[rng.integers(len(costs), size=size)]
        mask = rng.random((size, config.heads)) < config.bootstrap
        return (
            *(tensor(np.asarray(column)) for column in columns),
            tensor(cost),
            torch.as_tensor(mask),
        )


def option_value_loss(model: OptionValue, batch):
    state, goal, latent, landing, duration, budget, first, cost, mask = batch
    config = model.config
    model.update_target()
    arrived = first <= torch.minimum(duration, budget)
    expired = budget <= duration
    with torch.no_grad():
        left = (budget - duration).clamp_min(0)
        following = model.state(landing, goal, left, cost)
        # The goal is reached at least one action after landing, or the budget runs out.
        following = torch.minimum(torch.maximum(following, -cost * left), 1 - cost)
        target = torch.where(
            arrived,
            1 - cost * first.clamp_max(model.max_budget),
            torch.where(expired, -cost * budget, -cost * duration + following),
        )
        anchor = model.option(state, goal, latent, duration, budget, cost, target=True)
    values = model.state(state, goal, budget, cost)
    weight = torch.where(anchor >= values, config.expectile, 1 - config.expectile)
    q = model.option(state, goal, latent, duration, budget, cost)
    # Rows lead so that the trainer's per-row validity broadcasts over the heads.
    return (
        ((q - target).T, mask),
        ((weight.detach().sqrt() * (anchor - values)).T, mask),
    )


@torch.no_grad()
def value_range(
    model: OptionValue,
    options: HindsightOptions,
    latents: np.ndarray,
    cost: float,
    *,
    size: int = 8192,
    seed: int = 0,
) -> dict:
    """Quantiles of Q on labeled options, which no return can push outside [-1, 1]."""
    batch = OptionBatches(
        options, latents, replace(model.config, batch_size=size), (seed, 960)
    )(0)
    state, goal, latent, _, duration, budget, *_ = batch
    q = model.option(
        state, goal, latent, duration, budget, torch.full((size,), cost)
    ).mean(0)
    goal_rows = (latent == goal).all(-1)
    quantiles = (0.0, 0.01, 0.5, 0.99, 1.0)

    def summary(values):
        return np.quantile(values.numpy(), quantiles).round(4).tolist()

    return {
        "quantiles": list(quantiles),
        "all": summary(q),
        "goal_options": summary(q[goal_rows]),
        "subgoal_options": summary(q[~goal_rows]),
    }


def fit_option_value(
    options: HindsightOptions,
    latents: np.ndarray,
    max_budget: int,
    config: OptionValueConfig,
    *,
    seed: int,
    out: Path,
    attempts: OptionAttempts | None = None,
    online_share: float = 0.0,
    model: OptionValue | None = None,
    updates: int | None = None,
) -> OptionValue:
    """Fit from hindsight options, or continue `model` with the executor's attempts."""
    torch.manual_seed(seed)
    if model is None:
        model = OptionValue(latents.shape[-1], max_budget, config)
    return train_stage(
        model.train(),
        OptionBatches(options, latents, config, (seed, 950), attempts, online_share),
        option_value_loss,
        steps=config.updates if updates is None else updates,
        learning_rate=config.learning_rate,
        out=out,
        checkpoint_every=10_000,
    )
