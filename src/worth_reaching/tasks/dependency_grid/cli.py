"""Command-line entry for the dependency-grid experiment."""

import argparse
from dataclasses import replace
from pathlib import Path

from worth_reaching.recipes import completion_recipe


def _run(args: argparse.Namespace) -> dict:
    from worth_reaching.tasks.dependency_grid.experiment import run

    return run(
        args.out,
        split=args.split,
        layout_limit=args.layouts,
        repeats=args.repeats,
        seed=args.seed,
        slip_probability=args.slip_probability,
    )


def _latent(args: argparse.Namespace) -> dict:
    from worth_reaching.models.completion import TrainingConfig
    from worth_reaching.tasks.dependency_grid.dataset import QUERY_STEPS
    from worth_reaching.tasks.dependency_grid.latent import EvaluationConfig, run

    training = completion_recipe(
        replace(TrainingConfig(), queries_per_window=len(QUERY_STEPS)), args
    )
    evaluation = EvaluationConfig(
        split=args.split,
        layouts=args.layouts,
        seed=args.eval_seed,
        repeats=args.repeats,
        candidates=args.candidates,
        density_samples=args.density_samples,
        slip_probability=args.slip_probability,
    )
    return run(
        args.out,
        training,
        evaluation,
        collection_seed=args.collection_seed,
        demonstrations=args.demonstrations,
        flow_updates=args.flow_updates,
        selector_repeats=args.selector_repeats,
    )


def _regions(args: argparse.Namespace) -> dict:
    from worth_reaching.models.completion import TrainingConfig
    from worth_reaching.models.executor import ExecutorConfig
    from worth_reaching.models.region import RegionConfig
    from worth_reaching.tasks.dependency_grid.dataset import QUERY_STEPS
    from worth_reaching.tasks.dependency_grid.latent import EvaluationConfig
    from worth_reaching.tasks.dependency_grid.regions import (
        RegionRunConfig,
        SweepConfig,
        run,
    )

    training = completion_recipe(
        replace(TrainingConfig(), queries_per_window=len(QUERY_STEPS)), args
    )
    evaluation = EvaluationConfig(
        split=args.split,
        layouts=args.layouts,
        seed=args.eval_seed,
        repeats=args.repeats,
        candidates=args.candidates,
        slip_probability=args.slip_probability,
    )
    config = RegionRunConfig(
        RegionConfig(updates=args.region_updates),
        ExecutorConfig(updates=args.executor_updates),
        SweepConfig(
            recall_levels=tuple(args.recall_levels),
            region_samples=args.region_samples,
            coherence_realizations=args.coherence_realizations,
        ),
    )
    return run(
        args.out,
        training,
        evaluation,
        config,
        collection_seed=args.collection_seed,
        demonstrations=args.demonstrations,
        flow_updates=args.flow_updates,
    )


def register_commands(commands) -> None:
    command = commands.add_parser(
        "dependency-grid", help="Compare matched goal and order interventions"
    )
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), default="dev")
    command.add_argument("--layouts", type=int, help="Limit layouts for a smoke run")
    command.add_argument("--repeats", type=int, default=16)
    command.add_argument("--seed", type=int, default=7)
    command.add_argument("--slip-probability", type=float, default=0.10)
    command.set_defaults(run=_run)

    latent = commands.add_parser(
        "dependency-grid-latent",
        help="Train and compare goal proposals on matched starts",
    )
    latent.add_argument("--out", type=Path, required=True)
    latent.add_argument("--config", type=Path, help="TOML recipe with a [model] table")
    for name in (
        "seed",
        "encoder-updates",
        "updates",
        "predictor-val-every",
        "predictor-patience",
    ):
        latent.add_argument(f"--{name}", type=int)
    latent.add_argument("--flow-updates", type=int, default=400)
    latent.add_argument(
        "--selector-repeats",
        type=int,
        help="Fit a shared outcome scorer with this many paired repeats per training target",
    )
    latent.add_argument("--collection-seed", type=int, default=7)
    latent.add_argument(
        "--demonstrations",
        type=int,
        default=8,
        help="Episodes per layout, depth and goal",
    )
    latent.add_argument("--split", choices=("dev", "test"), default="dev")
    latent.add_argument(
        "--layouts", type=int, help="Limit the number of evaluation layouts"
    )
    latent.add_argument("--eval-seed", type=int, default=43301)
    latent.add_argument(
        "--repeats",
        type=int,
        default=16,
        help="Repeats per reference and evaluation bank",
    )
    latent.add_argument("--candidates", type=int, default=8)
    latent.add_argument("--density-samples", type=int, default=128)
    latent.add_argument("--slip-probability", type=float, default=0.10)
    latent.set_defaults(run=_latent, extra="training")

    regions = commands.add_parser(
        "dependency-grid-regions",
        help="Compare region goals with decoded point goals on matched starts",
    )
    regions.add_argument("--out", type=Path, required=True)
    regions.add_argument("--config", type=Path, help="TOML recipe with a [model] table")
    for name in (
        "seed",
        "encoder-updates",
        "updates",
        "predictor-val-every",
        "predictor-patience",
    ):
        regions.add_argument(f"--{name}", type=int)
    regions.add_argument("--flow-updates", type=int, default=400)
    regions.add_argument("--region-updates", type=int, default=3000)
    regions.add_argument("--executor-updates", type=int, default=2000)
    regions.add_argument(
        "--recall-levels",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 0.9, 0.95],
        help="Share of realized dev states each arrival rule accepts",
    )
    regions.add_argument(
        "--coherence-realizations",
        type=int,
        default=4,
        help="Realizations per flow region attempted as point goals",
    )
    regions.add_argument("--region-samples", type=int, default=32)
    regions.add_argument("--collection-seed", type=int, default=7)
    regions.add_argument("--demonstrations", type=int, default=8)
    regions.add_argument("--split", choices=("dev", "test"), default="dev")
    regions.add_argument("--layouts", type=int)
    regions.add_argument("--eval-seed", type=int, default=43301)
    regions.add_argument("--repeats", type=int, default=16)
    regions.add_argument("--candidates", type=int, default=8)
    regions.add_argument("--slip-probability", type=float, default=0.10)
    regions.set_defaults(run=_regions, extra="training")
