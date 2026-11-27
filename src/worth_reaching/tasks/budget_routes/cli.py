"""Register the budgeted latent composition experiment."""

import argparse
from pathlib import Path

from worth_reaching.recipes import completion_recipe


def _run(args: argparse.Namespace) -> dict:
    from worth_reaching.models.completion import TrainingConfig
    from worth_reaching.tasks.budget_routes.experiment import run

    config = completion_recipe(
        TrainingConfig(queries_per_window=2, latent_size=6), args
    )
    if config.queries_per_window != 2:
        raise ValueError("The budget-routes protocol uses two query slots")
    return run(
        args.out,
        config,
        split=args.split,
        layouts=args.layouts,
        repeats=args.repeats,
        candidates=args.candidates,
        demonstrations=args.demonstrations,
        exploration=args.exploration,
        flow_updates=args.flow_updates,
    )


def register_commands(commands) -> None:
    command = commands.add_parser(
        "budget-routes",
        help="Compare latent route composition across action budgets",
    )
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "test"), default="dev")
    command.add_argument("--layouts", type=int)
    command.add_argument("--repeats", type=int, default=64)
    command.add_argument("--candidates", type=int, default=24)
    command.add_argument("--demonstrations", type=int, default=24)
    command.add_argument("--exploration", type=int, default=96)
    command.add_argument("--flow-updates", type=int, default=400)
    command.add_argument("--config", type=Path, help="TOML recipe with a [model] table")
    for name in (
        "seed",
        "encoder-updates",
        "updates",
        "predictor-val-every",
        "predictor-patience",
    ):
        command.add_argument(f"--{name}", type=int)
    command.set_defaults(run=_run, extra="training")
