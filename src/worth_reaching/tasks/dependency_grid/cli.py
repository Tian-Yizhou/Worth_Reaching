"""Command-line entry for the dependency-grid experiment."""

import argparse
from pathlib import Path


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
