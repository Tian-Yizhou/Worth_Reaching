"""Command-line interface for PointMaze experiments."""

import argparse
from pathlib import Path

from worth_reaching.recipes import completion_recipe
from worth_reaching.tasks.pointmaze.spec import TASKS as POINTMAZE_TASKS


def _pointmaze_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.models.continuous_completion import BudgetedCompletionConfig
    from worth_reaching.tasks.pointmaze.experiment import run

    return run(
        args.data_dir,
        args.out,
        completion_recipe(
            BudgetedCompletionConfig(
                windows=4000, codec_updates=250, predictor_updates=600, flow_updates=300
            ),
            args,
        ),
        task_name=args.task,
        split=args.eval_split,
        repeats=args.eval_repeats,
        budgets=[args.eval_steps] if args.eval_steps is not None else args.eval_budgets,
        prediction_span=args.prediction_span,
        resume=args.resume,
    )


def _pointmaze_first_targets(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.tasks.pointmaze.probe import run

    return run(
        args.data_dir,
        args.runs,
        args.out,
        args.max_starts,
        compare_continuations=args.compare_continuations,
    )


def _pointmaze_latent_energy(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.tasks.pointmaze.energy import run

    return run(
        args.data_dir,
        args.runs,
        args.out,
        split=args.eval_split,
        repeats=args.eval_repeats,
        prefix_length=args.prefix_length,
        candidates=args.candidates,
        density_samples=args.density_samples,
    )


def register_commands(commands) -> argparse.ArgumentParser:
    pointmaze = commands.add_parser(
        "pointmaze-latent",
        help="Train and evaluate latent completion on Maze2D U-Maze, Medium or Large",
    )
    pointmaze.add_argument("--task", choices=POINTMAZE_TASKS, default="medium")
    pointmaze.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/minari")
    )
    pointmaze.add_argument("--out", type=Path, required=True)
    pointmaze.add_argument("--seed", type=int)
    pointmaze.add_argument("--windows", type=int)
    pointmaze.add_argument("--codec-updates", type=int)
    pointmaze.add_argument("--predictor-updates", type=int)
    pointmaze.add_argument("--flow-updates", type=int)
    pointmaze.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    pointmaze.add_argument("--eval-repeats", type=int, default=6)
    pointmaze_budgets = pointmaze.add_mutually_exclusive_group()
    pointmaze_budgets.add_argument(
        "--eval-steps",
        type=int,
        help="Action budget for one evaluation (defaults to the official limit)",
    )
    pointmaze_budgets.add_argument(
        "--eval-budgets",
        type=int,
        nargs="+",
        help="Compare multiple action budgets with the same checkpoint and resets",
    )
    pointmaze.add_argument(
        "--prediction-span",
        type=int,
        help="Maximum span for each model prediction (defaults to archive coverage)",
    )
    pointmaze.set_defaults(run=_pointmaze_latent, extra="pointmaze")
    first_targets = commands.add_parser(
        "pointmaze-first-targets",
        help="Evaluate frozen PointMaze first targets from matched resets",
    )
    first_targets.add_argument("--runs", nargs="+", type=Path, required=True)
    first_targets.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/minari")
    )
    first_targets.add_argument("--out", type=Path, required=True)
    first_targets.add_argument(
        "--max-starts", type=int, help="Limit starts for a smoke check"
    )
    first_targets.add_argument(
        "--compare-continuations",
        action="store_true",
        help="Use the first sampled target and compare direct and replanning continuations",
    )
    first_targets.set_defaults(run=_pointmaze_first_targets, extra="pointmaze")
    latent_energy = commands.add_parser(
        "pointmaze-latent-energy",
        help="Test frozen latent energy against matched PointMaze candidate attempts",
    )
    latent_energy.add_argument("--runs", nargs="+", type=Path, required=True)
    latent_energy.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/minari")
    )
    latent_energy.add_argument("--out", type=Path, required=True)
    latent_energy.add_argument("--eval-split", choices=("dev", "test"), default="test")
    latent_energy.add_argument("--eval-repeats", type=int, default=64)
    latent_energy.add_argument("--prefix-length", type=int, default=1)
    latent_energy.add_argument("--candidates", type=int, default=16)
    latent_energy.add_argument("--density-samples", type=int, default=256)
    latent_energy.set_defaults(run=_pointmaze_latent_energy, extra="pointmaze")
    return pointmaze
