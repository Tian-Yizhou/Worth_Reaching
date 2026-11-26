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


def _pointmaze_regions(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.models.executor import ExecutorConfig
    from worth_reaching.models.region import RegionConfig
    from worth_reaching.tasks.pointmaze.regions import MazeRegionConfig, run

    config = MazeRegionConfig(
        region=RegionConfig(updates=args.region_updates),
        executor=ExecutorConfig(
            updates=args.executor_updates, width=256, batch_size=256
        ),
        recall_levels=tuple(args.recall_levels),
        region_samples=args.region_samples,
        coherence_realizations=args.coherence_realizations,
        executor_pairs=args.executor_pairs,
    )
    return run(
        args.data_dir,
        args.checkpoint,
        args.out,
        config,
        split=args.eval_split,
        repeats=args.eval_repeats,
        candidates=args.candidates,
        prefix=args.prefix_length,
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
    regions = commands.add_parser(
        "pointmaze-regions",
        help="Compare region goals with decoded point goals from a frozen PointMaze run",
    )
    regions.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Budget directory of a pointmaze-latent run",
    )
    regions.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/minari")
    )
    regions.add_argument("--out", type=Path, required=True)
    regions.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    regions.add_argument("--eval-repeats", type=int, default=64)
    regions.add_argument("--candidates", type=int, default=8)
    regions.add_argument("--prefix-length", type=int, default=1)
    regions.add_argument("--region-updates", type=int, default=3000)
    regions.add_argument("--executor-updates", type=int, default=4000)
    regions.add_argument("--executor-pairs", type=int, default=200_000)
    regions.add_argument(
        "--recall-levels", type=float, nargs="+", default=[0.5, 0.75, 0.9]
    )
    regions.add_argument("--region-samples", type=int, default=32)
    regions.add_argument("--coherence-realizations", type=int, default=4)
    regions.set_defaults(run=_pointmaze_regions, extra="pointmaze")
    return pointmaze
