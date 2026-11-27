"""Command-line entry point for Worth Reaching experiments."""

import argparse
import json
from pathlib import Path

from worth_reaching.compute import ComputeConfig
from worth_reaching.recipes import completion_recipe
from worth_reaching.tasks.budget_routes.cli import (
    register_commands as register_budget_routes,
)
from worth_reaching.tasks.dependency_grid.cli import (
    register_commands as register_dependency_grid,
)
from worth_reaching.tasks.ogbench_manipulation.spec import SCENE, TASKS
from worth_reaching.tasks.pointmaze.cli import register_commands as register_pointmaze

EXPERIMENT_MODULES = {
    "gymnasium",
    "gymnasium_robotics",
    "h5py",
    "minari",
    "mujoco",
    "numpy",
    "ogbench",
}


def _devices(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.models.training import describe

    return describe()


def _attempt_compare(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.attempt_selection import compare_attempt_banks

    return compare_attempt_banks(args.train, args.dev, args.test, args.out, args.seed)


def _manipulation_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.ogbench_manipulation import ExecutorConfig, run
    from worth_reaching.models.continuous_completion import CompletionConfig

    return run(
        SCENE,
        args.data_dir,
        args.out,
        completion_recipe(
            CompletionConfig(
                windows=8000,
                max_horizon=SCENE.action_limit,
                codec_updates=350,
                predictor_updates=1000,
                flow_updates=450,
            ),
            args,
        ),
        ExecutorConfig(updates=args.executor_updates),
        split=args.eval_split,
        task=tuple(dict.fromkeys(args.task)) if args.task is not None else None,
        repeats=args.eval_repeats,
        limit=args.eval_steps,
        resume=args.resume,
        eval_workers=args.eval_workers,
    )


def _scene_reachability(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.scene_reachability import run

    return run(
        args.checkpoint,
        args.data_dir / f"{SCENE.dataset}-val.npz",
        args.out,
        seed=args.seed,
        offsets=args.offsets,
        budgets=args.budgets,
        cases_per_group=args.cases_per_group,
        replay_from=args.replay_from,
        waypoints=args.waypoints,
    )


def _scene_executor(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.scene_executor import run
    from worth_reaching.models.gcivl import GCIVLConfig

    return run(
        args.data_dir,
        args.out,
        GCIVLConfig(updates=args.updates, batch_size=args.batch_size),
        seed=args.seed,
        split=args.eval_split,
        tasks=tuple(dict.fromkeys(args.task)) if args.task is not None else TASKS,
        repeats=args.eval_repeats,
        eval_workers=args.eval_workers,
        resume=args.resume,
    )


def _scene_regions(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.scene_regions import SceneRegionConfig, run
    from worth_reaching.models.continuous_completion import CompletionConfig
    from worth_reaching.models.gcivl import GCIVLConfig
    from worth_reaching.models.region import RegionConfig

    config = SceneRegionConfig(
        completion=completion_recipe(
            CompletionConfig(max_horizon=SCENE.action_limit), args
        ),
        region=RegionConfig(
            updates=args.region_updates,
            width=args.region_width,
            batch_size=args.region_batch_size,
        ),
        latent_executor=GCIVLConfig(
            updates=args.latent_updates, batch_size=args.latent_batch_size
        ),
        recall_levels=tuple(args.recall_levels),
        region_samples=args.region_samples,
        decoder_windows=args.decoder_windows,
        stages=args.stages,
        sample_repeats=args.sample_repeats,
    )
    return run(
        args.data_dir,
        args.executor,
        args.out,
        config,
        split=args.eval_split,
        tasks=tuple(dict.fromkeys(args.task or (4, 5))),
        repeats=args.eval_repeats,
        workers=args.eval_workers,
        models_run=args.models,
    )


def _scene_evaluate(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.ogbench_manipulation import evaluate_frozen_scene

    return evaluate_frozen_scene(
        args.checkpoint,
        args.out,
        tasks=tuple(dict.fromkeys(args.task)) if args.task is not None else TASKS,
        split=args.eval_split,
        repeats=args.eval_repeats,
        limit=args.eval_steps,
        backend=args.backend,
        eval_workers=args.eval_workers,
        batch_size=args.batch_size,
        device=args.device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="wr")
    parser.add_argument(
        "--compute-config",
        type=Path,
        help="Read device settings before loading ML libraries",
    )
    commands = parser.add_subparsers(required=True)
    devices = commands.add_parser(
        "devices", help="Report the devices available for training"
    )
    devices.set_defaults(run=_devices, extra="training")
    register_dependency_grid(commands)
    register_budget_routes(commands)
    pointmaze = register_pointmaze(commands)
    scene = commands.add_parser(
        "scene-latent",
        help="Train and evaluate latent planning on state-based OGBench Scene",
    )
    scene.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/ogbench")
    )
    scene.add_argument("--out", type=Path, required=True)
    scene.add_argument("--seed", type=int)
    scene.add_argument("--windows", type=int)
    scene.add_argument("--executor-updates", type=int, default=2500)
    scene.add_argument("--codec-updates", type=int)
    scene.add_argument("--predictor-updates", type=int)
    scene.add_argument("--flow-updates", type=int)
    scene.add_argument(
        "--task",
        type=int,
        choices=TASKS,
        action="append",
        help="Select an official task (repeat to evaluate several in one run)",
    )
    scene.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    scene.add_argument("--eval-repeats", type=int, default=2)
    scene.add_argument(
        "--eval-workers",
        type=int,
        default=1,
        help="Number of CPU processes for independent matched starts",
    )
    scene.add_argument(
        "--eval-steps",
        type=int,
        help="Action limit (defaults to the official task limit)",
    )
    scene.set_defaults(run=_manipulation_latent, extra="ogbench")
    reachability = commands.add_parser(
        "scene-reachability", help="Probe a frozen Scene controller on achieved goals"
    )
    reachability.add_argument(
        "--checkpoint", type=Path, required=True, help="Directory from scene-latent"
    )
    reachability.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/ogbench")
    )
    reachability.add_argument("--out", type=Path, required=True)
    reachability.add_argument("--seed", type=int, default=803)
    reachability.add_argument("--offsets", nargs="+", type=int, default=[16, 32, 64])
    reachability.add_argument(
        "--budgets", nargs="+", type=int, default=[16, 32, 64, 128]
    )
    reachability.add_argument("--cases-per-group", type=int, default=6)
    reachability.add_argument(
        "--waypoints", choices=("mixed", "drawer-insertion"), default="mixed"
    )
    reachability.add_argument(
        "--replay-from",
        type=Path,
        help="Reuse recorded-action replays from an identical waypoint bank",
    )
    reachability.set_defaults(run=_scene_reachability, extra="ogbench")
    evaluate = commands.add_parser(
        "scene-evaluate",
        help="Evaluate frozen Scene weights on CPU or batched CUDA physics",
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--task", action="append", choices=TASKS, type=int)
    evaluate.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    evaluate.add_argument("--eval-repeats", type=int, default=4)
    evaluate.add_argument("--eval-steps", type=int, default=SCENE.action_limit)
    evaluate.add_argument("--backend", choices=("cpu", "warp"), default="cpu")
    evaluate.add_argument("--eval-workers", type=int, default=1)
    evaluate.add_argument("--batch-size", type=int, default=16)
    evaluate.add_argument("--device", default="cuda:0")
    evaluate.set_defaults(run=_scene_evaluate, extra="ogbench")
    executor = commands.add_parser(
        "scene-executor",
        help="Train a GCIVL Scene executor and measure direct success on official tasks",
    )
    executor.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/ogbench")
    )
    executor.add_argument("--out", type=Path, required=True)
    executor.add_argument("--seed", type=int, default=7)
    executor.add_argument("--updates", type=int, default=1_000_000)
    executor.add_argument("--batch-size", type=int, default=1024)
    executor.add_argument("--task", type=int, choices=TASKS, action="append")
    executor.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    executor.add_argument("--eval-repeats", type=int, default=10)
    executor.add_argument("--eval-workers", type=int, default=10)
    executor.add_argument(
        "--resume", type=Path, help="Continue from a Lightning executor checkpoint"
    )
    executor.set_defaults(run=_scene_executor, extra="ogbench")
    scene_regions = commands.add_parser(
        "scene-regions",
        help="Compare region and point goals in closed loop on Scene tasks",
    )
    scene_regions.add_argument(
        "--executor",
        type=Path,
        required=True,
        help="A scene-executor run whose GCIVL executor pursues observation goals",
    )
    scene_regions.add_argument(
        "--models",
        type=Path,
        help="Reuse the world model, decoder and latent executor of a scene-regions run",
    )
    scene_regions.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/ogbench")
    )
    scene_regions.add_argument("--out", type=Path, required=True)
    scene_regions.add_argument("--seed", type=int)
    for name in ("windows", "codec-updates", "predictor-updates", "flow-updates"):
        scene_regions.add_argument(f"--{name}", type=int)
    scene_regions.add_argument("--task", type=int, choices=TASKS, action="append")
    scene_regions.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    scene_regions.add_argument("--eval-repeats", type=int, default=20)
    scene_regions.add_argument("--eval-workers", type=int, default=1)
    scene_regions.add_argument("--latent-updates", type=int, default=1_000_000)
    scene_regions.add_argument("--latent-batch-size", type=int, default=1024)
    scene_regions.add_argument("--region-updates", type=int, default=3000)
    scene_regions.add_argument("--region-width", type=int, default=128)
    scene_regions.add_argument("--region-batch-size", type=int, default=64)
    scene_regions.add_argument("--decoder-windows", type=int, default=20000)
    scene_regions.add_argument(
        "--recall-levels", type=float, nargs="+", default=[0.5, 0.75, 0.9]
    )
    scene_regions.add_argument("--region-samples", type=int, default=32)
    scene_regions.add_argument("--stages", type=int, default=1)
    scene_regions.add_argument("--sample-repeats", type=int, default=2)
    scene_regions.set_defaults(run=_scene_regions, extra="ogbench")
    comparison = commands.add_parser(
        "attempt-compare", help="Fit and compare shared-bank selectors"
    )
    for split in ("train", "dev", "test"):
        comparison.add_argument(f"--{split}", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--seed", type=int, default=43301)
    comparison.set_defaults(run=_attempt_compare, extra="training")
    for command in (pointmaze, scene):
        command.add_argument(
            "--resume", type=Path, help="Continue from a compatible Lightning run"
        )
    for command in (pointmaze, scene, scene_regions):
        command.add_argument(
            "--config", type=Path, help="TOML completion recipe with a [model] table"
        )
        command.add_argument(
            "--predictor-val-every",
            type=int,
            help="Predictor updates between development checks",
        )
        command.add_argument(
            "--predictor-patience",
            type=int,
            help="Development checks without improvement before stopping",
        )
    args = parser.parse_args()

    try:
        ComputeConfig.load(args.compute_config).apply()
        summary = args.run(args)
    except ModuleNotFoundError as error:
        if error.name in {"lightning", "torch", "torchdata"}:
            parser.error(
                "Add the training (CUDA) or training-cpu extra alongside your experiment extras in uv sync --locked"
            )
        if error.name not in EXPERIMENT_MODULES:
            raise
        parser.error(
            f"Install experiment dependencies with uv sync --locked --extra {args.extra}"
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if hasattr(args, "out"):
        summary = {"run_id": args.out.name, "out": str(args.out), **summary}
    print(json.dumps(summary))
