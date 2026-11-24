"""Command-line entry point for Worth Reaching experiments."""

import argparse
import json
from pathlib import Path

from worth_reaching.compute import ComputeConfig
from worth_reaching.recipes import completion_recipe
from worth_reaching.tasks.antmaze.cli import register_commands as register_antmaze
from worth_reaching.tasks.ogbench_manipulation.spec import PUZZLE_BOARDS, SCENE, TASKS
from worth_reaching.tasks.pointmaze.cli import register_commands as register_pointmaze


def _devices(args: argparse.Namespace) -> dict[str, object]:
    if args.framework == "jax":
        from worth_reaching.models.parallel import probe

        return probe(args.batch_size)
    from worth_reaching.models.training import describe

    return describe()


def _run_pilot(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.keycorridor import PilotConfig, run_pilot

    return run_pilot(PilotConfig.load(args.config), args.out)


def _collect(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.tasks.craftax.collection import CollectionConfig, collect
    from worth_reaching.tasks.craftax.environment import TASK

    config = CollectionConfig.load(
        args.config,
        episodes=args.episodes,
        budget=args.budget,
        seed_start=args.seed_start,
    )
    return collect(
        config,
        args.out,
        expert_checkpoint=args.expert_checkpoint,
        task=args.task or TASK,
    )


def _attempt_compare(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.attempt_selection import compare_attempt_banks

    return compare_attempt_banks(args.train, args.dev, args.test, args.out, args.seed)


def _prepare_windows(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.data.completion import WindowConfig
    from worth_reaching.tasks.craftax.windows import prepare_craftax_windows

    config = WindowConfig(
        args.seed, args.windows_per_episode, args.min_span, args.max_span
    )
    return prepare_craftax_windows(args.collection, args.out, config)


def _train_completion(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.completion import run_completion_training
    from worth_reaching.models.completion import TrainingConfig

    config = completion_recipe(TrainingConfig(), args)
    return run_completion_training(
        args.out,
        config,
        source=args.windows,
        evaluate_test=args.evaluate_test,
        resume=args.resume,
    )


def _run_grid_routes(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.grid_execution import ExecutionConfig, run

    config = ExecutionConfig(
        seed=args.seed,
        updates=args.updates,
        candidates=args.candidates,
        repeats=args.repeats,
        slip_probability=args.slip_probability,
        split=args.split,
        layouts=args.layouts,
    )
    return run(args.checkpoint, args.out, config, resume=args.resume)


def _run_grid_feedback(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.grid_feedback import FeedbackConfig, run

    config = FeedbackConfig(
        args.seed,
        args.updates,
        args.candidates,
        args.repeats,
        args.passage_slip,
        args.split,
        args.layouts,
    )
    return run(args.checkpoint, args.flow_run, args.out, config, resume=args.resume)


def _kuka_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.kuka import run
    from worth_reaching.models.continuous_completion import CompletionConfig

    return run(
        args.source_dir,
        args.data_dir,
        args.out,
        completion_recipe(
            CompletionConfig(
                windows=4000,
                max_horizon=256,
                latent_size=24,
                codec_updates=250,
                predictor_updates=600,
                flow_updates=300,
            ),
            args,
        ),
        split=args.eval_split,
        repeats=args.eval_repeats,
        limit=args.eval_steps,
        resume=args.resume,
    )


def _manipulation_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.ogbench_manipulation import ExecutorConfig, run
    from worth_reaching.models.continuous_completion import CompletionConfig

    benchmark = PUZZLE_BOARDS[args.board] if args.board is not None else SCENE
    return run(
        benchmark,
        args.data_dir,
        args.out,
        completion_recipe(
            CompletionConfig(
                windows=8000,
                max_horizon=benchmark.action_limit,
                codec_updates=350,
                predictor_updates=1000,
                flow_updates=450,
            ),
            args,
        ),
        ExecutorConfig(updates=args.executor_updates),
        split=args.eval_split,
        task=args.task,
        repeats=args.eval_repeats,
        limit=args.eval_steps,
        resume=args.resume,
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
    devices.add_argument("--framework", choices=("torch", "jax"), default="torch")
    devices.add_argument(
        "--batch-size",
        type=int,
        default=17,
        help="Batch size for the JAX sharding probe",
    )
    devices.set_defaults(run=_devices, extra="training")
    pilot = commands.add_parser("pilot", help="Run the CPU KeyCorridor diagnostic")
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    pilot.set_defaults(run=_run_pilot, extra="minigrid")
    collection = commands.add_parser(
        "collect", help="Collect Craftax-Classic trajectories"
    )
    collection.add_argument("--config", type=Path, required=True)
    collection.add_argument("--out", type=Path, required=True)
    collection.add_argument("--episodes", type=int)
    collection.add_argument("--budget", type=int)
    collection.add_argument("--seed-start", type=int)
    collection.add_argument(
        "--task", help="Craftax terminal task (default: make_iron_pickaxe)"
    )
    collection.add_argument("--expert-checkpoint", type=Path)
    collection.set_defaults(run=_collect, extra="craftax", uses_jax=True)
    windows = commands.add_parser(
        "prepare-windows", help="Export masked windows from a Craftax collection"
    )
    windows.add_argument("--collection", type=Path, required=True)
    windows.add_argument("--out", type=Path, required=True)
    windows.add_argument("--seed", type=int, default=0)
    windows.add_argument("--windows-per-episode", type=int, default=8)
    windows.add_argument("--min-span", type=int, default=1)
    windows.add_argument("--max-span", type=int)
    windows.set_defaults(run=_prepare_windows, extra="craftax")
    training = commands.add_parser(
        "train-completion", help="Train and compare masked latent predictors"
    )
    source = training.add_mutually_exclusive_group(required=True)
    source.add_argument("--grid", action="store_true", help="Use paired grid routes")
    source.add_argument("--windows", type=Path, help="Use exported Craftax windows")
    training.add_argument("--out", type=Path, required=True)
    training.add_argument("--seed", type=int)
    training.add_argument("--encoder-updates", type=int)
    training.add_argument("--updates", type=int)
    training.add_argument("--history-length", type=int)
    training.add_argument("--queries-per-window", type=int)
    training.add_argument(
        "--evaluate-test", action="store_true", help="Score frozen grid test worlds"
    )
    training.set_defaults(run=_train_completion, extra="craftax")
    routes = commands.add_parser(
        "grid-routes",
        help="Sample latent grid routes and run matched waypoint attempts",
    )
    routes.add_argument("--checkpoint", type=Path, required=True)
    routes.add_argument("--out", type=Path, required=True)
    routes.add_argument("--seed", type=int, default=7)
    routes.add_argument("--updates", type=int, default=800)
    routes.add_argument("--candidates", type=int, default=8)
    routes.add_argument("--repeats", type=int, default=8)
    routes.add_argument("--slip-probability", type=float, default=0.15)
    routes.add_argument("--split", choices=("dev", "test"), default="dev")
    routes.add_argument(
        "--layouts", type=int, help="Limit evaluated layouts for a smoke run"
    )
    routes.set_defaults(run=_run_grid_routes, extra="craftax")
    feedback = commands.add_parser(
        "grid-feedback",
        help="Train from matched waypoint attempts and freeze the reranker",
    )
    feedback.add_argument("--checkpoint", type=Path, required=True)
    feedback.add_argument("--flow-run", type=Path, required=True)
    feedback.add_argument("--out", type=Path, required=True)
    feedback.add_argument("--seed", type=int, default=7)
    feedback.add_argument("--updates", type=int, default=250)
    feedback.add_argument("--candidates", type=int, default=8)
    feedback.add_argument("--repeats", type=int, default=8)
    feedback.add_argument("--passage-slip", type=float, default=0.85)
    feedback.add_argument("--split", choices=("dev", "test"), default="dev")
    feedback.add_argument("--layouts", type=int)
    feedback.set_defaults(run=_run_grid_feedback, extra="craftax")
    latent = register_antmaze(commands)
    pointmaze = register_pointmaze(commands)
    kuka = commands.add_parser(
        "kuka-latent",
        help="Train and evaluate latent completion on conditional Kuka stacking",
    )
    kuka.add_argument(
        "--source-dir", type=Path, default=Path("results/external/diffuser-kuka")
    )
    kuka.add_argument(
        "--data-dir",
        type=Path,
        default=Path("results/datasets/diffuser-kuka/kuka_dataset"),
    )
    kuka.add_argument("--out", type=Path, required=True)
    kuka.add_argument("--seed", type=int)
    kuka.add_argument("--windows", type=int)
    kuka.add_argument("--codec-updates", type=int)
    kuka.add_argument("--predictor-updates", type=int)
    kuka.add_argument("--flow-updates", type=int)
    kuka.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    kuka.add_argument("--eval-repeats", type=int, default=4)
    kuka.add_argument("--eval-steps", type=int, default=384)
    kuka.set_defaults(run=_kuka_latent, extra="stacking")
    scene = commands.add_parser(
        "scene-latent",
        help="Train and evaluate latent planning on state-based OGBench Scene",
    )
    scene.set_defaults(board=None)
    puzzle = commands.add_parser(
        "puzzle-latent",
        help="Train and evaluate latent planning on state-based OGBench Puzzle",
    )
    puzzle.add_argument("--board", choices=PUZZLE_BOARDS, default="4x4")
    for command in (scene, puzzle):
        command.add_argument(
            "--data-dir", type=Path, default=Path("results/datasets/ogbench")
        )
        command.add_argument("--out", type=Path, required=True)
        command.add_argument("--seed", type=int)
        command.add_argument("--windows", type=int)
        command.add_argument("--executor-updates", type=int, default=2500)
        command.add_argument("--codec-updates", type=int)
        command.add_argument("--predictor-updates", type=int)
        command.add_argument("--flow-updates", type=int)
        command.add_argument(
            "--task", type=int, choices=TASKS, help="Select one official task"
        )
        command.add_argument("--eval-split", choices=("dev", "test"), default="dev")
        command.add_argument("--eval-repeats", type=int, default=2)
        command.add_argument(
            "--eval-steps",
            type=int,
            help="Action limit (defaults to the official task limit)",
        )
        command.set_defaults(run=_manipulation_latent, extra="ogbench")
    comparison = commands.add_parser(
        "attempt-compare", help="Fit and compare shared-bank selectors"
    )
    for split in ("train", "dev", "test"):
        comparison.add_argument(f"--{split}", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--seed", type=int, default=43301)
    comparison.set_defaults(run=_attempt_compare, extra="craftax")
    for command in (training, routes, feedback, latent, pointmaze, kuka, scene, puzzle):
        command.add_argument(
            "--resume", type=Path, help="Continue from a compatible Lightning run"
        )
    for command in (training, latent, pointmaze, kuka, scene, puzzle):
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
        compute = ComputeConfig.load(args.compute_config)
        compute.apply()
        if (
            getattr(args, "uses_jax", False)
            or getattr(args, "framework", None) == "jax"
        ):
            compute.validate_backend()
        summary = args.run(args)
    except ModuleNotFoundError as error:
        if error.name in {"lightning", "torch", "torchdata"}:
            parser.error(
                "Add the training (CUDA) or training-cpu extra alongside your experiment extras in uv sync --locked"
            )
        if error.name not in {
            "gymnasium",
            "minigrid",
            "craftax",
            "flax",
            "jax",
            "mujoco",
            "minari",
            "h5py",
            "gymnasium_robotics",
            "pybullet",
            "numpy",
            "ogbench",
            "optax",
        }:
            raise
        extra = "craftax" if getattr(args, "framework", None) == "jax" else args.extra
        parser.error(
            f"Install experiment dependencies with uv sync --locked --extra {extra}"
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if hasattr(args, "out"):
        summary = {"run_id": args.out.name, "out": str(args.out), **summary}
    print(json.dumps(summary))
