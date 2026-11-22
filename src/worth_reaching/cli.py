"""Command-line entry point for Worth Reaching experiments."""

import argparse
import json
from pathlib import Path

from worth_reaching.compute import ComputeConfig
from worth_reaching.envs.ogbench_tasks import PUZZLE_BOARDS, SCENE, TASKS
from worth_reaching.envs.pointmaze_tasks import TASKS as POINTMAZE_TASKS


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
    from worth_reaching.data.collection import CollectionConfig, collect
    from worth_reaching.envs.craftax import TASK

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
    from worth_reaching.data.completion import WindowConfig, prepare_craftax_windows

    config = WindowConfig(
        args.seed, args.windows_per_episode, args.min_span, args.max_span
    )
    return prepare_craftax_windows(args.collection, args.out, config)


def _predictor_stopping(args: argparse.Namespace):
    from worth_reaching.models.training import PredictorStopping

    settings = {
        key: value
        for key, value in (
            ("val_every", args.predictor_val_every),
            ("patience", args.predictor_patience),
        )
        if value is not None
    }
    return PredictorStopping(**settings)


def _train_completion(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.completion import run_completion_training
    from worth_reaching.models.completion import TrainingConfig

    config = TrainingConfig(
        seed=args.seed,
        encoder_updates=args.encoder_updates,
        updates=args.updates,
        history_length=args.history_length,
        queries_per_window=args.queries_per_window,
        predictor_stopping=_predictor_stopping(args),
    )
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


def _antmaze_gcbc(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.antmaze import EvalConfig, TrainConfig, run

    train = TrainConfig(
        seed=args.seed,
        width=args.width,
        batch_size=args.batch_size,
        max_trajectories=args.max_trajectories,
        max_goal_steps=args.max_goal_steps,
    )
    evaluation = EvalConfig(args.eval_split, args.eval_repeats, args.eval_steps)
    return run(
        args.dataset,
        args.data_dir,
        args.out,
        train,
        evaluation,
        updates=args.updates,
        checkpoint_every=args.checkpoint_every,
        resume=args.resume,
        start_bank=args.starts,
    )


def _antmaze_waypoints(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.antmaze import EvalConfig
    from worth_reaching.evaluation.antmaze_waypoints import WaypointConfig, run

    evaluation = EvalConfig(args.eval_split, args.eval_repeats, args.eval_steps)
    waypoint = WaypointConfig(
        args.waypoint_spacing, args.waypoint_steps, args.arrival_radius
    )
    return run(args.checkpoint, args.out, evaluation, waypoint, args.starts)


def _antmaze_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.antmaze import EvalConfig
    from worth_reaching.evaluation.antmaze_latent import run
    from worth_reaching.models.maze_completion import CompletionConfig

    config = CompletionConfig(
        seed=args.seed,
        windows=args.windows,
        history=args.history,
        queries=args.queries,
        codec_updates=args.codec_updates,
        predictor_updates=args.predictor_updates,
        flow_updates=args.flow_updates,
        predictor_stopping=_predictor_stopping(args),
        candidates=args.candidates,
    )
    evaluation = EvalConfig(args.eval_split, args.eval_repeats, args.eval_steps)
    return run(
        args.dataset,
        args.data_dir,
        args.checkpoint,
        args.out,
        config,
        evaluation,
        args.starts,
        probe_first_candidates=args.probe_first_candidates,
        resume=args.resume,
    )


def _pointmaze_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.pointmaze import run
    from worth_reaching.models.maze_completion import CompletionConfig

    return run(
        args.data_dir,
        args.out,
        CompletionConfig(
            seed=args.seed,
            windows=args.windows,
            codec_updates=args.codec_updates,
            predictor_updates=args.predictor_updates,
            flow_updates=args.flow_updates,
            predictor_stopping=_predictor_stopping(args),
        ),
        task_name=args.task,
        split=args.eval_split,
        repeats=args.eval_repeats,
        limit=args.eval_steps,
        resume=args.resume,
    )


def _pointmaze_first_targets(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.pointmaze_probe import run

    return run(
        args.data_dir,
        args.runs,
        args.out,
        args.max_starts,
        compare_continuations=args.compare_continuations,
    )


def _pointmaze_latent_energy(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.pointmaze_energy import run

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


def _kuka_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.kuka import run
    from worth_reaching.models.maze_completion import CompletionConfig

    return run(
        args.source_dir,
        args.data_dir,
        args.out,
        CompletionConfig(
            seed=args.seed,
            windows=args.windows,
            max_horizon=256,
            latent_size=24,
            codec_updates=args.codec_updates,
            predictor_updates=args.predictor_updates,
            flow_updates=args.flow_updates,
            predictor_stopping=_predictor_stopping(args),
        ),
        split=args.eval_split,
        repeats=args.eval_repeats,
        limit=args.eval_steps,
        resume=args.resume,
    )


def _manipulation_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.ogbench_manipulation import ExecutorConfig, run
    from worth_reaching.models.maze_completion import CompletionConfig

    benchmark = PUZZLE_BOARDS[args.board] if args.board is not None else SCENE
    return run(
        benchmark,
        args.data_dir,
        args.out,
        CompletionConfig(
            seed=args.seed,
            windows=args.windows,
            max_horizon=benchmark.action_limit,
            codec_updates=args.codec_updates,
            predictor_updates=args.predictor_updates,
            flow_updates=args.flow_updates,
            predictor_stopping=_predictor_stopping(args),
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
    training.add_argument("--seed", type=int, default=7)
    training.add_argument("--encoder-updates", type=int, default=250)
    training.add_argument("--updates", type=int, default=600)
    training.add_argument("--history-length", type=int, default=16)
    training.add_argument("--queries-per-window", type=int, default=16)
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
    antmaze = commands.add_parser(
        "antmaze-gcbc",
        help="Train goal-conditioned cloning and score saved AntMaze starts",
    )
    antmaze.add_argument("--dataset", default="antmaze-large-navigate-v0")
    antmaze.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/ogbench")
    )
    antmaze.add_argument("--out", type=Path, required=True)
    antmaze.add_argument("--seed", type=int, default=7)
    antmaze.add_argument("--updates", type=int, default=1000)
    antmaze.add_argument("--checkpoint-every", type=int, default=1000)
    antmaze.add_argument("--width", type=int, default=256)
    antmaze.add_argument("--batch-size", type=int, default=256)
    antmaze.add_argument("--max-trajectories", type=int)
    antmaze.add_argument("--max-goal-steps", type=int)
    antmaze.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    antmaze.add_argument("--eval-repeats", type=int, default=4)
    antmaze.add_argument("--eval-steps", type=int, default=1000)
    antmaze.add_argument(
        "--starts", type=Path, help="Reuse a matching saved AntMaze start bank"
    )
    antmaze.add_argument(
        "--resume", type=Path, help="Continue from a compatible training run"
    )
    antmaze.set_defaults(run=_antmaze_gcbc, extra="ogbench")
    waypoints = commands.add_parser(
        "antmaze-waypoints",
        help="Compare direct and oracle waypoint execution with frozen GCBC",
    )
    waypoints.add_argument("--checkpoint", type=Path, required=True)
    waypoints.add_argument("--out", type=Path, required=True)
    waypoints.add_argument(
        "--starts", type=Path, help="Reuse a matching AntMaze start bank"
    )
    waypoints.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    waypoints.add_argument("--eval-repeats", type=int, default=4)
    waypoints.add_argument("--eval-steps", type=int, default=1000)
    waypoints.add_argument("--waypoint-spacing", type=int, default=1)
    waypoints.add_argument("--waypoint-steps", type=int, default=150)
    waypoints.add_argument("--arrival-radius", type=float, default=1.0)
    waypoints.set_defaults(run=_antmaze_waypoints, extra="ogbench")
    latent = commands.add_parser(
        "antmaze-latent",
        help="Train masked latent completion and execute decoded AntMaze candidates",
    )
    latent.add_argument("--dataset", default="antmaze-large-navigate-v0")
    latent.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/ogbench")
    )
    latent.add_argument(
        "--checkpoint", type=Path, required=True, help="Frozen GCBC executor run"
    )
    latent.add_argument("--out", type=Path, required=True)
    latent.add_argument(
        "--starts", type=Path, help="Reuse a matching AntMaze start bank"
    )
    latent.add_argument("--seed", type=int, default=7)
    latent.add_argument("--windows", type=int, default=16000)
    latent.add_argument("--history", type=int, default=4)
    latent.add_argument("--queries", type=int, default=8)
    latent.add_argument("--candidates", type=int, default=4)
    latent.add_argument(
        "--probe-first-candidates",
        action="store_true",
        help="Compare each sampled first target under a shared closed-loop continuation",
    )
    latent.add_argument("--codec-updates", type=int, default=500)
    latent.add_argument("--predictor-updates", type=int, default=1200)
    latent.add_argument("--flow-updates", type=int, default=700)
    latent.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    latent.add_argument("--eval-repeats", type=int, default=2)
    latent.add_argument("--eval-steps", type=int, default=1000)
    latent.set_defaults(run=_antmaze_latent, extra="ogbench")
    pointmaze = commands.add_parser(
        "pointmaze-latent",
        help="Train and evaluate latent completion on Maze2D U-Maze, Medium or Large",
    )
    pointmaze.add_argument("--task", choices=POINTMAZE_TASKS, default="medium")
    pointmaze.add_argument(
        "--data-dir", type=Path, default=Path("results/datasets/minari")
    )
    pointmaze.add_argument("--out", type=Path, required=True)
    pointmaze.add_argument("--seed", type=int, default=7)
    pointmaze.add_argument("--windows", type=int, default=4000)
    pointmaze.add_argument("--codec-updates", type=int, default=250)
    pointmaze.add_argument("--predictor-updates", type=int, default=600)
    pointmaze.add_argument("--flow-updates", type=int, default=300)
    pointmaze.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    pointmaze.add_argument("--eval-repeats", type=int, default=6)
    pointmaze.add_argument(
        "--eval-steps",
        type=int,
        help="Shared training span and evaluation budget (default determined from the archive)",
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
    kuka.add_argument("--seed", type=int, default=7)
    kuka.add_argument("--windows", type=int, default=4000)
    kuka.add_argument("--codec-updates", type=int, default=250)
    kuka.add_argument("--predictor-updates", type=int, default=600)
    kuka.add_argument("--flow-updates", type=int, default=300)
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
        command.add_argument("--seed", type=int, default=7)
        command.add_argument("--windows", type=int, default=8000)
        command.add_argument("--executor-updates", type=int, default=2500)
        command.add_argument("--codec-updates", type=int, default=350)
        command.add_argument("--predictor-updates", type=int, default=1000)
        command.add_argument("--flow-updates", type=int, default=450)
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
