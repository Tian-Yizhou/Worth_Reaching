"""Command-line interface for AntMaze experiments."""

import argparse
from pathlib import Path

from worth_reaching.recipes import completion_recipe


def _antmaze_gcbc(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.tasks.antmaze.baseline import EvalConfig, TrainConfig, run

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
    from worth_reaching.tasks.antmaze.baseline import EvalConfig
    from worth_reaching.tasks.antmaze.waypoints import WaypointConfig, run

    evaluation = EvalConfig(args.eval_split, args.eval_repeats, args.eval_steps)
    waypoint = WaypointConfig(
        args.waypoint_spacing, args.waypoint_steps, args.arrival_radius
    )
    return run(args.checkpoint, args.out, evaluation, waypoint, args.starts)


def _antmaze_latent(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.models.continuous_completion import CompletionConfig
    from worth_reaching.tasks.antmaze.baseline import EvalConfig
    from worth_reaching.tasks.antmaze.experiment import run

    config = completion_recipe(CompletionConfig(), args)
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


def register_commands(commands) -> argparse.ArgumentParser:
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
    latent.add_argument("--seed", type=int)
    latent.add_argument("--windows", type=int)
    latent.add_argument("--history", type=int)
    latent.add_argument("--queries", type=int)
    latent.add_argument("--candidates", type=int)
    latent.add_argument(
        "--probe-first-candidates",
        action="store_true",
        help="Compare each sampled first target under a shared closed-loop continuation",
    )
    latent.add_argument("--codec-updates", type=int)
    latent.add_argument("--predictor-updates", type=int)
    latent.add_argument("--flow-updates", type=int)
    latent.add_argument("--eval-split", choices=("dev", "test"), default="dev")
    latent.add_argument("--eval-repeats", type=int, default=2)
    latent.add_argument("--eval-steps", type=int, default=1000)
    latent.set_defaults(run=_antmaze_latent, extra="ogbench")
    return latent
