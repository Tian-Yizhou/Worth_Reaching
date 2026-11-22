"""Command-line entry point for Worth Reaching experiments."""

import argparse
import json
from pathlib import Path


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


def _train_completion(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.completion import run_completion_training
    from worth_reaching.models.completion import TrainingConfig

    config = TrainingConfig(
        seed=args.seed,
        encoder_updates=args.encoder_updates,
        updates=args.updates,
        history_length=args.history_length,
        queries_per_window=args.queries_per_window,
    )
    return run_completion_training(
        args.out, config, source=args.windows, evaluate_test=args.evaluate_test
    )


def main() -> None:
    parser = argparse.ArgumentParser(prog="wr")
    commands = parser.add_subparsers(required=True)
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
    collection.set_defaults(run=_collect, extra="craftax")
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
    comparison = commands.add_parser(
        "attempt-compare", help="Fit and compare shared-bank selectors"
    )
    for split in ("train", "dev", "test"):
        comparison.add_argument(f"--{split}", type=Path, required=True)
    comparison.add_argument("--out", type=Path, required=True)
    comparison.add_argument("--seed", type=int, default=43301)
    comparison.set_defaults(run=_attempt_compare, extra="craftax")
    args = parser.parse_args()

    try:
        summary = args.run(args)
    except ModuleNotFoundError as error:
        if error.name not in {
            "gymnasium",
            "minigrid",
            "craftax",
            "flax",
            "jax",
            "numpy",
            "optax",
        }:
            raise
        parser.error(
            f"Install experiment dependencies with uv sync --locked --extra {args.extra}"
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"run_id": args.out.name, "out": str(args.out), **summary}))
