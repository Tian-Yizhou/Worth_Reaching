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
        if error.name not in {"gymnasium", "minigrid", "craftax", "jax", "numpy"}:
            raise
        parser.error(
            f"Install experiment dependencies with uv sync --locked --extra {args.extra}"
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"run_id": args.out.name, "out": str(args.out), **summary}))
