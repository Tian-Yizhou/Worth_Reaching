"""Command-line entry point for Worth Reaching experiments."""

import argparse
import json
from pathlib import Path


def _run_pilot(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.keycorridor import PilotConfig, run_pilot

    return run_pilot(PilotConfig.load(args.config), args.out)


def _collect(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.data.collection import CollectionConfig, collect

    config = CollectionConfig.load(
        args.config,
        episodes=args.episodes,
        budget=args.budget,
        seed_start=args.seed_start,
    )
    return collect(config, args.out, expert_checkpoint=args.expert_checkpoint)


def _branch(args: argparse.Namespace) -> dict[str, object]:
    from worth_reaching.evaluation.craftax_branches import BranchConfig, label_branches

    config = BranchConfig(
        split=args.split,
        decisions=args.decisions,
        repeats=args.repeats,
        lead=args.lead,
        horizon=args.horizon,
        seed=args.seed,
    )
    return label_branches(args.dataset, args.out, args.checkpoint, config)


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
    collection.add_argument("--expert-checkpoint", type=Path)
    collection.set_defaults(run=_collect, extra="craftax")
    branching = commands.add_parser(
        "branch", help="Evaluate paired Craftax action branches"
    )
    branching.add_argument("--dataset", type=Path, required=True)
    branching.add_argument("--out", type=Path, required=True)
    branching.add_argument("--checkpoint", type=Path, required=True)
    branching.add_argument("--split", choices=("train", "dev", "test"), default="train")
    branching.add_argument("--decisions", type=int, default=2)
    branching.add_argument("--repeats", type=int, default=2)
    branching.add_argument("--lead", type=int, default=8)
    branching.add_argument("--horizon", type=int, default=16)
    branching.add_argument("--seed", type=int, default=0)
    branching.set_defaults(run=_branch, extra="craftax")
    args = parser.parse_args()

    try:
        summary = args.run(args)
    except ModuleNotFoundError as error:
        if error.name not in {"gymnasium", "minigrid", "craftax", "jax"}:
            raise
        parser.error(
            f"Install experiment dependencies with uv sync --locked --extra {args.extra}"
        )
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"run_id": args.out.name, "out": str(args.out), **summary}))
