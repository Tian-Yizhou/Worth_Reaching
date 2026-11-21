"""Command-line entry point for Worth Reaching experiments."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(prog="wr")
    commands = parser.add_subparsers(required=True)
    pilot = commands.add_parser("pilot", help="Run the CPU KeyCorridor diagnostic")
    pilot.add_argument("--config", type=Path, required=True)
    pilot.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    try:
        from worth_reaching.runner import PilotConfig, run_pilot
    except ModuleNotFoundError as error:
        if error.name not in {"gymnasium", "minigrid"}:
            raise
        parser.error(
            "Install the MiniGrid extra with uv sync --locked --extra minigrid"
        )

    try:
        summary = run_pilot(PilotConfig.load(args.config), args.out)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(json.dumps({"run_id": args.out.name, "out": str(args.out), **summary}))
