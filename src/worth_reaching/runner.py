"""Run and record the KeyCorridor measurement pilot."""

import json
import subprocess
import tomllib
from contextlib import closing
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path

from worth_reaching.contracts import Action
from worth_reaching.envs.minigrid import (
    CONTINUATION_ID,
    ENVIRONMENT_ID,
    TASK,
    KeyCorridorEpisode,
    continue_toward_ball,
)
from worth_reaching.evaluation.branches import Attempt, evaluate_action, replay_prefix


@dataclass(frozen=True, slots=True)
class PilotConfig:
    seed: int
    budget: int
    prefix: tuple[Action, ...]
    interventions: tuple[Action, ...]

    @classmethod
    def load(cls, path: Path) -> "PilotConfig":
        with path.open("rb") as stream:
            values = tomllib.load(stream)
        if values.keys() != {"seed", "budget", "prefix", "interventions"}:
            raise ValueError(
                "Config must contain seed, budget, prefix and interventions"
            )

        seed, budget = values["seed"], values["budget"]
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if type(budget) is not int or budget <= 0:
            raise ValueError("budget must be a positive integer")
        if not isinstance(values["prefix"], list) or not isinstance(
            values["interventions"], list
        ):
            raise ValueError("prefix and interventions must be action arrays")

        prefix = tuple(Action(value) for value in values["prefix"])
        interventions = tuple(Action(value) for value in values["interventions"])
        if len(prefix) >= budget:
            raise ValueError("The configured prefix must leave at least one action")
        if len(interventions) != 2 or len(set(interventions)) != 2:
            raise ValueError("interventions must contain two distinct actions")
        return cls(seed, budget, prefix, interventions)


def _code_revision() -> dict[str, str | bool | None]:
    repo = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}
    return {"commit": revision, "dirty": bool(status)}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run_pilot(config: PilotConfig, out: Path) -> dict[str, object]:
    if out.exists():
        raise FileExistsError(f"Output directory already exists: {out}")

    code_revision = _code_revision()
    with closing(KeyCorridorEpisode(seed=config.seed, budget=config.budget)) as episode:
        recorded = replay_prefix(episode, config.prefix)

    attempts: list[Attempt] = []
    for action in config.interventions:
        with closing(
            KeyCorridorEpisode(seed=config.seed, budget=config.budget)
        ) as episode:
            replayed = replay_prefix(episode, config.prefix)
            if replayed != recorded:
                raise RuntimeError(
                    "Replaying the starting prefix changed the public trace"
                )
            attempts.append(
                evaluate_action(
                    episode,
                    replayed,
                    action,
                    budget=config.budget,
                    continuation=continue_toward_ball,
                )
            )

    out.mkdir(parents=True)
    _write_json(
        out / "manifest.json",
        {
            "schema_version": 1,
            "run_id": out.name,
            "environment": ENVIRONMENT_ID,
            "environment_version": version("minigrid"),
            "gymnasium_version": version("gymnasium"),
            "task": TASK,
            "continuation_policy": CONTINUATION_ID,
            "sampling": "exact_replay",
            "config": asdict(config),
            "code_revision": code_revision,
            "environment_resets": 1 + len(attempts),
            "prefix_recording_steps": len(config.prefix),
            "prefix_replay_steps": len(config.prefix) * len(attempts),
            "branch_steps": sum(attempt.steps_elapsed for attempt in attempts),
        },
    )
    _write_json(out / "prefix.json", asdict(recorded))
    with (out / "attempts.jsonl").open("w", encoding="utf-8") as stream:
        for attempt in attempts:
            stream.write(json.dumps(asdict(attempt)) + "\n")

    summary = {
        "attempts": len(attempts),
        "successes": sum(attempt.success for attempt in attempts),
        "outcomes": {attempt.intervention: attempt.outcome for attempt in attempts},
    }
    _write_json(out / "summary.json", summary)
    return summary
