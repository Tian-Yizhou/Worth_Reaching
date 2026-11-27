"""Plan with latent options and a budget-conditioned value in Maze2D.

A frozen pointmaze-latent run supplies the world model. This runner trains a latent-goal
GCIVL executor on frozen encodings of archive states and an option value on hindsight
options from the same archive, or reuses both from an earlier run. Each arm then plays
the official resets in closed loop. Nothing defines intermediate goals or arrival: the
planner picks among sampled latent futures and the encoded terminal goal, and the
executor pursues the choice until the planner switches or samples again.

Direct tracking of the terminal goal with Diffuser's position and velocity tracker is
the floor and a privileged shortest-path oracle is the ceiling, as in E1. Direct
pursuit of the encoded goal with the latent executor isolates the executor.
"""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from hashlib import sha256
from importlib.metadata import version
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import torch

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import StateArchive
from worth_reaching.evaluation import options as shared
from worth_reaching.evaluation.closed_loop import (
    ORACLE,
    ClosedLoop,
    Controls,
    RegionModels,
)
from worth_reaching.models.gcivl import GCIVL, GCIVLConfig, fit_gcivl
from worth_reaching.models.option_value import (
    NEVER,
    OptionValue,
    OptionValueConfig,
    fit_option_value,
    label_options,
)
from worth_reaching.models.training import (
    describe,
    load_weights,
    restore_weights,
    save_weights,
    stream_seed,
    weights,
)
from worth_reaching.planning.options import (
    OptionPlanner,
    PlannerConfig,
    surprise_threshold,
)
from worth_reaching.tasks.pointmaze.dataset import (
    ACTION_SIZE,
    OBSERVATION_SIZE,
    load_dataset,
    split_trajectories,
)
from worth_reaching.tasks.pointmaze.energy import FrozenRun
from worth_reaching.tasks.pointmaze.environment import (
    GOAL_RADIUS,
    SEED_BASE,
    PointMaze,
)
from worth_reaching.tasks.pointmaze.experiment import rollout, track
from worth_reaching.tasks.pointmaze.geometry import MazeGeometry
from worth_reaching.tasks.pointmaze.spec import TASKS

SCHEMA = "pointmaze_options_v1"
WEIGHTS_FILE = "option-weights.pt"


@dataclass(frozen=True, slots=True)
class MazeOptionConfig:
    executor: GCIVLConfig = field(default_factory=lambda: GCIVLConfig(updates=300_000))
    value: OptionValueConfig = field(default_factory=OptionValueConfig)
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    eval_costs: tuple[float, ...] = (0.0, 0.001)
    """Time costs evaluated with the full planner; ablations use the last."""
    surprise_quantile: float = 0.95
    oracle_cells: int = 1
    oracle_speed: float = 2.0

    def __post_init__(self) -> None:
        if (
            not self.eval_costs
            or not set(self.eval_costs) <= set(self.value.time_costs)
            or not 0 < self.surprise_quantile < 1
            or self.oracle_cells < 1
            or self.oracle_speed <= 0
        ):
            raise ValueError(
                "Evaluate time costs the value was trained on, with a surprise "
                "quantile in (0, 1) and a positive oracle"
            )

    @property
    def recipe(self) -> dict:
        """The settings that determine the trained executor and value."""
        settings = {"executor": asdict(self.executor), "value": asdict(self.value)}
        return json.loads(json.dumps(settings))


def reached(states: np.ndarray, goals: np.ndarray) -> np.ndarray:
    """PointMaze's sparse success predicate on public positions."""
    return np.linalg.norm(states[..., :2] - goals[..., :2], axis=-1) <= GOAL_RADIUS


def arms(config: MazeOptionConfig) -> dict[str, PlannerConfig | str]:
    """Bounds, direct latent pursuit, the planner per time cost and its ablations."""
    planner = config.planner
    cost = config.eval_costs[-1]
    result: dict[str, PlannerConfig | str] = {
        shared.DIRECT: shared.DIRECT,
        ORACLE: ORACLE,
        shared.DIRECT_LATENT: replace(
            planner, time_cost=cost, subgoals=False, goal_candidate=True
        ),
    }
    for value in config.eval_costs:
        result[f"options@{value:g}"] = replace(planner, time_cost=value)
    result[f"no_goal@{cost:g}"] = replace(planner, time_cost=cost, goal_candidate=False)
    result[f"no_interrupt@{cost:g}"] = replace(planner, time_cost=cost, interrupt=False)
    result[f"fixed_duration@{cost:g}"] = replace(planner, time_cost=cost, slots=(0,))
    return result


def _archive(trajectories, fingerprint: str) -> StateArchive:
    finals = np.zeros(len(trajectories.observations), dtype=bool)
    finals[trajectories.ends] = True
    return StateArchive(
        trajectories.observations,
        trajectories.actions,
        trajectories.ends,
        np.flatnonzero(~finals),
        fingerprint,
    )


def _train(frozen: FrozenRun, train, validation, limit, config, out, fingerprint):
    model = frozen.model
    seed = model.config.seed
    latents = model.encode_observations(train.observations)
    executor = fit_gcivl(
        _archive(train, fingerprint),
        ACTION_SIZE,
        config.executor,
        seed=seed,
        out=out / "checkpoints" / "executor",
        goal_features=latents,
    )
    options = label_options(
        train,
        reached,
        config.value.options,
        max_budget=limit,
        max_horizon=model.config.max_horizon,
        queries=model.config.queries,
        config=config.value,
        rng=np.random.default_rng([seed, 7100]),
    )
    value = fit_option_value(
        options,
        latents,
        limit,
        config.value,
        seed=seed,
        out=out / "checkpoints" / "value",
    )
    save_weights(
        out / WEIGHTS_FILE, {"executor": weights(executor), "value": weights(value)}
    )
    arrived = options.first <= np.minimum(options.durations, options.budgets)
    return (
        executor,
        value,
        {
            "labeled_options": len(options.first),
            "labeled_arrival_share": float(arrived.mean()),
            "labeled_goal_reached_share": float((options.first < NEVER).mean()),
        },
    )


def _load(trained: Path, frozen: FrozenRun, limit: int, config: MazeOptionConfig):
    manifest = json.loads((trained / "manifest.json").read_text())
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"Expected a pointmaze-options run in {trained}")
    if manifest["recipe"] != config.recipe:
        raise ValueError(f"The executor or value recipe differs from {trained}")
    if manifest["source_weights_sha256"] != frozen.weights_sha256:
        raise ValueError(f"{trained} was trained on a different world model")
    values = load_weights(trained / WEIGHTS_FILE)
    latent_size = frozen.model.config.latent_size
    executor = GCIVL(OBSERVATION_SIZE, latent_size, ACTION_SIZE, config.executor)
    value = OptionValue(latent_size, limit, config.value)
    restore_weights(executor, values["executor"])
    restore_weights(value, values["value"])
    return executor, value, manifest.get("training", {})


_WORKER: dict = {}


def _start_worker(
    data_dir, task_name, completion, executor, value, arm_configs, config, limit
):
    torch.set_num_threads(1)
    task = TASKS[task_name]
    dataset, _ = load_dataset(data_dir, task.dataset_id)
    native = dataset.recover_environment(eval_env=True)
    env, geometry = PointMaze(native, task), MazeGeometry(native.unwrapped.maze)
    _WORKER.update(
        env=env,
        completion=completion,
        executor=executor,
        value=value,
        arms=arm_configs,
        limit=limit,
        oracle=Controls(
            act=track,
            act_latent=track,
            terminal=track,
            arrived=env.arrived,
            oracle=lambda observation, goal: geometry.waypoint(
                observation, goal, config.oracle_cells, config.oracle_speed
            ),
        ),
    )


def _policy(arm: str, seed: int):
    worker, setting = _WORKER, _WORKER["arms"][arm]
    completion, limit = worker["completion"], worker["limit"]
    key = stream_seed(completion.config.seed, seed, list(worker["arms"]).index(arm))
    if setting == shared.DIRECT:
        return track
    if setting == ORACLE:
        return ClosedLoop(
            RegionModels(completion, None),
            worker["oracle"],
            ORACLE,
            "forecast",
            key,
            limit,
            stages=1,
            region_samples=1,
            action_size=ACTION_SIZE,
        )
    return OptionPlanner(
        completion,
        worker["value"],
        worker["executor"].act,
        key,
        limit,
        setting,
        action_size=ACTION_SIZE,
    )


def _play(batch: list[tuple[str, int]]) -> list[dict]:
    rows = []
    for arm, seed in batch:
        policy = _policy(arm, seed)
        outcome, _ = rollout(_WORKER["env"], seed, _WORKER["limit"], policy)
        rows.append(
            shared.record(
                arm,
                seed,
                None,
                policy,
                outcome["success"],
                outcome["steps"],
                outcome["first_success_step"],
            )
        )
    return rows


def run(
    data_dir: Path,
    run_path: Path,
    out: Path,
    config: MazeOptionConfig,
    *,
    split: str,
    repeats: int,
    workers: int = 1,
    trained: Path | None = None,
) -> dict:
    frozen = FrozenRun.load(run_path)
    manifest = frozen.manifest
    limit = manifest["evaluation"]["action_limit"]
    if not 0 < repeats <= SEED_BASE["test"] - SEED_BASE["dev"] or workers < 1:
        raise ValueError("Use a positive number of reset seeds and workers")
    task = TASKS[manifest["task"]]
    dataset, hashes = load_dataset(data_dir, task.dataset_id)
    if hashes != manifest["dataset_files_sha256"]:
        raise ValueError("Dataset files disagree with the frozen run")
    train, validation, _ = split_trajectories(dataset)
    out.mkdir(parents=True, exist_ok=False)
    if trained is None:
        executor, value, training = _train(
            frozen, train, validation, limit, config, out, hashes["main_data.hdf5"]
        )
        weights_path = out / WEIGHTS_FILE
    else:
        executor, value, training = _load(trained, frozen, limit, config)
        weights_path = trained / WEIGHTS_FILE
    model = frozen.model
    windows = validation.sample_windows(
        np.random.default_rng(801),
        512,
        history=model.config.history,
        queries=model.config.queries,
        max_horizon=model.config.max_horizon,
        max_action_budget=model.config.max_action_budget,
    )
    surprise = surprise_threshold(
        model, windows, stream_seed(model.config.seed, 7200), config.surprise_quantile
    )
    config = replace(config, planner=replace(config.planner, surprise=surprise))
    arm_configs = arms(config)
    seeds = list(range(SEED_BASE[split], SEED_BASE[split] + repeats))
    write_json(
        out / "manifest.json",
        {
            "schema": SCHEMA,
            "task": manifest["task"],
            "dataset": task.dataset_id,
            "dataset_files_sha256": hashes,
            "versions": {
                name: version(name)
                for name in (
                    "minari",
                    "gymnasium-robotics",
                    "gymnasium",
                    "mujoco",
                    "torch",
                )
            },
            "source_run": str(frozen.directory),
            "source_weights_sha256": frozen.weights_sha256,
            "trained_run": str(trained) if trained else None,
            "weights_sha256": sha256(weights_path.read_bytes()).hexdigest(),
            "recipe": config.recipe,
            "training": training,
            "planner": asdict(config.planner),
            "surprise_threshold": surprise,
            "action_limit": limit,
            "split": split,
            "seeds": seeds,
            "arms": {
                name: setting if isinstance(setting, str) else asdict(setting)
                for name, setting in arm_configs.items()
            },
            "compute": {**describe(), "eval_workers": workers},
            "code_revision": code_revision(),
        },
    )
    planned = [(arm, seed) for seed in seeds for arm in arm_configs]
    shards = [planned[index::workers] for index in range(workers)]
    with ProcessPoolExecutor(
        len(shards),
        mp_context=get_context("spawn"),
        initializer=_start_worker,
        initargs=(
            data_dir,
            manifest["task"],
            model,
            executor,
            value,
            arm_configs,
            config,
            limit,
        ),
    ) as pool:
        rows = [row for shard in pool.map(_play, shards) for row in shard]
    rows.sort(key=lambda row: (row["arm"], row["seed"]))
    with gzip.open(out / "attempts.jsonl.gz", "wt", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, separators=(",", ":")) + "\n")
    summary = {
        "training": training,
        "surprise_threshold": surprise,
        "evaluation_environment_steps": sum(row["steps"] for row in rows),
        **shared.summarize(rows),
    }
    write_json(out / "summary.json", summary)
    return summary
