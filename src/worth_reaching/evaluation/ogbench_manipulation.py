"""Train a shared OGBench manipulation executor and test decoded plans."""

import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
from hashlib import sha256
from importlib.metadata import version
from itertools import product
from multiprocessing import get_context
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from flax import serialization

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import StateArchive, load_state_archive
from worth_reaching.models import gcbc
from worth_reaching.models.continuous_completion import (
    CompletionConfig,
    LearnedCompletion,
    fit,
)
from worth_reaching.models.training import (
    PredictorStopping,
    describe,
    regression,
    restore_weights,
    stage_resume,
    stream_seed,
    train_stage,
    validate_resume,
    weights,
)
from worth_reaching.planning.receding_goals import Policy, RecedingGoals
from worth_reaching.tasks.ogbench_manipulation.environment import (
    STATE_SPEC,
    Manipulation,
    ManipulationStart,
)
from worth_reaching.tasks.ogbench_manipulation.spec import (
    ACTION_SIZE,
    SCENE,
    SEED_OFFSETS,
    TASKS,
    ManipulationTask,
)


@dataclass(frozen=True, slots=True)
class ExecutorConfig:
    updates: int = 2500
    width: int = 256
    batch_size: int = 256
    learning_rate: float = 0.0003
    short_goal_steps: int = 64

    def __post_init__(self) -> None:
        if self.updates < 1:
            raise ValueError("Use at least one executor update")


def _fit_executor(
    train: StateArchive,
    normalization: tuple[np.ndarray, np.ndarray],
    seed: int,
    config: ExecutorConfig,
    out: Path,
    resume: Path | None,
) -> tuple[Policy, dict]:
    mean, scale = normalization
    torch.manual_seed(seed)
    model = train_stage(
        gcbc.GCBC(config.width, len(mean), ACTION_SIZE, relative_coordinates=0),
        gcbc.BehaviorBatches(
            train,
            mean,
            scale,
            (seed, 600),
            config.batch_size,
            config.short_goal_steps,
            alternate_horizons=True,
        ),
        regression,
        steps=config.updates,
        learning_rate=config.learning_rate,
        out=out,
        resume=resume,
    )
    return gcbc.policy_from_model(model, mean, scale), weights(model)


def rollout(
    env: Manipulation, initial: np.ndarray, goal: np.ndarray, policy: Policy, limit: int
) -> tuple[dict, np.ndarray, np.ndarray]:
    observation = initial
    states, actions = [observation], []
    for step in range(1, limit + 1):
        action = policy(observation, goal)
        observation, success, done = env.step(action)
        states.append(observation)
        actions.append(action)
        if success or done:
            break
    return {"success": success, "steps": step}, np.stack(states), np.stack(actions)


def _rollout_cases(
    benchmark: ManipulationTask,
    cases: list[tuple[int, int]],
    actor: Policy,
    model: LearnedCompletion,
    limit: int,
):
    env = Manipulation(benchmark)
    try:
        for task, seed in cases:
            start = env.capture(task, seed)
            planner = RecedingGoals(
                actor,
                model,
                stream_seed(model.config.seed, task, seed),
                limit,
                action_size=ACTION_SIZE,
            )
            direct = rollout(env, start.observation, start.goal, actor, limit)
            latent = rollout(env, env.restore(start), start.goal, planner, limit)
            yield start, {"direct": direct, "latent": latent}, planner.plans
    finally:
        env.close()


def _evaluate_shard(
    cases: list[tuple[int, int]],
    *,
    benchmark: ManipulationTask,
    config: CompletionConfig,
    executor: ExecutorConfig,
    weights_path: Path,
    limit: int,
):
    torch.set_num_threads(1)
    actor, model = _restore_models(
        weights_path.read_bytes(), benchmark, config, executor
    )
    return list(_rollout_cases(benchmark, cases, actor, model, limit))


def _restore_models(packed, benchmark, config, executor):
    values = serialization.msgpack_restore(packed)
    model = LearnedCompletion.from_weights(config, values)
    actor_model = gcbc.GCBC(
        executor.width, benchmark.observation_size, ACTION_SIZE, relative_coordinates=0
    )
    actor = gcbc.policy_from_model(
        restore_weights(actor_model, values["executor"]), model.mean, model.scale
    )
    return actor, model


def _cpu_shards(cases, worker_count, **settings):
    shards = [cases[i::worker_count] for i in range(worker_count)]
    evaluate = partial(_evaluate_shard, **settings)
    with ProcessPoolExecutor(worker_count, mp_context=get_context("spawn")) as pool:
        indexed = {
            (start.task, start.seed): (start, arms, plans)
            for shard in pool.map(evaluate, shards)
            for start, arms, plans in shard
        }
    yield from (indexed[key] for key in cases)


def _warp_arm(batch, starts, policies, limit):
    batch.reset()
    paths = [[start.observation] for start in starts]
    commands = [[] for _ in starts]
    success = [False for _ in starts]
    for _ in range(limit):
        actions = {
            i: policy(paths[i][-1], start.goal)
            for i, (policy, start) in enumerate(zip(policies, starts, strict=True))
            if not success[i]
        }
        if not actions:
            break
        for i, (observation, reached) in batch.step(actions).items():
            success[i] = reached
            paths[i].append(observation)
            commands[i].append(actions[i])
    return [
        ({"success": hit, "steps": len(actions)}, np.stack(path), np.stack(actions))
        for hit, path, actions in zip(success, paths, commands, strict=True)
    ]


def _warp_cases(cases, actor, model, limit, batch_size, device):
    from worth_reaching.tasks.ogbench_manipulation.scene_warp import SceneWarpBatch

    envs = [Manipulation(SCENE) for _ in range(min(batch_size, len(cases)))]
    batch = None
    try:
        for offset in range(0, len(cases), batch_size):
            chunk = cases[offset : offset + batch_size]
            active_envs = envs[: len(chunk)]
            starts = [
                env.capture(*key) for env, key in zip(active_envs, chunk, strict=True)
            ]
            if batch is None:
                batch = SceneWarpBatch(envs, device)
            direct = _warp_arm(batch, starts, [actor] * len(starts), limit)
            planners = [
                RecedingGoals(
                    actor,
                    model,
                    stream_seed(model.config.seed, task, seed),
                    limit,
                    action_size=ACTION_SIZE,
                )
                for task, seed in chunk
            ]
            for env, start in zip(active_envs, starts, strict=True):
                env.restore(start)
            latent = _warp_arm(batch, starts, planners, limit)
            for start, direct_result, latent_result, planner in zip(
                starts, direct, latent, planners, strict=True
            ):
                yield (
                    start,
                    {"direct": direct_result, "latent": latent_result},
                    planner.plans,
                )
    finally:
        for env in envs:
            env.close()


def _summarize(episodes: list[dict], tasks: tuple[int, ...]) -> dict:
    return {
        "episodes": len(episodes),
        "successes": sum(row["success"] for row in episodes),
        "success_rate": float(np.mean([row["success"] for row in episodes])),
        "per_task": {
            str(task): {
                "episodes": sum(row["task"] == task for row in episodes),
                "successes": sum(
                    row["task"] == task and row["success"] for row in episodes
                ),
            }
            for task in tasks
        },
    }


def run(
    benchmark: ManipulationTask,
    data_dir: Path,
    out: Path,
    config: CompletionConfig,
    executor: ExecutorConfig,
    *,
    split: str,
    task: int | tuple[int, ...] | None,
    repeats: int,
    limit: int | None,
    resume: Path | None = None,
    eval_workers: int = 1,
) -> dict:
    if limit is None:
        limit = benchmark.action_limit
    if (
        not config.queries < limit <= benchmark.action_limit
        or not 0 < repeats <= SEED_OFFSETS["test"]
    ):
        raise ValueError(
            f"Choose an action budget in {config.queries + 1}..{benchmark.action_limit} "
            f"for {benchmark.environment} and disjoint reset seeds"
        )
    if eval_workers < 1:
        raise ValueError("Use at least one evaluation worker")
    train, validation = [
        load_state_archive(
            data_dir / f"{benchmark.dataset}{suffix}.npz",
            observation_size=benchmark.observation_size,
            action_size=ACTION_SIZE,
        )
        for suffix in ("", "-val")
    ]
    if train.fingerprint == validation.fingerprint:
        raise ValueError("Training and validation must use distinct OGBench archives")
    tasks = TASKS if task is None else (task,) if isinstance(task, int) else task
    start_seed = benchmark.seed_base + SEED_OFFSETS[split]
    seeds = range(start_seed, start_seed + repeats)
    cases = list(product(tasks, seeds))
    worker_count = min(eval_workers, len(cases))
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": f"ogbench_{benchmark.family}_latent_v2",
            "environment": benchmark.environment,
            "observation": "official_state",
            "versions": {
                name: version(name) for name in ("ogbench", "gymnasium", "mujoco")
            },
            "dataset": benchmark.dataset,
            "train_sha256": train.fingerprint,
            "validation_sha256": validation.fingerprint,
            "training": asdict(config),
            "executor": asdict(executor),
            "evaluation": {
                "split": split,
                "tasks": tasks,
                "seeds": list(seeds),
                "action_limit": limit,
            },
            "selection": "first sampled route, earliest queried state",
            "compute": {**describe(), "eval_workers": worker_count},
            "code_revision": code_revision(),
        },
    )
    mean = train.observations.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(train.observations.std(axis=0, dtype=np.float64), 1e-3).astype(
        np.float32
    )
    previous = validate_resume(
        out, resume, legacy_defaults={"training": {"learning_rate": 0.001}}
    )
    actor, actor_weights = _fit_executor(
        train,
        (mean, scale),
        config.seed,
        executor,
        out / "checkpoints" / "executor",
        stage_resume(previous, "executor"),
    )
    settings = dict(
        history=config.history, queries=config.queries, max_horizon=config.max_horizon
    )
    windows = train.sample_windows(
        np.random.default_rng([config.seed, 800]), config.windows, **settings
    )
    heldout = validation.sample_windows(np.random.default_rng(801), 512, **settings)
    model = fit(windows, heldout, mean, scale, config, out / "checkpoints", previous)
    metrics = model.validate(heldout, compare_xy=False)
    for label, cap in (("short", executor.short_goal_steps), ("all", None)):
        batch = validation.sample(np.random.default_rng(803), 1024, max_goal_steps=cap)
        errors = actor(batch.observations, batch.goals) - batch.actions
        metrics[f"executor_{label}_action_mse"] = float(np.square(errors).mean())
    write_json(out / "validation.json", metrics)
    weights_path = out / "weights.msgpack"
    weights_path.write_bytes(
        serialization.to_bytes(
            {
                **model.weights,
                "executor": actor_weights,
                "mean": mean,
                "scale": scale,
                "flow_scale": model.flow_scale.numpy(),
            }
        )
    )

    if worker_count == 1:
        results = _rollout_cases(benchmark, cases, actor, model, limit)
    else:
        results = _cpu_shards(
            cases,
            worker_count,
            benchmark=benchmark,
            config=config,
            executor=executor,
            weights_path=weights_path,
            limit=limit,
        )
    return _write_evaluation(out, tasks, results, metrics=metrics)


def _write_evaluation(
    out: Path, tasks: tuple[int, ...], results, *, metrics=None
) -> dict:
    episodes = {"direct": [], "latent": []}
    starts: list[ManipulationStart] = []
    trajectories = {}
    began = perf_counter()
    with gzip.open(out / "plans.jsonl.gz", "wt", encoding="utf-8") as stream:
        for start, arms, plans in results:
            starts.append(start)
            task_id, seed = start.task, start.seed
            for arm, (result, states, actions) in arms.items():
                episodes[arm].append({"task": task_id, "seed": seed, **result})
                trajectories[f"{arm}_{task_id}_{seed}_observations"] = states
                trajectories[f"{arm}_{task_id}_{seed}_actions"] = actions
            for plan in plans:
                stream.write(
                    json.dumps(
                        {"task": task_id, "seed": seed, **plan}, separators=(",", ":")
                    )
                    + "\n"
                )
    np.savez_compressed(
        out / "starts.npz",
        ids=[(s.task, s.seed) for s in starts],
        observation=np.stack([s.observation for s in starts]),
        goal=np.stack([s.goal for s in starts]),
        physics=np.stack([s.physics for s in starts]),
        state_spec=STATE_SPEC,
    )
    np.savez_compressed(out / "trajectories.npz", **trajectories)
    write_json(out / "episodes.json", episodes)
    paired = list(zip(episodes["direct"], episodes["latent"], strict=True))
    summary = {
        **({"validation": metrics} if metrics is not None else {}),
        **{arm: _summarize(rows, tasks) for arm, rows in episodes.items()},
        "only_direct": sum(a["success"] and not b["success"] for a, b in paired),
        "only_latent": sum(b["success"] and not a["success"] for a, b in paired),
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(a["steps"] + b["steps"] for a, b in paired),
        "evaluation_wall_seconds": perf_counter() - began,
    }
    write_json(out / "summary.json", summary)
    return summary


def evaluate_frozen_scene(
    checkpoint: Path,
    out: Path,
    *,
    tasks: tuple[int, ...],
    split: str,
    repeats: int,
    limit: int,
    backend: str,
    eval_workers: int,
    batch_size: int,
    device: str,
) -> dict:
    source = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if (
        source.get("schema") != "ogbench_scene_latent_v2"
        or source.get("environment") != SCENE.environment
    ):
        raise ValueError("Expected a state-based scene-latent v2 checkpoint")
    training = source["training"].copy()
    training["predictor_stopping"] = PredictorStopping(
        **training.get("predictor_stopping", {})
    )
    config, executor = (
        CompletionConfig(**training),
        ExecutorConfig(**source["executor"]),
    )
    if (
        not config.queries < limit <= SCENE.action_limit
        or not 0 < repeats <= SEED_OFFSETS["test"]
    ):
        raise ValueError(
            f"Use {config.queries + 1}..{SCENE.action_limit} steps and a valid reset count"
        )
    if eval_workers < 1 or batch_size < 1:
        raise ValueError("Use a positive worker count and batch size")
    start_seed = SCENE.seed_base + SEED_OFFSETS[split]
    seeds = list(range(start_seed, start_seed + repeats))
    cases = list(product(tasks, seeds))
    workers = min(eval_workers, len(cases))
    weights_path = checkpoint / "weights.msgpack"
    packed = weights_path.read_bytes()
    torch.set_num_threads(1)
    actor, model = _restore_models(packed, SCENE, config, executor)
    dependencies = ["ogbench", "mujoco", "torch"]
    if backend == "warp":
        try:
            import warp  # noqa: F401
            from mujoco import mjx  # noqa: F401
        except ImportError as error:
            raise ValueError(
                "Install the scene-warp extra for CUDA physics evaluation"
            ) from error
        dependencies += ["mujoco-mjx", "warp-lang"]
        results = _warp_cases(cases, actor, model, limit, batch_size, device)
    elif workers == 1:
        results = _rollout_cases(SCENE, cases, actor, model, limit)
    else:
        results = _cpu_shards(
            cases,
            workers,
            benchmark=SCENE,
            config=config,
            executor=executor,
            weights_path=weights_path,
            limit=limit,
        )
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "ogbench_scene_frozen_eval_v1",
            "environment": SCENE.environment,
            "observation": "official_state",
            "versions": {name: version(name) for name in dependencies},
            "backend": backend,
            "compute": (
                {"eval_workers": workers}
                if backend == "cpu"
                else {
                    "device": device,
                    "batch_size": min(batch_size, len(cases)),
                    "experimental": True,
                }
            ),
            "evaluation": {
                "split": split,
                "tasks": tasks,
                "seeds": seeds,
                "action_limit": limit,
            },
            "checkpoint": str(checkpoint),
            "weights_sha256": sha256(packed).hexdigest(),
            "code_revision": code_revision(),
        },
    )
    return _write_evaluation(out, tasks, results)
