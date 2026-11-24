"""Measure a frozen Scene actor's arrival at achieved validation waypoints."""

import json
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
from flax.serialization import msgpack_restore
from ogbench.manipspace.envs.scene_env import SceneEnv

from worth_reaching.data.artifacts import code_revision, write_json
from worth_reaching.data.ogbench import StateArchive, load_state_archive
from worth_reaching.models.gcbc import GCBC, policy_from_model
from worth_reaching.models.training import restore_weights
from worth_reaching.planning.receding_goals import Policy
from worth_reaching.tasks.ogbench_manipulation.spec import ACTION_SIZE, SCENE

# Slots and scale factors in OGBench 1.1.x's state observation.
EFFECTOR = slice(12, 15)
CONTACT = 18
CUBE = slice(19, 22)
CUBE_QUATERNION = slice(22, 26)
CUBE_YAW = slice(26, 28)
BUTTON_STATES = [28, 29, 32, 33]
SLIDES = [36, 38]
ARRIVAL_METERS = 0.04
RESTORE_TOLERANCE = 1e-3


def _cube_yaw_is_singular(observation: np.ndarray) -> bool:
    w, x, y, z = observation[CUBE_QUATERNION].astype(np.float64)
    # Yaw is the direction of the cube's local x-axis projected onto the XY plane.
    projection = np.hypot(w * w + x * x - y * y - z * z, 2 * (w * z + x * y))
    return bool(projection <= 8 * np.finfo(np.float32).eps)


def _restoration_errors(initial: np.ndarray, archived: np.ndarray, row: int) -> dict:
    error = np.abs(initial - archived)
    checked = np.ones(error.shape, dtype=bool)
    # The archive omits contact forces, which MuJoCo regenerates after stepping.
    checked[CONTACT] = False
    singular = _cube_yaw_is_singular(initial) and _cube_yaw_is_singular(archived)
    if singular:
        # The quaternion still defines the cube's orientation at the Euler singularity.
        checked[CUBE_YAW] = False
    mismatched = np.flatnonzero(
        ~np.isfinite(error) | (checked & (error > RESTORE_TOLERANCE))
    )
    if mismatched.size:
        raise ValueError(
            f"Scene state at archive row {row} did not restore "
            f"(observation indices {mismatched.tolist()})"
        )
    return {
        "initial_contact_error": float(error[CONTACT]),
        "initial_cube_yaw_error": float(error[CUBE_YAW].max()),
        "initial_cube_yaw_singular": singular,
    }


def _scene_match(observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
    cube = np.linalg.norm(observation[..., CUBE] - goal[..., CUBE], axis=-1)
    slides = np.abs(observation[..., SLIDES] - goal[..., SLIDES])
    return (
        (cube <= 10 * ARRIVAL_METERS)
        & (slides <= np.array([18, 15]) * ARRIVAL_METERS).all(axis=-1)
        & (observation[..., BUTTON_STATES] == goal[..., BUTTON_STATES]).all(axis=-1)
    )


def _effector_match(observation: np.ndarray, goal: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(observation[..., EFFECTOR] - goal[..., EFFECTOR], axis=-1)
    return distance <= 10 * ARRIVAL_METERS


def _arrived(observations: np.ndarray, goal: np.ndarray) -> np.ndarray:
    return _scene_match(observations, goal) & _effector_match(observations, goal)


def _arrival(path: np.ndarray, goal: np.ndarray) -> tuple[int | None, bool]:
    matching = _arrived(path, goal)
    hits = np.flatnonzero(matching)
    return int(hits[0]) if hits.size else None, bool(matching[-1])


def _read_replays(
    reference: Path, archive: StateArchive, cases: list, seed: int
) -> list:
    saved = json.loads((reference / "manifest.json").read_text(encoding="utf-8"))
    if (
        saved.get("schema")
        not in ("ogbench_scene_reachability_v1", "ogbench_scene_reachability_v2")
        or saved.get("validation_sha256") != archive.fingerprint
        or saved.get("seed") != seed
        or any(
            saved.get("versions", {}).get(name) != version(name)
            for name in ("ogbench", "mujoco")
        )
    ):
        raise ValueError(
            "The replay cache requires the same Scene archive, simulator versions and reset seed"
        )
    rows = json.loads((reference / "attempts.json").read_text(encoding="utf-8"))
    replays = []
    with np.load(reference / "trajectories.npz", allow_pickle=False) as data:
        for i, (attempt, (group, offset, episode, row)) in enumerate(
            zip(rows, cases, strict=True)
        ):
            if (
                attempt["group"],
                attempt["offset"],
                attempt["episode"],
                attempt["start_row"],
                attempt["target_row"],
            ) != (group, offset, episode, row, row + offset):
                raise ValueError("The replay cache and sampled waypoint rows differ")
            path, actions = (
                data[f"{i}_replay_observations"],
                data[f"{i}_replay_actions"],
            )
            if (
                path.shape != (len(actions) + 1, SCENE.observation_size)
                or not 0 < len(actions) <= offset
                or not np.array_equal(
                    actions, archive.actions[row : row + len(actions)]
                )
            ):
                raise ValueError(f"Invalid cached Scene replay at archive row {row}")
            replays.append((path, actions))
    return replays


def _waypoints(archive: StateArchive, offsets: list[int], count: int, seed: int):
    rng = np.random.default_rng(seed)
    episodes = np.searchsorted(archive.ends, archive.rows)
    for offset in offsets:
        valid = archive.rows + offset <= archive.ends[episodes]
        rows, roots = archive.rows[valid], episodes[valid]
        start, goal = archive.observations[rows], archive.observations[rows + offset]
        scene_match = _scene_match(start, goal)
        groups = {
            "scene_change": ~scene_match,
            "pose_only": scene_match & ~_effector_match(start, goal),
        }
        for group, eligible in groups.items():
            eligible_rows, eligible_roots = rows[eligible], roots[eligible]
            unique = np.unique(eligible_roots)
            for root in rng.choice(unique, size=min(count, len(unique)), replace=False):
                row = int(rng.choice(eligible_rows[eligible_roots == root]))
                yield group, offset, int(root), row


def _rollout(
    env: SceneEnv,
    initial: np.ndarray,
    goal: np.ndarray,
    steps: int,
    act: Policy,
    *,
    stop_on_arrival: bool,
):
    observations, actions = [initial], []
    for _ in range(steps):
        action = act(observations[-1], goal)
        observation, _, done, truncated, _ = env.step(action)
        observations.append(np.asarray(observation, dtype=np.float32))
        actions.append(action)
        if done or truncated or (stop_on_arrival and _arrived(observations[-1], goal)):
            break
    path = np.stack(observations)
    return path, np.stack(actions), *_arrival(path, goal)


def _summary(attempts: list[dict], offsets: list[int], budgets: list[int]) -> dict:
    groups = {}
    for offset in offsets:
        for group in ("scene_change", "pose_only"):
            rows = [r for r in attempts if (r["offset"], r["group"]) == (offset, group)]
            reproduced = [r for r in rows if r["replay_matches_at_offset"]]
            groups[f"{group}_{offset}"] = {
                "cases": len(rows),
                "replay_matches_at_offset": len(reproduced),
                "actor_arrivals_by_budget": {
                    str(b): sum(
                        r["actor_arrival"] is not None and r["actor_arrival"] <= b
                        for r in rows
                    )
                    for b in budgets
                },
                "actor_arrivals_on_reproduced_cases_by_budget": {
                    str(b): sum(
                        r["actor_arrival"] is not None and r["actor_arrival"] <= b
                        for r in reproduced
                    )
                    for b in budgets
                },
            }
    return {
        "cases": len(attempts),
        "unique_validation_episodes": len({r["episode"] for r in attempts}),
        "groups": groups,
        "training_environment_steps": 0,
        "evaluation_environment_steps": sum(
            r["actor_steps"] + (0 if r["replay_reused"] else r["replay_steps"])
            for r in attempts
        ),
        "replay_steps_reused": sum(
            r["replay_steps"] for r in attempts if r["replay_reused"]
        ),
    }


def run(
    checkpoint: Path,
    archive_path: Path,
    out: Path,
    *,
    seed: int,
    offsets: list[int],
    budgets: list[int],
    cases_per_group: int,
    replay_from: Path | None = None,
) -> dict:
    if (
        seed < 0
        or cases_per_group < 1
        or any(
            not steps
            or min(steps) < 1
            or max(steps) > SCENE.action_limit
            or len(steps) != len(set(steps))
            for steps in (offsets, budgets)
        )
    ):
        raise ValueError(
            "Use a nonnegative seed, positive case count, and distinct offsets and budgets in 1..750"
        )
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema") != "ogbench_scene_latent_v2"
        or manifest.get("environment") != SCENE.environment
    ):
        raise ValueError("Expected a state-based scene-latent v2 checkpoint")
    archive = load_state_archive(
        archive_path, observation_size=SCENE.observation_size, action_size=ACTION_SIZE
    )
    if archive.fingerprint == manifest["train_sha256"]:
        raise ValueError("Use an archive held out from controller training")
    packed = (checkpoint / "weights.msgpack").read_bytes()
    weights = msgpack_restore(packed)
    model = GCBC(
        manifest["executor"]["width"],
        SCENE.observation_size,
        ACTION_SIZE,
        relative_coordinates=0,
    )
    mean, scale = weights["mean"], weights["scale"]
    if (
        mean.shape != (SCENE.observation_size,)
        or scale.shape != mean.shape
        or not np.isfinite(mean).all()
        or not (np.isfinite(scale) & (scale > 0)).all()
    ):
        raise ValueError("Invalid Scene GCBC normalization")
    actor = policy_from_model(restore_weights(model, weights["executor"]), mean, scale)

    with np.load(archive_path, allow_pickle=False) as data:
        qpos, qvel, buttons = (data[key] for key in ("qpos", "qvel", "button_states"))
    cases = list(_waypoints(archive, offsets, cases_per_group, seed))
    if not cases:
        raise ValueError(
            "No nontrivial within-episode targets at the requested offsets"
        )
    replays = (
        None
        if replay_from is None
        else _read_replays(replay_from, archive, cases, seed)
    )

    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "manifest.json",
        {
            "schema": "ogbench_scene_reachability_v2",
            "environment": SCENE.environment,
            "mode": "data_collection",
            "split": "validation",
            "observation": "official_state",
            "versions": {
                name: version(name) for name in ("ogbench", "mujoco", "torch")
            },
            "validation_archive": str(archive_path),
            "validation_sha256": archive.fingerprint,
            "executor_checkpoint": str(checkpoint),
            "executor_weights_sha256": sha256(packed).hexdigest(),
            "executor_training_sha256": manifest["train_sha256"],
            "replay_source": str(replay_from) if replay_from is not None else None,
            "seed": seed,
            "cases_per_group": cases_per_group,
            "offsets": offsets,
            "budgets": budgets,
            "arrival": {
                "radius_meters": ARRIVAL_METERS,
                "features": [
                    "cube_xyz",
                    "drawer",
                    "window",
                    "lock_bits",
                    "effector_xyz",
                ],
            },
            "code_revision": code_revision(),
        },
    )
    attempts, trajectories = [], {}
    env = SceneEnv(env_type="scene", mode="data_collection", terminate_at_goal=False)
    try:
        for i, (group, offset, episode, row) in enumerate(cases):
            end = row + offset
            goal = archive.observations[end]
            actions = iter(archive.actions[row:end])
            attempt = {
                "case": i,
                "episode": episode,
                "group": group,
                "offset": offset,
                "start_row": row,
                "target_row": end,
                "replay_reused": replays is not None,
            }
            trajectories[f"{i}_target"] = goal
            replay_start = None
            for arm, steps, policy in (
                ("replay", offset, lambda _ob, _goal: next(actions)),
                ("actor", max(budgets), actor),
            ):
                use_cached = arm == "replay" and replays is not None
                if use_cached:
                    path, commands = replays[i]
                    initial = path[0]
                else:
                    env.reset(seed=seed)
                    env.set_state(qpos[row], qvel[row], buttons[row])
                    initial = np.asarray(env.compute_observation(), dtype=np.float32)
                if replay_start is None:
                    replay_start = initial
                    attempt.update(
                        _restoration_errors(initial, archive.observations[row], row)
                    )
                elif not np.array_equal(initial, replay_start):
                    raise ValueError(
                        f"Scene attempts at archive row {row} did not start together"
                    )
                if use_cached:
                    arrival, ends_at_goal = _arrival(path, goal)
                else:
                    path, commands, arrival, ends_at_goal = _rollout(
                        env,
                        initial,
                        goal,
                        steps,
                        policy,
                        stop_on_arrival=arm == "actor",
                    )
                attempt[f"{arm}_steps"] = len(commands)
                attempt[f"{arm}_arrival"] = arrival
                if arm == "replay":
                    attempt["replay_matches_at_offset"] = (
                        ends_at_goal and len(commands) == offset
                    )
                trajectories[f"{i}_{arm}_observations"] = path
                trajectories[f"{i}_{arm}_actions"] = commands
            attempts.append(attempt)
    finally:
        env.close()
    write_json(out / "attempts.json", attempts)
    np.savez_compressed(out / "trajectories.npz", **trajectories)
    summary = _summary(attempts, offsets, budgets)
    write_json(out / "summary.json", summary)
    return summary
