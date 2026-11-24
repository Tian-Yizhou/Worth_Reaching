# Updating Python experiment scripts

Use the `wr` commands to launch experiments and the Python modules below to read trajectories, prepare windows or inspect checkpoints in a notebook. Install the same extras as the corresponding experiment. The import paths changed when the simulator adapters and task-specific loaders moved into task families. [Experiment setup](experiments.md#choose-a-workflow)

## Craftax collection and windows

Set `collection` to the output directory of `wr collect`, or run the [short collection example](experiments.md#collect-craftax-trajectories) to create one. This example prints the first training episode's world ID and action count, if the split contains any episodes.

```bash
uv run --no-sync python - "$collection" <<'PY'
import sys
from pathlib import Path

from worth_reaching.tasks.craftax.dataset import load_craftax_archive

archive = load_craftax_archive(Path(sys.argv[1]))
for spec, trace in archive.iter_episodes("train"):
    print(spec.world_id, len(trace["steps"]))
    break
PY
```

For a directory produced by `wr prepare-windows`, use `load_completion_corpus` from `worth_reaching.tasks.craftax.windows`. The shared `CompletionInput`, `CompletionTarget` and `WindowConfig` types live in `worth_reaching.data.completion`. [Window reader example](experiments.md#prepare-masked-trajectory-windows)

## Import path reference

All paths in the tables are relative to `worth_reaching`. For example, `tasks.pointmaze.spec` means `from worth_reaching.tasks.pointmaze.spec import TASKS`.

| Former module | Current module |
| --- | --- |
| `envs.minigrid` | `tasks.keycorridor.environment` |
| `envs.grid_routes` | `tasks.grid_routes.environment` |
| `envs.craftax` | `tasks.craftax.environment` |
| `data.collection` | `tasks.craftax.collection` for `CollectionConfig` and `collect`. Import `episode_keys` and `episode_spec` from `tasks.craftax.dataset`. |
| `data.craftax_trajectories` | `tasks.craftax.dataset` |
| `policies.craftax_expert` | `tasks.craftax.expert` |
| `data.completion` (Craftax I/O functions) | `tasks.craftax.windows` for `prepare_craftax_windows` and `load_completion_corpus` |
| `envs.kuka` | `tasks.kuka.environment` |
| `data.kuka` | `tasks.kuka.dataset` |
| `envs.ogbench_manipulation` | `tasks.ogbench_manipulation.environment` |
| `envs.ogbench_tasks` | `tasks.ogbench_manipulation.spec` |
| `envs.scene` | `tasks.ogbench_manipulation.scene` |

### Maze tasks and shared models

| Former module | Current module |
| --- | --- |
| `envs.pointmaze` | `tasks.pointmaze.environment` |
| `envs.pointmaze_tasks` | `tasks.pointmaze.spec` |
| `data.pointmaze` | `tasks.pointmaze.dataset` |
| `evaluation.pointmaze` | `tasks.pointmaze.experiment` |
| `evaluation.pointmaze_probe` | `tasks.pointmaze.probe` |
| `evaluation.pointmaze_energy` | `tasks.pointmaze.energy` |
| `envs.antmaze` | `tasks.antmaze.environment` |
| `data.antmaze` | `tasks.antmaze.dataset` |
| `evaluation.antmaze` | `tasks.antmaze.baseline` |
| `evaluation.antmaze_latent` | `tasks.antmaze.experiment` |
| `evaluation.antmaze_waypoints` | `tasks.antmaze.waypoints` |
| `models.gcbc` (AntMaze checkpoint loader and schema constants) | `tasks.antmaze.checkpoint` |
| `data.maze_windows` | `data.trajectory_windows`. Update the type name `MazeTrajectories` to `ContinuousTrajectories`. |
| `models.maze_completion` | `models.continuous_completion` |
| `evaluation.maze_planning` | `planning.receding_goals` |

To load a saved AntMaze actor, import `load_policy` from `worth_reaching.tasks.antmaze.checkpoint` and pass the checkpoint directory. Scripts that construct a fresh `GCBC` from `worth_reaching.models.gcbc` now supply the network width, observation size and action size. AntMaze defines its sizes in `worth_reaching.tasks.antmaze.dataset`. [Checkpoint and evaluation protocol](experiments.md#train-the-antmaze-gcbc-baseline)
