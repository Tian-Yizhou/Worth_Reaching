# Worth Reaching

**Can a world model discover intermediate goals in latent space that substantially improve the probability of final task success?**

Worth Reaching studies this question for long-horizon tasks. Given an observation history, a terminal goal, and a remaining action budget, we use a world model trained on demonstrations to find individual latent goals and sequences of goals worth pursuing. A latent goal names a region of states defined by a generative decoder, not a single target. The demonstrations require no hand-labeled subgoals.

[Setup](#setup) · [Quick start](#quick-start) · [Architecture](#architecture) · [Experiment guide](docs/experiments.md)

## Approach

- **Learn latent trajectory completion.** Mask a segment of a demonstration and predict the latent representations of its missing states from the starting history, endpoint condition, and time span.
- **Treat goals as latent regions.** A conditional generative decoder `p(s | z)` defines which states realize a latent goal `z`. A latent-conditioned executor pursues `z` directly and judges arrival in latent space. The decoder reveals afterwards what the model discovered.
- **Learn the value of attempting a goal.** Fit a budget-conditioned option value in latent space from attempt outcomes, including recoveries after missed goals, and plan sequences of goals with it.
- **Measure contribution to success.** Compare pursuing a candidate with going directly to the terminal goal from matched starts under the same executor and action budget. Measure intermediate-goal arrival and final task success separately.

[Research plan](docs/latent-subgoal-research.md) · [Trajectory completion design](docs/trajectory-completion.md) · [Implementation plan](PLAN.md)

## Setup

Use Python 3.12 or later and uv. From the repository root, install the package and its `wr` command. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked --no-dev
```

To install experiment dependencies, rerun the sync command with the extras for your task. For training, include `training-cpu` or `training` for CUDA on Linux. The command below sets up PointMaze with CPU training. [Experiment extras](docs/experiments.md#choose-a-workflow) · [Compute configuration](docs/experiments.md#compute-devices)

```bash
uv sync --locked --no-dev --extra training-cpu --extra pointmaze
```

## Quick start

The command format is `uv run --no-sync wr COMMAND [OPTIONS]`. Use `--help` to list the available commands and inspect their options.

```bash
uv run --no-sync wr --help
uv run --no-sync wr pointmaze-latent --help
```

Choose an experiment below for its data preparation and run commands.

## Experiments

| Task | What you can test |
| --- | --- |
| [Dependency grid](docs/experiments.md#compare-ordered-targets-in-the-dependency-grid) | Measure the effect of target order and goal switches, then [compare learned selection](docs/experiments.md#learn-to-select-a-goal-from-terminal-outcomes) on latent and replay proposals. |
| [Budgeted routes](docs/experiments.md#compose-latent-goals-under-action-budgets) | Compare latent goal composition and replay across budgets in a stochastic task with several routes to the same endpoint. |
| [Selector comparison](docs/experiments.md#compare-selectors-on-matched-attempts) | Fit and compare selectors on a shared bank of matched attempts. |
| [Maze2D / PointMaze](docs/experiments.md#run-the-diffuser-maze2d-pilots) | Compare direct and latent plans in U-Maze, Medium and Large, sweep the action budget, and [test frozen energy](docs/experiments.md#test-latent-energy-against-terminal-success) against terminal success. |
| [OGBench Scene](docs/experiments.md#run-state-based-ogbench-scene) | Evaluate manipulation plans on the official state-based goals and [probe Scene waypoint reachability](docs/experiments.md#test-scene-waypoint-reachability). |

## Architecture

The Python package lives under `src/worth_reaching/`. Task adapters supply observations, actions, and terminal goal formats. Shared models learn from those inputs, and planners turn model predictions into goals for a task controller. The evaluator controls simulator resets and measures outcomes from matched starts.

| Module | Responsibility |
| --- | --- |
| [`tasks/`](src/worth_reaching/tasks) | Own simulator adapters, task definitions, dataset loaders, and task-specific experiment code and scripts. |
| [`models/`](src/worth_reaching/models) | Encode observations, predict masked latent states, sample trajectories, and train or restore weights. |
| [`data/`](src/worth_reaching/data) | Define shared trajectory and window formats, read OGBench state archives, and record experiment artifacts. |
| [`planning/`](src/worth_reaching/planning) | Execute intermediate targets with a supplied controller and replan from new observations. |
| [`evaluation/`](src/worth_reaching/evaluation) | Replay trajectories, compare matched attempts, and score waypoint arrival and final success. |
| [`cli.py`](src/worth_reaching/cli.py), [`recipes.py`](src/worth_reaching/recipes.py), [`compute.py`](src/worth_reaching/compute.py) | Dispatch `wr` commands, resolve model settings, and select compute devices. |
| [`configs/`](configs) | Store TOML task, model, and compute settings. |

[Interface details](docs/architecture.md) · [Model configuration](docs/experiments.md#configure-a-completion-model) · [Contributing](CONTRIBUTING.md)
