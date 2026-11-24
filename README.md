# Worth Reaching

**Can a world model discover intermediate goals in latent space that substantially improve the probability of final task success?**

Worth Reaching studies this question for long-horizon tasks. Given an observation history, a terminal goal, and a remaining action budget, we use a world model trained on demonstrations to find individual latent goals and sequences of goals worth pursuing. The demonstrations require no hand-labeled subgoals.

[Setup](#setup) · [Quick start](#quick-start) · [Architecture](#architecture) · [Experiment guide](docs/experiments.md)

## Approach

- **Learn latent trajectory completion.** Mask a segment of a demonstration and predict the latent representations of its missing states from the starting history, endpoint condition, and time span.
- **Discover candidate goals.** Sample multiple latent trajectories with a conditional flow. Decode selected states into targets a controller can pursue. Test whether the frozen model's energy helps identify useful candidates before execution.
- **Measure contribution to success.** Compare pursuing a candidate with going directly to the terminal goal from matched starts under the same controller and action budget. Measure intermediate-goal arrival and final task success separately.

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
| [Dependency grid](docs/experiments.md#compare-ordered-targets-in-the-dependency-grid) | Measure the effect of target order and [compare learned selection](docs/experiments.md#learn-to-select-a-goal-from-terminal-outcomes) on latent and replay proposals. |
| [MiniGrid KeyCorridor](docs/experiments.md#replay-a-keycorridor-decision) | Branch at one decision and inspect how a single action changes the outcome at the deadline. |
| [Two-goal grid](docs/experiments.md#sample-and-execute-alternative-routes) | Switch the terminal goal on the same layout, sample alternate routes and train from execution feedback. |
| [Craftax-Classic](docs/experiments.md#collect-craftax-trajectories) | Collect expert and random demonstrations, then train on masked windows from episodes that craft an iron pickaxe and survive through the deadline. |
| [Maze2D / PointMaze](docs/experiments.md#run-the-diffuser-maze2d-pilots) | Compare direct and latent plans in U-Maze, Medium and Large, and probe first-target selection on Medium. |
| [OGBench AntMaze](docs/experiments.md#train-the-antmaze-gcbc-baseline) | Train a goal-conditioned actor and reuse it to compare direct execution, waypoint following and learned plans. |
| [Kuka stacking](docs/experiments.md#run-conditional-kuka-stacking) | Condition a plan on a requested four-block order and score the resulting tower in the Diffuser simulator. |
| [OGBench Scene](docs/experiments.md#run-state-based-ogbench-scene) and [Puzzle](docs/experiments.md#run-state-based-ogbench-puzzle) | Evaluate manipulation plans on the official state-based goals with a shared executor per task. |

## Architecture

The Python package lives under `src/worth_reaching/`. Task adapters supply observations, actions, and terminal goal formats. Shared models learn from those inputs, and planners turn model predictions into goals for a task controller. The evaluator controls simulator resets and measures outcomes from matched starts.

| Module | Responsibility |
| --- | --- |
| [`tasks/`](src/worth_reaching/tasks) | Own simulator adapters, task definitions, dataset loaders, and task-specific experiment code. |
| [`models/`](src/worth_reaching/models) | Encode observations, predict masked latent states, sample trajectories, and train or restore weights. |
| [`data/`](src/worth_reaching/data) | Define shared trajectory and window formats, read OGBench state archives, and record experiment artifacts. |
| [`planning/`](src/worth_reaching/planning) | Execute intermediate targets with a supplied controller and replan from new observations. |
| [`evaluation/`](src/worth_reaching/evaluation) | Replay trajectories, compare matched attempts, and score waypoint arrival and final success. |
| [`contracts.py`](src/worth_reaching/contracts.py) | Define the public observations, actions, and policy inputs for the KeyCorridor diagnostic. |
| [`cli.py`](src/worth_reaching/cli.py), [`recipes.py`](src/worth_reaching/recipes.py), [`compute.py`](src/worth_reaching/compute.py) | Dispatch `wr` commands, resolve model settings, and select compute devices. |
| [`configs/`](configs) | Store TOML task, model, and compute settings. |

[Interface details](docs/architecture.md) · [Model configuration](docs/experiments.md#configure-a-completion-model) · [Python import migration](docs/python-imports.md) · [Contributing](CONTRIBUTING.md)

## Acknowledgments

The Craftax expert collector uses the Craftax-Classic PPO-RNN weights released by the ReMDM Planner project. [Checkpoint source](https://github.com/mathisweil/craftax-ReMDM-planner)
