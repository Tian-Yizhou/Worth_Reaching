# Worth Reaching

**Can a world model discover intermediate goals in latent space that substantially improve the probability of final task success?**

Worth Reaching studies this question for long-horizon tasks. Given an observation history, a terminal goal and a remaining action budget, a world model trained on unsegmented trajectories samples latent futures, and a learned value decides which one to pursue, for how long and when to change course. No one defines the intermediate goals or the test for reaching them.

[Setup](#setup) · [Quick start](#quick-start) · [Architecture](#architecture) · [Experiment guide](docs/experiments.md)

## Approach

- **Learn latent trajectory completion.** Mask a segment of a trajectory and predict the latent representations of its missing states from the starting history, the terminal condition and the time span. A residual flow samples several latent futures per query offset.
- **Pursue latent goals as options.** One executor conditioned on a latent `z` pursues any candidate for at most `k` actions. A latent goal means where pursuing it lands, so there is no decoder target and no arrival test.
- **Choose with a budget-conditioned option value.** `Q(h, g, z, k, b)` scores each candidate and its duration against the remaining budget, with the encoded terminal goal always among the candidates. It is fitted offline on hindsight options and corrected with the executor's own attempts. The same value decides when to switch to a cached candidate or sample new ones.
- **Measure contribution to success.** Compare with direct execution, direct pursuit of the encoded goal and, where one exists, a privileged planner, from matched starts under the same budget.

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
| [Maze2D / PointMaze](docs/experiments.md#run-the-diffuser-maze2d-pilots) | Train latent completion in U-Maze, Medium and Large, [plan with latent options](docs/experiments.md#plan-with-latent-options-in-maze2d) against direct tracking and a shortest-path oracle, and [test frozen energy](docs/experiments.md#test-latent-energy-against-terminal-success) against terminal success. |
| [OGBench Scene](docs/experiments.md#run-state-based-ogbench-scene) | Qualify a goal-conditioned executor on the official state-based goals and [plan with latent options](docs/experiments.md#plan-with-latent-options-in-scene). |

## Architecture

The Python package lives under `src/worth_reaching/`. Task adapters supply observations, actions, and terminal goal formats. Shared models learn from those inputs, and planners turn model predictions into goals for a task controller. The evaluator controls simulator resets and measures outcomes from matched starts.

| Module | Responsibility |
| --- | --- |
| [`tasks/`](src/worth_reaching/tasks) | Own simulator adapters, task definitions, dataset loaders, and task-specific experiment code and scripts. |
| [`models/`](src/worth_reaching/models) | Encode observations, predict masked latent states, sample trajectories, and train or restore weights. |
| [`data/`](src/worth_reaching/data) | Define shared trajectory and window formats, read OGBench state archives, and record experiment artifacts. |
| [`planning/`](src/worth_reaching/planning) | Choose, pursue and interrupt latent options with a budget-conditioned value, and pursue declared goals in the controlled tasks. |
| [`evaluation/`](src/worth_reaching/evaluation) | Run closed-loop and option-planner comparisons, correct the value online, compare matched attempts, and score final success. |
| [`cli.py`](src/worth_reaching/cli.py), [`recipes.py`](src/worth_reaching/recipes.py), [`compute.py`](src/worth_reaching/compute.py) | Dispatch `wr` commands, resolve model settings, and select compute devices. |
| [`configs/`](configs) | Store TOML task, model, and compute settings. |

[Interface details](docs/architecture.md) · [Model configuration](docs/experiments.md#configure-a-completion-model) · [Contributing](CONTRIBUTING.md)
