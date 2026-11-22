# Worth Reaching

Worth Reaching studies how agents discover intermediate goals for long-horizon tasks. The proposed approach learns to fill in missing portions of demonstration trajectories in latent space. Given a starting history and a terminal goal, it samples multiple routes and revises its plan as it acts. The first benchmark asks an agent to craft an iron pickaxe within a fixed action budget in Craftax-Classic and stay alive through the deadline. [Research design](docs/trajectory-completion.md)

## Research contribution

The project asks whether agents can discover reliably executable goals that are rare in demonstrations, and whether sequencing those goals improves terminal success. The evaluation measures both the probability of reaching an intermediate goal and the probability of finishing the task after attempting it. Matched simulator starts allow several choices to be tested with the same information and budget.

The planned comparison evaluates generated routes against direct execution and replay proposals on held-out worlds under matched budgets. The repository provides MiniGrid and Craftax environments, trajectory collection, and matched simulator branches for prospective evaluation. [Implementation plan](PLAN.md) · [Architecture](docs/architecture.md)

## Quick start

Use Python 3.12 or newer and run the commands below from the repository root. Both examples run on CPU. [Install uv](https://docs.astral.sh/uv/getting-started/installation/)

### Run the KeyCorridor pilot

```bash
uv sync --locked --group dev --extra minigrid
mkdir -p results
run_dir="$(mktemp -d results/minigrid-smoke.XXXXXX)/keycorridor"
uv run --no-sync wr pilot --config configs/minigrid/keycorridor.toml --out "$run_dir"
cat "$run_dir/summary.json"
```

The pilot replays one starting history and applies two different actions before handing control to the same policy. Turning right leads to success on the final step. Moving forward consumes a step against a wall and leads to timeout.

### Collect Craftax-Classic trajectories

```bash
uv sync --locked --group dev --extra minigrid --extra craftax
mkdir -p results
run_dir="$(mktemp -d results/craftax-smoke.XXXXXX)/craftax"
JAX_PLATFORMS=cpu uv run --no-sync wr collect \
  --config configs/craftax/iron_pickaxe.toml \
  --task make_iron_pickaxe_and_survive --out "$run_dir"
cat "$run_dir/summary.json"
```

The default recipe records four random-policy episodes of at most 32 actions each. Each run saves the public trajectories, world splits and terminal outcomes under the repository's Git-ignored `results/` directory.

### Prepare masked trajectory windows

Point the command at a completed craft-and-survive collection with successful demonstrations. The export gives each successful training or development episode a window at its initial observation and additional windows at sampled steps. Mask lengths vary with the available action budget. [Window format and loader](docs/experiments.md#prepare-masked-trajectory-windows)

```bash
collection=results/datasets/your-craft-and-survive-collection
out="$(mktemp -d results/completion.XXXXXX)/windows"
uv run --no-sync wr prepare-windows --collection "$collection" --out "$out" --seed 7
```

### Longer server runs

The experiment guide covers matched expert and random collection, artifact locations, and selection on shared banks of prospective attempts. [Running experiments](docs/experiments.md)

## Contributing

Choose a focused change from the implementation plan and include a reproducible check. For the full CPU test suite, install both environment extras as shown in the Craftax quick start. [Contribution guide](CONTRIBUTING.md) · [Implementation plan](PLAN.md)

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src tests scripts
JAX_PLATFORMS=cpu uv run --no-sync pytest
```

## Acknowledgments

Worth Reaching uses MiniGrid for diagnostic tasks and Craftax for the main benchmark. Its expert collector uses the Craftax-Classic PPO-RNN weights released with the ReMDM Planner project. [MiniGrid](https://minigrid.farama.org/) · [Craftax](https://proceedings.mlr.press/v235/matthews24a.html) · [ReMDM Planner](https://github.com/mathisweil/craftax-ReMDM-planner)
