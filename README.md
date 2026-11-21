# Worth Reaching

Worth Reaching is a reinforcement learning research project about discovering intermediate goals for long-horizon tasks. Its first benchmark asks an agent to craft an iron pickaxe in Craftax-Classic within a fixed action budget. Different routes can leave the agent with similar opportunities to finish, even when their positions or inventories differ. We study whether a learned latent region can represent those opportunities and guide the agent's next decision.

## Research contribution

The project asks whether agents can discover reliably executable goals that are rare in demonstrations, and whether sequencing those goals improves terminal success. The evaluation measures both the probability of reaching an intermediate goal and the probability of finishing the task after attempting it. Matched simulator starts allow several choices to be tested with the same information and budget.

The planned comparison tests learned regions against fixed latent neighborhoods, direct planning and human-defined milestones. The current implementation includes a deterministic MiniGrid pilot, random and PPO-RNN trajectory collection in Craftax-Classic, and paired action branches that measure how a decision changes the chance of crafting the iron pickaxe. [Implementation plan](PLAN.md) · [Architecture](docs/architecture.md)

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
  --config configs/craftax/iron_pickaxe.toml --out "$run_dir"
cat "$run_dir/summary.json"
```

The default recipe records four random-policy episodes of at most 32 actions each. Each run saves the public trajectories, world splits and terminal outcomes under the repository's Git-ignored `results/` directory.

### Longer server runs

The experiment guide covers matched expert and random collection, paired action evaluation, and a shared-bank scoring pilot for replay-derived waypoints. [Running experiments](docs/experiments.md) · [Waypoint scoring pilot](docs/experiments.md#shared-bank-waypoint-scoring-pilot)

## Contributing

Choose a focused change from the implementation plan and include a reproducible check. For the full CPU test suite, install both environment extras as shown in the Craftax quick start. [Contribution guide](CONTRIBUTING.md) · [Implementation plan](PLAN.md)

```bash
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src tests scripts
JAX_PLATFORMS=cpu uv run --no-sync pytest
```

## Acknowledgments

Worth Reaching uses MiniGrid for diagnostic tasks and Craftax for the main benchmark. Its expert collector uses the Craftax-Classic PPO-RNN weights released with the ReMDM Planner project. [MiniGrid](https://minigrid.farama.org/) · [Craftax](https://proceedings.mlr.press/v235/matthews24a.html) · [ReMDM Planner](https://github.com/mathisweil/craftax-ReMDM-planner)
