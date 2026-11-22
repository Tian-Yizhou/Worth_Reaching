# Worth Reaching

Worth Reaching studies how agents discover intermediate goals for long-horizon tasks. The proposed approach learns to fill in missing portions of demonstration trajectories in latent space. Given a starting history and a terminal goal, it samples multiple routes and revises its plan as it acts. The first benchmark asks an agent to craft an iron pickaxe within a fixed action budget in Craftax-Classic and stay alive through the deadline. [Research design](docs/trajectory-completion.md)

## Research contribution

The project asks whether agents can discover reliably executable goals that are rare in demonstrations, and whether sequencing those goals improves terminal success. The evaluation measures both the probability of reaching an intermediate goal and the probability of finishing the task after attempting it. Matched simulator starts allow several choices to be tested with the same information and budget.

The planned comparison evaluates generated routes against direct execution and replay proposals on held-out worlds under matched budgets. The repository includes MiniGrid and Craftax diagnostics and paired AntMaze and PointMaze benchmarks. [Implementation plan](PLAN.md) · [Architecture](docs/architecture.md)

## Quick start

Use Python 3.12 or newer and run the commands below from the repository root. Install the environment extras for the experiments you plan to run. On an NVIDIA Linux server, include `--extra cuda12` to enable GPU training. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) · [Compute configuration](docs/experiments.md#compute-devices)

```bash
uv sync --locked --group dev --extra minigrid --extra craftax --extra ogbench --extra pointmaze --extra cuda12
uv run --no-sync wr devices
```

JAX automatically uses the available backend. Training uses every visible device by default. Edit a local copy of `configs/compute.toml` to select devices and pass it with `wr --compute-config PATH`.

### Run the KeyCorridor pilot

```bash
mkdir -p results
run_dir="$(mktemp -d results/minigrid-smoke.XXXXXX)/keycorridor"
uv run --no-sync wr pilot --config configs/minigrid/keycorridor.toml --out "$run_dir"
cat "$run_dir/summary.json"
```

The pilot replays one starting history and applies two different actions before handing control to the same policy. Turning right leads to success on the final step. Moving forward consumes a step against a wall and leads to timeout.

### Collect Craftax-Classic trajectories

```bash
mkdir -p results
run_dir="$(mktemp -d results/craftax-smoke.XXXXXX)/craftax"
uv run --no-sync wr collect \
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

### Train and compare latent predictors

The grid diagnostic generates matched demonstrations for two terminal goals. At evaluation both predictions use the same map, starting observation and action budget. [Training and goal switch protocol](docs/experiments.md#train-and-test-latent-trajectory-completion)

```bash
mkdir -p results
run_dir="$(mktemp -d results/grid-completion.XXXXXX)"
uv run --no-sync wr train-completion --grid --out "$run_dir/model"
```

### Execute sampled latent routes

Use the grid checkpoint from the previous command as the frozen predictor. A conditional flow samples several routes from the same reset and tests their waypoints under matched disturbances. [Route generation and execution protocol](docs/experiments.md#sample-and-execute-alternative-routes)

```bash
uv run --no-sync wr grid-routes \
  --checkpoint "$run_dir/model" --out "$run_dir/routes" --layouts 2
```

### Train the route generator from execution feedback

Use the checkpoint and sampled flow from the preceding commands. This diagnostic learns from matched attempts on training layouts and compares generators with one fixed reranker. [Feedback protocol](docs/experiments.md#train-from-execution-feedback)

```bash
uv run --no-sync wr grid-feedback \
  --checkpoint "$run_dir/model" --flow-run "$run_dir/routes" --out "$run_dir/feedback" \
  --updates 10 --layouts 2
```

### Longer server runs

The experiment guide covers matched expert and random collection, artifact locations, and selection on shared banks of prospective attempts. [Running experiments](docs/experiments.md)

### Train a state-based AntMaze baseline

Use the official state AntMaze-Large archives to train a goal-conditioned continuous-action policy. The development smoke run uses a saved reset and goal for each of the five benchmark tasks. [Dataset setup and matched evaluation](docs/experiments.md#train-the-antmaze-gcbc-baseline)

```bash
mkdir -p results
run_dir="$(mktemp -d results/antmaze-gcbc.XXXXXX)"
uv run --no-sync wr antmaze-gcbc \
  --dataset antmaze-large-navigate-v0 --out "$run_dir/navigate" \
  --updates 1000 --eval-repeats 1 --eval-steps 100
```

### Execute learned AntMaze plans

Train masked trajectory completion on the official demonstrations and compare decoded intermediate goals with direct execution using the same frozen actor. This smoke run reuses the reset bank above. [Learned planner protocol](docs/experiments.md#run-learned-latent-planning-in-antmaze)

```bash
uv run --no-sync wr antmaze-latent \
  --checkpoint "$run_dir/navigate" --starts "$run_dir/navigate/starts.npz" \
  --out "$run_dir/latent" --eval-repeats 1 --eval-steps 100 \
  --windows 128 --codec-updates 10 --predictor-updates 10 --flow-updates 10
```

### Run a Maze2D pilot

Train and evaluate the same latent planner on Farama's maintained versions of D4RL Maze2D U-Maze, Medium and Large. Select a task with `--task umaze`, `--task medium` or `--task large`. The pilot pairs direct and latent planning from the same resets using a common controller. [Dataset setup and scoring protocol](docs/experiments.md#run-the-diffuser-maze2d-pilots)

```bash
MINARI_DATASETS_PATH="$PWD/results/datasets/minari" \
  uv run --no-sync minari download D4RL/pointmaze/umaze-v2
run_dir="$(mktemp -d results/pointmaze-diffuser.XXXXXX)"
uv run --no-sync wr pointmaze-latent --task umaze --out "$run_dir/umaze"
```

### Run conditional Kuka stacking

Install the `stacking` extra alongside your existing extras. The pilot learns from Diffuser's released demonstrations and tests four-block orders on the author-released simulator. [Setup and scoring protocol](docs/experiments.md#run-conditional-kuka-stacking)

```bash
bash scripts/prepare_diffuser_kuka.sh
run_dir="$(mktemp -d results/diffuser-kuka.pilot.XXXXXX)"
uv run --no-sync wr kuka-latent --out "$run_dir/conditional"
```

### Run state-based OGBench Scene

Use the official Scene play archives to train a shared executor and compare direct and latent planning from the five official goals. [Data setup and scoring protocol](docs/experiments.md#run-state-based-ogbench-scene)

```bash
uv run --no-sync python -c 'import ogbench; ogbench.download_datasets(["scene-play-v0"], dataset_dir="results/datasets/ogbench")'
run_dir="$(mktemp -d results/ogbench-scene.XXXXXX)"
uv run --no-sync wr scene-latent --out "$run_dir/state" --eval-repeats 1
```

## Contributing

Choose a focused change from the implementation plan and include a reproducible check. Install the environment extras for the full CPU test suite. [Contribution guide](CONTRIBUTING.md) · [Implementation plan](PLAN.md)

```bash
uv sync --locked --group dev --extra minigrid --extra craftax --extra ogbench --extra pointmaze
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src tests scripts
JAX_PLATFORMS=cpu uv run --no-sync pytest
```

## Acknowledgments

Worth Reaching uses MiniGrid for diagnostic tasks and Craftax for the main benchmark. Its expert collector uses the Craftax-Classic PPO-RNN weights released with the ReMDM Planner project. [MiniGrid](https://minigrid.farama.org/) · [Craftax](https://proceedings.mlr.press/v235/matthews24a.html) · [ReMDM Planner](https://github.com/mathisweil/craftax-ReMDM-planner)
