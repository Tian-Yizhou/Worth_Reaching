# Worth Reaching

Worth Reaching studies how an agent given a distant terminal objective can discover and sequence intermediate goals in a learned latent space. A goal is a region that can admit several different histories. A useful region offers the current controller feasible ways to enter it, and attempting it improves the prospect of completing the task.

Consider an agent trying to craft an iron pickaxe before nightfall. Several visible positions may offer iron. Approaching one may leave enough health and time to finish crafting, while approaching another may exhaust the agent. A latent waypoint could represent a family of positions from which the next step is feasible. We want to know whether a model can discover that distinction, act on it, and revise the plan when new observations arrive.

## The first paper

1. **Can the agent discover regions that are rare in demonstrations and reliable to execute?** Compare demonstration support with the outcomes of repeated, predeclared attempts.
2. **Can the agent plan from a distant final goal by choosing its own intermediate regions?** Compare learned regions, fixed latent neighborhoods, flat planning and designer milestones under matched information and budgets.

The project evaluates three ways to propose regions. A conditional generator learns from unsegmented demonstrations. Online RL improves a pretrained flow proposer using terminal outcomes. A hierarchical world model uses coarse predictions of candidate macro-actions to propose regions for fine-scale execution. Director demonstrated learned latent goals in Crafter. Our experiments test what makes a region useful and when it improves end-to-end planning. [Director](https://danijar.com/project/director/) · [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208)

The proposed first benchmark is Craftax-Classic with symbolic observations. MiniGrid provides a small diagnostic task. Balatro motivates a later test involving variable-size inventories, ordered interactions and irreversible purchases. The first experiment can proceed independently of the Balatro simulator.

## Project documents

| Document | Use |
| --- | --- |
| [Implementation plan](PLAN.md) | Development milestones, dependencies and evidence for completion |
| [Architecture](docs/architecture.md) | Proposed components, data contracts and information boundaries |
| [Environment choices](docs/experiment_env.md) | Candidate environments, adaptations and reasons to select each |
| [Contributing](CONTRIBUTING.md) | Development workflow and reproducibility rules |
| [Agent instructions](AGENTS.md) | Entry point for assistants working in this repository |

## Development setup

The project uses uv and requires Python 3.12 or newer. Install uv and sync the locked CPU dependencies. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv sync --locked --group dev --extra minigrid
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src tests
uv run --no-sync pytest
```

## Run the KeyCorridor pilot

```bash
run_dir="$(mktemp -d)/keycorridor"
uv run --no-sync wr pilot --config configs/minigrid/keycorridor.toml --out "$run_dir"
```

The fixed seed and 12-action prefix leave the agent in the open doorway, facing north with two actions remaining. Forcing `right` lets the fixed continuation policy pick up the ball on the last action. Forcing `forward` spends a step against the wall and the episode times out as the policy turns. Expected outcomes are `right: success` and `forward: timeout`. [KeyCorridor environment](https://minigrid.farama.org/environments/babyai/KeyCorridor/)

The run directory contains `manifest.json` with the resolved configuration, simulator versions and interaction counts, `prefix.json` with the recorded public trajectory, `attempts.jsonl` with the two branches, and `summary.json` with the outcome counts. The tests rerun the CLI, compare both public traces byte for byte, and perturb a hidden tile while checking that policy input stays identical.

## Collect Craftax-Classic trajectories

The adapter uses Craftax-Classic Symbolic with the native 17-action space and a fixed action budget. The first transition that earns the iron-pickaxe achievement is success, including on the final allowed action. Death from depleted health or lava takes precedence over timeout. The adapter reads the terminal transition from the native environment before any reset. [Craftax implementation](https://github.com/MichaelTMatthews/Craftax)

Run the short CPU check from the repository root. The default recipe collects four random-policy episodes with at most 32 actions each.

```bash
uv sync --locked --group dev --extra minigrid --extra craftax
JAX_PLATFORMS=cpu uv run --no-sync pytest
JAX_PLATFORMS=cpu uv run --no-sync wr collect \
  --config configs/craftax/iron_pickaxe.toml --out "$(mktemp -d)/craftax"
```

Each run writes the versioned configuration in `manifest.json`, the world assignments in `splits.json`, one complete public trajectory per line in `trajectories.jsonl.gz`, and the final outcome and interaction counts in `summary.json`. The split manifest is written before collecting any episode. A stable hash of the world identity assigns about 80% of worlds to train, 10% to dev and 10% to test. Overlapping seed ranges reuse the same world and split. Use a different `seed_start` in the TOML file to collect additional worlds. A run is complete when its summary is present.

### Longer server runs

Start a tmux session on the server. From the repository root inside that session, run the commands below. The example reserves at most one million environment actions. Use a new output name for each collection.

```bash
tmux new-session -s craftax-collect
```

```bash
export WR_ARTIFACTS="${WR_ARTIFACTS:-$HOME/wr-artifacts}"
mkdir -p "$WR_ARTIFACTS"
uv sync --locked --group dev --extra minigrid --extra craftax
run_dir="$WR_ARTIFACTS/craftax-$(date -u +%Y%m%dT%H%M%SZ)"
set -o pipefail
JAX_PLATFORMS=cpu uv run --no-sync wr collect \
  --config configs/craftax/iron_pickaxe.toml --episodes 1000 --budget 1000 \
  --out "$run_dir" 2>&1 | tee "$run_dir.log"
```

Keep raw trajectories, logs and checkpoints in `WR_ARTIFACTS`. To include the small run metadata in a Mac handoff, copy selected files after collection finishes.

```bash
share_dir="local-file/results/$(basename "$run_dir")"
mkdir -p "$share_dir"
cp "$run_dir/manifest.json" "$run_dir/splits.json" "$run_dir/summary.json" "$share_dir/"
```
