# Running experiments

Use Python 3.12 or newer and run from the repository root. The [quick start](../README.md#quick-start) covers the CPU smoke checks. Artifacts stay in `results/` on the server. Git and the Mac sync scripts exclude the entire directory.

| Directory | Contents |
| --- | --- |
| `results/datasets/` | Complete trajectory collections and their logs |
| `results/completion.*/` | Masked trajectory windows and export manifests |
| `results/grid-completion.*/` | Paired grid predictor weights, world assignments and evaluation |
| `results/checkpoints/` | Policy and model weights |
| `results/external/` | Isolated benchmark source trees and environments |
| `results/_archive/` | Archived experiment runs, code snapshots and path maps |

Use a new output directory for each run. Keep long collection and training runs in tmux.

## Collect Craftax trajectories

Use `make_iron_pickaxe_and_survive` for trajectory completion. The agent must craft the iron pickaxe by the deadline and live through the last action. The `make_iron_pickaxe` option scores the first craft and remains available for reading or collecting earlier task variants. [Collection contract](architecture.md#craftax-collection-contract)

The following commands start matching expert and random collections with the same world seeds and step budget. The published PPO-RNN checkpoint supplies the expert actions.

```bash
uv sync --locked --group dev --extra craftax
python3 scripts/fetch_craftax_expert.py

tmux new-session -d -s wr-craftax-expert -c "$PWD" \
  'bash scripts/run_craftax_collection.sh expert 1000 256 25000 make_iron_pickaxe_and_survive'
tmux new-session -d -s wr-craftax-random -c "$PWD" \
  'bash scripts/run_craftax_collection.sh random 1000 256 25000 make_iron_pickaxe_and_survive'
```

Attach with `tmux attach -t wr-craftax-expert` or `tmux attach -t wr-craftax-random`. The launcher prints the resolved path and writes the log next to the run.

A completed run contains `manifest.json` for configuration and provenance, `splits.json` for world assignments, `trajectories.jsonl.gz` for observations and actions, and `summary.json` for counts and outcomes. The loader `worth_reaching.data.craftax_trajectories.load_craftax_archive` validates the manifest and streams complete episodes with `archive.iter_episodes("train")`. Fit models on the training split, choose settings in development, and reserve the test split for final evaluation.

## Prepare masked trajectory windows

Run `wr prepare-windows --collection PATH --out PATH` on a completed `make_iron_pickaxe_and_survive` collection. Set `--seed` to reproduce the sampling recipe. Use `--windows-per-episode`, `--min-span` and `--max-span` to control the number and lengths of windows. By default the mask length ranges from one step to the last eligible interior step at each start.

The export stores the sampling recipe and source split fingerprint in `manifest.json`, episode and window counts in `summary.json`, and one compressed array file per selected episode in `train/` or `dev/`. Windows reference slices in the episode file. The reader separates public inputs, masked targets and episode identifiers. [Window contract](architecture.md#masked-trajectory-window-contract)

```python
from pathlib import Path

from worth_reaching.data.completion import load_completion_corpus

corpus = load_completion_corpus(Path("results/completion.example/windows"))
example = next(corpus.iter_examples("train"))
inputs, target = example.inputs, example.target
```

The optional `corpus.iter_examples("train", with_terminal_example=True)` provides the recorded terminal observation for training with endpoint examples. Use the default task descriptor for training for goal-only deployment. Development windows measure prediction on successful demonstrations from separate worlds. This corpus has one task descriptor. Testing whether predictions respond to the goal itself requires demonstrations with distinct goals. Measure task success later through rollouts from natural resets.

## Train and test latent trajectory completion

Run `wr train-completion --grid --out PATH` for the two-goal diagnostic or `wr train-completion --windows PATH --out PATH` for the exported Craftax corpus. The trainer runs both the goal-conditioned model and its history-and-time control. `--seed`, `--encoder-updates`, `--updates`, `--history-length` and `--queries-per-window` specify the training and batching recipe. The defaults use 250 encoder updates, 600 predictor updates per arm, a 16-observation history and up to 16 query positions per window.

The grid has a wall with two passages and one colored destination on each side. Both destination positions and the passages are public. Each start has two shortest-path demonstrations. Both routes use the same 14-action budget and wait at their respective destination after arrival. The red and blue destinations trade sides across starts. The fixed split groups all nine starts per layout, with 32 training, 8 development and 16 test layouts. The paired reset observations, available time and query offsets are identical across the two goals at each start.

The route comparison scores the steps where the two demonstrations occupy different cells before either arrives. The summary reports the number of starts with such steps. The score compares how far each decoded prediction is from its matching demonstration and the alternate goal's demonstration, using Manhattan distance in cells and weighting eligible starts equally. Ties count for half a win. Exact-cell accuracy after rounding measures how often a forecast lands on the demonstrated cell. The interval for the matched-versus-swapped error difference resamples whole layouts within a training run. Use multiple training seeds to check variation across runs.

The default command evaluates development worlds. Add `--evaluate-test` for the grid after fixing the training recipe. Craftax uses its development windows and one task descriptor. Its latent and observation prediction errors check the training path on that corpus.

Each output contains `manifest.json` for the settings and source, `splits.json` for the world assignments, `weights.msgpack` for the shared codec and both predictors, and `summary.json` for metrics. The grid uses a generated reference route per goal. Closed-loop task success and discovery of alternative routes are evaluated in later milestones.

## Compare selectors on matched attempts

Given completed `shared_attempts_v1` banks, compare selection by arrival rate and by complete-attempt terminal success. Each candidate uses the same declared random repeat groups at its start. The command fits selectors from train attempts, calibrates filtering on the development bank, and scores choices on independent test worlds. [Attempt data contract](architecture.md#shared-attempt-evaluation-contract)

```bash
bank=results/my-attempt-bank
out="$(mktemp -d results/attempt-compare.XXXXXX)"
uv run --no-sync wr attempt-compare \
  --train "$bank/train" --dev "$bank/dev" --test "$bank/test" \
  --out "$out/comparison"
```

The summary reports terminal success, arrival, success after a missed target and paired differences by world. Each actual attempt includes the time spent pursuing the target and a continuation with the remaining budget. [Research design](trajectory-completion.md)
