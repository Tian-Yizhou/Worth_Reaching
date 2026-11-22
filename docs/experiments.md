# Running experiments

Use Python 3.12 or newer and run from the repository root. The [quick start](../README.md#quick-start) covers the CPU smoke checks. Artifacts stay in `results/` on the server. Git and the Mac sync scripts exclude the entire directory.

| Directory | Contents |
| --- | --- |
| `results/datasets/` | Complete trajectory collections and their logs |
| `results/completion.*/` | Masked trajectory windows and export manifests |
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
