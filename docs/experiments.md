# Running experiments

Use Python 3.12 or newer and run from the repository root. The [quick start](../README.md#quick-start) covers the CPU smoke checks. Artifacts stay in `results/` on the server. Git and the Mac sync scripts exclude the entire directory.

| Directory | Contents |
| --- | --- |
| `results/datasets/` | Complete trajectory collections and their logs |
| `results/completion.*/` | Masked trajectory windows and export manifests |
| `results/grid-completion.*/` | Paired grid predictor weights, world assignments and evaluation |
| `results/grid-routes.*/` | Latent flow checkpoints and matched attempt records |
| `results/grid-feedback.*/` | Training attempts, target masks, updated flows and held-out feedback comparisons |
| `results/antmaze-gcbc.*/` | GCBC checkpoints, matched reset banks and per-task rollout outcomes |
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

Each output contains `manifest.json` for the settings and source, `splits.json` for the world assignments, `weights.msgpack` for the shared codec and both predictors, and `summary.json` for metrics. The grid uses a generated reference route per goal.

## Sample and execute alternative routes

Run `wr grid-routes --checkpoint PATH --out PATH` with a grid completion checkpoint. By default it fits a conditional latent flow in 800 updates, samples eight routes per start and terminal goal, and evaluates eight matched repeats per candidate. The generator always trains on the original training layouts. Each training case contributes a shortest route through the passage on the destination side. A second demonstration through the opposite passage is included with probability 0.25 when feasible within 14 steps. Another eight layouts, disjoint from train, dev and test, provide demonstrations drawn by the same recipe with a separate random stream.

The runner selects the middle entry of the ordered distinct free cells decoded before the destination, excluding the starting cell. Its attempt deadline is the forecast step plus one, capped to leave time for continuation. An unusable sample falls back to the direct controller. Arrival uses valid waypoint attempts as the denominator. Terminal success includes all attempts, and success after missing a valid waypoint is counted separately.

The grid executor plans shortest paths using the public map. It recomputes after each action and turns to the terminal goal after arrival or the waypoint deadline. The open-loop arm replays its nominal action sequence. In each repeat all arms share a stream that independently replaces each action with a wait with probability 0.15. All arms use the 14-action budget and succeed if they occupy the specified destination at the deadline. The direct baseline targets the final destination from reset. The deterministic baseline decodes one route from the frozen predictor. The first flow draw provides a single-route comparison without using rollout scores for selection.

The reference frequency counts one passage per demonstration. The candidate pool covers a route family when one valid target both arrives and succeeds through that passage in at least half its repeats. Coverage of both passages requires two different targets or deadlines. The low-support family threshold is 25% of reference episodes. This is a diagnostic of demonstrated passage families. The paired terminal difference between feedback and open loop includes a layout bootstrap interval.

The default split is development. Once settings are fixed, use `--split test` and independent `--seed` values. `--layouts`, `--candidates`, `--repeats` and `--updates` can shorten a smoke run. `flow.msgpack`, `decisions.jsonl`, `manifest.json` and `summary.json` record weights, proposed latent targets, repeat masks, attempt results, and checkpoint provenance.

## Train from execution feedback

Run `wr grid-feedback --checkpoint PATH --flow-run PATH --out PATH` with the completion and flow checkpoints. The diagnostic retains the 14-action task and the 15% action-to-wait probability. An attempted entry into an inner wall passage also independently becomes a wait with probability 85%. The executor in every waypoint arm continues to use feedback and shortest paths, followed by the same direct continuation. The direct baselines use shortest paths or finite-horizon dynamic programming with the public transition probabilities.

The frozen flow draws candidates on training layouts. A supervised reranker predicts arrival and terminal success from public path lengths, deadline slack and risky passage entries. Terminal success counts the full rollout after an attempt, including recovery after a miss. For the generator update, a candidate must arrive in at least 25% of repeats and outperform the mean terminal success of valid candidates from the same start and goal. Its sampling weight is the product of the positive terminal advantage and the arrival rate. Feedback supervises the latent slot of the attempted waypoint. The training objective also uses the full demonstrated trajectories.

The frozen flow, demonstration-only continuation and feedback update share the same reranker and executor. The two updated flows have matched demonstration batches and optimizer steps. Evaluation uses matched base noise and repeat masks across arms, distinct from the training streams. Proposals are selected before rollouts. Report selection by predicted terminal success and selection by predicted arrival on the shared frozen pool. Report terminal success and low-support pool coverage under the same candidate and rollout counts. Bootstrap paired differences by layout.

Development is the default split. Use `--split test` for confirmation after fixing settings. The run saves training and held-out attempts, credited slots, repeat masks, the frozen reranker, the two updated weight sets and separate interaction counts. [Execution contract](architecture.md#execution-feedback)

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

## Train the AntMaze GCBC baseline

Install the locked OGBench extra. The following command downloads the state-based train and validation archives when they are absent. [OGBench dataset API](https://github.com/seohongpark/ogbench#dataset-apis)

```bash
uv sync --locked --group dev --extra ogbench
uv run --no-sync python -c 'import ogbench; ogbench.download_datasets(["antmaze-large-navigate-v0", "antmaze-large-stitch-v0"], dataset_dir="results/datasets/ogbench")'
```

`wr antmaze-gcbc` trains an MLP by action MSE using public observations, hindsight goals drawn within each trajectory and the relative goal displacement. Training statistics normalize the public inputs. The command scores held-out action MSE on the official validation archive and closed-loop success on the five simulator tasks. The zero-action control is a rollout sanity check. [OGBench reference GCBC](https://github.com/seohongpark/ogbench/blob/master/impls/agents/gcbc.py)

```bash
run_dir="$(mktemp -d results/antmaze-gcbc.XXXXXX)"
JAX_PLATFORMS=cpu uv run --no-sync wr antmaze-gcbc \
  --dataset antmaze-large-navigate-v0 --updates 2000 --width 128 --out "$run_dir/navigate"
JAX_PLATFORMS=cpu uv run --no-sync wr antmaze-gcbc \
  --dataset antmaze-large-stitch-v0 --updates 2000 --width 128 \
  --starts "$run_dir/navigate/starts.npz" --out "$run_dir/stitch"
```

Reuse `starts.npz` across comparisons with the same evaluation split and repeat count. The bank stores the exact reset observations, goals and integration state. The default evaluates four repeats for each task on development seeds with the 1,000-action horizon. Use `--eval-split test` after locking the recipe. Lower `--eval-steps` or `--eval-repeats` for smoke runs. A success comes from the benchmark's binary success event. The episode log also stores the initial, closest and final Euclidean distances in public XY coordinates.

Each run saves its recipe and archive hashes in `manifest.json`, validation checkpoints in `history.json`, parameters and normalization separately, full optimizer state in `training.msgpack`, and exact starts and outcomes in `starts.npz` and `episodes.json`. To resume, pass `--resume PREVIOUS_RUN`, use the same training recipe and archives, specify a new output directory and set `--updates` to the new total. Setting that total to the checkpoint step reruns evaluation at the saved weights. The batch stream is indexed by update so it resumes at the same batch as an uninterrupted run.
