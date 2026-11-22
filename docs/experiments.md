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
| `results/antmaze-latent.*/` | Completion weights, sampled and decoded plans, and paired terminal outcomes |
| `results/antmaze-waypoints.*/` | Paired direct and oracle waypoint execution outcomes |
| `results/pointmaze-diffuser.*/` | PointMaze archive splits, decoded plans, executed trajectories and paired outcomes |
| `results/diffuser-kuka.pilot.*/` | Conditional stacking clip splits, decoded plans, executed states and tower scores |
| `results/benchmark-suite.*/` | Per-task development runs and logs from the overnight launcher |
| `results/checkpoints/` | Policy and model weights |
| `results/external/` | Isolated benchmark source trees and environments |
| `results/_archive/` | Archived experiment runs, code snapshots and path maps |

Use a new output directory for each run. Keep long collection and training runs in tmux.

## Compute devices

On supported NVIDIA Linux machines, install the locked CUDA 12 extra with the environment extras you need. On CPU machines, omit `--extra cuda12` from the same install command. Use `uv run --no-sync` after installation to reuse the selected extras. [JAX CUDA installation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu)

```bash
uv sync --locked --group dev --extra minigrid --extra craftax --extra ogbench --extra cuda12
uv run --no-sync wr devices
```

The defaults in `configs/compute.toml` use `platform = "auto"` and `gpus = "all"`. JAX uses all visible GPUs when CUDA is available and falls back to CPU on CPU-only hosts. Explicit GPU selections report an error if the requested devices are unavailable. Use `platform = "cpu"` for a CPU run.

Copy the configuration to `local-file/compute.toml` for a machine-specific override. An explicit `--compute-config` takes precedence over `WR_COMPUTE_CONFIG`, which takes precedence over the repository default. For a two-GPU run, a local file can contain:

```toml
[compute]
platform = "gpu"
gpus = [0, 2]
```

```bash
uv run --no-sync wr --compute-config local-file/compute.toml devices
```

Indices are zero-based within any existing `CUDA_VISIBLE_DEVICES` list. The configuration never expands that list. With no launcher mask, the default CUDA ordering is PCI bus order. CUDA renumbers the selected devices inside the process, and the report includes the effective visibility mask. Add the same global flag before any experiment command, or export `WR_COMPUTE_CONFIG` for a tmux session. The `devices` command runs a sharded reduction and reports the rows placed on each device.

The completion predictor, route flow, feedback update and AntMaze GCBC trainer use JAX data parallelism on a single host. Batch size is global, so the same update uses the same number of training examples on one, two or more devices. Run manifests record the detected backend, device count and visibility mask. Simulator rollout loops execute serially. [JAX automatic parallelization](https://docs.jax.dev/en/latest/parallel.html)

## Collect Craftax trajectories

Use `make_iron_pickaxe_and_survive` for trajectory completion. The agent must craft the iron pickaxe by the deadline and live through the last action. The `make_iron_pickaxe` option scores the first craft and remains available for reading or collecting earlier task variants. [Collection contract](architecture.md#craftax-collection-contract)

The following commands start matching expert and random collections with the same world seeds and step budget. The published PPO-RNN checkpoint supplies the expert actions.

```bash
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

Install the locked OGBench extra using the compute setup above. The following command downloads the state-based train and validation archives when they are absent. [OGBench dataset API](https://github.com/seohongpark/ogbench#dataset-apis)

```bash
uv run --no-sync python -c 'import ogbench; ogbench.download_datasets(["antmaze-large-navigate-v0", "antmaze-large-stitch-v0"], dataset_dir="results/datasets/ogbench")'
```

`wr antmaze-gcbc` trains an MLP by action MSE using public observations, hindsight goals drawn within each trajectory and the relative goal displacement. Training statistics normalize the public inputs. The command scores held-out action MSE on the official validation archive and closed-loop success on the five simulator tasks. The zero-action control is a rollout sanity check. [OGBench reference GCBC](https://github.com/seohongpark/ogbench/blob/master/impls/agents/gcbc.py)

```bash
run_dir="$(mktemp -d results/antmaze-gcbc.XXXXXX)"
uv run --no-sync wr antmaze-gcbc \
  --dataset antmaze-large-navigate-v0 --updates 2000 --width 128 --out "$run_dir/navigate"
uv run --no-sync wr antmaze-gcbc \
  --dataset antmaze-large-stitch-v0 --updates 2000 --width 128 \
  --starts "$run_dir/navigate/starts.npz" --out "$run_dir/stitch"
```

Reuse `starts.npz` across comparisons with the same evaluation split and repeat count. The bank stores the exact reset observations, goals and integration state. The default evaluates four repeats for each task on development seeds with the 1,000-action horizon. Use `--eval-split test` after locking the recipe. Lower `--eval-steps` or `--eval-repeats` for smoke runs. A success comes from the benchmark's binary success event. The episode log also stores the initial, closest and final Euclidean distances in public XY coordinates.

Each run saves its recipe and archive hashes in `manifest.json`, validation checkpoints in `history.json`, parameters and normalization separately, full optimizer state in `training.msgpack`, and exact starts and outcomes in `starts.npz` and `episodes.json`. To resume, pass `--resume PREVIOUS_RUN`, use the same training recipe and archives, specify a new output directory and set `--updates` to the new total. Setting that total to the checkpoint step reruns evaluation at the saved weights. The batch stream is indexed by update so it resumes at the same batch as an uninterrupted run.

## Run learned latent planning in AntMaze

Train the completion model on windows from the official navigation training archive. A window contains recent public observations and actions, a later observation from the same trajectory as its goal, and masked intermediate states at queried time offsets. The official validation archive supplies separate windows. Use a frozen GCBC checkpoint trained on navigation or stitching demonstrations. This example uses the stitching executor trained above.

```bash
checkpoint="$run_dir/stitch"
latent_dir="$(mktemp -d results/antmaze-latent.XXXXXX)"
uv run --no-sync wr antmaze-latent \
  --dataset antmaze-large-navigate-v0 --checkpoint "$checkpoint" \
  --out "$latent_dir/pilot"
```

By default the evaluator creates two development resets per official task. To reuse an existing bank, pass `--starts PATH` with the bank's `--eval-split` and `--eval-repeats`. Both arms get the same reset, original terminal goal, frozen actor and 1,000-action budget. The direct arm pursues the terminal goal throughout. At each replan the latent arm samples four routes and decodes eight states per route. Its fixed selection takes the first route's earliest state, then switches on the corresponding time offset. Each replan conditions on the observations and actions that actually occurred. When the remaining budget is too short to query an interior window, the actor pursues the terminal goal directly. The planner uses public inputs, and the benchmark's binary success event determines each episode's outcome.

`plans.jsonl.gz` records the latent and decoded routes, selected slot, planned deadline and number of actions executed toward that target. The run also saves `weights.msgpack`, `manifest.json`, `starts.npz` and `episodes.json`. The `summary.json` reports paired terminal success and closest distance. The separate `validation.json`, written before rollouts, reports XY errors for encoding and decoding actual observations, decoding the world-model prediction, sampled trajectories and linear interpolation. Best-sample error picks a single sampled route per reference window using the true future. It measures reconstruction of a recorded route under privileged offline scoring.

Add `--probe-first-candidates` to branch from each matched start and try the earliest state of every route in the same initial sample. All branches share the first attempt's deadline and the continuation policy, which observes each branch's actual trajectory. The evaluator records all terminal outcomes in `first_candidates.json` and counts starts where the candidate choice changes success, a later candidate succeeds when the first fails, or any candidate succeeds when direct execution fails. The counts use outcomes from all branches to diagnose a frozen proposal pool. Use `--eval-split test` only after selecting the recipe on development data.

## Evaluate AntMaze waypoint execution

This executor diagnostic asks how far a frozen GCBC actor can follow coordinates supplied by the evaluator. The route comes from privileged simulator geometry. Both arms use the same checkpoint, resets, exact terminal goals, simulator success event and action limit.

```bash
checkpoint=results/antmaze-gcbc.example/navigate
run_dir="$(mktemp -d results/antmaze-waypoints.XXXXXX)"
uv run --no-sync wr antmaze-waypoints --checkpoint "$checkpoint" \
  --starts "$checkpoint/starts.npz" --out "$run_dir/navigate"
```

The evaluator follows the official maze distance map along one shortest cell path, keeps every corner, and samples intermediate cells at most `--waypoint-spacing` apart along straight segments. The default spacing is one cell. It copies the terminal goal vector and replaces only XY with the next cell center. The actor receives the current public state and that constructed target. The diagnostic switches to the next coordinate when observed XY is within `--arrival-radius` (default 1.0). After `--waypoint-steps` actions (default 150) without reaching the coordinate, it switches to the original terminal goal. The radius controls the handoff in this diagnostic. Terminal scores use the benchmark's official success event.

`episodes.json` records the declared path, attempted waypoints, timeouts, and remaining terminal outcomes. The summary reports paired terminal success counts and a task-cluster bootstrap interval. Development starts are the default. Use `--eval-split test` to build an independent bank after freezing settings.

## Run the Diffuser Maze2D pilots

Diffuser's Maze2D experiment uses endpoint conditioning with four state coordinates and follows the generated states with position and velocity feedback. These pilots use Farama's maintained evaluation specifications. The paper reports scores on the original D4RL `maze2d-*-v1` environments and datasets. A same-version benchmark will run both methods on the same environments under matched budgets. [Diffuser task configuration](https://github.com/jannerm/diffuser/blob/maze2d/config/maze2d.py) · [Diffuser controller and evaluation](https://github.com/jannerm/diffuser/blob/maze2d/scripts/plan_maze2d.py)

| `--task` | Gymnasium environment | Action limit | Minari dataset |
| --- | --- | ---: | --- |
| `umaze` | `PointMaze_UMaze-v3` | 300 | [`D4RL/pointmaze/umaze-v2`](https://minari.farama.org/datasets/D4RL/pointmaze/umaze-v2/#evaluation-environment-specs) |
| `medium` (default) | `PointMaze_Medium-v3` | 600 | [`D4RL/pointmaze/medium-v2`](https://minari.farama.org/datasets/D4RL/pointmaze/medium-v2/#evaluation-environment-specs) |
| `large` | `PointMaze_Large-v3` | 800 | [`D4RL/pointmaze/large-v2`](https://minari.farama.org/datasets/D4RL/pointmaze/large-v2/#evaluation-environment-specs) |

```bash
uv sync --locked --extra pointmaze
export MINARI_DATASETS_PATH="$PWD/results/datasets/minari"
mkdir -p results
run_dir="$(mktemp -d results/pointmaze-diffuser.XXXXXX)"
for task in umaze medium large; do
  dataset="D4RL/pointmaze/$task-v2"
  if [[ ! -f "$MINARI_DATASETS_PATH/$dataset/data/main_data.hdf5" ]]; then
    uv run --no-sync minari download "$dataset"
  fi
  uv run --no-sync wr pointmaze-latent --task "$task" --out "$run_dir/$task"
done
```

Include `--extra pointmaze` alongside any installed extras you want to keep, including `--extra cuda12` for GPU. Each Minari archive contains a continuing collection. The loader uses the first 80% of episodes by ID for fitting, skips the boundary episode and holds out the suffix for offline reconstruction and prediction diagnostics. The evaluator uses paired seeded resets on the fixed-goal, continuing task. Both arms use the task's action limit and the same position and velocity controller. Use `--eval-steps` for a shorter run. Direct execution tracks the terminal goal. Latent execution uses the earliest decoded state of the first sampled route and replans when the queried time offset elapses. Terminal success means the native success event occurred by the action limit. Sparse return is summed across the whole episode, including success steps. Use `--eval-split test` after fixing the recipe.

The run stores the data hashes and versions in `manifest.json`, archive episode IDs in `split.json`, sampled latent and decoded routes in `plans.jsonl.gz`, executed public states in `trajectories.npz`, and scores in `episodes.json` and `summary.json`.

## Run conditional Kuka stacking

Install the locked `stacking` extra alongside your existing extras. PyBullet builds from source on Linux with CPython 3.12, using a C/C++ compiler and Python development headers. A uv-managed Python 3.12 installation supplies those headers when the system installation lacks them.

```bash
uv sync --locked --group dev --extra minigrid --extra craftax --extra ogbench --extra pointmaze --extra stacking --extra cuda12
bash scripts/prepare_diffuser_kuka.sh
run_dir="$(mktemp -d results/diffuser-kuka.pilot.XXXXXX)"
uv run --no-sync wr kuka-latent --out "$run_dir/conditional"
```

The setup script pins the author's `kuka` branch, applies a compatibility patch for Python 3.12 and headless PyBullet, and verifies the archive checksum. The loader uses the released 39-coordinate state clips, keeps every second frame and caps each clip at 300 frames, matching the author's loader. It splits complete files before sampling windows. Each clip can show an individual placement in a partly completed scene. At deployment, the goal specifies a complete four-block order. The goal encoding leaves horizontal translation free and includes block positions without joint angles or grasp flags. [Author release and setup](https://github.com/jannerm/diffuser/tree/kuka) · [Released trajectory loader](https://github.com/jannerm/diffuser/blob/kuka/denoising_diffusion_pytorch/datasets/tamp.py)

Both arms use the same reset, complete terminal order and executor. One follows the decoded forecast and the other follows the first sampled route, taking the earliest queried state and replanning at its time offset. Each control cycle uses ten PyBullet simulation steps. The evaluator uses 384 control cycles, the author's three 128-step planning chunks. At the deadline all three adjacent pairs must be in the requested order, in physical contact, and released from the gripper. The upstream conditional placement score is logged separately. `--eval-split test` selects disjoint reset seeds. [Author conditional evaluator](https://github.com/jannerm/diffuser/blob/kuka/scripts/conditional_kuka_planning_eval.py)

The run records `manifest.json`, the clip split and hashes, latent and decoded plans, public trajectories, and both terminal metrics. The two arms measure the effect of proposal sampling within the shared latent planner. The author's released Diffuser checkpoint can supply a future same-environment comparison.
