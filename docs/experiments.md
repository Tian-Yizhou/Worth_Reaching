# Running experiments

Use Python 3.12 or newer and run from the repository root. Install the project and the extras for your workflow. For a single CPU rollout with a known outcome, start with KeyCorridor. [Project setup](../README.md#setup) · [KeyCorridor instructions](#replay-a-keycorridor-decision)

## Choose a workflow

Each row lists the experiment extra to pass to `uv sync --locked`. For model training, also select `--extra training-cpu` or `--extra training`. Include all extras you want to keep in the same sync command. Run `uv run --no-sync wr --help` to list commands and `uv run --no-sync wr COMMAND --help` for their options.

| Workflow | Experiment extra | Inputs and next step |
| --- | --- | --- |
| [Dependency grid](#compare-ordered-targets-in-the-dependency-grid) | Base install | Run matched target attempts at two depths with no dataset or checkpoint. |
| [Grid prediction](#train-and-test-latent-trajectory-completion) | `craftax` (Flax serialization) | Generate demonstrations with `train-completion --grid`, use its checkpoint with [grid-routes](#sample-and-execute-alternative-routes), then pass both runs to [grid-feedback](#train-from-execution-feedback). |
| [Craftax collection](#collect-craftax-trajectories) | `craftax` | Collect worlds with a random policy or the expert checkpoint. [Export windows](#prepare-masked-trajectory-windows) from successful demonstrations and pass them to `train-completion --windows`. |
| [PointMaze](#run-the-diffuser-maze2d-pilots) | `pointmaze` | Download the corresponding Minari dataset. Train with `pointmaze-latent`, then reuse frozen Medium runs for the [energy comparison](#test-latent-energy-against-terminal-success). |
| [AntMaze](#train-the-antmaze-gcbc-baseline) | `ogbench` | Train the GCBC actor on the official archive. Reuse its checkpoint and saved starts for [latent planning](#run-learned-latent-planning-in-antmaze) or [waypoint evaluation](#evaluate-antmaze-waypoint-execution). |
| [Kuka](#run-conditional-kuka-stacking) | `stacking` | Prepare the Diffuser code and demonstrations with the supplied script, then run `kuka-latent`. |
| OGBench [Scene](#run-state-based-ogbench-scene) and [Puzzle](#run-state-based-ogbench-puzzle) | `ogbench` | Download the play archive for the selected task and run `scene-latent` or `puzzle-latent`. |

For scripts that consume these artifacts in Python, see the [task-package import guide](python-imports.md).

## Replay a KeyCorridor decision

This pilot requires only the MiniGrid extra. It replays the same public action history, branches on the next action, and hands both continuations to the same policy.

```bash
uv sync --locked --extra minigrid
mkdir -p results
run_dir="$(mktemp -d results/keycorridor.XXXXXX)"
uv run --no-sync wr pilot --config configs/minigrid/keycorridor.toml --out "$run_dir/pilot"
cat "$run_dir/pilot/summary.json"
```

With the supplied configuration, turning right succeeds on the final step. Moving forward into the wall ends in a timeout.

## Compare ordered targets in the dependency grid

Run a short CPU diagnostic after the [base installation](../README.md#setup).

```bash
mkdir -p results
run_dir="$(mktemp -d results/dependency-grid.XXXXXX)"
uv run --no-sync wr dependency-grid --out "$run_dir/dev"
```

Each layout supplies a shared start and two terminal destinations at both depths. Depth one requires the entry key to pass the final gate. Depth two requires the entry key to reach the inner chamber and the inner key to pass the final gate. Keys are acquired on contact. Some keys lie on a geometric route, and others require a detour. The controller tracks shortest paths on the public map geometry. The simulator blocks movement through locked gates.

Per goal, the diagnostic runs direct tracking, each lane's individual entry and inner targets, and both orders of each pair. The total allowance is 20 actions. The first and second target deadlines are at actions 7 and 14. When the attempt ends, the same local controller continues toward the terminal goal with the remaining budget. The same per-step slip draws apply to every branch, both goals, and both depths at a given layout and repeat.

`manifest.json` records the layout IDs, settings, and controller. `decisions.jsonl` records the declared targets, matched draws, and outcomes. `summary.json` reports terminal success and the difference from direct tracking by depth and goal, arrivals, successes after misses, and strict ranking reversals when the goal switches. At depth one, the reversal compares the two entry targets. At depth two, it compares the two entry-then-inner sequences. The diagnostic's targets are predeclared key cells. Learning and selecting latent targets is the next milestone. [Research plan](latent-subgoal-research.md)

Development uses 16 layouts and 16 repeats by default. To shorten a smoke check, append `--layouts 1 --repeats 2`. Use the disjoint `--split test` layouts after fixing the protocol.

## Check a development installation

From the repository root, install the CPU dependencies used by CI and run the repository checks.

```bash
uv sync --locked --group dev --extra training-cpu --extra minigrid --extra craftax --extra ogbench --extra pointmaze
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src tests scripts
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync pytest
```

## Run artifacts

Artifacts stay in `results/` on the server. Git and the Mac sync scripts exclude the entire directory.

| Directory | Contents |
| --- | --- |
| `results/datasets/` | Complete trajectory collections and their logs |
| `results/completion.*/` | Masked trajectory windows and export manifests |
| `results/grid-completion.*/` | Paired grid predictor weights, world assignments and evaluation |
| `results/grid-routes.*/` | Latent flow checkpoints and matched attempt records |
| `results/grid-feedback.*/` | Training attempts, target masks, updated flows and held-out feedback comparisons |
| `results/dependency-grid.*/` | Layouts, matched target attempts and terminal success by goal and depth |
| `results/antmaze-gcbc.*/` | GCBC checkpoints, matched reset banks and per-task rollout outcomes |
| `results/antmaze-latent.*/` | Completion weights, sampled and decoded plans, and paired terminal outcomes |
| `results/antmaze-waypoints.*/` | Paired direct and oracle waypoint execution outcomes |
| `results/pointmaze-diffuser.*/` | PointMaze archive splits, decoded plans, executed trajectories and paired outcomes |
| `results/diffuser-kuka.pilot.*/` | Conditional stacking clip splits, decoded plans, executed states and tower scores |
| `results/ogbench-scene.*/` | Frozen Scene executor and completion weights, matched starts, decoded plans and rollout outcomes |
| `results/ogbench-puzzle.*/` | Puzzle artifacts indexed by board, including public rollouts and paired successes |
| `results/benchmark-suite.*/` | Per-task development runs and logs from the overnight launcher |
| `results/checkpoints/` | Policy and model weights |
| `results/external/` | Isolated benchmark source trees and environments |
| `results/_archive/` | Archived experiment runs, code snapshots and path maps |

Use a new output directory for each run. Keep long collection and training runs in tmux.

## Compute devices

Install `training` for the locked PyTorch CUDA build on Linux, or `training-cpu` for CPU development and CI. Use `uv run --no-sync` after installation to reuse the selected extras. Add `cuda12` when you also need GPU acceleration for JAX Craftax collection. [JAX CUDA installation](https://docs.jax.dev/en/latest/installation.html#nvidia-gpu) · [PyTorch installation](https://pytorch.org/get-started/locally/)

```bash
uv sync --locked --group dev --extra training --extra minigrid --extra craftax --extra ogbench
uv run --no-sync wr devices
```

The defaults in `configs/compute.toml` use `platform = "auto"` and `gpus = "all"`. Lightning uses all visible GPUs when CUDA is available and falls back to CPU on CPU-only hosts. Explicit GPU selections report an error if the requested devices are unavailable. Use `platform = "cpu"` for a CPU run.

Copy the configuration to `local-file/compute.toml` for a machine-specific override. An explicit `--compute-config` takes precedence over `WR_COMPUTE_CONFIG`, which takes precedence over the repository default. For a two-GPU run, a local file can contain:

```toml
[compute]
platform = "gpu"
gpus = [0, 2]
```

```bash
uv run --no-sync wr --compute-config local-file/compute.toml devices
```

Indices are zero-based within any existing `CUDA_VISIBLE_DEVICES` list. The configuration never expands that list. With no launcher mask, the default CUDA ordering is PCI bus order. CUDA renumbers the selected devices inside the process, and the report includes the effective visibility mask. Add the same global flag before any experiment command, or export `WR_COMPUTE_CONFIG` for a tmux session. Run `wr devices --framework jax` to check the Craftax backend.

The completion predictor, route flow, feedback update and GCBC trainer use Lightning's single-host DDP. Batch size is global, so each update uses the same examples on one or more devices. Run manifests record the framework, backend, device count and visibility mask. Simulator rollout loops execute serially. [Lightning distributed strategies](https://lightning.ai/docs/pytorch/stable/reference/accelerators/gpu_intermediate)

## Configure a completion model

`wr train-completion`, `antmaze-latent`, `pointmaze-latent`, `kuka-latent`, `scene-latent` and `puzzle-latent` accept `--config PATH` after the subcommand. A TOML `[model]` table overrides the command's defaults. Explicit command-line options take precedence. The run manifest saves the resolved settings alongside the dataset and split information.

```bash
run_dir="$(mktemp -d results/pointmaze-recipe.XXXXXX)"
uv run --no-sync wr pointmaze-latent --config configs/pointmaze/completion-smoke.toml \
  --task medium --seed 7 --eval-split dev --eval-repeats 1 --out "$run_dir/medium"
```

For a prepared Craftax window corpus, use `wr train-completion --windows PATH --config configs/craftax/completion-smoke.toml --out RUN`. Both example files use two optimizer updates per stage to check the training path.

Both model families accept `width`, `latent_size`, `batch_size`, `learning_rate`, `seed`, `predictor_val_every` and `predictor_patience`. For the continuous model, use `history`, `queries`, `windows`, `codec_updates`, `predictor_updates`, `flow_updates` and `candidates`. Its batch size sets the codec and predictor batches. For Craftax and grid predictor training, use `history_length`, `queries_per_window`, `encoder_updates` and `updates`. Each task supplies its public input and goal format. In PointMaze, use `--prediction-span` for the maximum prediction span and `--eval-budgets` to evaluate several action budgets using one set of weights.

## Select a predictor checkpoint

The predictor monitors `val_latent_mse`, the mean squared error at valid masked query positions in fixed development windows encoded by the frozen codec. Set `--predictor-val-every` to change the default interval of 200 updates and `--predictor-patience` to change the default of five validation checks without improvement. The update budget caps training. A budget shorter than the interval is validated at its last update. The grid and Craftax goal-conditioned and goal-blind predictors share validation windows and stopping settings and select their checkpoints independently. [Lightning EarlyStopping](https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.callbacks.EarlyStopping)

The predictor checkpoint directory contains `best.ckpt`, selected by validation error, and `selection.json`, which records the best score, the selected update, the actual final update and the stop reason. Continuous benchmarks use the best predictor's forecasts to fit the route generator. Grid route training uses the exported best goal-conditioned predictor. Test rollouts stay outside checkpoint selection. [Lightning ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.callbacks.ModelCheckpoint)

## Resume training

Each stage writes Lightning checkpoints and CSV metrics under `checkpoints/<stage>/` in the run directory. `last.ckpt` records the latest saved training state, and `final.ckpt` marks completion. Predictor stages save progress at validation checks and restore the best weights when completed. Pass `--resume PREVIOUS_RUN` with the original data and settings and a new `--out` directory. Interrupted stages recover optimizer and callback state, and evaluation runs after training.

```bash
old_run=results/pointmaze-diffuser.example/umaze
new_run="$(mktemp -d results/pointmaze-resume.XXXXXX)/umaze"
uv run --no-sync wr pointmaze-latent --task umaze --resume "$old_run" --out "$new_run"
```

Add any non-default flags from the original run to the resume command. Published Flax GCBC and grid weights remain readable for frozen inference. Resumption with optimizer state uses a Lightning checkpoint from the same training recipe. [Lightning checkpoint format](https://lightning.ai/docs/pytorch/stable/reference/common/checkpointing_basic)

## Collect Craftax trajectories

Use `make_iron_pickaxe_and_survive` for trajectory completion. The agent must craft the iron pickaxe by the deadline and live through the last action. The `make_iron_pickaxe` option scores the first craft and remains available for reading or collecting earlier task variants. [Collection contract](architecture.md#craftax-collection-contract)

With the `craftax` extra installed, this short random-policy collection exercises the recorder and split assignment. For training examples, use a collection with successful episodes.

```bash
mkdir -p results
collection="$(mktemp -d results/craftax-demo.XXXXXX)/collection"
uv run --no-sync wr collect --config configs/craftax/iron_pickaxe.toml \
  --task make_iron_pickaxe_and_survive --out "$collection"
cat "$collection/summary.json"
```

The following commands start matching expert and random collections with the same world seeds and step budget. The published PPO-RNN checkpoint supplies the expert actions.

```bash
python3 scripts/fetch_craftax_expert.py

tmux new-session -d -s wr-craftax-expert -c "$PWD" \
  'bash scripts/run_craftax_collection.sh expert 1000 256 25000 make_iron_pickaxe_and_survive'
tmux new-session -d -s wr-craftax-random -c "$PWD" \
  'bash scripts/run_craftax_collection.sh random 1000 256 25000 make_iron_pickaxe_and_survive'
```

Attach with `tmux attach -t wr-craftax-expert` or `tmux attach -t wr-craftax-random`. The launcher prints the resolved path and writes the log next to the run.

A completed run contains `manifest.json` for configuration and provenance, `splits.json` for world assignments, `trajectories.jsonl.gz` for observations and actions, and `summary.json` for counts and outcomes. The loader `worth_reaching.tasks.craftax.dataset.load_craftax_archive` validates the manifest and streams complete episodes with `archive.iter_episodes("train")`. Fit models on the training split, choose settings in development, and reserve the test split for final evaluation.

## Prepare masked trajectory windows

Run `wr prepare-windows --collection PATH --out PATH` on a completed `make_iron_pickaxe_and_survive` collection. Set `--seed` to reproduce the sampling recipe. Use `--windows-per-episode`, `--min-span` and `--max-span` to control the number and lengths of windows. By default the mask length ranges from one step to the last eligible interior step at each start.

The export stores the sampling recipe and source split fingerprint in `manifest.json`, episode and window counts in `summary.json`, and one compressed array file per selected episode in `train/` or `dev/`. Windows reference slices in the episode file. The reader separates public inputs, masked targets and episode identifiers. [Window contract](architecture.md#masked-trajectory-window-contract)

```python
from pathlib import Path

from worth_reaching.tasks.craftax.windows import load_completion_corpus

corpus = load_completion_corpus(Path("results/completion.example/windows"))
example = next(corpus.iter_examples("train"))
inputs, target = example.inputs, example.target
```

The optional `corpus.iter_examples("train", with_terminal_example=True)` provides the recorded terminal observation for training with endpoint examples. Use the default task descriptor for training for goal-only deployment. Development windows measure prediction on successful demonstrations from separate worlds. This corpus has one task descriptor. Testing whether predictions respond to the goal itself requires demonstrations with distinct goals. Measure task success later through rollouts from natural resets.

## Train and test latent trajectory completion

Run `wr train-completion --grid --out PATH` for the two-goal diagnostic or `wr train-completion --windows PATH --out PATH` for the exported Craftax corpus. The trainer runs both the goal-conditioned model and its history-and-time control. `--seed`, `--encoder-updates`, `--updates`, `--history-length` and `--queries-per-window` specify the training and batching recipe. The defaults use 250 encoder updates, up to 600 predictor updates per arm, a 16-observation history and up to 16 query positions per window. [Predictor checkpoint selection](#select-a-predictor-checkpoint)

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

Each run saves its recipe and archive hashes in `manifest.json`, validation checkpoints in `history.json`, parameters and normalization separately, full optimizer state under `checkpoints/executor/`, and exact starts and outcomes in `starts.npz` and `episodes.json`. To resume, pass `--resume PREVIOUS_RUN`, use the same training recipe and archives, specify a new output directory and set `--updates` to the new total. Setting that total to the checkpoint step reruns evaluation at the saved weights. The batch stream is indexed by update so it resumes at the same batch as an uninterrupted run.

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

| `--task` | Gymnasium environment | Official action limit | Minari dataset |
| --- | --- | ---: | --- |
| `umaze` | `PointMaze_UMaze-v3` | 300 | [`D4RL/pointmaze/umaze-v2`](https://minari.farama.org/datasets/D4RL/pointmaze/umaze-v2/#evaluation-environment-specs) |
| `medium` (default) | `PointMaze_Medium-v3` | 600 | [`D4RL/pointmaze/medium-v2`](https://minari.farama.org/datasets/D4RL/pointmaze/medium-v2/#evaluation-environment-specs) |
| `large` | `PointMaze_Large-v3` | 800 | [`D4RL/pointmaze/large-v2`](https://minari.farama.org/datasets/D4RL/pointmaze/large-v2/#evaluation-environment-specs) |

```bash
uv sync --locked --extra training --extra pointmaze
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

Include `--extra pointmaze` alongside any installed extras you want to keep. Each Minari archive contains a continuing collection. The loader uses the first 80% of episodes by ID for fitting, skips the boundary episode and holds out the suffix for offline reconstruction and prediction diagnostics. The evaluator uses paired seeded resets on the fixed-goal, continuing task and a shared position and velocity controller.

Evaluation defaults to the task's official action limit. `--eval-steps` sets a single action budget within that limit. `--eval-budgets` evaluates one trained model on several budgets with the same seeded starts and terminal goals. Each budget gets its own rollout with that budget supplied to the planner. Terminal success counts a native success event at any step through the budget, including on the last action. Each episode records the first success step.

The model's maximum prediction span defaults to the 90th percentile of training episode lengths eligible for masking, capped by the official limit and the longest validation episode. Use `--prediction-span` to specify a span covered by both training and validation. A quarter of each sample set starts at the first observation of a trajectory and covers that full span. Full-span training windows can carry different remaining action budgets up to the official limit while keeping their observed endpoints and queries fixed. This supplies examples of reaching an endpoint within a larger budget. At deployment the model receives the actual remaining action budget, and its query offsets are computed over the smaller of the remaining budget and the configured prediction span. The manifest records both training and evaluation coverage.

Direct execution tracks the terminal goal. Latent execution follows the earliest decoded state of the first sampled route and replans when the queried time offset elapses. Sparse return is summed across the action budget. Use `--eval-split test` after fixing the recipe. Archived `v2` and `v3` runs can still be probed under their recorded protocols. State the evaluation budget and prediction span when comparing scores.

The run stores the data hashes and versions in `manifest.json`, archive episode IDs in `split.json`, sampled latent and decoded routes in `plans.jsonl.gz`, executed public states in `trajectories.npz`, and scores in `episodes.json` and `summary.json`.

#### Sweep the action budget in Maze2D Medium

```bash
run_dir="$(mktemp -d results/pointmaze-budgets.XXXXXX)"
uv run --no-sync wr pointmaze-latent --task medium --eval-split test \
  --eval-budgets 300 400 500 600 --out "$run_dir/medium"
```

The sweep writes one set of trained weights and a `summary.json` with results by budget. Each `budget-<steps>/` contains its own manifest, terminal outcomes, trajectories, and plans. Each plan records both `remaining_budget` and `prediction_span`. Pass a budget subdirectory to the frozen checkpoint probes for that budget.

### Diagnose frozen Maze2D first targets

Use completed runs with the same task, action budget, prediction span, reset bank, environment versions, and dataset. The probe restores the predictor, codec, and flow from the saved weights and uses the original first query time and sampling seed. Here is a two-seed Medium run with the official evaluation budget and an archive-derived prediction span.

```bash
run_dir="$(mktemp -d results/pointmaze-horizon-aligned.XXXXXX)"
for seed in 7 19; do
  uv run --no-sync wr pointmaze-latent --task medium --seed "$seed" \
    --windows 16000 --codec-updates 1200 --predictor-updates 4800 \
    --flow-updates 1600 --eval-repeats 24 --out "$run_dir/medium-$seed"
done
uv run --no-sync wr pointmaze-first-targets \
  --runs "$run_dir"/medium-{7,19} --out "$run_dir/first-targets"
```

For each reset, compare direct terminal tracking, the decoded forecast, and the first state of each sampled route. Every candidate uses the same feedback controller and switches to direct terminal tracking on arrival or at the first query deadline. Arrival applies the benchmark's native sparse-goal rule to public XY. Terminal success uses the original environment goal and the budget recorded by the training run.

A diagnostic branch uses the public state at the first query time from the same-reset archived trajectory that achieved terminal success earliest after that time. A reset without such a trajectory has no reference branch. The reference uses future outcomes and is evaluated under the same controller and continuation as the generated targets. The summary counts both successes with and without arrival in the allotted time, plus the number of starts where any sampled candidate succeeds in hindsight. The per-start targets, outcomes and archived reference sources are in `episodes.json`, and public rollout states are in `trajectories.npz`. Use `--max-starts 1` for a smoke test.

### Compare continuations after the same Maze2D prefix

```bash
uv run --no-sync wr pointmaze-first-targets --compare-continuations \
  --runs "$run_dir"/medium-{7,19} --out "$run_dir/continuations"
```

Each model selects the earliest state of its first sampled route. Both branches use the same feedback controller to pursue that target until arrival or the first query deadline. The evaluator replays the same reset and verifies that the actions and public observations match through the handoff. One branch then tracks the terminal goal directly. The other begins replanning at the handoff using the actual observation and action history and the remaining budget. Subsequent replans use the model's original time-based schedule and selection rule. The direct-from-reset arm measures terminal tracking without the shared prefix.

The summary reports paired terminal wins, losses, and results after arrival and after a miss. The action budget is matched. The replanning branch uses additional model calls, which are reported with rollout time. The episode log records each handoff and subsequent selected target. The trajectory archive contains public branch states and the verified shared actions.

### Test latent energy against terminal success

Use the checkpoints from the two-seed Medium run above, or select a budget subdirectory from each checkpoint's sweep. Each start supplies a public reset observation, terminal goal, and remaining budget. The frozen world model and its completion flow draw candidates and an independent reference sample. Both draws are truncated to the prefix that will be attempted. The score is the negative log Gaussian-smoothed marginal density of that prefix under the flow draws, divided by the number of latent coordinates and reported up to a fixed additive constant. Smoothing uses the frozen residual normalization and Scott's bandwidth in those coordinates. The predictor's squared latent error is recorded as a separate ranking on the same candidates. This setup compares rankings without fitting either score to execution outcomes.

```bash
uv run --no-sync wr pointmaze-latent-energy --runs "$run_dir"/medium-{7,19} \
  --eval-split test --eval-repeats 64 --prefix-length 1 \
  --candidates 16 --density-samples 256 --out "$run_dir/energy-single"
uv run --no-sync wr pointmaze-latent-energy --runs "$run_dir"/medium-{7,19} \
  --eval-split test --eval-repeats 64 --prefix-length 2 \
  --candidates 16 --density-samples 256 --out "$run_dir/energy-pair"
```

The evaluator first writes all candidates, energy scores, and selections to `candidate-bank.json` and records its hash. It then replays each reset and attempts every declared candidate. The shared controller pursues the decoded targets in order, using the original absolute query deadlines. The benchmark's native sparse rule measures arrival in public XY. The first miss or the final arrival hands control to direct terminal tracking for the rest of the budget. The pair experiment scores and attempts both positions in order. Any suffix after the attempted prefix is integrated out by projecting the independent reference draws onto that prefix.

`episodes.json` records attempted targets by candidate index, stage arrival or miss, steps spent, remaining budget, and terminal success. `summary.json` compares lowest and highest energy, lowest forecast error, the first flow draw, and the expected success of uniform choice on the shared bank. It also reports the direct-from-reset baseline and includes terminal successes after a missed target. The experiment measures conditional model energy across candidates within a start and prefix length. Reference draws estimate model density, while demonstration prevalence requires a separate measurement on the source trajectories. Set `--eval-split dev --eval-repeats 1 --candidates 3 --density-samples 16` for a smoke check.

## Run conditional Kuka stacking

Install the locked `stacking` extra alongside your existing extras. PyBullet builds from source on Linux with CPython 3.12, using a C/C++ compiler and Python development headers. A uv-managed Python 3.12 installation supplies those headers when the system installation lacks them.

```bash
uv sync --locked --group dev --extra training --extra minigrid --extra craftax --extra ogbench --extra pointmaze --extra stacking
bash scripts/prepare_diffuser_kuka.sh
run_dir="$(mktemp -d results/diffuser-kuka.pilot.XXXXXX)"
uv run --no-sync wr kuka-latent --out "$run_dir/conditional"
```

The setup script pins the author's `kuka` branch, applies a compatibility patch for Python 3.12 and headless PyBullet, and verifies the archive checksum. The loader uses the released 39-coordinate state clips, keeps every second frame and caps each clip at 300 frames, matching the author's loader. It splits complete files before sampling windows. Each clip can show an individual placement in a partly completed scene. At deployment, the goal specifies a complete four-block order. The goal encoding leaves horizontal translation free and includes block positions without joint angles or grasp flags. [Author release and setup](https://github.com/jannerm/diffuser/tree/kuka) · [Released trajectory loader](https://github.com/jannerm/diffuser/blob/kuka/denoising_diffusion_pytorch/datasets/tamp.py)

Both arms use the same reset, complete terminal order and executor. One follows the decoded forecast and the other follows the first sampled route, taking the earliest queried state and replanning at its time offset. Each control cycle uses ten PyBullet simulation steps. The evaluator uses 384 control cycles, the author's three 128-step planning chunks. At the deadline all three adjacent pairs must be in the requested order, in physical contact, and released from the gripper. The upstream conditional placement score is logged separately. `--eval-split test` selects disjoint reset seeds. [Author conditional evaluator](https://github.com/jannerm/diffuser/blob/kuka/scripts/conditional_kuka_planning_eval.py)

The run records `manifest.json`, the clip split and hashes, latent and decoded plans, public trajectories, and both terminal metrics. The two arms measure the effect of proposal sampling within the shared latent planner. The author's released Diffuser checkpoint can supply a future same-environment comparison.

## Run state-based OGBench Scene

Install the locked `ogbench` extra and select a Lightning training extra. With the default compute configuration, `wr devices` reports the selected backend and all visible GPUs. The Scene executor and completion model both use the same Lightning training path. Evaluation steps the MuJoCo environment on the host. [Compute configuration](#compute-devices)

```bash
uv run --no-sync python -c 'import ogbench; ogbench.download_datasets(["scene-play-v0"], dataset_dir="results/datasets/ogbench")'
run_dir="$(mktemp -d results/ogbench-scene.XXXXXX)"
uv run --no-sync wr scene-latent --out "$run_dir/state" --eval-repeats 1
```

`scene-v0` supplies a 40-feature public state, five continuous action coordinates and five goal configurations involving the cube, drawer, window and locks. Both arms reuse exactly the same seeded start and goal. Each rollout stops on the official success event or the 750-action limit. Use `--task 4` to isolate the cube-in-drawer task, `--eval-steps` to shorten a smoke test, and `--eval-split test` after fixing the experiment recipe. [OGBench Scene](https://seohong.me/projects/ogbench/#scene) · [Dataset and task protocol](https://github.com/seohongpark/ogbench)

Executor training alternates hindsight-goal batches capped at 64 transitions and batches that sample any later observation in the same training episode. Both arms use that actor after training. The direct arm tracks the terminal observation. The latent arm takes the earliest state of the first sampled decoded route and replans when its time offset elapses. The official input features and action history feed the planner. Extra archive fields for MuJoCo state and buttons are outside the training loader.

The manifest identifies the official train and validation archives by checksum. Offline validation reports action MSE and normalized reconstruction and prediction MSE. The other artifacts contain exact starts, public trajectories with executed actions, sampled plans, weights and paired success by task.

## Run state-based OGBench Puzzle

The robot presses buttons to match the full goal pattern. The evaluator uses OGBench's five official goal configurations and scores the native success event. The board selects the official dataset, observation width and action budget. [OGBench Puzzle](https://seohong.me/projects/ogbench/#puzzle) · [Official task and dataset specifications](https://arxiv.org/html/2410.20092#A5.SS1)

| `--board` | Training archive | State features | Action limit |
| --- | --- | ---: | ---: |
| `3x3` | `puzzle-3x3-play-v0` | 55 | 500 |
| `4x4` (default) | `puzzle-4x4-play-v0` | 83 | 500 |
| `4x5` | `puzzle-4x5-play-v0` | 99 | 1,000 |
| `4x6` | `puzzle-4x6-play-v0` | 115 | 1,000 |

```bash
board=4x4
uv run --no-sync python -c 'import ogbench, sys; ogbench.download_datasets([f"puzzle-{sys.argv[1]}-play-v0"], dataset_dir="results/datasets/ogbench")' "$board"
run_dir="$(mktemp -d results/ogbench-puzzle.XXXXXX)"
uv run --no-sync wr puzzle-latent --board "$board" --out "$run_dir/$board" --eval-repeats 1
```

The runner uses the same executor recipe and artifact layout as Scene. Both arms share the exact start, public goal, frozen executor and action budget. The latent arm generates candidate intermediate observations and replans from feedback. The archive's extra physics and button-state arrays stay outside training. Use `--task` to select one official goal, `--eval-steps` to shorten a smoke test and `--eval-split test` after fixing the recipe.
