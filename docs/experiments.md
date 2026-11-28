# Running experiments

Use Python 3.12 or newer and run from the repository root. Install the project and the extras for your workflow. For a CPU run with no dataset or checkpoint, start with the dependency grid. [Project setup](../README.md#setup) · [Dependency grid](#compare-ordered-targets-in-the-dependency-grid)

## Choose a workflow

Each row lists the experiment extra to pass to `uv sync --locked`. For model training, also select `--extra training-cpu` or `--extra training`. Include all extras you want to keep in the same sync command. Run `uv run --no-sync wr --help` to list commands and `uv run --no-sync wr COMMAND --help` for their options.

| Workflow | Experiment extra | Inputs and next step |
| --- | --- | --- |
| [Dependency grid](#compare-ordered-targets-in-the-dependency-grid) | Base install | Run matched target attempts at two depths with no dataset or checkpoint. |
| [Dependency-grid latent goals](#test-frozen-latent-goals-in-the-dependency-grid) | Training only | Collect public trajectories, train a completion model, and compare frozen selection and replay. |
| [Region goals in the grid](#compare-region-goals-with-decoded-point-goals) | Training only | Train the region decoder and goal executors, then compare region and point goals on the same latent proposals. |
| [Budgeted routes](#compose-latent-goals-under-action-budgets) | Training only | Compose latent goals and compare routes across action budgets. |
| [Selector comparison](#compare-selectors-on-matched-attempts) | Training only | Fit and compare selectors on exported `shared_attempts_v1` banks. |
| [PointMaze](#run-the-diffuser-maze2d-pilots) | `pointmaze` | Download the corresponding Minari dataset. Train with `pointmaze-latent`, then reuse frozen Medium runs for the [energy comparison](#test-latent-energy-against-terminal-success). |
| [OGBench Scene](#run-state-based-ogbench-scene) | `ogbench` | Download the Scene play archive and run `scene-latent`. Add `scene-warp` for batched CUDA evaluation. |
| [Latent options in Maze2D](#plan-with-latent-options-in-maze2d) | `pointmaze` | Reuse a frozen `pointmaze-latent` budget directory, then train the executor and option value and compare the planner with direct execution and the oracle. |
| [Latent options in Scene](#plan-with-latent-options-in-scene) | `ogbench` | Reuse a `scene-regions` run's world model and latent executor, then fit the option value and compare the planner with direct execution. |

Runs save model weights as PyTorch tensors in `weights.pt`.

## Compare ordered targets in the dependency grid

Run a short CPU diagnostic after the [base installation](../README.md#setup).

```bash
mkdir -p results
run_dir="$(mktemp -d results/dependency-grid.XXXXXX)"
uv run --no-sync wr dependency-grid --out "$run_dir/dev"
```

Each layout supplies a shared start and two terminal destinations at both depths. Depth one requires the entry key to pass the final gate. Depth two requires the entry key to reach the inner chamber and the inner key to pass the final gate. Keys are acquired on contact. Some keys lie on a geometric route, and others require a detour. The controller tracks shortest paths on the public map geometry. The simulator blocks movement through locked gates.

Per goal, the diagnostic runs direct tracking, each lane's individual entry and inner targets, and both orders of each pair. The total allowance is 20 actions. The first and second target deadlines are at actions 7 and 14. When the attempt ends, the same local controller continues toward the terminal goal with the remaining budget. The same per-step slip draws apply to every branch, both goals, and both depths at a given layout and repeat.

`manifest.json` records the layout IDs, settings, and controller. `decisions.jsonl` records the declared targets, matched draws, and outcomes. `summary.json` reports terminal success and the difference from direct tracking by depth and goal, arrivals, successes after misses, and strict ranking reversals when the goal switches. At depth one, the reversal compares the two entry targets. At depth two, it compares the two entry-then-inner sequences. The diagnostic's targets are predeclared key cells. [Learned target comparison](#test-frozen-latent-goals-in-the-dependency-grid) · [Research plan](latent-subgoal-research.md)

Development uses 16 layouts and 16 repeats by default. To shorten a smoke check, append `--layouts 1 --repeats 2`. Use the disjoint `--split test` layouts after fixing the protocol.

## Test frozen latent goals in the dependency grid

Install the training and serialization extras. The command generates its own trajectories and writes them alongside the paired attempts.

```bash
uv sync --locked --extra training-cpu
mkdir -p results
run_dir="$(mktemp -d results/dependency-grid-latent.XXXXXX)"
uv run --no-sync wr dependency-grid-latent --seed 7 --out "$run_dir/seed-7"
```

The collector uses terminal-directed shortest-action search with sampled tie-breaking and action slips. It saves complete public observations and intended actions for every episode. Successful train trajectories supply the reconstruction encoder, masked predictor and residual flow. Success on the disjoint development trajectories filters validation windows, which are used for predictor stopping. The model observes position, the public key locations, acquired-key flags, depth, the terminal goal and the remaining budget.

The model forecasts at actions 4 and 8. At depth one the evaluator attempts the first decoded cell, and at depth two it attempts both in order. The first invalid, repeated, start or terminal cell ends the proposed prefix. An empty prefix invokes direct tracking. A nonempty prefix uses the same geometric controller, absolute deadlines at actions 7 and 14, and 20-action budget as the dependency-grid diagnostic. Nearest-layout replay proposes the cells observed at the same query offsets from successful training episodes with the same depth and terminal goal. Each replay candidate comes from a distinct training layout.

`candidate-bank.json` is saved before evaluation. It records the decoded proposals and frozen choices by minimum and maximum conditional flow energy, minimum latent forecast error, and minimum Manhattan distance among the longest valid prefixes. Both model-based scores use the requested latent slots before rounding and truncation. `attempts.jsonl` records matched outcomes for every candidate, the goal-conditioned and goal-blind forecasts, and timed targets that a terminal-directed demonstration achieved on the evaluation layout. The achieved-target replay is an evaluator diagnostic. The best-candidate references for flow and training-layout replay are selected on their own independent repeats and scored on the common evaluation repeats. The uniform rules report the mean over their full candidate banks. The nearest-replay rule selects the first candidate in the distance-ranked replay bank before seeing outcomes.

`summary.json` reports terminal success, arrival, success after a missed target, and candidate prefix lengths. The arrival rate is the fraction of all attempts that reached the full nonempty prefix. Paired intervals resample layouts and keep both goal cases together. The run counts train and development collection, achieved-target collection, reference selection, and evaluation steps separately. [Research plan](latent-subgoal-research.md#2-separate-discovery-selection-and-execution)

Use `--seed` to vary training while keeping the demonstrations, evaluation starts, and repeat streams fixed. Use `--collection-seed` to generate a different demonstration archive deliberately. For a smoke check, append `--layouts 1 --repeats 2 --candidates 2 --density-samples 4 --demonstrations 1 --encoder-updates 2 --updates 2 --flow-updates 2`. After settling the protocol on development layouts, pass `--split test` to evaluate the disjoint layouts with the same train and development split.

## Learn to select a goal from terminal outcomes

Add `--selector-repeats 16` to the latent comparison to fit a shared scorer. The command uses the same installation and model settings.

```bash
mkdir -p results
run_dir="$(mktemp -d results/dependency-grid-outcome.XXXXXX)"
uv run --no-sync wr dependency-grid-latent --seed 7 --selector-repeats 16 --out "$run_dir/seed-7"
```

The generator and geometric controller are frozen during selector training. On each training layout and goal, both candidate pools share a bank of slip draws. Identical executable prefixes, including the empty prefix, are tried once per repeat. Replay proposals for a training start are drawn from other training layouts. The scorer fits a regularized binomial model to terminal success counts. It uses the same goal-relative geometry, public key locations, target coordinates, and deadlines for both sources. A successful continuation after a target miss still supplies a positive outcome.

The fitted scorer ranks each evaluation pool once before rollouts, with direct tracking available in both pools. A score tie with direct tracking selects direct. `selector.npz` records the fitted weights and normalization. `selector-training-attempts.jsonl` contains the training labels and `candidate-bank.json` contains the frozen evaluation scores and choices. The summary compares `flow_outcome` and `replay_outcome` on matched evaluation repeats, reports how often each chose direct tracking, and counts selector training interactions separately. Uniform selection remains a fixed control.

For a short smoke run, add `--selector-repeats 2` to the smoke settings above. The comparison evaluates one high-level choice per start with a fixed continuation. [Research plan](latent-subgoal-research.md#2-separate-discovery-selection-and-execution)

## Compare region goals with decoded point goals

This is experiment E1. It trains the dependency-grid world model as above, then a region decoder `p(s | z)` and two goal-conditioned executors from the same demonstrations. Every arm attempts the same latent candidates, the forecast and the flow samples, from matched starts and slip draws. After the stages end, every arm continues with the shared shortest-path controller toward the terminal goal. [Research plan](latent-subgoal-research.md#4-compare-region-goals-with-decoded-point-goals)

```bash
uv sync --locked --extra training-cpu
run_dir="$(mktemp -d results/dependency-grid-regions-dev.XXXXXX)"
for seed in 7 19 31 43 59; do
  uv run --no-sync wr dependency-grid-regions --seed "$seed" --out "$run_dir/seed-$seed"
done
```

| Arm | Executor and goal | Arrival |
| --- | --- | --- |
| `point_bfs` | Shortest-path controller on the decoded, rounded cell | Exact cell |
| `point_learned` | Observation-conditioned executor on the decoded observation | Exact cell |
| `latent_point@r` | Latent-conditioned executor on `z` | `‖enc(s) − z‖ ≤ ε` |
| `region_gaussian@r` | Latent-conditioned executor on `z` | Support under a diagonal Gaussian matched to the region samples at least κ |
| `region_flow@r` | Latent-conditioned executor on `z` | Kernel support under region decoder samples at least κ |
| `region_decoded@r` | Observation-conditioned executor on one decoder sample per repeat | Kernel support under region decoder samples at least κ |
| `sample_cell` | Observation-conditioned executor on one decoder sample per repeat | Exact cell of the sample |
| `decoded_support@r` | Observation-conditioned executor on the decoded observation | Kernel support under region decoder samples at least κ |

The region decoder is a conditional flow trained on latents interpolated between the world model's prediction and the encoding of the realized state, so the forecast decodes to a wide region and a flow sample to a narrow one. It also conditions on the decision-time observation and generates the change from it, so the fixed layout decodes exactly. Both executors share an architecture and are trained by behavior cloning on hindsight goals within eight steps of each demonstrated action. The latent executor receives encoded goals. The kernel bandwidth and Gaussian floor are half a cell.

Each arrival rule's threshold is calibrated on development windows so that it accepts a fixed share `r` of realized states, set by `--recall-levels` (default 0.25, 0.5, 0.75, 0.9 and 0.95). Rules are therefore compared at equal recall. `summary.json` records each threshold and its false arrival rate on mismatched development states. `sample_cell` and `decoded_support` separate the target from the arrival rule: together with `point_learned` and `region_decoded` they cross a decoded or sampled target with exact-cell or region-support arrival under the same executor.

The coherence diagnostic draws `--coherence-realizations` samples from each flow region, attempts each as a fixed point goal on the evaluation slip draws, and reports the variance of value within regions, between regions of the same start, and the binomial noise of the repeats. A coherent region has realizations of similar value.

`candidate-bank.json` and `candidate-regions.npz` record every candidate's latents, decoded observations, point targets, region samples and per-repeat realizations before any attempt. `region-weights.pt` stores the decoder, the Gaussian scale and both executors. `summary.json` reports, for each arm and depth, terminal success, arrival, success after a miss and attempt length for the forecast, the uniform average over flow candidates and the candidate selected on independent reference repeats. Paired differences from `point_learned` carry layout-bootstrap intervals. It also reports executor accuracy on development demonstrations. For a smoke run, append `--layouts 1 --demonstrations 1 --encoder-updates 2 --updates 2 --flow-updates 2 --region-updates 2 --executor-updates 2 --repeats 2 --candidates 2`.

## Compose latent goals under action budgets

Install `training-cpu` as for the dependency-grid latent workflow. Run the controlled task with two-, three-, and four-action budgets.

```bash
mkdir -p results
run_dir="$(mktemp -d results/budget-routes.XXXXXX)"
uv run --no-sync wr budget-routes --seed 7 --out "$run_dir/seed-7"
```

The shortcut reaches the same terminal as the bypass and the gate routes. A successful supply stop can raise the gate route's terminal odds. The start may already be supplied. The planner observes its inventory and a coarse risk profile. World-specific success probabilities and future draws belong to the evaluator. Repeats are matched across arms and share the same random prefixes across budgets.

The shared completion model compresses 11 public features into six latent dimensions and predicts the states at offsets 1 and 2. Scripted successful reset trajectories cover shortcut, bypass and gate routes. They omit the complete supply-then-gate route. A local continuation from the supply state records that segment. Independent random training exploration supplies one-step transition counts, so the scorer can observe individual transitions from the held-out full route. During validation the worlds vary by risk draw and starting inventory on a common public graph.

The composer takes distinct targets from the same latent samples used by the joint baseline, tries singles and legal ordered pairs, and retains the source slot for each target. Joint samples, composed prefixes, and matched-context training replay share one public graph controller and one learned transition scorer. This first scorer propagates empirical public-state transitions to compute the probability of terminal success within the remaining budget. Planning energy is the negative log of that probability. A separate baseline ranks the original joint samples by flow density. All three pools share a candidate cap and the manifest records the sample budget and per-case rankings. The fixed-pool diagnostic ranks the same four-action candidate bank at all three budgets.

The replan arm starts with the selected composition. On reaching its first target it can replace the suffix using the observed inventory and remaining budget. It reuses proposed nodes and reports extra scoring calls. Rarity counts ordered pairs in successful training demonstrations with the same public profile, start inventory and budget, once per eligible episode. Both targets need individual support. Pairs at or below 5% support qualify. The evaluator deduplicates implied execution paths within each world and budget, screens rare paths using independent reference repeats, then counts a confirmed path if it reaches both targets and succeeds on at least half of the evaluation repeats and improves terminal success over direct tracking by at least 10 percentage points.

`candidate-bank.json` records the frozen choices, latent samples, support counts, and planning energy. For a selected pair, it also declares the two single targets and the reversed order as paired controls. `attempts.json` records actual actions and outcomes. `summary.json` reports paired terminal success, budget-dependent choices, suffix changes, and rare-path coverage. For a smoke run, append `--layouts 1 --repeats 4 --candidates 4 --demonstrations 4 --exploration 24 --encoder-updates 2 --updates 2 --flow-updates 2`. Reserve `--split test` for the fixed confirmation protocol. [Research plan](latent-subgoal-research.md#3-compose-alternative-routes-under-a-budget)

## Check a development installation

From the repository root, install the CPU dependencies used by CI and run the repository checks.

```bash
uv sync --locked --group dev --extra training-cpu --extra ogbench --extra pointmaze
uv run --no-sync ruff check .
uv run --no-sync ruff format --check src tests
CUDA_VISIBLE_DEVICES= JAX_PLATFORMS=cpu uv run --no-sync pytest
```

## Run artifacts

Artifacts stay in `results/` on the server. Git and the Mac sync scripts exclude the entire directory.

| Directory | Contents |
| --- | --- |
| `results/datasets/` | Complete trajectory collections and their logs |
| `results/dependency-grid.*/` | Layouts, matched target attempts and terminal success by goal and depth |
| `results/dependency-grid-latent.*/` | Public trajectories, frozen weights, candidate declarations, and matched terminal outcomes |
| `results/pointmaze-regions-*/` | Maze2D world models, region decoders, latent executors and matched attempts by arm |
| `results/dependency-grid-regions-*/` | Region decoders, executors, declared regions and matched attempts by arm |
| `results/budget-routes.*/` | Route candidate banks, matched attempts and results by budget |
| `results/attempt-compare.*/` | Shared-bank selector fits and paired comparisons |
| `results/pointmaze-diffuser.*/` | PointMaze archive splits, decoded plans, executed trajectories and paired outcomes |
| `results/ogbench-scene.*/` | Frozen Scene executor and completion weights, matched starts, decoded plans and rollout outcomes |
| `results/checkpoints/` | Policy and model weights |
| `results/external/` | Isolated benchmark source trees and environments |
| `results/_archive/` | Archived experiment runs, code snapshots and path maps |

Use a new output directory for each run. Keep long collection and training runs in tmux.

## Compute devices

Install `training` for the locked PyTorch CUDA build on Linux, or `training-cpu` for CPU development and CI. Use `uv run --no-sync` after installation to reuse the selected extras. [PyTorch installation](https://pytorch.org/get-started/locally/)

```bash
uv sync --locked --group dev --extra training --extra ogbench --extra pointmaze
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

Indices are zero-based within any existing `CUDA_VISIBLE_DEVICES` list. The configuration never expands that list. With no launcher mask, the default CUDA ordering is PCI bus order. CUDA renumbers the selected devices inside the process, and the report includes the effective visibility mask. Add the same global flag before any experiment command, or export `WR_COMPUTE_CONFIG` for a tmux session.

The completion predictor, route flow and GCBC trainer use Lightning's single-host DDP. Batch size is global, so each update uses the same examples on one or more devices. Run manifests record the framework, backend, device count and visibility mask. Simulator rollout loops execute serially. [Lightning distributed strategies](https://lightning.ai/docs/pytorch/stable/reference/accelerators/gpu_intermediate)

## Configure a completion model

`dependency-grid-latent`, `budget-routes`, `pointmaze-latent` and `scene-latent` accept `--config PATH` after the subcommand. A TOML `[model]` table overrides the command's defaults. Explicit command-line options take precedence. The run manifest saves the resolved settings alongside the dataset and split information.

```bash
run_dir="$(mktemp -d results/pointmaze-recipe.XXXXXX)"
uv run --no-sync wr pointmaze-latent --config configs/pointmaze/completion-smoke.toml \
  --task medium --seed 7 --eval-split dev --eval-repeats 1 --out "$run_dir/medium"
```

The example file uses two optimizer updates per stage to check the training path.

Both model families accept `width`, `latent_size`, `batch_size`, `learning_rate`, `seed`, `predictor_val_every` and `predictor_patience`. For the continuous model, use `history`, `queries`, `windows`, `codec_updates`, `predictor_updates`, `flow_updates` and `candidates`. Its batch size sets the codec and predictor batches. For the discrete grid and routes predictor, use `history_length`, `queries_per_window`, `encoder_updates` and `updates`. Each task supplies its public input and goal format. In PointMaze, use `--prediction-span` for the maximum prediction span and `--eval-budgets` to evaluate several action budgets using one set of weights.

## Select a predictor checkpoint

The predictor monitors `val_latent_mse`, the mean squared error at valid masked query positions in fixed development windows encoded by the frozen codec. Set `--predictor-val-every` to change the default interval of 200 updates and `--predictor-patience` to change the default of five validation checks without improvement. The update budget caps training. A budget shorter than the interval is validated at its last update. The grid's goal-conditioned and goal-blind predictors share validation windows and stopping settings and select their checkpoints independently. [Lightning EarlyStopping](https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.callbacks.EarlyStopping)

The predictor checkpoint directory contains `best.ckpt`, selected by validation error, and `selection.json`, which records the best score, the selected update, the actual final update and the stop reason. Continuous benchmarks use the best predictor's forecasts to fit the route generator. Test rollouts stay outside checkpoint selection. [Lightning ModelCheckpoint](https://lightning.ai/docs/pytorch/stable/optional-api/api/lightning.pytorch.callbacks.ModelCheckpoint)

## Resume training

Each stage writes Lightning checkpoints and CSV metrics under `checkpoints/<stage>/` in the run directory. `last.ckpt` records the latest saved training state, and `final.ckpt` marks completion. Predictor stages save progress at validation checks and restore the best weights when completed. Pass `--resume PREVIOUS_RUN` with the original data and settings and a new `--out` directory. Interrupted stages recover optimizer and callback state, and evaluation runs after training.

```bash
old_run=results/pointmaze-diffuser.example/umaze
new_run="$(mktemp -d results/pointmaze-resume.XXXXXX)/umaze"
uv run --no-sync wr pointmaze-latent --task umaze --resume "$old_run" --out "$new_run"
```

Add any non-default flags from the original run to the resume command. Frozen runs read `weights.pt`. Runs saved before this format used Flax `weights.msgpack` and need a one-time conversion. Resumption with optimizer state uses a Lightning checkpoint from the same training recipe. [Lightning checkpoint format](https://lightning.ai/docs/pytorch/stable/reference/common/checkpointing_basic)

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

The sweep writes one set of trained weights and a `summary.json` with results by budget. Each `budget-<steps>/` contains its own manifest, terminal outcomes, trajectories, and plans. Each plan records both `remaining_budget` and `prediction_span`. Pass a budget subdirectory to the frozen energy comparison for that budget.

### Test latent energy against terminal success

Train two Medium checkpoints with the official evaluation budget and an archive-derived prediction span, or select a budget subdirectory from each checkpoint's sweep.

```bash
run_dir="$(mktemp -d results/pointmaze-horizon-aligned.XXXXXX)"
for seed in 7 19; do
  uv run --no-sync wr pointmaze-latent --task medium --seed "$seed" \
    --windows 16000 --codec-updates 1200 --predictor-updates 4800 \
    --flow-updates 1600 --eval-repeats 24 --out "$run_dir/medium-$seed"
done
```
 Each start supplies a public reset observation, terminal goal, and remaining budget. The frozen world model and its completion flow draw candidates and an independent reference sample. Both draws are truncated to the prefix that will be attempted. The score is the negative log Gaussian-smoothed marginal density of that prefix under the flow draws, divided by the number of latent coordinates and reported up to a fixed additive constant. Smoothing uses the frozen residual normalization and Scott's bandwidth in those coordinates. The predictor's squared latent error is recorded as a separate ranking on the same candidates. This setup compares rankings without fitting either score to execution outcomes.

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

### Compare region goals with point goals in Maze2D

This is E1 on a continuous task. It reads a frozen `pointmaze-latent` budget directory, checks the dataset checksum against it, and trains a region decoder and a latent-conditioned executor on the same archive. [Grid version](#compare-region-goals-with-decoded-point-goals)

```bash
uv run --no-sync wr pointmaze-regions --checkpoint "$run_dir/medium" --out "$run_dir/medium-regions"
```

Every arm plans in closed loop. From the public history it declares the first query of one completion as a stage, with that query's offset as the deadline, and replans when the stage arrives or times out. `forecast` arms decode the deterministic forecast, and `sample` arms draw a fresh flow sample at every replan with two episodes per start. When the remaining budget falls to the query count, Diffuser's position and velocity tracker pursues the terminal goal. The tracker is also the observation executor, so no arm hands control to a learned policy that knows the task. `point_native` and `sample_native` track the decoded state or one region realization until the benchmark's sparse XY predicate fires. `decoded_support@r` and `sample_support@r` track the same targets with region-support arrival. `latent_point@r`, `region_gaussian@r` and `region_flow@r` give the latent executor z itself. Regions and kernels live in observations normalized by the training scale, with a bandwidth of 0.25. Thresholds are calibrated to equal recall on validation windows.

Two bounds share the tracker and the schedule. `direct` tracks the terminal goal from the reset. `oracle` replans one cell ahead along the shortest path on the privileged layout, with a target velocity of 2 toward the waypoint so that the tracker does not brake at every cell. A one-shot protocol that tracks only the first target and then the terminal goal cannot separate arms: in the earlier run, even a perfect first waypoint raised success by at most 0.03. Each arm reports success, paired differences from `direct` and from `point_native` with start-bootstrap intervals, its share of the gap between `direct` and `oracle`, and stage arrival. It also reports two layout diagnostics that no model sees: the share of declared targets inside a wall and the share that are closer to the goal along free cells. `attempts.jsonl.gz` keeps every episode with its first plan. Use `--eval-workers` to spread episodes across CPU processes. A smoke run uses `--eval-repeats 1 --sample-repeats 1 --region-updates 2 --executor-updates 2 --executor-pairs 64`.

Train the world model with `--config configs/pointmaze/completion-scaled.toml`. It widens the codec, predictor and flow to 512 units and trains on 200,000 windows with about 50 times more updates than the development recipe. The original models put 20% to 54% of first targets inside walls.

### Plan with latent options in Maze2D

This is the first milestone of E2. It reads a frozen `pointmaze-latent` budget directory and trains on its archive a latent-goal GCIVL executor, whose goals are frozen encodings of archive observations, and a budget-conditioned option value. It then corrects the value with the executor's own attempts. `--trained` reuses the executor and both values from an earlier `pointmaze-options` run with the same executor and value recipe and world model, so planner settings can be evaluated without retraining; collection settings may differ and the manifest keeps the earlier run's recipe. [Option value protocol](latent-subgoal-research.md#5-learn-budget-conditioned-option-value-in-latent-space)

```bash
uv run --no-sync wr pointmaze-options --checkpoint "$run_dir/models/medium-7" --out "$run_dir/options/medium-7" --eval-workers 32
```

Nothing defines an intermediate goal or its arrival. An option pursues a latent `z` for at most `k` actions. The value `Q(s, g, z, k, b, λ)` is first fitted on hindsight options from the archive: every segment pursued the encoding of its end state, and the encoded relabeled goal is an option pursued for the whole budget. Targets are `1 − λd` on reaching the goal after `d` actions, `−λb` when the budget runs out, and `−λk + V` at the next state otherwise, with `V` an upper expectile of `Q` over the options pursued from the same state. The backups have no discount, so `V` is clipped to the returns still possible, `[−λ(b − k), 1 − λ]`. Without the clip each backup added the expectile's bias and values reached 25 offline and 2,400 after correction. `value_range` in the summary gives quantiles of `Q` on labeled options after offline fitting and after each round. The time cost `λ` is a value input sampled from `--time-costs`. Four heads trained on bootstrap masks give a spread.

The archive never pursued the terminal goal on purpose, so the offline value misjudges options the executor would pursue. `--online-rounds` rounds each run the planner on `--online-episodes` training resets, starting at seed 9001 and disjoint from the development and test seeds. At a decision the planner pursues a random sampled candidate with probability `--explore`, drawn uniformly over proposal budgets and then within the budget. Each training reset is also played once by pursuing the encoded terminal goal for the whole budget without interruption (`--no-goal-attempts` turns this off). In the first full run the goal entered collection only through exploration, was switched away from after a median of one to four steps, and so never labeled `Q(goal, k = b)`; its value stayed near zero on Scene Tasks 2 and 3 where direct latent pursuit succeeded 0.6 to 0.7 of the time. Every option the executor pursued until the first success becomes an attempt with its landing state and outcome; an interrupted option counts as the option with the duration it ran, and the option that reached the goal keeps the duration it was commanded. Recording the actions it took instead made every attempt commanded for the whole budget a failure, which drove the goal's corrected value to about `−λb` once goal pursuit joined collection. The value continues fitting for `--online-updates` updates per round on batches that draw half their rows from the attempts. `executor-attempts.npz` keeps them, and `rounds` in the summary records the planner's success, goal choices, the success of goal pursuit and attempts per round.

At a decision the planner samples `--samples` flow routes, takes every query offset of every route as a candidate, adds the encoded terminal goal, and pursues the candidate with the highest mean value plus `--risk` times the spread. The `fast` arms also ask the world model for routes under the `--fast-budgets` fractions of the remaining budget, which adds faster candidates; the value still scores every candidate with the true remaining budget. Each step the planner re-scores the current option and the cached candidates, including the goal, and switches when one is better by more than `--margin` spreads. It samples again when the option's duration runs out, its value drops by more than `--margin` spreads, or the encoded observation is farther from the chosen route's predicted latent than the 95th percentile on validation windows.

The arms are `direct` (tracker toward the goal), `oracle` (shortest-path waypoints) and `direct_latent` (the latent executor toward the encoded goal). For each value, `offline` and `online`, `options@λ` runs at each `--eval-costs` value and `fast@λ` at the last one. Three ablations of `online/fast` at the last cost remove the goal candidate (`fast_no_goal`), switching and the value and surprise triggers (`fast_no_interrupt`), or every offset but the first (`fast_fixed_duration`). Each arm reports success, first success step, paired differences from `direct` and `direct_latent`, its share of the oracle gap, candidate generations and switches per episode, decisions by trigger, the share of decisions that chose the goal and the mean chosen duration. Each planner row in `attempts.jsonl.gz` keeps `decision_log`, with the trigger, the chosen option and its source, whether it was explored, its value, the goal's value and the best sampled value at every decision, and `switch_log` with the values before and after each switch. A smoke run uses `--eval-repeats 1 --executor-updates 2 --executor-batch-size 16 --value-updates 2 --value-options 64 --samples 2 --online-rounds 1 --online-episodes 2 --online-updates 2`.

## Run state-based OGBench Scene

Install the locked `ogbench` extra and select a Lightning training extra. With the default compute configuration, `wr devices` reports the selected backend and all visible GPUs. The Scene executor and completion model both use the same Lightning training path. Evaluation steps the MuJoCo environment on the host. [Compute configuration](#compute-devices)

```bash
uv run --no-sync python -c 'import ogbench; ogbench.download_datasets(["scene-play-v0"], dataset_dir="results/datasets/ogbench")'
run_dir="$(mktemp -d results/ogbench-scene.XXXXXX)"
uv run --no-sync wr scene-latent --out "$run_dir/state" --eval-repeats 1
```

`scene-v0` supplies a 40-feature public state, five continuous action coordinates and five goal configurations involving the cube, drawer, window and locks. Both arms reuse exactly the same seeded start and goal. Each rollout stops on the official success event or the 750-action limit. Use `--task 4` to isolate the cube-in-drawer task, `--eval-steps` to shorten a smoke test, and `--eval-split test` after fixing the experiment recipe. [OGBench Scene](https://seohong.me/projects/ogbench/#scene) · [Dataset and task protocol](https://github.com/seohongpark/ogbench)

Repeat `--task` to evaluate selected goals with one set of trained weights. `--task 4 --task 5` evaluates the drawer and rearrangement goals using the same controller and planner.

Use `--eval-workers 4` to distribute independent starts across four CPU processes after training. Each process restores the same frozen weights and runs both arms on each of its starts. The manifest records the actual worker count under `compute.eval_workers`.

Executor training alternates hindsight-goal batches capped at 64 transitions and batches that sample any later observation in the same training episode. Both arms use that actor after training. The direct arm tracks the terminal observation. The latent arm takes the earliest state of the first sampled decoded route and replans when its time offset elapses. The official input features and action history feed the planner. Extra archive fields for MuJoCo state and buttons are outside the training loader.

The manifest identifies the official train and validation archives by checksum. Offline validation reports action MSE and normalized reconstruction and prediction MSE. The other artifacts contain exact starts, public trajectories with executed actions, sampled plans, weights and paired success by task.

### Qualify a value-based Scene executor

The behavior-cloned executor above reaches no official Scene goal. `scene-executor` trains goal-conditioned implicit V-learning (GCIVL) as in OGBench: two expectile value heads toward `r + γ V̄(s', g)` with `r = 0` at the goal and −1 elsewhere, a Polyak target, and advantage-weighted regression of the actor with temperature 10. Value goals mix the current state, a geometric later state and a random state with probabilities 0.2, 0.5 and 0.3. Actor goals are uniform later states of the same trajectory. The networks use three layer-normalized hidden layers of 512 units and batches of 1024 raw observations. [OGBench GCIVL](https://github.com/seohongpark/ogbench/blob/master/impls/agents/gcivl.py)

```bash
uv run --no-sync wr scene-executor --seed 7 --updates 1000000 \
  --eval-repeats 20 --eval-workers 20 --out "$run_dir/executor-7"
```

The executor tracks the official goal observation from each seeded reset until the native success event or 750 actions. `summary.json` reports success by task and the value at goals 1, 10, 50 and 200 steps ahead in validation trajectories. One update takes about 17 ms on an RTX 2080 Ti, so a million updates take about five hours.


### Compare region goals with point goals in Scene

This is closed-loop E1 on official Tasks 4 and 5. A qualified `scene-executor` run supplies the GCIVL executor, which pursues observation targets and the terminal goal. `scene-regions` trains a latent completion model with the span set to the 750-action limit, a region decoder on 20,000 training windows, and a latent-goal GCIVL whose goals are frozen encodings of archive observations. Arrival thresholds are calibrated to equal recall on validation windows, as in Maze2D. [Maze2D version](#compare-region-goals-with-point-goals-in-maze2d)

```bash
uv run --no-sync wr scene-regions --executor "$executor_run" --out "$run_dir/regions" \
  --config configs/scene/completion-scaled.toml --region-updates 20000 \
  --region-width 512 --region-batch-size 512 --eval-workers 48
```

The arms and modes match Maze2D. The observation arms hand decoded states or region realizations to the frozen executor. The latent arms hand `z` to the latent-goal executor. Native arrival at a declared target applies the official scene predicate and a 4 cm end-effector match after projecting the target's button states onto one-hot values. Scene has no privileged path planner, so direct execution is the only bound. A target makes progress when more of the cube, drawer, window and two buttons match the goal in the target than in the current observation. Region supports are log densities, which avoids underflow in the 40-feature observation and leaves quantile-calibrated decisions unchanged.

`--models` reuses the world model, decoder and latent executor of an earlier run with the same recipe, so evaluation settings can change without retraining. `summary.json` reports every arm overall and by task. A smoke run uses `--config configs/pointmaze/completion-smoke.toml --latent-updates 2 --region-updates 2 --decoder-windows 8 --eval-repeats 1 --sample-repeats 1`.

### Plan with latent options in Scene

This is the Scene version of the Maze2D option planner. It reuses the world model and latent-goal executor of a `scene-regions` run, and that run's `scene-executor` run for `direct`. It fits the option value on hindsight options from the Scene archive, whose terminal success is the official scene predicate, and corrects it online as in Maze2D. Training resets cycle over the evaluated tasks with seeds from 63001, between the development and test ranges, and each round reports the environment steps it used. [Maze2D version](#plan-with-latent-options-in-maze2d)

```bash
uv run --no-sync wr scene-options --regions "$regions_run/seed-7" --out "$run_dir/scene-7" --eval-workers 32
```

The value, online and planner options match `pointmaze-options`. The default tasks are 2 to 5 with 20 development resets each. There is no oracle, so the arms are `direct`, `direct_latent`, both values with and without faster routes, and the three ablations. `summary.json` reports every arm overall and by task. `--trained` reuses both values of an earlier run with the same value recipe and world model. A smoke run uses `--task 2 --eval-repeats 1 --value-updates 2 --value-options 64 --samples 2 --online-rounds 1 --online-episodes 2 --online-updates 2`.
### Compare native CPU and batched CUDA Scene evaluation

`scene-evaluate` loads a completed `scene-latent` run and evaluates the frozen controller and planner. Use the same checkpoint, tasks, split, seeds and budgets for both backends. On Linux x86-64, add `--extra scene-warp` to the Scene install for CUDA physics.

```bash
checkpoint=results/path-to-scene-run
run_dir="$(mktemp -d results/scene-backends.XXXXXX)"
uv run --no-sync wr scene-evaluate --checkpoint "$checkpoint" --task 4 --task 5 \
  --eval-repeats 8 --eval-workers 4 --out "$run_dir/cpu"
uv run --no-sync wr scene-evaluate --checkpoint "$checkpoint" --task 4 --task 5 \
  --eval-repeats 8 --backend warp --batch-size 16 --out "$run_dir/warp"
```

The Warp backend batches physics substeps and contact forces on one CUDA device. It uses OGBench's native reset, inverse kinematics, button and lock transitions, and success rule on the host. Warp's single-precision solver can change trajectories and outcome labels. The run records the backend and elapsed evaluation time, including worker or graph setup, and saves both arms' public observations and actions. Use Warp for development throughput and numerical comparisons. Use native MuJoCo for official scores until agreement on contact-heavy traces and terminal outcomes is established.

### Test Scene waypoint reachability

Pass an existing `scene-latent` run directory to evaluate its frozen controller. The command uses `scene-play-v0-val.npz` and steps MuJoCo on the host. [Scene setup](#run-state-based-ogbench-scene)

```bash
run_dir="$(mktemp -d results/scene-reachability.XXXXXX)"
uv run --no-sync wr scene-reachability --checkpoint results/path-to-scene-run \
  --out "$run_dir/probe" --offsets 16 32 64 --budgets 16 32 64 128
```

For each demonstration offset, the evaluator samples one nontrivial waypoint per selected validation episode in each group. `scene_change` requires a change to the cube, drawer, window or locks. `pose_only` requires moving the end effector while the scene already matches. Arrival requires the cube and end-effector positions and both slides to be within 4 cm of their respective targets, with both lock states matching exactly. The controller receives the live public observation and the recorded public target.

Both arms restore the same archived start using the private physics and lock arrays. The archive lacks MuJoCo's contact-force cache, so the evaluator reports the initial contact discrepancy. At cube orientations where yaw is undefined at archive precision, it checks the cube quaternion and records the derived yaw discrepancy. Both arms receive exactly the same restored observation.

The evaluator reports whether recorded actions reproduce the target at the recorded offset. The controller stops at first arrival or the largest budget. The first arrival supplies the outcome at each budget, including on the last permitted action. `summary.json` gives counts for all cases and for the subset the replay reproduces. The public trajectory prefixes and per-case outcomes allow further inspection. This development diagnostic uses play-archive states and measures local arrival. Official Task 4 and 5 resets and terminal success are evaluated with `scene-latent`.

When comparing checkpoints on the same waypoint bank, pass `--replay-from PATH` to reuse the first run's recorded-action branches. The evaluator checks the archive hash, simulator versions, reset seed, waypoint rows and recorded actions. It accounts for reused steps separately from new simulator interactions.

### Test Scene drawer insertion in isolation

Use `--waypoints drawer-insertion --offsets 64 --budgets 32 64 128` with `scene-reachability` to give the frozen controller recorded targets reached after inserting the cube. Choose `--cases-per-group 100` to include all eligible validation episodes in the standard archive. The start has an open drawer at least 12 cm out and a cube below 4.5 cm on the table, at least 10 cm from the moving drawer target. The sampled clip keeps the drawer at least 10 cm open and ends at the fifth consecutive placement frame. All offsets stay within each source episode.

Placement requires the drawer to be at least 12 cm open, the cube to be within 4 cm of the drawer-relative placement center, and the end effector to be more than 5.5 cm from the cube for five consecutive observations. The controller receives the recorded public goal observation. The score uses placement and release so it can distinguish control of the cube from reproduction of the demonstrated arm pose. The paired recorded-action replay checks the start and target under the same simulator. The diagnostic measures this skill on validation clips.

