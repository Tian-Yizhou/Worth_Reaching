# Research architecture

Use an environment-independent planner and a simulator-backed evaluator. At decision time each policy receives the public history, objective, remaining budget and any legitimate public action constraints. The evaluator owns simulator snapshots, future random keys and outcome labels. This boundary allows the same planner to operate in a grid world, a crafting game and eventually a card game.

## Proposed components

| Component | Inputs and outputs | Responsibility |
| --- | --- | --- |
| Environment adapter | Reset or step produces public observation, action specification, terminal flags and scoreable events | Map native actions and observations, apply task termination, pin rules and wrapper versions. Provide optional privileged branching to the evaluator through a separate interface. |
| Trajectory recorder and split manifest | Episodes and policy versions produce trajectories, snapshot references and world-group splits | Preserve action order, termination reason and data provenance. All descendants of one root episode belong to its split. |
| History encoder and world model | Public observation and action history produces a latent state. The starting history, terminal condition and time information determine predictions for the missing middle. | Learn task-relevant middle representations from complete training traces with variable-length masks. Action-conditioned forward predictions can also support execution checks. |
| Candidate proposer | Public history, terminal objective and budget produce candidate latent routes and next targets | Sample different middles for the same start and goal using a conditional generator. Use the same candidate interface for replay comparisons and retain proposal provenance. |
| Goal representation and executor | Target representation, public history and budget produce primitive actions and a stop reason | Allow different concrete histories and actions to realize one abstract target. Declare the arrival rule and deadline before each prospective attempt. |
| High-level planner | Current latent state, terminal goal and budget produce a target, duration or direct plan | Generate alternatives, score their likely consequences, execute the first segment and replan from observed outcomes. |
| Branch evaluator and experiment runner | Frozen starts, interventions, policies and random streams produce attempts and aggregate results | Collect interventional labels, enforce permissions, meter interactions and compute, and estimate paired outcomes. |

The `src/worth_reaching/` package groups code by responsibility.

| Package | Current responsibility |
| --- | --- |
| `tasks/<family>/environment.py` | Adapt the simulator and its task rules, expose public observations and actions, and support evaluator-controlled resets |
| `data/` | Represent public completion examples and masked trajectory windows, read shared OGBench archives and record experiment artifacts |
| `models/` | Encode observations, predict masked latent positions and sample residual trajectories |
| `models/training.py` | Configure Lightning trainers, step-indexed batches, checkpoints and logs |
| `tasks/craftax/` | Collect expert and random episodes, validate world splits and export craft-and-survive windows |
| `tasks/kuka/dataset.py` | Read Kuka demonstration clips and encode block-stacking goals |
| `evaluation/` | Replay, matched continuations, shared-bank comparisons and paired route scoring |
| `cli.py` | Connect configurations to collection, window export, training and evaluation |
| `tasks/pointmaze/` | Own the Maze2D task specifications, dataset and simulator adapters, controller, experiment commands and paired probes |
| `tasks/antmaze/` | Load AntMaze archives and saved actors, train the GCBC baseline, and run latent and waypoint comparisons from matched starts |
| `tasks/dependency_grid/` | Define switchable goals with ordered prerequisites, collect public trajectories, and evaluate declared or learned targets through paired execution |
| `tasks/budget_routes/` | Sample latent goals, compose executable prefixes and assess rare routes under a shared learned transition scorer |
| `planning/` | Execute decoded goals from public observation and action histories through a supplied controller |

Group tasks that share simulator behavior in one family. For example, Scene and Puzzle share the OGBench reset and step implementation in `tasks/ogbench_manipulation/environment.py`. Their presets select the observation schema, goals and action limit. A new family can follow the PointMaze or AntMaze layout and register its commands from a lightweight `cli.py`, so listing commands works without installing simulator extras.

PointMaze and AntMaze runners assemble the task adapter, completion model and controller. The model consumes public histories, goal observations, remaining action budgets and query offsets. The task loader checks the observation and action shapes against its archive schema. AntMaze, Kuka, Scene and Puzzle also use the continuous completion model in `models/continuous_completion.py` and the window types in `data/trajectory_windows.py`. Craftax and the grid diagnostic use the discrete-action predictor in `models/completion.py`.

The trajectory completion model will use public data from `data/`. Its executor will consume public observations and emit primitive actions. Evaluation owns simulator snapshots and restores them for matched repeats. [Trajectory completion design](trajectory-completion.md)

## Training on multiple devices

New model training uses PyTorch Lightning. Lightning runs DDP on one host and manages checkpointing and logging for each stage. A step-indexed sampler supplies the same global batch across device counts. A row mask excludes padding, and the loss uses the global count of valid elements to preserve its weighting under DDP. A stateful dataloader saves its cursor with the optimizer and model. The process that launches training handles serial simulator evaluation after the workers finish. Craftax simulation and the released expert policy use JAX. [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/core-api/trainer) · [Compute configuration](experiments.md#compute-devices)

## Execution and data contracts

1. A collector creates root episodes and records the public prefix, behavior policy and observable task outcomes. An adapter can additionally emit an opaque snapshot handle into a private store.
2. The branch evaluator names an action or waypoint and a deadline before rollout. It restores a start, forks the random stream, applies the intervention, then runs the declared continuation policy. Exact replay and fresh stochastic continuations are distinct sampling modes.
3. An attempt record carries the root world and episode IDs, prefix index, public history reference, option or action ID, region and proposer versions where applicable, policy and encoder versions, budget, random-stream group, arrival outcome, terminal outcome, elapsed steps and termination reason. A training loader exposes an explicit allowlist of fields. A hindsight-achieved goal and a prospectively attempted goal have distinct labels.
4. At inference, the planner encodes the latest public history, scores alternatives using its learned models, and gives the executor a target and time budget. The next observation closes the loop. For a direct planner the executor targets the terminal objective. Results record when a waypoint completes, times out, becomes infeasible, or is replaced.
5. Each run stores an experiment configuration, dependency versions, code revision, seed and split manifests, dataset or checkpoint identifiers, and metric definitions. Complete traces, logs and checkpoints live in the Git-ignored `results/` directory at the repository root. Commit small manifests or representative fixtures where redistribution permits.

## First interface checks

Test termination at the action limit and success on the last allowed action. Check that restoring a branch reproduces its public prefix and leaves the source episode intact. Perturb privileged fields while holding the public history fixed and compare the serialized policy input. Verify that an executor evaluated against a frozen waypoint uses the same encoder version and arrival rule that defined that waypoint.

## Craftax collection contract

Craftax-Classic passes symbolic observations and action IDs to the collection path. The adapter owns the native state. The evaluator can fork a live snapshot and supply a transition key for each step. World generation, transition randomness and policy sampling use separate streams. In `make_iron_pickaxe`, the first craft takes precedence over death or exhaustion of the budget. In `make_iron_pickaxe_and_survive`, death ends the episode. A living agent at the deadline succeeds if it has crafted the pickaxe. Both tasks share world identifiers and splits for a matched seed. The compound archive records its own task, rules version and first-craft step. Policies receive the same symbolic observations.

Each compressed trajectory record contains the episode ID, initial observation, public transitions and outcome. The split manifest maps the episode ID to its root world and split before collection. The run manifest identifies the observation schema, rules and action vocabulary. The final summary is written after the trajectory file closes.

Random and PPO-RNN collection share the same environment and transition seed streams for each world. Their policy sampling streams are separate from the simulator. PPO-RNN receives only the current public symbolic observation and its per-episode recurrent memory. The manifest records the checkpoint step and actor digest, and the split label is assigned by world regardless of policy.

`tasks/craftax/dataset.py` loads completed random and expert collections through the same interface. It validates task rules and verifies episode IDs and split assignments against the original world seeds. Its split iterator yields complete public episode records without loading a policy checkpoint or constructing a simulator.

`evaluation/craftax.py` reconstructs a live start from recorded actions and verifies each replayed observation. It returns the recurrent memory immediately before the current observation. A continuation can act directly or run a declared action prefix first. Forced actions still advance policy memory from each public observation. Match repeat keys across alternatives to compare outcomes on the same transition and policy random streams.

## Masked trajectory window contract

For a successful craft-and-survive episode with budget `B`, index observations from `o[0]` and let action `a[t]` produce `o[t]`. A window starts at an observed step `s` and queries the next `L` positions, with `s + L < B`. The input contains observations through `o[s]`, actions through `a[s]`, the terminal task and the remaining budget `B - s`. The target contains observations and incoming actions for steps `s + 1` through `s + L`. The query offsets `1..L` describe which positions to predict. They do not depend on when the pickaxe was crafted.

Each successful episode contributes a window from the reset and up to seven later starts by default. The seed and world ID determine the sampled starts and lengths. Each export inherits the archive's world split. Train and development exports use terminal outcomes to select demonstrations. Model inputs contain the public task descriptor. The training shard can additionally supply `o[B]` through an explicit endpoint-example option. Development windows use the task descriptor alone. The final action is outside the windows, and the test split stays in the source archive for evaluation. [Window export and loader](experiments.md#prepare-masked-trajectory-windows)

## Latent predictor

The first trainer learns an observation autoencoder on masked training observations and freezes its encoder and decoder. Both predictors use that same target representation, initial predictor weights and sampled batches. A GRU summarizes public observations and their incoming actions. The predictor receives that history representation, the time remaining and the requested time offsets. The goal-conditioned arm also receives the public task identifier. It is trained to predict the encoded observation at each requested position.

The trainer pads the recent history and uses evenly spaced queries across longer masks to bound batch size. Only the public-input packer feeds prediction. Target observations are encoded separately for training and metrics. The decoder translates predictions into observation space for diagnostics. The model saves inference weights, architecture and training settings. [Training and goal switch protocol](experiments.md#train-and-test-latent-trajectory-completion)

## Sampled routes and attempts

The grid flow learns the difference between a frozen prediction and an encoded demonstration for the complete interior window. It conditions on the starting observation, the goal and the frozen prediction. Each draw decodes to a route and declares one intermediate cell and attempt deadline. A feedback executor uses the observed cell to choose the next action. After arrival or timeout the same controller continues toward the terminal objective with the remaining budget. The evaluation runner supplies repeat streams, pairs those streams across candidates and saves the attempted latent target separately from the observed results. The route diagnostic uses a fixed grid horizon and exact public cell arrival. [Route generation and execution protocol](experiments.md#sample-and-execute-alternative-routes)

The dependency-grid adapter defines public geometry, position, and acquired keys. A separate experiment runner declares target cells, applies matched slip draws, and records realized arrivals, elapsed steps, and terminal success. A changed terminal goal or depth reuses the same layout, reset, and slip draws. The diagnostic uses disjoint layout IDs for train, development, and test. [Ordered-target diagnostic](experiments.md#compare-ordered-targets-in-the-dependency-grid)

The learned dependency-grid runner constructs `CompletionExample` windows with a public reset and two time-indexed observation targets. It assembles the shared discrete completion model and residual flow. Target extraction uses decoded coordinates and public walls. The optional scorer is fitted on paired training-layout attempts and ranks both pools from public goal-relative features. Its choices are saved before evaluation outcomes, reference repeats, and achieved-target trajectories are collected. [Frozen latent-goal comparison](experiments.md#test-frozen-latent-goals-in-the-dependency-grid) · [Outcome-based selection](experiments.md#learn-to-select-a-goal-from-terminal-outcomes)

Budgeted routes reuses the discrete completion model and residual flow. Latent samples map to public graph nodes for execution. The composer deduplicates targets by executable node and keeps the originating sample and slot. A transition model estimated from training exploration predicts budgeted terminal success using public node, inventory and risk profile. Evaluation owns the world-specific probabilities and future draws. [Budgeted route composition](experiments.md#compose-latent-goals-under-action-budgets)

## Execution feedback

Training rollouts record arrival and terminal success for predeclared targets. The supervised reranker fits on training attempts and accepts only public candidate features. Flow refinement pairs complete demonstration targets with masked feedback targets whose loss applies to the attempted latent position. A demonstration-only update controls for the additional optimizer steps. The evaluator keeps repeat masks and held-out outcomes outside the input to all selectors. [Feedback training protocol](experiments.md#train-from-execution-feedback)

## AntMaze baseline

The raw OGBench archive reader extracts public observations and actions. Terminal markers split stored trajectories and exclude their final padding actions. The hindsight sampler pairs an action at a nonfinal row with a later observation from the same trajectory. The actor maps the current observation and public goal observation to a continuous action. Training and validation use separate archives.

The evaluator records task and reset IDs, MuJoCo integration state, task goal and the exact public tensors from the reset. Each rollout restores the snapshot and starts with the saved goal. The actor receives only the latest public observation and the public goal. The evaluator uses the benchmark's success event, including its occurrence on the final allowed action. Development and test use disjoint reset seeds. [AntMaze baseline protocol](experiments.md#train-the-antmaze-gcbc-baseline)

The learned AntMaze planner first fits an observation encoder and decoder by reconstruction, then freezes them. A masked completion predictor uses the public observation and action history, terminal goal and remaining budget. A conditional flow samples residual trajectories around the prediction in latent space. The decoder maps the sampled latents back to full public observations. The development runner gives the frozen GCBC actor the earliest observation from the first sampled route and replans when that slot's queried time offset elapses. Each replan uses the actual observation and action history. The evaluator scores the original terminal objective over the full rollout. [Learned planner protocol](experiments.md#run-learned-latent-planning-in-antmaze) · [Research design](trajectory-completion.md)

The executor diagnostic uses privileged maze geometry inside the evaluator to predeclare an XY route. Its position tolerance controls the handoff between adjacent waypoints in that diagnostic. [Waypoint protocol](experiments.md#evaluate-antmaze-waypoint-execution)

## PointMaze benchmark

The learned PointMaze and AntMaze planners share the masked window sampler, completion model and time-based replanning policy. The PointMaze task determines the Minari archive, evaluation specification and official action limit. The loader retains four public position and velocity features and two continuous actions. The evaluator restores a natural reset from its seed and passes the public goal position with zero target velocity. The same feedback controller consumes either that goal or a decoded intermediate observation. The planner receives public observations and actions. The simulator's goal geometry and sparse rewards are available to the evaluator for setup and scoring. [Maze2D protocol](experiments.md#run-the-diffuser-maze2d-pilots)

The official action limit is PointMaze's default evaluation budget. The model conditions on the actual remaining budget and queries a prediction span capped by trajectory coverage. Full-span training windows can share an observed endpoint and query times while allowing different remaining budgets. Budget sweeps share a checkpoint, seeded resets, executor, and goal predicate. The evaluator records the first native success event and the span used for each prediction. [Budget sweep](experiments.md#sweep-the-action-budget-in-maze2d-medium)

## Kuka stacking benchmark

The Kuka adapter runs the pinned upstream PyBullet simulator with headless Gymnasium compatibility. Its planner uses the public 39-coordinate state, a terminal block-position condition with free horizontal translation, and time. The demonstration files contain states, so the shared masked completion model uses observation history alone. The executor tracks decoded arm joint targets and attachment requests using the author's motor-control and grasp interface. The evaluator keeps access to simulator contacts and supplies the same seeded start and block order to each arm. [Conditional stacking protocol](experiments.md#run-conditional-kuka-stacking)

## OGBench manipulation benchmarks

Scene and Puzzle share the OGBench state archive loader, continuous goal-conditioned cloning model, masked completion and receding-goal planner. The benchmark preset specifies the observation width, archive and official action limit. The loader reads public observations, five-dimensional actions and episode markers. The evaluator seeds a persistent action space because OGBench samples actions while constructing the goal observation. It regenerates the exact public start and goal and verifies MuJoCo integration state before the second arm. Simulator integration state and native success events stay inside evaluation. [Scene protocol](experiments.md#run-state-based-ogbench-scene) · [Puzzle protocol](experiments.md#run-state-based-ogbench-puzzle)

## Shared-attempt evaluation contract

The shared-attempt comparison fits its scoring heads on training starts and selects the reachability threshold on development starts. Each external simulator can export a `shared_attempts_v1` bank with disjoint root `world_id`s, fixed public features, a `direct` arm, and ordered matched trials for each candidate. An optional `native_choice` identifies the native selector's choice from the same bank. `public_bank` projects each record to the selection input. Selection uses only these projected fields. The evaluator uses the held-out arrival and terminal labels to score the frozen choices.
