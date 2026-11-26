# Research architecture

Use an environment-independent planner and a simulator-backed evaluator. At decision time each policy receives the public history, objective, remaining budget and any legitimate public action constraints. The evaluator owns simulator snapshots, future random keys and outcome labels. This boundary allows the same planner to operate in grid tasks, mazes and manipulation.

## Proposed components

| Component | Inputs and outputs | Responsibility |
| --- | --- | --- |
| Environment adapter | Reset or step produces public observation, action specification, terminal flags and scoreable events | Map native actions and observations, apply task termination, pin rules and wrapper versions. Provide optional privileged branching to the evaluator through a separate interface. |
| Trajectory recorder and split manifest | Episodes and policy versions produce trajectories, snapshot references and world-group splits | Preserve action order, termination reason and data provenance. All descendants of one root episode belong to its split. |
| History encoder and world model | Public observation and action history produces a latent state. The starting history, terminal condition and time information determine predictions for the missing middle. | Learn task-relevant middle representations from complete training traces with variable-length masks. Action-conditioned forward predictions can also support execution checks. |
| Candidate proposer | Public history, terminal objective and budget produce candidate latent routes and next targets | Sample different middles for the same start and goal using a conditional generator. Use the same candidate interface for replay comparisons and retain proposal provenance. |
| Goal representation and executor | Latent goal `z`, public history and budget produce primitive actions and a stop reason | A latent goal names the region of states given by a generative decoder `p(s | z)`. The planned executor conditions on `z` directly and judges arrival in latent space by encoder consistency or region support, so different concrete histories and actions can realize one goal. Declare the arrival rule and deadline before each prospective attempt. Current runners decode `z` to a single observation and use observation-conditioned executors. [Method](latent-subgoal-research.md#method) |
| High-level planner | Current latent state, terminal goal and budget produce a target, duration or direct plan | Generate alternatives, score them with a budget-conditioned option value `Q(h, z, b)`, execute the first segment and replan from observed outcomes. |
| Branch evaluator and experiment runner | Frozen starts, interventions, policies and random streams produce attempts and aggregate results | Collect interventional labels, enforce permissions, meter interactions and compute, and estimate paired outcomes. |

The `src/worth_reaching/` package groups code by responsibility.

| Package | Current responsibility |
| --- | --- |
| `tasks/<family>/environment.py` | Adapt the simulator and its task rules, expose public observations and actions, and support evaluator-controlled resets |
| `tasks/<family>/scripts/` | Hold task-specific setup and collection launchers and their supporting files |
| `data/` | Represent public completion examples and masked trajectory windows, read shared OGBench archives and record experiment artifacts |
| `models/` | Encode observations, predict masked latent positions and sample residual trajectories |
| `models/training.py` | Configure Lightning trainers, step-indexed batches, checkpoints and logs |
| `evaluation/` | Shared-bank selector comparison, region calibration and arms, the closed-loop goal pursuit shared by Maze2D and Scene, and Scene evaluation and reachability |
| `cli.py` | Connect configurations to training and evaluation commands |
| `tasks/pointmaze/` | Own the Maze2D task specifications, dataset and simulator adapters, controller, experiment commands and paired probes |
| `tasks/dependency_grid/` | Define switchable goals with ordered prerequisites, collect public trajectories, and evaluate declared or learned targets through paired execution |
| `tasks/budget_routes/` | Sample latent goals, compose executable prefixes and assess rare routes under a shared learned transition scorer |
| `planning/goals.py` | Share stage semantics for every task: pursue declared goals in order until arrival, deadline or termination, abandon the rest after a miss, and record how each stage ended. The task supplies the target its executor consumes and the arrival rule |
| `planning/receding_goals.py` | Replan decoded goals from public observation and action histories for the continuous tasks |

Group tasks that share simulator behavior in one family. For example, Scene uses the OGBench reset and step implementation in `tasks/ogbench_manipulation/environment.py`, and its preset selects the observation schema, goals and action limit. A new family can follow the PointMaze or dependency-grid layout and register its commands from a lightweight `cli.py`, so listing commands works without installing simulator extras.

PointMaze and Scene runners assemble the task adapter, completion model and controller. The model consumes public histories, goal observations, remaining action budgets and query offsets. The task loader checks the observation and action shapes against its archive schema. Both use the continuous completion model in `models/continuous_completion.py` and the window types in `data/trajectory_windows.py`. The dependency grid and budgeted routes use the discrete-action predictor in `models/completion.py`.

The trajectory completion model will use public data from `data/`. Its executor will consume public observations and emit primitive actions. Evaluation owns simulator snapshots and restores them for matched repeats. [Trajectory completion design](trajectory-completion.md)

## Training on multiple devices

New model training uses PyTorch Lightning. Lightning runs DDP on one host and manages checkpointing and logging for each stage. A step-indexed sampler supplies the same global batch across device counts. A row mask excludes padding, and the loss uses the global count of valid elements to preserve its weighting under DDP. A stateful dataloader saves its cursor with the optimizer and model. Scene can evaluate independent starts in CPU processes after training. Frozen Scene evaluation also supports an experimental CUDA physics path through Warp. Each evaluator keeps both arms of a matched start together. [Lightning Trainer](https://lightning.ai/docs/pytorch/stable/core-api/trainer) · [Scene evaluation backends](experiments.md#compare-native-cpu-and-batched-cuda-scene-evaluation)

## Execution and data contracts

1. A collector creates root episodes and records the public prefix, behavior policy and observable task outcomes. An adapter can additionally emit an opaque snapshot handle into a private store.
2. The branch evaluator names an action or waypoint and a deadline before rollout. It restores a start, forks the random stream, applies the intervention, then runs the declared continuation policy. Exact replay and fresh stochastic continuations are distinct sampling modes.
3. An attempt record carries the root world and episode IDs, prefix index, public history reference, option or action ID, region and proposer versions where applicable, policy and encoder versions, budget, random-stream group, arrival outcome, terminal outcome, elapsed steps and termination reason. A training loader exposes an explicit allowlist of fields. A hindsight-achieved goal and a prospectively attempted goal have distinct labels.
4. At inference, the planner encodes the latest public history, scores alternatives using its learned models, and gives the executor a target and time budget. The next observation closes the loop. For a direct planner the executor targets the terminal objective. Results record when a waypoint completes, times out, becomes infeasible, or is replaced.
5. Each run stores an experiment configuration, dependency versions, code revision, seed and split manifests, dataset or checkpoint identifiers, and metric definitions. Complete traces, logs and checkpoints live in the Git-ignored `results/` directory at the repository root. Commit small manifests or representative fixtures where redistribution permits.

## First interface checks

Test termination at the action limit and success on the last allowed action. Check that restoring a branch reproduces its public prefix and leaves the source episode intact. Perturb privileged fields while holding the public history fixed and compare the serialized policy input. Verify that an executor evaluated against a frozen waypoint uses the same encoder version and arrival rule that defined that waypoint.

## Latent predictor

The first trainer learns an observation autoencoder on masked training observations and freezes its encoder and decoder. Both predictors use that same target representation, initial predictor weights and sampled batches. A GRU summarizes public observations and their incoming actions. The predictor receives that history representation, the time remaining and the requested time offsets. The goal-conditioned arm also receives the public task identifier. It is trained to predict the encoded observation at each requested position.

The trainer pads the recent history and uses evenly spaced queries across longer masks to bound batch size. Only the public-input packer feeds prediction. Target observations are encoded separately for training and metrics. The decoder translates predictions into observation space for diagnostics. The model saves inference weights, architecture and training settings. [Predictor checkpoint selection](experiments.md#select-a-predictor-checkpoint)

## Sampled routes and attempts

The residual flow learns the difference between a frozen prediction and an encoded demonstration for the complete interior window. It conditions on the starting observation, the goal and the frozen prediction. Each draw decodes to a route and declares intermediate targets with attempt deadlines. After arrival or timeout the same controller continues toward the terminal objective with the remaining budget. The evaluation runner supplies repeat streams, pairs those streams across candidates and saves the attempted latent target separately from the observed results.

The dependency-grid adapter defines public geometry, position, and acquired keys. A separate experiment runner declares target cells, applies matched slip draws, and records realized arrivals, elapsed steps, and terminal success. A changed terminal goal or depth reuses the same layout, reset, and slip draws. The diagnostic uses disjoint layout IDs for train, development, and test. [Ordered-target diagnostic](experiments.md#compare-ordered-targets-in-the-dependency-grid)

The learned dependency-grid runner constructs `CompletionExample` windows with a public reset and two time-indexed observation targets. It assembles the shared discrete completion model and residual flow. Target extraction uses decoded coordinates and public walls. The optional scorer is fitted on paired training-layout attempts and ranks both pools from public goal-relative features. Its choices are saved before evaluation outcomes, reference repeats, and achieved-target trajectories are collected. [Frozen latent-goal comparison](experiments.md#test-frozen-latent-goals-in-the-dependency-grid) · [Outcome-based selection](experiments.md#learn-to-select-a-goal-from-terminal-outcomes)

Budgeted routes reuses the discrete completion model and residual flow. Latent samples map to public graph nodes for execution. The composer deduplicates targets by executable node and keeps the originating sample and slot. A transition model estimated from training exploration predicts budgeted terminal success using public node, inventory and risk profile. Evaluation owns the world-specific probabilities and future draws. [Budgeted route composition](experiments.md#compose-latent-goals-under-action-budgets)

## PointMaze benchmark

The learned PointMaze and Scene planners share the masked window sampler, completion model and time-based replanning policy. The PointMaze task determines the Minari archive, evaluation specification and official action limit. The loader retains four public position and velocity features and two continuous actions. The evaluator restores a natural reset from its seed and passes the public goal position with zero target velocity. The same feedback controller consumes either that goal or a decoded intermediate observation. The planner receives public observations and actions. The simulator's goal geometry and sparse rewards are available to the evaluator for setup and scoring. [Maze2D protocol](experiments.md#run-the-diffuser-maze2d-pilots)

The official action limit is PointMaze's default evaluation budget. The model conditions on the actual remaining budget and queries a prediction span capped by trajectory coverage. Full-span training windows can share an observed endpoint and query times while allowing different remaining budgets. Budget sweeps share a checkpoint, seeded resets, executor, and goal predicate. The evaluator records the first native success event and the span used for each prediction. [Budget sweep](experiments.md#sweep-the-action-budget-in-maze2d-medium)

## OGBench manipulation benchmarks

Scene uses the OGBench state archive loader, continuous goal-conditioned cloning model, masked completion and receding-goal planner. The benchmark preset specifies the observation width, archive and official action limit. The loader reads public observations, five-dimensional actions and episode markers. The evaluator seeds a persistent action space because OGBench samples actions while constructing the goal observation. It regenerates the exact public start and goal and verifies MuJoCo integration state before the second arm. Simulator integration state and native success events stay inside evaluation. [Scene protocol](experiments.md#run-state-based-ogbench-scene)

## Shared-attempt evaluation contract

The shared-attempt comparison fits its scoring heads on training starts and selects the reachability threshold on development starts. Each external simulator can export a `shared_attempts_v1` bank with disjoint root `world_id`s, fixed public features, a `direct` arm, and ordered matched trials for each candidate. An optional `native_choice` identifies the native selector's choice from the same bank. `public_bank` projects each record to the selection input. Selection uses only these projected fields. The evaluator uses the held-out arrival and terminal labels to score the frozen choices.
