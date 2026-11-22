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
| `envs/` | Versioned simulator adaptations and terminal task rules |
| `data/` | Collect complete public trajectories, validate world splits and export masked training and development windows |
| `policies/` | Recurrent expert inference from public observations |
| `evaluation/` | Replay, matched continuations and comparisons of shared candidate banks |
| `cli.py` | Connect configurations to `pilot`, `collect`, `prepare-windows` and `attempt-compare` |

The trajectory completion model will use public data from `data/`. Its executor will consume public observations and emit primitive actions. Evaluation owns simulator snapshots and restores them for matched repeats. [Trajectory completion design](trajectory-completion.md)

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

`data/craftax_trajectories.py` loads completed random and expert collections through the same interface. It validates task rules and verifies episode IDs and split assignments against the original world seeds. Its split iterator yields complete public episode records without loading a policy checkpoint or constructing a simulator.

`evaluation/craftax.py` reconstructs a live start from recorded actions and verifies each replayed observation. It returns the recurrent memory immediately before the current observation. A continuation can act directly or run a declared action prefix first. Forced actions still advance policy memory from each public observation. Match repeat keys across alternatives to compare outcomes on the same transition and policy random streams.

## Masked trajectory window contract

For a successful craft-and-survive episode with budget `B`, index observations from `o[0]` and let action `a[t]` produce `o[t]`. A window starts at an observed step `s` and queries the next `L` positions, with `s + L < B`. The input contains observations through `o[s]`, actions through `a[s]`, the terminal task and the remaining budget `B - s`. The target contains observations and incoming actions for steps `s + 1` through `s + L`. The query offsets `1..L` describe which positions to predict. They do not depend on when the pickaxe was crafted.

Each successful episode contributes a window from the reset and up to seven later starts by default. The seed and world ID determine the sampled starts and lengths. Each export inherits the archive's world split. Train and development exports use terminal outcomes to select demonstrations. Model inputs contain the public task descriptor. The training shard can additionally supply `o[B]` through an explicit endpoint-example option. Development windows use the task descriptor alone. The final action is outside the windows, and the test split stays in the source archive for evaluation. [Window export and loader](experiments.md#prepare-masked-trajectory-windows)

## Shared-attempt evaluation contract

The shared-attempt comparison fits its scoring heads on training starts and selects the reachability threshold on development starts. Each external simulator can export a `shared_attempts_v1` bank with disjoint root `world_id`s, fixed public features, a `direct` arm, and ordered matched trials for each candidate. An optional `native_choice` identifies the native selector's choice from the same bank. `public_bank` projects each record to the selection input. Selection uses only these projected fields. The evaluator uses the held-out arrival and terminal labels to score the frozen choices.
