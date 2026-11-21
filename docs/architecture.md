# Research architecture

Use an environment-independent planner and a simulator-backed evaluator. At decision time each policy receives the public history, objective, remaining budget and any legitimate public action constraints. The evaluator owns simulator snapshots, future random keys and outcome labels. This boundary allows the same planner to operate in a grid world, a crafting game and eventually a card game.

## Proposed components

| Component | Inputs and outputs | Responsibility |
| --- | --- | --- |
| Environment adapter | Reset or step produces public observation, action specification, terminal flags and scoreable events | Map native actions and observations, apply task termination, pin rules and wrapper versions. Provide optional privileged branching to the evaluator through a separate interface. |
| Trajectory recorder and split manifest | Episodes and policy versions produce trajectories, snapshot references and world-group splits | Preserve action order, termination reason and data provenance. All descendants of one root episode belong to its split. |
| History encoder and dynamics | Public observation/action prefix produces a latent state. A latent state and candidate action produce a distribution of subsequent features and termination. | Handle memory, stochastic evolution and optional event-scale predictions. Encode order explicitly when the rules use it. |
| Goal representation and executor | Target region, history and deadline produce primitive actions and a stop reason | Fit fixed neighborhoods or outcome-informed regions from training data. Freeze the membership rule for each comparison, then learn or plan short action sequences toward the region. |
| Candidate proposer | Public history or latent state, terminal objective and budget produce region candidates and deadlines | Propose candidates from replay, a demonstration-trained generator, an RL-tuned generator, or coarse world-model predictions. Use the same candidate interface and retain proposal provenance. |
| High-level planner | Current latent state, terminal goal and budget produce a target, duration or direct plan | Generate alternatives, score their likely consequences, execute the first segment and replan from observed outcomes. |
| Branch evaluator and experiment runner | Frozen starts, interventions, policies and random streams produce attempts and aggregate results | Collect interventional labels, enforce permissions, meter interactions and compute, and estimate paired outcomes. |

The `src/worth_reaching/` package groups code by responsibility. `envs/` adapts simulators, `data/` handles trajectory collection and experiment records, `policies/` selects actions from public observations, and `evaluation/` runs replay and interventions. `cli.py` connects configurations to these modules. Shared public types describe trajectories, actions and goals. Learned dynamics and planning can have their own packages as they are introduced. Only evaluation and training-data paths can obtain simulator snapshots. A bounded simulator planner is a separately labeled baseline.

## Execution and data contracts

1. A collector creates root episodes and records the public prefix, behavior policy and observable task outcomes. An adapter can additionally emit an opaque snapshot handle into a private store.
2. The branch evaluator names an action or waypoint and a deadline before rollout. It restores a start, forks the random stream, applies the intervention, then runs the declared continuation policy. Exact replay and fresh stochastic continuations are distinct sampling modes.
3. An attempt record carries the root world and episode IDs, prefix index, public history reference, option or action ID, region and proposer versions where applicable, policy and encoder versions, budget, random-stream group, arrival outcome, terminal outcome, elapsed steps and termination reason. A training loader exposes an explicit allowlist of fields. A hindsight-achieved goal and a prospectively attempted goal have distinct labels.
4. At inference, the planner encodes the latest public history, scores alternatives using its learned models, and gives the executor a target and time budget. The next observation closes the loop. For a direct planner the executor targets the terminal objective. Results record when a waypoint completes, times out, becomes infeasible, or is replaced.
5. Each run stores an experiment configuration, dependency versions, code revision, seed and split manifests, dataset or checkpoint identifiers, and metric definitions. Complete traces, logs and checkpoints live in the Git-ignored `results/` directory at the repository root. Commit small manifests or representative fixtures where redistribution permits.

## First interface checks

Test termination at the action limit and success on the last allowed action. Check that restoring a branch reproduces its public prefix and leaves the source episode intact. Perturb privileged fields while holding the public history fixed and compare the serialized policy input. Verify that an executor evaluated against a frozen waypoint uses the same encoder version and arrival rule that defined that waypoint.

## Craftax collection contract

Craftax-Classic passes symbolic observations and action IDs to the collection path. The adapter owns the native state. The evaluator can fork a live snapshot and supply a transition key for each step. World generation, transition randomness and policy sampling use separate streams. The terminal label is assigned from the step that first achieves the iron pickaxe, leaves the agent dead or in lava, or exhausts the budget, in that order.

Each compressed trajectory record contains the episode ID, initial observation, public transitions and outcome. The split manifest maps the episode ID to its root world and split before collection. The run manifest identifies the observation schema, rules and action vocabulary. The final summary is written after the trajectory file closes.

Random and PPO-RNN collection share the same environment and transition seed streams for each world. Their policy sampling streams are separate from the simulator. PPO-RNN receives only the current public symbolic observation and its per-episode recurrent memory. The manifest records the checkpoint step and actor digest, and the split label is assigned by world regardless of policy.

The action branch diagnostic reconstructs each simulator start using the recorded reset and transition keys. It verifies the public prefix and rebuilds PPO-RNN memory from public observations, including the observation at the forced decision. Each matched repeat shares new transition and policy keys across candidate arms. The output records public-history references and outcomes, while snapshots stay in the evaluator's memory.
