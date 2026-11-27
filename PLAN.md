# Implementation plan

Learn to find intermediate goals in latent space that improve final task success. At deployment, the planner receives a public observation and action history, a terminal objective and the remaining action budget. A world model proposes intermediate goals as latent vectors by completing the latent trajectory between the history and the terminal condition. [Research plan](docs/latent-subgoal-research.md)

## Central claim

**An intermediate goal is a latent region, and its worth is the value of attempting it.** The world model proposes a latent goal `z` without naming it. A conditional generative decoder `p(s | z)` defines its meaning as a distribution over concrete states, so `z` names a region of states rather than a single target. The executor is conditioned on `z` itself and judges arrival in latent space. The decoder only reveals afterwards which goal the model discovered. The value of a region is the probability of terminal success after the shared executor attempts it until arrival or a deadline and the shared continuation uses the actual remaining budget. Success after a missed goal counts.

| Part | Role in the paper |
| --- | --- |
| **(A) Goals as generative latent regions** | Core claim. The world model discovers regions of state space that matter for terminal success, the executor pursues them without a hand-specified target, and value belongs to the region. |
| **(C) Budget-conditioned option value in latent space** | Value estimation method. Learn `Q(h, z, b)` with semi-Markov Bellman backups over single-segment attempts, and plan sequences of regions with the learned `V(h, b)`. |
| **(B) Plausibility is not value, and the budget changes the choice** | Supporting empirical findings. Frozen model energy does not identify valuable goals, and the goal worth reaching changes with the budget. |

Every part has a precedent: DisCo RL represents goals as state distributions, Goal-Space Planning and STAR use subgoal sets, SAGE and HWM predict single latent subgoals, and HiTS and horizon-aware C-Learning cover deadlines and horizon-conditioned values. The contribution is the combination: regions predicted from the current history, terminal goal and budget, and valued by attempting them under that budget. [Method](docs/latent-subgoal-research.md#method) · [Positioning against prior work](docs/latent-subgoal-research.md#positioning-against-prior-work)

## Method components

| Component | Design | Status |
| --- | --- | --- |
| Latent goal proposer | The masked completion model predicts `ẑ(h, g, b, q)` at query offset `q` from the public history `h`, terminal condition `g` and budget `b`. Candidates come from several query offsets and, where needed, residual flow samples around the prediction. | Implemented with a frozen reconstruction codec and residual flow. |
| Region decoder `p(s | z)` | A conditional diffusion or flow decoder trained on pairs of predicted latents and the true future states at the queried offset, so it learns the full conditional distribution of realizations rather than a mean. It interprets goals and supplies the region-support arrival rule. | Planned. The current decoder is deterministic. |
| Latent-conditioned executor `π(a | h, z)` | Trained on demonstrations with latent goals at matched offsets. It never receives a decoded or hand-projected target. | Planned. Current executors take decoded states projected to cells, nodes or observations. |
| Latent arrival rule | Two candidates, compared before choosing one: **encoder consistency**, `‖enc(s) − z‖ ≤ ε`, and **region support**, `log p(s | z) ≥ κ`. Thresholds are calibrated on development data. | Planned. Current arrival uses exact cells or nodes, or hand-set Scene tolerances. |
| Option value `Q(h, z, b)` | Semi-Markov Bellman backup `Q(h, z, b) = E[V(h', b − τ)]` with `V(h, b) = max(Q_direct(h, b), max_z Q(h, z, b))`, fitted on latent features from single-segment attempts that record the elapsed time `τ` and landing history `h'` whether or not the region was reached. The Monte Carlo logistic selector remains the reference. | Planned. Current selectors use Monte Carlo outcomes on hand-designed features or a public-state transition table. |
| Sequences of regions | Generate ordered region sequences guided by the learned value, compared with enumeration plus dynamic programming at matched inference budgets. Enumeration is exact at depth two and becomes infeasible at depths three and four. | Planned. Current composition enumerates up to 24 ordered pairs. |

## Completed evidence with decoded point goals

These studies decoded each latent proposal into a single public target, projected it to a cell, a graph node or an observation, and used hand-designed selector features. They support (B) and the value definition. They do not yet test goals as latent regions.

| Study | Result | Record |
| --- | --- | --- |
| Dependency grid benchmark | On 16 development layouts, swapping the goal reversed the ranking of the two lane-specific ordered targets on every layout. The goal-matched order succeeded on 511 of 512 attempts, the reversed order on 199 and local direct tracking on 112. | [Benchmark](docs/latent-subgoal-research.md#1-establish-a-controlled-multi-stage-benchmark) |
| Frozen energy versus uniform choice | Minimum flow energy and uniform selection were within noise on development layouts for both model seeds. Reference-selected flow candidates reached 46.5% and 62.5%, so the pool contained useful targets that the frozen score missed. | [Development comparison](docs/latent-subgoal-research.md#2-separate-discovery-selection-and-execution) |
| Outcome selection on test layouts | Outcome scoring raised two-stage latent success over minimum energy by 18.75 and 25.00 points for the two model seeds. The intervals were [6.25, 34.38] and [9.38, 43.75]. | [Confirmation](docs/latent-subgoal-research.md#2-separate-discovery-selection-and-execution) |
| Budgeted route composition on test worlds | At four actions, composition reached 93.16% against 84.38% for joint samples, single targets and replay. The gain was 8.79 points [1.17, 16.80]. Composition confirmed 12 low-support paths per seed, and the selected route changed with the budget on 8 of 8 worlds. | [Confirmation](docs/latent-subgoal-research.md#3-compose-alternative-routes-under-a-budget) |
| Scene executor diagnostic (development, provisional) | Across six executor checkpoints at different scales, drawer insertion within 128 actions rose from 9 of 108 to 39 of 108 validation clips with an observation-conditioned executor. | [Waypoint protocol](docs/experiments.md#test-scene-drawer-insertion-in-isolation) |

## Paper experiments by priority

**P0** experiments are required for the submission. **P1** experiments strengthen specific claims. **P2** work is optional or deferred to later papers.

| Priority | Work | Completion evidence |
| --- | --- | --- |
| P0 · E1 Region goals versus decoded point goals | In the dependency grid and budget routes, train the region decoder and the latent-conditioned executor. On the same latent proposals, compare the current decoded point goal, a single latent point with latent-distance arrival in the style of SAGE and HWM, a Gaussian region in the style of DisCo RL, and diffusion-decoded region goals under each arrival rule. Sweep the arrival threshold to trace region granularity. | Arrival, value of attempting and paired terminal success by goal form on test layouts. Region coherence: the spread of attempt value across realizations sampled from `p(s | z)`. Decoded examples of the regions the selector prefers. [Protocol](docs/latent-subgoal-research.md#4-compare-region-goals-with-decoded-point-goals) |
| P0 · E2 Budget-conditioned option value in latent space | Fit `Q(h, z, b)` with semi-Markov Bellman backups on latent features. Compare with the Monte Carlo selector on hand features, minimum frozen energy, likelihood-based selection in the style of DSP, uniform choice and the reference-selected candidate. Ablate recovery after a miss, budget conditioning and a HiTS-style discounted high level. Refit on 1, 2, 4, 8 and 16 attempts per candidate and give extra demonstrations or executor fine-tuning the same environment steps. | Paired terminal success on test banks, the gap to the hand-feature selector, and terminal success against selector training attempts with the matched-interaction baselines. [Protocol](docs/latent-subgoal-research.md#5-learn-budget-conditioned-option-value-in-latent-space) |
| P0 · E3 Supervision sources on one bank | On frozen region-goal banks, compare the option value with a trajectory-labeled energy in the style of HDFlow, a reachability filter with continuation value in the style of HECRL, arrival only, route length in the style of HorizonFlow and latent endpoint distance in the style of Fast-JEPA. Extend `wr attempt-compare` with the missing arms. | Paired terminal success with world-bootstrap intervals, arrival and miss-then-success rates, and a prespecified analysis of disagreement cases under a shared bank, executor, continuation and attempt set. [Protocol](docs/latent-subgoal-research.md#6-compare-supervision-sources-on-a-shared-bank) |
| P0 · E4 Terminal comparison in OGBench Scene | Qualify the latent-conditioned executor on drawer insertion. Then compare direct execution, minimum frozen energy, option-value selection on region goals, matched-time replay, and targets extracted from Diffuser or Hierarchical Diffuser at matched times on Tasks 4 and 5. Decode the selected regions to check whether the model discovered an open drawer. | Paired terminal success on official resets with intervals across starts and training seeds, after the executor passes a development arrival threshold and direct execution leaves room above floor and below ceiling. [Protocol](docs/latent-subgoal-research.md#7-test-region-goals-in-scene) |
| P0 · E5 Controlled Maze2D comparison with Diffuser | Run our planner and Diffuser on the same environments, data, resets and action budgets of 300 to 600 in Medium and Large, with region goals and the option value added to the frozen-energy pilot. | Paired terminal success by budget, plus the budget at which each method's selected first goal changes. [Protocol](docs/latent-subgoal-research.md#8-compare-with-diffuser-in-maze2d) |
| P1 · E6 Depth and sequence generation | Extend the dependency grid to three and four ordered prerequisites. Compare enumeration with dynamic programming, value-guided generation of region sequences, and closed-loop endpoint scoring at matched inference budgets. | Paired success and ranking quality by depth, with model calls and latency. [Protocol](docs/latent-subgoal-research.md#9-generate-region-sequences-as-depth-grows) |
| P1 · E7 Test the representation | Compare the reconstruction encoder with a predictive encoder trained with SIGReg under the same proposer, decoder, executor and value. | Collapse diagnostics, probes for keys and inventory, region coherence and terminal success. [Protocol](docs/latent-subgoal-research.md#10-isolate-the-representation-contribution) |
| P1 · E8 Vary demonstration coverage and topology | In budget routes, set the full-route frequency in demonstrations to 0%, 5% and 20% while keeping the local segments. Add new graph topologies. | Composition gain and confirmed low-support paths as functions of coverage, on more than one topology. [Protocol](docs/latent-subgoal-research.md#11-vary-demonstration-coverage) |
| P1 · E9 Plan several decisions in closed loop | Choose a new region after each arrival, timeout or detected change, with the option value bootstrapped across decision points. | Terminal success against the single-decision planner and fixed-interval replanning at matched model calls. [Protocol](docs/latent-subgoal-research.md#12-plan-several-decisions-in-closed-loop) |
| P2 | Update the generator from attempt outcomes with weighted flow training or Flow-GRPO. Compare with HDFlow's published visual AntMaze results as a cross-paper reference, since its code is unavailable. Use Craftax for a compound stochastic objective and goal transfer. | Reported separately from the main claim. |

Start with E1 on CPU in the dependency grid, where regions, arrival and value can be inspected exactly. E2 and E3 reuse the E1 attempt banks. In parallel on GPU, train the region decoder and latent-conditioned executor for Scene and Maze2D. Freeze each protocol on development data before running its test split.

## Experiment infrastructure

The paper uses these public entry points. The region decoder, latent-conditioned executor and option value will be added to them.

| Command | Role |
| --- | --- |
| `wr dependency-grid`, `wr dependency-grid-latent` | Benchmark checks, latent proposals, frozen energy, and outcome selection with `--selector-repeats`. [Grid protocol](docs/experiments.md#compare-ordered-targets-in-the-dependency-grid) |
| `wr budget-routes` | Budget-conditioned composition, fixed-pool route changes, rare-path confirmation and suffix replanning. [Routes protocol](docs/experiments.md#compose-latent-goals-under-action-budgets) |
| `wr attempt-compare` | Selector comparison on `shared_attempts_v1` banks. The base for E3. [Selector comparison](docs/experiments.md#compare-selectors-on-matched-attempts) |
| `wr scene-latent`, `wr scene-evaluate`, `wr scene-reachability` | Scene training, terminal evaluation and executor qualification. [Scene protocol](docs/experiments.md#run-state-based-ogbench-scene) |
| `wr pointmaze-latent`, `wr pointmaze-latent-energy` | Maze2D planning, budget sweeps and frozen-energy tests. [Maze2D protocol](docs/experiments.md#run-the-diffuser-maze2d-pilots) |

The shared masked trajectory completion model supplies the proposer for every task. It trains a public-history encoder and predictor to complete the hidden middle representations from both ends, then samples alternatives with residual flow matching. [Trajectory completion design](docs/trajectory-completion.md) · [Model configuration](docs/experiments.md#configure-a-completion-model)

Runners outside the paper's main line were removed: Craftax collection and windows, KeyCorridor replay, the two-goal grid and its feedback training, AntMaze, Kuka stacking, OGBench Puzzle and the Maze2D first-target probe. Retrieve them from Git history if a follow-up needs them. [Running experiments](docs/experiments.md) · [Environments](docs/experiment_env.md)

## Evaluation protocol

Keep exploratory development worlds separate from test worlds. Freeze decision rules, arrival thresholds and region granularity before the final comparison. Declare candidates and scores in a candidate bank before rollouts. Compare methods on matched starts and independent repeat streams. When isolating proposal, goal form or selection, share the executor, continuation, terminal objective and candidate budget. Matched interventions measure the true value of attempting a goal and serve as the evaluation ground truth for every learned value. Record all environment interactions used for training, selector fitting, proposal updates and evaluation, and give baselines the same interaction allowance when they can use it. [Architecture](docs/architecture.md) · [Running experiments](docs/experiments.md)
