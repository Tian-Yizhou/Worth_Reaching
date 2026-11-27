# Implementation plan

Learn to find intermediate goals in latent space that improve final task success. At deployment, the planner receives a public observation and action history, a terminal goal and the remaining action budget. A world model samples latent futures, and a budget-conditioned option value decides which one to pursue, for how long and when to change course. [Research plan](docs/latent-subgoal-research.md)

## Central claim

**A latent goal means what happens when it is pursued, and one budget-conditioned value decides which goal is worth reaching.** One executor conditioned on a latent `z` pursues it for at most `k` actions, so there is no decoder target and no arrival test. `Q(h, g, z, k, b)` chooses the candidate and its duration, treats the encoded terminal goal as a candidate in every comparison, and decides when to switch or sample new candidates. It is fitted offline on hindsight options and corrected with the executor's own attempts.

| Part | Role in the paper |
| --- | --- |
| **(A) Latent goals as options** | A latent goal is defined by its outcomes under execution, with no decoder target, arrival test or replanning schedule. |
| **(C) Budget-conditioned option value** | The method. Semi-Markov expectile backups on hindsight options, bounded to the returns possible, then online correction with the executor's attempts. |
| **(B) Plausibility is not value, and the budget changes the choice** | Supporting findings from the controlled tasks, E1 and the time-cost sweep. |

Every component has a precedent: Director, HIQL, SAGE and HWM propose latent subgoals for latent-conditioned executors, HiTS attaches durations to subgoals, TDM and horizon-aware C-Learning learn horizon-conditioned values, and options with interruption go back to Sutton, Precup and Singh. The contribution is choosing the goal, its duration and when to switch with one value under the global budget, with the terminal goal always competing. [Method](docs/latent-subgoal-research.md#method) · [Positioning against prior work](docs/latent-subgoal-research.md#positioning-against-prior-work)

## Method components

| Component | Design | Status |
| --- | --- | --- |
| World model | Masked completion predicts latents at query offsets from the history, terminal condition and budget, with a frozen reconstruction codec; a residual flow samples several futures per offset. | Implemented. Offsets are fixed fractions of the span; geometric offsets are planned. |
| Latent-goal executor | GCIVL with frozen encodings of archive observations as goals. | Implemented for Maze2D and Scene. |
| Option value `Q(h, g, z, k, b, λ)` | Hindsight options from the archive, in-sample semi-Markov expectile backups clipped to the returns possible, time cost as an input, bootstrap-masked heads. | Implemented (`models/option_value.py`). |
| Online correction | Rounds of the planner on training resets with exploration over proposal budgets, plus uninterrupted goal pursuit on every reset; the value keeps training on the executor's attempts. | Implemented for Maze2D and Scene; goal pursuit added after the first full run. |
| Option planner | Candidates at every query offset under the remaining and shorter budgets plus the encoded goal; switching among cached candidates on the ensemble spread; sampling again on duration, value drop or latent surprise. | Implemented (`planning/options.py`). |

## Completed evidence

| Study | Result | Record |
| --- | --- | --- |
| Dependency grid benchmark | On 16 development layouts, swapping the goal reversed the ranking of the two lane-specific ordered targets on every layout. The goal-matched order succeeded on 511 of 512 attempts, the reversed order on 199 and local direct tracking on 112. | [Benchmark](docs/latent-subgoal-research.md#1-establish-a-controlled-multi-stage-benchmark) |
| Frozen energy versus uniform choice | Minimum flow energy and uniform selection were within noise on development layouts for both model seeds. Reference-selected flow candidates reached 46.5% and 62.5%, so the pool contained useful targets that the frozen score missed. | [Development comparison](docs/latent-subgoal-research.md#2-separate-discovery-selection-and-execution) |
| Outcome selection on test layouts | Outcome scoring with hand-designed features raised two-stage latent success over minimum energy by 18.75 and 25.00 points for the two model seeds, with intervals [6.25, 34.38] and [9.38, 43.75]. | [Confirmation](docs/latent-subgoal-research.md#2-separate-discovery-selection-and-execution) |
| Budgeted route composition on test worlds | At four actions, composition reached 93.16% against 84.38% for joint samples, single targets and replay, a gain of 8.79 points [1.17, 16.80]. Composition confirmed 12 low-support paths per seed, and the selected route changed with the budget on 8 of 8 worlds. | [Confirmation](docs/latent-subgoal-research.md#3-compose-alternative-routes-under-a-budget) |
| E1 region goals against point goals | Region goals did not beat point goals in the grid, Maze2D or Scene. In Maze2D, decoded targets from larger world models reached 1.00 in Medium against 0.58 for direct tracking, but every model arm first succeeded near the action limit, which is the ablation without a time cost. In Scene, unselected targets lowered Tasks 2 and 3 from 0.625 to at most 0.05. | [E1 results](docs/latent-subgoal-research.md#4-compare-region-goals-with-decoded-point-goals) |
| E2 option planner in Maze2D (seed 7 probe) | With bounded backups, online correction and shorter-budget proposals, the planner reached 1.00 in Medium (first success at step 148 against 181 for direct latent pursuit) and 0.91 in Large (0.45 direct latent, 0.48 direct, 1.00 oracle). Either change alone was weaker; removing interruption cost 0.13 in Large. | [Option value](docs/latent-subgoal-research.md#5-learn-budget-conditioned-option-value-in-latent-space) |
| E2 full development run | Maze2D Large: 1.00 on three seeds against 0.48 direct, 0.57 direct latent and 1.00 oracle, mostly from shorter-budget proposals; Medium at ceiling. Scene: 0.42 against 0.34 direct, with Task 5 from 0.00 to 0.31 and Task 4 from 0.10 to 0.34, but Tasks 2 and 3 from 0.63 to about 0.52 because the value underrated the goal candidate and chose it in 2% of decisions. | [Option value](docs/latent-subgoal-research.md#5-learn-budget-conditioned-option-value-in-latent-space), [Scene](docs/latent-subgoal-research.md#6-run-the-option-planner-in-scene) |
| Scene executor | A GCIVL executor trained for 1M updates reached 0.41 over the five tasks on 20 development starts per task and two seeds, in line with OGBench's GCIVL. Tasks 4 and 5 stayed at 0.10 and 0.00. | [Executor protocol](docs/experiments.md#qualify-a-value-based-scene-executor) |

## Paper experiments by priority

**P0** experiments are required for the submission. **P1** experiments strengthen specific claims. **P2** work is optional or deferred to later papers.

| Priority | Work | Completion evidence |
| --- | --- | --- |
| P0 · E2 Option planner in Maze2D | Medium and Large on three world-model seeds: offline and corrected values with and without shorter-budget proposals, time costs 0 and 0.001, and ablations of the goal candidate, interruption and the chosen duration. | Paired success and first success step against direct execution, direct latent pursuit and the oracle on every seed, with `Q` bounded. [Protocol](docs/latent-subgoal-research.md#5-learn-budget-conditioned-option-value-in-latent-space) |
| P0 · E2 Option planner in Scene | Tasks 2 to 5 on four world-model seeds with the same arms except the oracle. | At or above direct execution on Tasks 2 and 3 on every seed, and a gain on Tasks 4 and 5. [Protocol](docs/latent-subgoal-research.md#6-run-the-option-planner-in-scene) |
| P0 · Interactions | Success against online attempts, with matched-step baselines: more demonstrations, and executor fine-tuning toward the terminal goal. | A learning curve per task and the step budget at which the baselines catch up, if they do. |
| P0 · Baselines and ablations | HIQL, likelihood-based selection in the style of DSP and Diffuser in Maze2D. Ablate budget conditioning, use a HiTS-style discounted high level and fixed-interval replanning. | Paired success by method and budget. [Diffuser and HIQL](docs/latent-subgoal-research.md#7-compare-with-diffuser-and-hiql-in-maze2d) |
| P1 · Geometric offsets | Retrain the world models with offsets growing geometrically up to the remaining budget. | Success, first success step and the share of choices from shorter-budget proposals against fraction offsets. |
| P1 · Supervision sources | On a shared bank, the option value against HDFlow-style trajectory energy, a HECRL-style reachability filter, arrival only, route length and endpoint distance. | Paired success with world-bootstrap intervals and a disagreement analysis. [Protocol](docs/latent-subgoal-research.md#8-compare-supervision-sources-on-a-shared-bank) |
| P1 · Depth, representation and coverage | Three and four prerequisites in the dependency grid, a SIGReg predictive encoder, and route coverage of 0%, 5% and 20% with new topologies. | [Depth](docs/latent-subgoal-research.md#9-scale-to-deeper-tasks) · [Representation](docs/latent-subgoal-research.md#10-isolate-the-representation-contribution) · [Coverage](docs/latent-subgoal-research.md#11-vary-demonstration-coverage) |
| P2 | Update the proposer from attempt outcomes with weighted flow training or Flow-GRPO. Compare with HDFlow's published visual AntMaze results. Use Craftax for a compound stochastic objective and goal transfer. | Reported separately from the main claim. |

Run each study on development resets first and freeze its settings before the test split.

## Experiment infrastructure

The paper uses these public entry points.

| Command | Role |
| --- | --- |
| `wr dependency-grid`, `wr dependency-grid-latent` | Benchmark checks, latent proposals, frozen energy, and outcome selection with `--selector-repeats`. [Grid protocol](docs/experiments.md#compare-ordered-targets-in-the-dependency-grid) |
| `wr budget-routes` | Budget-conditioned composition, fixed-pool route changes, rare-path confirmation and suffix replanning. [Routes protocol](docs/experiments.md#compose-latent-goals-under-action-budgets) |
| `wr attempt-compare` | Selector comparison on `shared_attempts_v1` banks. The base for the supervision-source comparison. [Selector comparison](docs/experiments.md#compare-selectors-on-matched-attempts) |
| `wr scene-latent`, `wr scene-evaluate`, `wr scene-reachability`, `wr scene-executor`, `wr scene-regions`, `wr scene-options` | Scene training, terminal evaluation, executor qualification, closed-loop E1 and the option planner. [Scene protocol](docs/experiments.md#run-state-based-ogbench-scene) |
| `wr pointmaze-latent`, `wr pointmaze-latent-energy`, `wr pointmaze-regions`, `wr pointmaze-options` | Maze2D planning, budget sweeps, frozen-energy tests, closed-loop E1 and the option planner. [Maze2D protocol](docs/experiments.md#run-the-diffuser-maze2d-pilots) |

The shared masked trajectory completion model supplies the proposer for every task. It trains a public-history encoder and predictor to complete the hidden middle representations from both ends, then samples alternatives with residual flow matching. [Trajectory completion design](docs/trajectory-completion.md) · [Model configuration](docs/experiments.md#configure-a-completion-model)

Runners outside the paper's main line were removed: Craftax collection and windows, KeyCorridor replay, the two-goal grid and its feedback training, AntMaze, Kuka stacking, OGBench Puzzle and the Maze2D first-target probe. Retrieve them from Git history if a follow-up needs them. [Running experiments](docs/experiments.md) · [Environments](docs/experiment_env.md)

## Evaluation protocol

Keep training, development and test resets disjoint, and freeze settings on development resets before the test split. Compare methods on the same starts with paired differences, and share the world model, executor, budget and candidate count when isolating one component. Direct execution is the floor and a privileged planner the ceiling where one exists. Record environment interactions for training, online correction and evaluation separately, and give baselines the same interaction allowance when they can use it. In the controlled tasks, matched interventions from saved resets measure the true value of attempting a goal. [Architecture](docs/architecture.md) · [Running experiments](docs/experiments.md)
