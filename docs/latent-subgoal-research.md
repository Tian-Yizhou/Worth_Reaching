# Research plan for discovering useful latent subgoals

**Can a world model identify intermediate targets in latent space that improve final task success when pursued?** The planner receives the public observation and action history, the terminal goal, and the remaining action budget. It learns from trajectories without hand-labeled subgoals. The terminal objective can admit several valid routes or outcomes.

The value of a candidate goal ξ is the probability of terminal success after assigning ξ to the shared executor, pursuing it until arrival, deadline or termination, and continuing with the shared policy on the actual remaining budget. The advantage of ξ is its value minus the value of direct execution. Success after a missed goal counts. This definition includes the time spent pursuing the goal and the chance of recovering after a miss.

## Hypotheses

1. **Goal-dependent value.** From a matched start, pursuing some intermediate targets or ordered sequences improves terminal success under a shared executor and action budget. Changing the terminal goal can change which targets help.
2. **Plausibility differs from value.** A pretrained world model can propose useful targets, but its frozen energy, its prediction error and the demonstration frequency of a target identify plausible futures rather than valuable ones. A selector trained on the outcomes of matched attempts identifies the valuable ones before rollout. Supervision from attempts also differs from supervision from recorded trajectory outcomes, arrival alone, route length or distance to a goal example.
3. **Budget-dependent composition.** Separately proposed goals can be recombined into ordered routes. Scoring those routes by attempt value under the actual budget finds reliable routes that joint samples and replay rarely contain, and the chosen route changes with the budget.
4. **Task depth.** As the number of dependent decisions grows, predicting a distant latent endpoint may become less reliable for ranking plans. Test whether explicit intermediate targets preserve useful rankings and improve closed-loop success across task depths. Define depth by prerequisite decisions and log path length, action budget, and inference horizon separately.

## Positioning against prior work

Each component of the method has a precedent. Diffuser completes trajectories between fixed endpoints. HorizonFlow samples variable-length latent subgoal routes. HDFlow generates latent subgoals in a learned world model and guides them with an energy. HWM and H-JEPA use higher-level latent predictions as subgoals. Fast-JEPA ranks complete action sequences by a predicted endpoint. HECRL filters subgoals by reachability before ranking them by value. Planning to Practice, CTA and SSD compose goals or fragments. The contribution of this project is therefore not latent trajectory completion itself.

| Claim | Nearest work and its selection signal | Distinguishing evidence |
| --- | --- | --- |
| Goal value is defined and measured by intervention. | HDFlow uses an energy trained on successful and failed trajectories. HorizonFlow selects by route length. HECRL filters by reachability, then ranks by value from the achieved subgoal. Fast-JEPA uses latent distance between a predicted endpoint and a goal example. | Outcome selection beats minimum frozen energy on held-out layouts. A shared-bank comparison of supervision sources isolates the label (section 4). |
| Plausibility is not value. | Energy-based planning ranks candidates by model energy, so low energy stands in for a good future. | Frozen energy ties uniform choice while the same pool contains much better candidates (section 2). |
| The budget changes which goal is worth reaching. | HorizonFlow prefers shorter routes. Fast-JEPA ranks by endpoint distance to the goal and uses deadlines to test execution. | One fixed pool selects different routes at different budgets on every test world (section 3). |
| Recombining separately proposed goals finds reliable routes that demonstrations rarely contain. | CTA and SSD stitch fragments without separating route support from component support. | Composition confirms more low-support paths with independent repeats than joint sampling or replay (sections 3 and 11). |
| Matched interventions with predeclared candidates make the comparison causal. | Prior work reports end-to-end success of complete planners. | Candidate banks, common random numbers and world-level intervals in every comparison. |

The current selectors use public geometry in the grid and a public-state transition table in budget routes. The world model only proposes candidates there. Section 5 moves value estimation into the latent space so the world model also supports selection. Reviewers will also ask for continuous-control evidence (sections 7 and 8) and for the number of simulator attempts the selector consumes (section 6).

## Related work and controls

| Work | Role in the experiment |
| --- | --- |
| Diffuser | Jointly generates state and action trajectories. It is a dense trajectory baseline. For a controlled proposal comparison, extract intermediate targets at matched times and hand them to the same executor. Recompute binary success using the same task protocol when comparing with our Maze2D runs. [Paper](https://proceedings.mlr.press/v162/janner22a.html) |
| HDFlow | Freezes a learned RSSM, then generates high-level latent subgoals with diffusion and low-level segments with rectified flow. Its separate energy model uses successful and failed trajectories. Compare trajectory-level labels with paired-attempt labels on a shared candidate bank when both kinds of supervision are available. Treat published visual results as cross-paper references unless the protocols align. [Paper](https://arxiv.org/html/2605.04525v1#S4) |
| Fast-JEPA | Uses intermediate latent predictions to construct an action proposal and directly predicts candidate endpoints from the observed anchor to rank complete sequences by latent distance to a goal image. It avoids recursive state prediction at scoring time. Its standard experiments plan 25 actions per query with a 50-action episode limit. Its DynaGoal benchmark shifts goals, imposes deadlines and delays observations during execution. Build a closed-loop endpoint-scoring control, use endpoint distance as a selection arm on the shared bank, and measure rank quality as task depth increases. [Paper](https://arxiv.org/html/2611.25927v1) |
| HorizonFlow | Samples variable-length latent subgoal routes between the current state and a goal and prefers candidates with fewer generated tokens. Use length as a selection arm on the shared bank and name the proposer that produced the bank. [Paper](https://arxiv.org/html/2609.36896v1) |
| HECRL | Factored Subgoal Diffusion filters generated subgoals by estimated reachability and ranks survivors by value from the subgoal. `wr attempt-compare` implements this factorized filter on shared attempts. [Paper](https://arxiv.org/html/2602.02722) |
| BGP | Shows that stochastic transitions can make an offline route look feasible. Independent repeats confirm low-support paths before they count as discoveries. [Paper](https://arxiv.org/abs/2610.40130) |
| LeJEPA | Uses prediction plus SIGReg to prevent representation collapse. Test the regularizer in a dynamics model and probe whether the representation retains task-relevant history. LeWorldModel is an action-conditioned implementation reference, and Fast-JEPA also uses SIGReg. [LeJEPA](https://arxiv.org/abs/2511.08544) · [LeWorldModel](https://arxiv.org/abs/2603.19312) |
| FF-JEPA | Predicts the next latent subgoal and uses an action-conditioned model for short-range planning. The comparison motivates testing explicitly goal-dependent subgoal value, compositional transfer, and matched interventions. [Paper](https://arxiv.org/abs/2606.09311) |

## Starting evidence

- In the 391-action Maze2D Medium pilot, direct tracking succeeded on 23 of 64 starts. Among its 41 failures, retrospective inspection found at least one candidate that arrived and then succeeded on 15 and 10 starts for the two checkpoints. Selecting the lowest conditional flow energy before rollout yielded 25 and 18 successes, respectively, tying the highest-energy rule within each checkpoint. Better decoded trajectory error and candidate coverage motivate further testing of selection. [Source and protocol](experiments.md#test-latent-energy-against-terminal-success)
- The two-goal grid shows that changing the endpoint changes the predicted route. Direct execution succeeds on about 99.6% of the ordinary grid's development attempts. In the stochastic passage pilot, terminal-outcome ranking improved on arrival ranking for the frozen proposer, while the end-to-end rates were similar to the risk-aware direct control. These runs motivate a task in which reaching superficially similar states can leave very different options for later decisions. [Grid diagnostics](experiments.md#train-from-execution-feedback)

## Experiment sequence

Sections 1 to 3 are complete. Sections 4 to 8 are required for the paper, and sections 9 to 12 strengthen specific claims. [Priorities](../PLAN.md#paper-experiments-by-priority)

### 1. Establish a controlled multi-stage benchmark

Use the dependency grid. Two destinations share the same start, and routes require either one or two keys and gates. Some keys lie on a direct route, and others require a detour. The two-depth comparison keeps map geometry and action budget fixed for each layout. Random action slips are paired across branches and goal switches. Use layout-disjoint splits. After verifying the benchmark, collect trajectories that cover the individual skills and reserve some combinations of skills and layouts for separate generalization tests. [Run the diagnostic](experiments.md#compare-ordered-targets-in-the-dependency-grid)

Use predeclared key locations as diagnostic targets. Compare direct tracking, individual targets, and both orders of the target pair. Record whether a candidate that helps reach one terminal goal loses its advantage when the goal changes. A full-state planner can help verify benchmark solvability. An executor following public geometry provides a local control for the matched attempts.

**Exit criterion.** The public runner reproduces arrival, timeout, recovery, and final-action success. Matched attempts expose goal-dependent effects and order-dependent effects at depth two. Verify the effect on development layouts before training a selector.

The initial CPU run used 16 development layouts, both goals, and 16 paired repeats per start. At depth two, swapping the goal reversed the ranking of the two lane-specific ordered targets on all 16 layouts. The goal-matched entry-then-inner sequence succeeded on 511 of 512 attempts, compared with 199 of 512 for the reversed sequence and 112 of 512 for local direct tracking. These results check the task design with predeclared targets. The run is recorded under `results/dependency-grid-dev-02/`.

### 2. Separate discovery, selection, and execution

Train the existing masked latent predictor on public trajectories as the first representation baseline. Propose goals from latent trajectory completion and compare with matched-time replay targets and geometric or observation-space candidates. Use the same candidate counts, target deadlines, executor, and budget. Include a diagnostic with executable targets from recorded trajectories to measure controller competence.

Declare candidates and scores before rollout. From each start, run the same controller toward each candidate until arrival, timeout, or native termination. Use the same continuation on the observed state with the actual remaining budget. Count terminal successes after misses. Compare no intermediate target, uniform candidate selection, lowest frozen energy, lowest latent prediction error, and matched arrival or proximity controls on the same pool. Use reversed energy as a score direction check. Assess retrospective candidate capacity separately, with independent repeat pools to select and evaluate the reference in stochastic conditions.

**Exit criterion.** Measure both candidate coverage and the gain achieved by prospective selection on held-out worlds. If the pool has useful targets and frozen ranking leaves a clear gap, compare outcome-based scoring learned from training branches with the frozen rules. If coverage is weak, inspect achieved-target execution and compare alternative proposal sources before refining a scorer.

The development comparison used two training seeds with identical demonstrations, 16 layouts, both goals, eight proposals per source, and 16 repeats per reference and evaluation bank. At depth two, uniform flow candidates succeeded on 28.8% and 33.2% of attempts. Minimum energy selected candidates with 28.1% and 34.4% success. Both paired intervals for the energy gain over uniform include zero. Choosing with independent reference repeats exposed more capacity in the flow pools at 46.5% and 62.5%. Matched-time training replay reached 64.8% under the same reference rule and 37.1% when selecting the nearest-layout candidate before execution. The evaluator's replay of timed targets achieved on the development layout reached 59.2%. This comparison motivates retaining replay in both the proposal and scoring experiments. The final test layouts are reserved for the confirmatory protocol. [Run the latent comparison](experiments.md#test-frozen-latent-goals-in-the-dependency-grid)

The fixed-demonstration summaries are under `results/dependency-grid-latent-dev-seed7-02/` and `results/dependency-grid-latent-dev-seed19-02/`.

Fit the same terminal-success scorer for both proposal pools. A training decision observes the public start, goal and budget, assigns one prefix or direct execution, and receives the final binary outcome. Keep the world model and low-level executor frozen. Use repeated training-layout attempts to estimate this single-choice value and freeze the scores before evaluation. Compare the results against uniform selection, frozen energy, nearest replay, and the diagnostic best-of-bank references. A later closed-loop study can learn another choice after observing arrival, timeout, or a changed state. [Run the outcome-based comparison](experiments.md#learn-to-select-a-goal-from-terminal-outcomes)

For the single-choice confirmation, use the existing split, collection seed 7, model seeds 7 and 19, evaluation seed 43301, eight candidates per pool, 16 repeats per training, reference, and evaluation bank, and a 10% action-slip rate. Keep the model update budgets and training procedure from the development runs. The primary contrast at depth two is outcome scoring versus frozen minimum-energy scoring on the same latent pool. The comparison of latent versus replay pools under the shared scorer measures the proposal source effect. Report both model seeds and paired intervals that resample whole layouts. Use the test outcomes to report this fixed comparison.

On the 16 test layouts, outcome scoring raised two-stage latent success from 37.50% to 56.25% for seed 7 and from 43.75% to 68.75% for seed 19. The paired gains were 18.75 and 25.00 percentage points, with 95% layout-bootstrap intervals of [6.25, 34.38] and [9.38, 43.75]. Using the same scorer with replay yielded 62.30% and 59.38%. The latent-minus-replay differences were −6.05 and +9.38 points, and both intervals included zero. This establishes a benefit from outcome-based ranking in the dependency grid. The experiment used public landmark geometry, a fixed continuation, and one high-level decision. The results are under `results/dependency-grid-outcome-test-seed7-01/` and `results/dependency-grid-outcome-test-seed19-01/`.

Track recurrence in successful trajectories as a complementary measure of progress. For an initially closed drawer, an interpretable landmark is a sufficiently open drawer before insertion. Count a recurring state once per eligible trajectory and compare matched successes and failures at the same task stage. Prospective attempts estimate the gain from explicitly pursuing that region under the shared controller.

### 3. Compose alternative routes under a budget

The central case is a single start and terminal objective with several feasible routes whose value depends on resources, uncertainty and time. Generate latent goals from public histories. Use a model of controlled transitions to score possible target orders by the chance of terminal success under the actual budget. Compare the original joint samples, recombined goals, individual targets, and matched-context replay with one executor and a common proposal cap. Reuse a fixed candidate bank across budgets to isolate selection from changing proposals.

Use the budgeted-routes task to separate rare combinations from unfamiliar individual targets. Include the local segments in training and omit the full reset-to-terminal route from the scripted demonstration distribution. Record ordered and component support separately. Screen each proposed low-support execution path on independent reference repeats, then estimate its reliability and paired terminal gain on fresh repeats. The controlled comparison retains a public-state transition model for scoring and a learned latent model for proposing targets. It provides a baseline for later latent dynamics and more varied layouts. [Run the experiment](experiments.md#compose-latent-goals-under-action-budgets)

For the initial confirmation, freeze the development configuration and use seeds 7 and 19 on the eight test worlds. Keep the collection, proposal and evaluation seeds fixed, the two query offsets, the 24-candidate cap, 64 repeats per reference and evaluation bank, and the budgets of 2, 3 and 4 actions. The primary comparison is composition versus joint sampling and replay at four actions under the common scorer. Report paired success intervals by world and coverage of independently verified rare paths. The fixed-pool route changes diagnose budget conditioning. Suffix replanning is a separate comparison with its own query count.

On the shared eight test worlds, the fixed-pool scorer changed its selected route across budgets on every world for both model seeds. At four actions, composition achieved 93.16% terminal success. Joint sampling, single targets, and matched-context replay each achieved 84.38%. The paired gain over joint sampling and replay was 8.79 percentage points for both seeds, with a world-bootstrap interval of [1.17, 16.80]. Independent repeats confirmed 12 low-support world–path pairs per seed for composition, versus zero and three for joint sampling and zero for replay. Replanning after the observed first target raised success to 95.70%, with a paired gain of 2.54 points [0.39, 5.66]. On the four worlds where the planner selected a pair, supply then gate achieved 88.67%, versus 62.89% in reverse order, 71.09% for supply alone, and 25.78% for gate alone. This study uses one graph topology, scripted demonstration support, and a public-state transition scorer. Runs with the paired order audit are recorded in `results/budget-routes-test-seed7-02/` and `results/budget-routes-test-seed19-02/`.

### 4. Compare supervision sources on a shared bank

This comparison isolates the label that trains the selector. Freeze the dependency-grid and budget-routes candidate banks from the confirmation runs. Every arm sees the same candidates, public features, executor, continuation and training attempts, and chooses one candidate or direct execution per start before evaluation.

| Arm | Training signal and decision rule |
| --- | --- |
| Value of attempting a goal | Predict terminal success after the attempt from all training attempts, including misses and timeouts. Choose the highest prediction and keep direct execution unless a candidate is strictly better. |
| Trajectory-labeled energy, as in HDFlow | Fit a contrastive energy on latent goals extracted from successful and failed recorded trajectories at matched times. The labels come from whole trajectories, not from attempts of the candidate. Choose the lowest energy. |
| Factorized filter, as in HECRL | Estimate arrival and the value of continuing from the achieved goal. Filter by an arrival threshold set on development layouts, then rank by continuation value. |
| Arrival only | Choose the candidate with the highest predicted arrival. |
| Route length, as in HorizonFlow | Choose the shortest candidate route that the proposer generated. |
| Endpoint distance, as in Fast-JEPA | Predict the terminal latent after the attempt and choose the smallest distance to the encoded terminal condition. |
| Frozen energy and uniform choice | The existing reference arms. |

`wr attempt-compare` already implements the attempt value, factorized filter and arrival-only arms on `shared_attempts_v1` banks. Export the grid and routes attempts in that format and add the remaining arms. Fit every learned arm with the same model class and tuning budget on training layouts and select thresholds on development layouts.

**Exit criterion.** Report paired terminal success on test layouts with world-bootstrap intervals, together with arrival, miss-then-success and the share of starts where each pair of arms disagrees. A gain of the attempt value over the trajectory-labeled energy and the factorized filter on the same bank attributes the improvement to the label. Prespecify the disagreement analysis: cases where the factorized filter rejects a candidate whose attempts often recover after a miss, and cases where a frequent trajectory goal attempts poorly.

### 5. Learn attempt value from latent features

The current selectors use hand-designed public features. In the grid, these are landmark distances and slack. In budget routes, planning energy comes from dynamic programming over a public-state transition table. Replace both with quantities the world model provides: the encoded history, the predicted latent at the query time, the candidate latent, the remaining budget and the query offset. Fit a small value head on the same training attempts. For routes, learn a latent transition model on public trajectories and run the same dynamic program over latent states.

Keep the proposer, executor, candidate banks and training attempts fixed. Report the latent selector against the hand-feature selector, frozen energy and the reference-selected candidate on the same test layouts.

**Exit criterion.** The latent selector recovers most of the hand-feature gain over frozen energy. Report the remaining gap with paired intervals. If the gap is large, probe which public variables the latent features fail to encode and relate the result to the representation study in section 10.

### 6. Account for selector interactions

The outcome selector uses simulator attempts that frozen scores do not need. Count those attempts per selector and report them in every table. Refit the selector on 1, 2, 4, 8 and 16 training repeats per candidate and plot test terminal success against attempts. Give two baselines the same number of environment steps: additional demonstrations for the proposer and executor, and fine-tuning of the executor on its own rollouts toward the terminal goal.

**Exit criterion.** The curve shows how many attempts the selector needs to beat frozen energy and the matched-interaction baselines. If the baselines catch up at the same budget, report the selector as a way to spend interactions rather than as a free gain.

### 7. Test the value of attempting goals in Scene

The cube-in-drawer task makes the motivation concrete. Opening the drawer is a valuable intermediate goal, while a likely future under the demonstration distribution can leave the drawer closed. Run the terminal comparison on official Tasks 4 and 5 after qualifying the executor.

Use the achieved-waypoint diagnostic to measure which targets a frozen executor can reach from a recorded start by a given deadline. Use matched replay to check that the archived segment can be reproduced and report arrival by demonstration offset and action budget. Keep this local control diagnostic separate from the official terminal-task evaluation. [Scene waypoint protocol](experiments.md#test-scene-waypoint-reachability)

After qualifying an executor, collect prospective attempts on decoded proposals and matched-time replay goals from the same starts. Fit an arrival probability conditioned on public history, the proposed waypoint and its deadline. Use `−log(max(p_arrival, ε))` as a soft energy and compare it with an arrival mask calibrated on development attempts. Reserve unconditional masks for known invalid outputs. A rare proposal or a miss by one executor provides limited evidence of physical infeasibility. Use only rollout outcomes as labels. Keep episode-level splits and report arrival calibration on held-out candidates, including low-support candidates.

Compare the mask and soft energy with an unfiltered bank under equal proposal and action budgets, sharing the executor and terminal-success scorer. Measure which candidates each rule discards, how many viable routes it retains, and paired terminal success. For sequences, update the next segment's reachability from the actual arrival history and remaining time. Terminal outcome remains a separate label, including when an attempt misses its waypoint and later recovers.

The terminal comparison runs on official resets with these arms: direct execution, minimum frozen energy on latent proposals, outcome selection on latent proposals, matched-time replay with the same selector, and targets extracted at matched times from Diffuser or Hierarchical Diffuser and handed to the same executor. Fit the selector on training resets from separate seeds and report terminal success with intervals across starts and training seeds.

**Exit criterion.** Proceed to the terminal comparison only when insertion arrival passes a threshold fixed on development clips and direct terminal success on development resets is neither near zero nor near one. Development drawer insertion within 128 actions rose from 9 of 108 to 39 of 108 clips across six executor scales, so the executor is not yet qualified.

### 8. Compare with Diffuser in Maze2D

Run our planner and Diffuser on the same environment versions, dataset, resets, controller and action budgets. Sweep budgets from 300 to 600 in Medium and include Large. Add the outcome selector to the existing frozen-energy comparison and keep direct execution, minimum energy and uniform choice on the same candidate banks.

**Exit criterion.** Report paired terminal success by budget for every arm and the budgets at which the selected first target changes. The pilot found useful candidates in the pool for 15 and 10 of 41 direct failures, while minimum energy tied the highest-energy rule. A gain of the outcome selector over minimum energy at matched budgets would reproduce the grid result on a standard benchmark.

### 9. Compare explicit subgoals with endpoint scoring

Extend the dependency grid to three and four ordered prerequisites at fixed map size, then compare a direct terminal-latent scoring planner with an explicit subgoal planner. Give both access to the same public feedback, representation, training data, primitive-action budget and comparable opportunities to replan. Keep the candidate construction and set fixed in a scoring ablation. Measure how both scores rank actual outcomes as task depth grows. In the end-to-end comparison, report success across matched inference budgets, with candidate counts, model calls, and latency measured separately.

**Exit criterion.** Estimate the paired success-rate difference by depth with uncertainty and report candidate ranking at the same depths. A short-task tie and a long-task gain would support the depth hypothesis. Similar scaling curves would constrain that claim.

### 10. Isolate the representation contribution

Compare the current frozen reconstruction encoder with an end-to-end predictive encoder regularized by SIGReg. Hold the planner, data and goal interface fixed and match capacity where possible. Track embedding covariance or effective rank as collapse diagnostics. Probe whether the representation distinguishes public histories with different inventory, lock state or grasp status, and whether those distinctions predict future feasibility. Test the same paired terminal-success endpoint. Add a within-architecture regularizer ablation to attribute changes specifically to SIGReg. Until this comparison is done, describe the method as masked latent prediction with a frozen reconstruction codec rather than as a JEPA representation.

### 11. Vary demonstration coverage

In budget routes, keep every local segment in the demonstrations and set the frequency of the complete reset-to-terminal route to 0%, 5% and 20% at a fixed demonstration count. Add at least two further graph topologies with different numbers of resources and gates. Freeze the support catalogue on training episodes before proposing candidates.

**Exit criterion.** Report the composition gain over joint sampling and replay, and the number of independently confirmed low-support paths, as functions of route coverage on each topology. A gain that shrinks as coverage rises would show that composition matters where demonstrations are sparse.

### 12. Plan several decisions in closed loop

The confirmed results use one high-level decision per start, plus one replanning step in budget routes. Let the planner choose a new goal after each arrival, timeout or detected change in the public state, and refit the selector on attempts collected at those later decision points. Compare event-triggered replanning with replanning at a fixed interval and with the single-decision planner at matched model calls and action budget.

**Exit criterion.** Report terminal success, the number of decisions per episode, and model calls. A gain over the single-decision planner at matched calls supports the sequential use of attempt value.

## Reporting

The primary outcome is the paired change in terminal success under a fixed action budget. Report it across held-out layouts, task depths, and training seeds, with uncertainty clustered by layout. Record arrival, attempt time, missed-target recoveries, goal-switch rankings, and candidate coverage as diagnostics. Choose practical effect thresholds and analysis rules on development data, then freeze them before confirmation. Meter environment interactions used for training, selector fitting, feedback, and evaluation separately. Keep snapshots, simulator random streams, reference rankings, and test outcomes on the evaluator side of the interface.

[Implementation plan](../PLAN.md) · [Architecture](architecture.md) · [Running experiments](experiments.md)
