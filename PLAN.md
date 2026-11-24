# Implementation plan

Learn to find intermediate goals in latent space that improve final task success. At deployment, the planner receives a public observation and action history, a terminal objective and the remaining action budget. The controlled studies test ordered prerequisites, switchable goals, and different routes to a common terminal. [Research plan](docs/latent-subgoal-research.md)

| Priority | Work | Completion evidence |
| --- | --- | --- |
| 1. Establish the dependency benchmark | Run paired interventions at one and two stages and switch the goal from the same start. | The dependency grid reports terminal effects by depth, goal, and target order through `wr dependency-grid`. |
| 2. Test latent goal discovery | Collect public dependency-grid trajectories and compare latent, replay and direct execution with `wr dependency-grid-latent`. | Paired rollouts quantify candidate coverage and how much of it frozen scoring can recover. Select the confirmation protocol on development layouts before evaluating the test split. |
| 2a. Learn to select a goal from terminal outcomes | Add `--selector-repeats 16` to fit one scorer on paired training attempts and apply it to both proposal sources. | Compare one prospective choice per case against frozen energy, nearest-layout replay, and direct tracking under matched repeats. |
| 2b. Compose alternative routes under a budget | Run `wr budget-routes` with one terminal and several possible routes. Compare ordered pairs of proposed latent goals, original joint samples, and replay. | Keep one pool fixed across budgets to check route changes. Verify low-support paths with independent repeats and measure closed-loop suffix changes. |
| 2c. Verify the Scene executor | Run `wr scene-reachability` on archived validation waypoints. Pair the frozen controller with action replay and compare arrival across action budgets. | Reproduce the waypoint with recorded actions and report controller arrival for pose-only and scene-changing targets before evaluating generated routes. |
| 3. Test the effect of depth | Compare explicit intermediate goals against closed-loop terminal scoring with matched feedback and inference budgets. | The comparison reports paired terminal success and ranking quality by task depth. |
| 4. Test the representation | Compare a reconstruction encoder and a predictive encoder trained with SIGReg under the same planner and goal interface. | Report collapse diagnostics, probes for decision-relevant state, and terminal success. |

## Existing experiment infrastructure

The masked trajectory completion and benchmark work below supplies training and evaluation components for the new comparison. [Trajectory completion design](docs/trajectory-completion.md)

| Milestone | Work | Completion evidence |
| --- | --- | --- |
| 1. Prepare trajectory windows | Split complete public trajectories by world. Sample start and endpoint contexts, variable-length middle masks and terminal goal conditions. | Training targets come from the masked segments of training trajectories. Held-out outcomes and privileged simulator state stay on the evaluation side. |
| 2. Learn latent trajectory completion | Train a public-history encoder and a world model to predict the hidden middle representations from both ends. Compare with a model given the starting context and time alone. | Held-out predictions respond to the terminal condition and preserve differences relevant to executing and continuing a route. |
| 3. Generate and execute alternatives | Sample several latent plans for the same conditions. Use a common executor to realize each plan through actions and observation feedback. | Repeated prospective attempts establish which proposed routes are reachable and which support terminal success. Measure demonstration support on separate training reference episodes. |
| 4. Improve proposals with execution feedback | Compare candidate attempts from the same starting history under a shared executor and continuation. Use terminal outcomes to rerank candidates and update the generator. | Hold the reranker fixed and measure whether the updated generator proposes more viable low-support targets on held-out worlds. |
| 5. Compare closed-loop planners | Run controlled comparisons with released baselines such as Diffuser. Evaluate our method on visual tasks from HDFlow and cite its published results, with HorizonFlow as a supplementary comparison where protocols align. Evaluate goal-only planning and planning given a rough initial plan. | Report paired terminal success and reliable low-support target discovery for controlled comparisons. Identify published scores and document protocol alignment. Use shared-executor and shared-bank comparisons to locate gains. |

Milestone 1 exports Craftax windows through `wr prepare-windows`. Its output supplies the public inputs and masked targets for milestone 2. [Window export and loader](docs/experiments.md#prepare-masked-trajectory-windows)

Milestone 2 has a masked latent predictor and a paired grid diagnostic. The diagnostic switches the goal at a common start, scores the predicted intermediate route, and compares a predictor trained with the goal against a predictor trained with history and time alone. The same trainer accepts the single-goal Craftax windows. [Training and goal switch protocol](docs/experiments.md#train-and-test-latent-trajectory-completion)

Milestone 3 adds a grid diagnostic that samples several latent routes for one goal, tries an intermediate target from each sample under paired disturbances, and reports arrival and terminal success against the same route executed open loop. Separate reference layouts measure demonstration support for each passage. [Route generation and execution protocol](docs/experiments.md#sample-and-execute-alternative-routes)

Milestone 4 has a feedback-training diagnostic with stochastic passage entry. It compares a frozen flow, an equally trained demonstration-only update and an update that adds matched execution feedback. All three use a fixed reranker. The next benchmark uses state-based AntMaze for controlled navigation and stitching comparisons. [Feedback protocol](docs/experiments.md#train-from-execution-feedback) · [Benchmark sequence](docs/experiment_env.md#robotics-benchmark-sequence)

Milestone 5 starts with state AntMaze-Large. `wr antmaze-gcbc` reads the official navigation and stitching archives, trains a goal-conditioned executor and evaluates it from a reusable set of simulator starts and public goals. [AntMaze baseline protocol](docs/experiments.md#train-the-antmaze-gcbc-baseline)

`wr antmaze-latent` learns masked trajectory completion from public AntMaze demonstrations. It samples latent routes, decodes an intermediate observation for the frozen executor, and replans from the observed rollout history. The development comparison pairs this loop with direct execution under the same reset, terminal goal and action limit. [Learned planner protocol](docs/experiments.md#run-learned-latent-planning-in-antmaze) · [Research design](docs/trajectory-completion.md)

`wr pointmaze-latent` runs the shared completion model on the maintained Maze2D U-Maze, Medium and Large tasks. The pilots pair direct execution and decoded latent targets using Diffuser's position and velocity controller. A controlled Diffuser comparison will use the same archive, environment and resets for each task. [Maze2D protocol](docs/experiments.md#run-the-diffuser-maze2d-pilots)

`wr kuka-latent` uses Diffuser's released manipulation clips and Kuka simulator to test complete four-block orders. The pilot compares tracking decoded forecasts and sampled trajectories from matched starts and scores the physical tower at the action limit. [Conditional stacking protocol](docs/experiments.md#run-conditional-kuka-stacking)

`wr scene-latent` pairs direct goal tracking and decoded intermediate targets with one learned executor across the five state-based OGBench Scene tasks. This establishes the state-observation comparison for a later visual experiment. [Scene protocol](docs/experiments.md#run-state-based-ogbench-scene)

`wr puzzle-latent` applies the shared OGBench manipulation runner to the four state-based Puzzle boards. The official goal configurations give a starting point for studying how performance varies with the number of presses required. [Puzzle protocol](docs/experiments.md#run-state-based-ogbench-puzzle)

## Evaluation protocol

Keep exploratory development worlds separate from test worlds. Freeze decision rules before the final comparison. Compare methods on matched starts and independent repeat streams. When isolating proposal quality, share the executor, terminal objective and candidate budget. Record all environment interactions used for training, proposal updates and evaluation. [Architecture](docs/architecture.md) · [Running experiments](docs/experiments.md)
