# Experimental environments

The paper uses two controlled discrete tasks to establish the claims and two continuous-control tasks to test them on standard benchmarks. Other environments are deferred and have no runner in the repository. [Research plan](latent-subgoal-research.md) · [Priorities](../PLAN.md#paper-experiments-by-priority)

| Environment | Role and rationale | Adaptation or limitation |
| --- | --- | --- |
| Dependency grid | Ordered keys and gates with two switchable destinations from one start. Tests goal-dependent value, frozen energy and outcome selection, and later task depth. | Layout-disjoint splits and paired action slips. [Protocol](experiments.md#compare-ordered-targets-in-the-dependency-grid) |
| Budget routes | One terminal with several resource and gate routes whose value depends on the budget. Tests composition, budget-dependent choice and low-support routes. | One graph topology so far. A coverage sweep and new topologies are planned. [Protocol](experiments.md#compose-latent-goals-under-action-budgets) |
| D4RL Maze2D (maintained PointMaze) | Controlled comparison with Diffuser on the same environments, data, resets and budgets. | Farama's maintained versions differ from the original D4RL `v1` environments, so both methods run on the same versions. [Protocol](experiments.md#run-the-diffuser-maze2d-pilots) |
| OGBench Scene Tasks 4 and 5 | Cube-in-drawer and rearrangement, where opening the drawer is a valuable intermediate goal. | The executor must pass the drawer-insertion diagnostic before the terminal comparison. [Scene protocol](experiments.md#run-state-based-ogbench-scene) |

## Deferred environments

These environments are candidates for follow-up work. Their earlier runners were removed. Retrieve them from Git history if a follow-up needs the collection or evaluation code.

| Environment | Role and rationale |
| --- | --- |
| OGBench Visual AntMaze | Compare with HDFlow's published Table 6 results. Report the visual inputs, goals, data and budget differences when citing published scores. [HDFlow](https://arxiv.org/html/2605.04525v1) |
| Craftax-Classic Symbolic | Craft an iron pickaxe by a fixed deadline and survive through it. A compound stochastic objective for goal transfer. [Craftax](https://proceedings.mlr.press/v235/matthews24a.html) |
| LIBERO-Long (LIBERO-10) | Visually grounded planning from a compound language goal, such as placing the yellow-and-white mug inside the microwave and closing its door. [LIBERO paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) |

For matched OGBench evaluations, save the simulator start and the exact public goal from one reset, then give each arm the same saved goal. The local OGBench 1.1.5 smoke found that state AntMaze reproduced its starting and goal observations exactly, while visual AntMaze showed single-channel pixel differences on some resets.
