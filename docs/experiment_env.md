# Experimental environments

Use MiniGrid KeyCorridor to check evaluation contracts, Craftax-Classic Symbolic to test the compound objective, and OGBench AntMaze for comparisons with published planning methods. [Research design](trajectory-completion.md)

| Environment | Role and rationale | Adaptation or limitation |
| --- | --- | --- |
| MiniGrid BabyAI KeyCorridor | A key opens a locked door guarding the target. The action branches make replay and termination easy to inspect. | The CPU diagnostic uses a fixed starting layout and keeps the partial observation. [Task](https://minigrid.farama.org/environments/babyai/KeyCorridor/) |
| Craftax-Classic Symbolic | Craft an iron pickaxe by a fixed deadline and survive through it. Repeat attempts from the same simulator state with fresh random keys. | Use public symbolic observations for collection and model decisions. Score at the deadline or death. [Craftax](https://proceedings.mlr.press/v235/matthews24a.html) · [Task rules](architecture.md#craftax-collection-contract) |
| OGBench AntMaze state observations | Compare shared trajectory-planning baselines under one evaluation protocol. The stitch data tests composition of short fragments. | Use the official dataset, goal set and action budget. Track extra environment interaction used for training separately. [OGBench](https://arxiv.org/html/2410.20092) |
| OGBench Visual AntMaze | Evaluate how the framework performs on tasks from HDFlow. | Report results with the documented visual inputs and goals, and name the data and budget differences when citing published scores. [HDFlow](https://arxiv.org/html/2605.04525v1) |
| OGBench Scene | Test object configurations that require sequential manipulation of locks, doors and a cube. | Use state observations for the initial comparison. [Scene protocol](experiments.md#run-state-based-ogbench-scene) |
| OGBench Puzzle | Test long chains of button presses toward a specified board pattern. | Use the benchmark's official goals. A later analysis can group goals by minimum presses within each board. [Puzzle protocol](experiments.md#run-state-based-ogbench-puzzle) |

For Craftax, repeated attempts from a snapshot estimate variation conditional on that hidden world. Use held-out worlds to measure generalization across layouts and initial conditions.

## Robotics benchmark sequence

Start with the state-based OGBench `antmaze-large-navigate-v0` dataset to establish a continuous-action executor and goal-conditioned behavior cloning baseline. Use `antmaze-large-stitch-v0` to test whether candidate routes connect short motion fragments. Both datasets use the `antmaze-large-v0` simulator. Track the official goal set and action budget, separate interaction used for feedback from evaluation, and use the same executor in the proposer comparison. [Baseline command](experiments.md#train-the-antmaze-gcbc-baseline) · [OGBench tasks and datasets](https://seohong.me/projects/ogbench/#locomotion-tasks)

For comparison with HDFlow, then evaluate visual AntMaze-Medium and Large with pixel observations and document the input, dataset and budget match to Table 6. OGBench Cube is an option for a manipulation follow-up after establishing the shared navigation pipeline. HDFlow also evaluates furniture assembly in simulation and on the robot, and robot manipulation in RLBench. [HDFlow experiments and dataset protocol](https://arxiv.org/html/2605.04525v2#S5.SS3)

The server has an OGBench test environment under `results/external/ogbench-cpu/`. A reset and step passed for all five goals of Large AntMaze and Cube-Single in both state and visual modes. AntMaze uses eight action coordinates and a 1,000-action limit. Cube-Single uses five action coordinates and a 200-action limit. The smoke used the simulator without loading an offline dataset. Store benchmark datasets under `results/datasets/ogbench/` by passing that directory to the official loader when starting the full comparison.

For matched evaluations, save the simulator start and the exact public goal from one reset, then give each arm the same saved goal. The local OGBench 1.1.5 smoke seeded the environment, NumPy and the action space. State AntMaze reproduced its starting and goal observations exactly. Visual AntMaze showed single-channel pixel differences on some resets, and Cube-Single goals varied across repeated resets.

## Deferred benchmark

Reserve **LIBERO-Long (LIBERO-10)** for a later test of visually grounded planning from a compound language goal. A concrete case is placing the yellow-and-white mug inside the microwave and closing its door. Use the official demonstrations and complete-task success, and compare policies on matched resets and action budgets once the need for a language-conditioned benchmark arises. [LIBERO paper](https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html) · [Official LIBERO-10 tasks](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/bddl_files/libero_10/tasks_info.txt)
