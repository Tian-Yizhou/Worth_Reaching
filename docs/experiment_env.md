# Experimental environments

Use MiniGrid KeyCorridor to check evaluation contracts, Craftax-Classic Symbolic to test the compound objective, and OGBench AntMaze for comparisons with published planning methods. [Research design](trajectory-completion.md)

| Environment | Role and rationale | Adaptation or limitation |
| --- | --- | --- |
| MiniGrid BabyAI KeyCorridor | A key opens a locked door guarding the target. The action branches make replay and termination easy to inspect. | The CPU diagnostic uses a fixed starting layout and keeps the partial observation. [Task](https://minigrid.farama.org/environments/babyai/KeyCorridor/) |
| Craftax-Classic Symbolic | Craft an iron pickaxe by a fixed deadline and survive through it. Repeat attempts from the same simulator state with fresh random keys. | Use public symbolic observations for collection and model decisions. Score at the deadline or death. [Craftax](https://proceedings.mlr.press/v235/matthews24a.html) · [Task rules](architecture.md#craftax-collection-contract) |
| OGBench AntMaze state observations | Compare shared trajectory-planning baselines under one evaluation protocol. The stitch data tests composition of short fragments. | Use the official dataset, goal set and action budget. Track extra environment interaction used for training separately. [OGBench](https://arxiv.org/html/2410.20092) |
| OGBench Visual AntMaze | Evaluate how the framework performs on tasks from HDFlow. | Report results with the documented visual inputs and goals, and name the data and budget differences when citing published scores. [HDFlow](https://arxiv.org/html/2605.04525v1) |

For Craftax, repeated attempts from a snapshot estimate variation conditional on that hidden world. Use held-out worlds to measure generalization across layouts and initial conditions.
