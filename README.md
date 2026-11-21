# Worth Reaching

Worth Reaching studies how an agent given a distant terminal objective can discover and sequence intermediate goals in a learned latent space. A goal is a region that can admit several different histories. A useful region offers the current controller feasible ways to enter it, and attempting it improves the prospect of completing the task.

Consider an agent trying to craft an iron pickaxe before nightfall. Several visible positions may offer iron. Approaching one may leave enough health and time to finish crafting, while approaching another may exhaust the agent. A latent waypoint could represent a family of positions from which the next step is feasible. We want to know whether a model can discover that distinction, act on it, and revise the plan when new observations arrive.

## The first paper

1. **Can the agent discover regions that are rare in demonstrations and reliable to execute?** Compare demonstration support with the outcomes of repeated, predeclared attempts.
2. **Can the agent plan from a distant final goal by choosing its own intermediate regions?** Compare learned regions, fixed latent neighborhoods, flat planning and designer milestones under matched information and budgets.

The project evaluates three ways to propose regions. A conditional generator learns from unsegmented demonstrations. Online RL improves a pretrained flow proposer using terminal outcomes. A hierarchical world model uses coarse predictions of candidate macro-actions to propose regions for fine-scale execution. Director demonstrated learned latent goals in Crafter. Our experiments test what makes a region useful and when it improves end-to-end planning. [Director](https://danijar.com/project/director/) · [Hierarchical Planning with Latent World Models](https://arxiv.org/abs/2604.03208)

The proposed first benchmark is Craftax-Classic with symbolic observations. MiniGrid provides a small diagnostic task. Balatro motivates a later test involving variable-size inventories, ordered interactions and irreversible purchases. The first experiment can proceed independently of the Balatro simulator.

## Project documents

| Document | Use |
| --- | --- |
| [Implementation plan](PLAN.md) | Development milestones, dependencies and evidence for completion |
| [Architecture](docs/architecture.md) | Proposed components, data contracts and information boundaries |
| [Environment choices](docs/experiment_env.md) | Candidate environments, adaptations and reasons to select each |
| [Contributing](CONTRIBUTING.md) | Development workflow and reproducibility rules |
| [Agent instructions](AGENTS.md) | Entry point for assistants working in this repository |

The repository is at the design stage. The plans describe proposed experiments. Results and runnable commands belong alongside the implementation that produces them.

## Development setup

The project will use uv and require Python 3.12 or newer. To prepare a development machine now, install uv and check it with `uv --version`. The MiniGrid implementation PR will add the project configuration, lockfile and commands for the CPU pilot. [uv installation](https://docs.astral.sh/uv/getting-started/installation/)
