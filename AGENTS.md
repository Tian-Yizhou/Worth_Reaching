# Working in Worth Reaching

Read [README.md](README.md), [PLAN.md](PLAN.md) and [CONTRIBUTING.md](CONTRIBUTING.md) at the start of work. Consult the [architecture](docs/architecture.md) before changing interfaces and the team's local research protocol at `docs/research_idea/research-protocol.md` before changing a baseline, training signal or metric.

Implement the smallest milestone that exercises the normal public entry point. Keep privileged simulator data on the evaluation side of the interface. Report which commands were run, what they established, and any checks that still require data or hardware. Use the branch and review workflow for significant changes and leave merges to the maintainer.

Run short experiments for smoke checks and preliminary results. Provide commands or scripts for long experiments so the maintainer can launch them in tmux. Keep trajectories, logs, checkpoints and summaries in the server checkout's Git-ignored `results/` directory.
