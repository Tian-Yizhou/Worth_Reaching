# Running experiments

Run these commands from the repository root. The server checkout keeps trajectories, logs, summaries and checkpoints in the Git-ignored `results/` directory. For the smallest CPU examples, start with the quick start. [README](../README.md#quick-start)

## MiniGrid replay pilot

The KeyCorridor configuration fixes the world seed, a 12-action prefix and a two-action remaining budget. Both branches replay the prefix before executing the specified first action and then use the same continuation policy. The evaluator records the initial observation and public prefix in `prefix.json`, the branch transitions in `attempts.jsonl`, and the outcomes in `summary.json`. The tests verify exact replay and check that changing an unseen tile leaves the policy input identical. [Pilot configuration](../configs/minigrid/keycorridor.toml)

## Craftax-Classic task

The symbolic adapter uses Craftax-Classic's 17 primitive actions. It scores the first transition that earns the iron-pickaxe achievement as success, including on the last allowed action. If an episode ends without success, death from depleted health or lava takes precedence over timeout. The evaluator reads the terminal transition before any reset. [Craftax implementation](https://github.com/MichaelTMatthews/Craftax) · [Collection contract](architecture.md#craftax-collection-contract)

The split manifest fixes world assignments before collection. A stable hash gives approximate train, dev and test shares of 80%, 10% and 10%. Overlapping seed ranges refer to the same worlds and preserve their splits across policies. Use `--seed-start` or edit the TOML recipe to select additional worlds. Fit models on the training split, select settings on the development split, and reserve the test split for evaluation. [Collection configuration](../configs/craftax/iron_pickaxe.toml)

## Matched expert and random collection

The paired runs below use the same 1,000 world seeds and a maximum of 1,000 actions per episode. The expert is a published PPO-RNN trained with the upstream Craftax achievement reward. Its collection run uses the same iron-pickaxe terminal rule as the random reference. The expert manifest records the checkpoint step, source revision and actor digest. [Published checkpoint](https://huggingface.co/mathisweil/remdm-craftax-checkpoints/tree/5b66c77b497777e4ffdcbf20c3563402e404daf8) · [PPO-RNN implementation](https://github.com/MichaelTMatthews/Craftax_Baselines/blob/main/ppo_rnn.py)

```bash
uv sync --locked --group dev --extra minigrid --extra craftax
python3 scripts/fetch_craftax_expert.py

tmux new-session -d -s wr-craftax-expert -c "$PWD" \
  'bash scripts/run_craftax_collection.sh expert 1000 1000 20000'
tmux new-session -d -s wr-craftax-random -c "$PWD" \
  'bash scripts/run_craftax_collection.sh random 1000 1000 20000'
```

Attach with `tmux attach -t wr-craftax-expert` or `tmux attach -t wr-craftax-random`. Detach with `Ctrl-b d`. The download script saves weights under `results/checkpoints/`. The launcher prints the run directory at startup and the summary path on completion. [Checkpoint download script](../scripts/fetch_craftax_expert.py) · [Collection launcher](../scripts/run_craftax_collection.sh)

## Craftax output files

| File | Contents |
| --- | --- |
| `manifest.json` | Resolved configuration, code revision, environment and policy provenance |
| `splits.json` | Episode IDs, original world seeds and split assignments |
| `trajectories.jsonl.gz` | One compressed JSON record per episode with public observations, actions and outcome |
| `summary.json` | Episode counts, action counts and terminal outcomes |
| `<run-name>.log` | Console output, stored next to the run directory when using the collection launcher |

The collector writes `summary.json` after the trajectory file closes. Use the summary to identify completed runs and join the two policies' episode records by world ID through their split manifests.
