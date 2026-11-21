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

## Paired action branches

`wr branch` replays a public history from a completed expert collection and restores the corresponding simulator state. The diagnostic selects the first successful episodes in the requested split and branches a fixed number of actions before their recorded success. A lead of eight places the forced action at the first of the final eight recorded steps. The two interventions are the recorded action and a no-op. If the recorded action is a no-op, the alternative is left.

The same PPO-RNN checkpoint continues both arms from memory reconstructed over the public prefix. Each repeat uses fresh dynamics and policy keys, matched by step across the two arms. The horizon includes the forced action and is capped by the original budget remaining. Reaching the diagnostic horizon while still active is labeled `horizon_limit`. Success rates apply to the sampled decision histories and declared horizon.

For a short CPU check, point `dataset` to a completed expert collection.

```bash
dataset=results/craftax-expert-20261121T082718Z-531436
run_dir="$(mktemp -d results/craftax-branches.XXXXXX)/labels"
JAX_PLATFORMS=cpu uv run --no-sync wr branch \
  --dataset "$dataset" --checkpoint results/checkpoints/craftax-classic-ppo-rnn \
  --split train --decisions 2 --repeats 2 --lead 8 --horizon 16 --out "$run_dir"
```

For the larger diagnostic, start a server tmux session and run the commands below inside it. The recipe schedules up to 3,200 attempts at 100 distinct training worlds.

```bash
tmux new-session -s wr-craftax-branches
```

```bash
set -o pipefail
dataset=results/craftax-expert-20261121T082718Z-531436
run_root="$(mktemp -d results/craftax-branches.XXXXXX)"
JAX_PLATFORMS=cpu uv run --no-sync wr branch \
  --dataset "$dataset" --checkpoint results/checkpoints/craftax-classic-ppo-rnn \
  --split train --decisions 100 --repeats 16 --lead 8 --horizon 16 \
  --out "$run_root/labels" 2>&1 | tee "$run_root/run.log"
```

`decisions.jsonl` records the source episode, prefix length and two declared actions. `attempts.jsonl` records each action and repeat, elapsed steps, remaining actions and outcome. Join attempts by decision ID and repeat to compare the paired outcomes. For a shorter deadline, count successful attempts whose elapsed steps fit the deadline. `manifest.json` identifies the source run, split manifest and continuation checkpoint. The final `summary.json` records per-arm counts at the configured horizon, paired success outcomes and separate counts for replay and branch actions.

## Action-ranking baselines

`wr rank` fits three scoring heads on a completed train branch run and evaluates them on a separate dev or test branch run from the same source collection. The command compares an action-only head, public-history return regression and public-history pairwise ranking with uniform choice. The two history heads use the same representation and candidates. Features are the current observation, the mean of up to four observations, the recent action histogram and the remaining episode budget. Each input ends at the declared decision. Success by the specified deadline is the training target for return regression. Discordant paired outcomes supply the pairwise loss.

Collect paired train and dev branches first. Use distinct worlds, a common lead and horizon, and at least two repeats per evaluation decision. Substitute the paths of completed runs below.

```bash
dataset=results/craftax-expert-20261121T082718Z-531436
train_branches=results/craftax-train-branches/labels
dev_branches=results/craftax-dev-branches/labels
out="$(mktemp -d results/craftax-ranking.XXXXXX)/ranking"
uv run --no-sync wr rank \
  --dataset "$dataset" --train-branches "$train_branches" \
  --eval-branches "$dev_branches" --deadline 8 --out "$out"
```

Even-numbered evaluation repeats select a reference action. Odd-numbered repeats estimate the success of the reference and each score-based choice. Tied scores use the mean success of the two choices. The report gives success estimates, signed reference-minus-model gaps, a world bootstrap interval and agreement where the reference ordering passes a paired sign test. The manifest records the label inputs and training settings, `weights.npz` stores the heads, and `scores.jsonl` stores evaluation scores by decision. Use larger development samples before interpreting differences between methods.
