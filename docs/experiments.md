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

## Shared-bank waypoint scoring pilot

The pilot compares empirical selectors on the same Craftax decision histories. Start 16 actions before the recorded iron-pickaxe success in each selected expert episode. The candidate bank consists of fixed-radius regions around the public observations two, five and eight actions ahead in that recording. Each candidate uses the recorded action prefix of the corresponding length as its executor. The executor stops on arrival or when it exhausts that prefix. The frozen PPO-RNN then uses the remaining steps of a shared 24-action terminal horizon. A stress condition replaces each scripted action with a no-op with a declared probability.

The direct selector ranks candidates by iron-pickaxe success across all calibration attempts. The factorized selector filters candidates whose calibration arrival rate is below 50%, then ranks the rest by success among arrivals. If none qualify, the frozen expert continues directly from the start. Both selectors use the same candidate bank and matched keys. An independent repeat pool estimates success after selection.

```bash
dataset=results/craftax-expert-20261121T082718Z-531436
run_root="$(mktemp -d results/craftax-waypoint-pilot.XXXXXX)"
for slip_probability in 0 0.25; do
  JAX_PLATFORMS=cpu uv run --no-sync wr waypoint-pilot \
    --dataset "$dataset" --checkpoint results/checkpoints/craftax-classic-ppo-rnn \
    --split dev --worlds 6 --repeats 4 --seed 0 \
    --slip-probability "$slip_probability" --out "$run_root/slip-$slip_probability"
done
```

Each condition uses four calibration and four evaluation repeats per world. The manifest records the selection recipe and region radius. `worlds.jsonl` records candidate scripts and selected options. `attempts.jsonl` records arrival, executor steps, remaining actions and terminal outcome. The summary reports selector success, paired differences, recovery after missed waypoints and a bootstrap interval by world. The estimates apply to replay-derived candidates on the selected near-completion histories. The selectors use per-start simulator estimates during calibration.

A follow-up uses the next six eligible development worlds with the same candidates, matching and stress conditions. A 16-action horizon matches the recorded distance from the decision to terminal success. Add `--skip-worlds 6 --horizon 16` to both invocations and use a new output directory. The world offset is applied before replay, preserving separate worlds across the two pilots.

### Shared expert continuation and earlier starts

Use `--offer-expert` to give both selectors a fourth option that runs the frozen PPO-RNN immediately. Both selectors calibrate this option by its terminal success rate, and a tie favors immediate continuation. The 50% arrival threshold and success among arrivals still govern how the factorized selector scores waypoint options. The summary records how often each selector chooses immediate continuation and measures their success on a separate pool of matched repeats.

Revisit the six 16-action development starts under 25% slip with the shared expert option. For the earlier-start diagnostic, use step 96 across both source cohorts and give the terminal objective 128 actions. Eligibility requires that the source recording contains all three candidate observations before termination. The source outcome defines the cohort and is logged for evaluation. Both selectors score options from fresh calibration attempts.

```bash
dataset=results/craftax-expert-20261121T082718Z-531436
run_root="$(mktemp -d results/craftax-waypoint-shared-expert.XXXXXX)"
common=(--dataset "$dataset" --checkpoint results/checkpoints/craftax-classic-ppo-rnn
        --split dev --seed 0 --slip-probability 0.25 --offer-expert)
JAX_PLATFORMS=cpu uv run --no-sync wr waypoint-pilot "${common[@]}" \
  --worlds 6 --skip-worlds 6 --horizon 16 --repeats 4 --out "$run_root/near"
for outcome in success death; do
  JAX_PLATFORMS=cpu uv run --no-sync wr waypoint-pilot "${common[@]}" \
    --prefix-steps 96 --source-outcome "$outcome" --horizon 128 \
    --worlds 4 --repeats 3 --out "$run_root/early-$outcome"
done
```

### Waypoint scoring mechanism study

The study measures whether including successes after a waypoint miss improves candidate selection and whether a hard arrival filter helps. It uses the archived observation regions, action scripts and shared expert continuation. The primary comparison is direct versus factorized scoring at 25% slip and a 16-action horizon on 24 additional near-success development worlds. The same worlds also run at 0% and 50% slip and a 32-action horizon. Earlier starts use eight additional successful source histories and all six death histories at step 96, with 64- and 128-action horizons at 0% and 25% slip. Two of the death histories are new to this pilot. A six-world radius control at 25% slip repeats execution with radii 0.001, 0.002 and 0.004. The radius governs the stopping decision, so each new radius gets fresh execution under the same random keys.

Each run uses seed 1729 and eight calibration and eight evaluation repeats per world and option. Shorter horizons censor terminal outcomes from the same rollouts once the complete waypoint script fits within both budgets. One explicit 16-action rerun verifies the logged records produced by censoring. The offline analysis checks the recorded selector choices and holds the evaluation pool fixed across all scoring rules.

| Selector | Waypoint score | Arrival requirement |
| --- | --- | --- |
| Direct | Terminal success across all attempts | Any |
| Factorized | Terminal success among arrivals | At least 50% |
| Joint arrival and success | Probability of arrival and terminal success in the same attempt | Any |
| Direct with filter | Terminal success across all attempts | At least 50% |
| Conditional | Terminal success among arrivals | At least one observed arrival |

All rules score the expert by its unconditional success and favor it on ties. Direct versus joint scoring isolates the training target's treatment of successful misses in this empirical pilot. Direct with and without the filter tests the gate on a common score. Two disjoint four-repeat calibration halves measure decision stability on the same evaluation pool. Report paired effects and bootstrap intervals by world. Candidate-level counts are descriptive because different options share starts and keys. Candidates depend on the archived future and the study uses development worlds and per-start simulator estimates.

The primary condition also runs with 24 calibration and 24 evaluation repeats. Keep the evaluation pool fixed and compare selectors using calibration blocks of 4, 8, 12 and 24 repeats. Each depth averages over every disjoint block of that size and estimates its paired gap by world. The analysis verifies that the first eight repeats of each pool agree exactly with the corresponding original records.

```bash
dataset=results/craftax-expert-20261121T082718Z-531436
run_root="$(mktemp -d results/craftax-waypoint-study.XXXXXX)"
bash scripts/run_craftax_waypoint_study.sh \
  "$dataset" results/checkpoints/craftax-classic-ppo-rnn "$run_root/raw"
uv run --no-sync python scripts/analyze_craftax_waypoint_study.py \
  "$run_root/raw" --out "$run_root/analysis.json"
```
