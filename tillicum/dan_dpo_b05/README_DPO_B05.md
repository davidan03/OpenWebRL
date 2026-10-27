# DPO β = 0.5 on matched pairs (tillicum)

OpenWebRL-4B-SFT fine-tuned with **plain DPO (no SFT term), β = 0.5, lr 5e-5** on the same
4,078 matched pairs as your `rpo-matched-lr5e-5` (chosen and rejected make the same number
of tool calls). Everything else is your RPO recipe; only β (0.1 → 0.5) and the SFT weight
(1 → 0) differ.

| Online-Mind2Web, 300 tasks, gpt-5.6-terra judge, Zixian's protocol | Runs (successes/300) | Mean | vs base, 95% CI |
|---|---|---|---|
| Base OpenWebRL-4B-SFT (your 4 runs 10-20 to 10-23 + our run 10-26) | 90, 88, 88, 89, 80 | 87.0 (29.0%) | – |
| **DPO β 0.5, matched pairs** | 106, 87, 96 | **96.33 (32.1%)** | +3.1 pts [+0.1, +6.2], p = 0.045 |
| RPO β 0.1, matched pairs (3 our runs + 3 yours) | 107, 89, 96, 107, 95, 90 | 97.33 (32.4%) | +3.4 pts [+0.8, +6.1], p = 0.008 |
| RPO β 0.1, on-policy local-ARM pairs only ([data](https://huggingface.co/datasets/davidan03/openwebrl-arm-onpolicy-pairs)) | 106, 96, 105 | 102.33 (34.1%) | +5.1 pts [+1.8, +8.5], p = 0.002 |
| SFT on successful episodes, no ARM | 83, 86, 75 | 81.33 (27.1%) | −1.9 pts [−4.5, +0.7], p = 0.16 |

Invalid episodes count as failures. CIs and p-values come from a bootstrap over the 300 tasks
on per-task means. Paired: DPO β 0.5 − no-ARM SFT = +5.0 pts [+1.8, +8.2], p = 0.002;
DPO β 0.5 − on-policy RPO = −2.0 pts [−5.3, +1.2], p = 0.23 (tie).

**Dates matter here.** DPO run 1 (106) ran on 10-23; runs 2 and 3 (87, 96) on 10-26. A base
run on 10-26, same script and judge (job 390345), scored **80**, against 88.75 for the four
earlier base runs. Against that same-day base, DPO runs 2 and 3 are +3.8 pts [−0.7, +8.3],
p = 0.11. Re-judging run 1's episodes with terra on 10-26 gives 102 (95.8% agreement with the
live verdicts), so the judge is stable; the day-to-day drift is in the episodes. Every arm's
first run was its best (first runs average 106.5, later runs 94.3, over four arms), so treat
single runs as noisy.

## Settings

| | DPO β 0.5 (this) | your RPO matched |
|---|---|---|
| Pairs | 4,078 matched, `joint_pairs.train.jsonl` sha256 `113615b2…` | same file (same dropped pair, `Piotr:ep9__t6`) |
| Objective | DPO, full-response score | same |
| β | **0.5** | 0.1 |
| SFT weight (α) | **0** | 1 |
| LR | 5e-5, warmup 512 examples, ends at 0.5× | same |
| LoRA | r 16 / alpha 32 / dropout 0.05 | same |
| Batch / updates | 32 pairs, 1 epoch = 128 updates (final batch 14) | same |
| Reference | OpenWebRL-4B-SFT with adapters disabled | same |
| Seed / order | 42, order sha256 `08b40f5b…` | same |

## Reproducing with your scripts (simplest)

Your `train_joint.sbatch` already takes `BETA`. Leave `SFT_WEIGHT` unset (= 0):

```bash
OBJECTIVE=dpo LR=5e-5 BETA=0.5 OWRL_JOINT_DATA=$OUT RUN=dpo-matched-lr5e-5-b0.5 \
  sbatch --time=05:00:00 --exclude=g013,g021 tillicum/train_joint.sbatch
```

Then gate and eval as in [README_RPO_MATCHED.md](https://github.com/piotr-teterwak/OpenWebRL/blob/piotr_arm/tillicum/README_RPO_MATCHED.md),
with base in the same eval job. Our trainer matched yours update for update when we
reproduced your RPO run (first 20 updates: same `winner_ce` to 3 decimals, same LR
schedule; OM2W 116 vs your 120), so the two pipelines should agree.

## Our exact scripts

The files next to this README are byte-identical copies of what ran, so their paths still point
at our directories (`/gpfs/scrubbed/dan29/...`, `/gpfs/projects/krishna/dan29/...`). Change
`EXP`, `WT`, `ENV`, `X`, `HF_HOME`, the `peft_0181` and `ms-playwright` paths and the
`--output` log path to yours before submitting.

| File | What it is |
|---|---|
| `train_dpo_r1_ddp.py` | Zixian's `scripts/train_arm_joint_ddp.py` @ `fe65405` + 6 listed edits (docstring); edit 6 is your RPO term from `2c0c342`, a no-op at α 0. `diff <(git show fe65405:scripts/train_arm_joint_ddp.py) tillicum/dan_dpo_b05/train_dpo_r1_ddp.py` shows all of them. sha256 `2fb3de03…` = `trainer_copy_sha256` in the config |
| `filter_joint_eqcount.py` | equal-tool-call-count filter (5,540 → 4,078) |
| `make_dpo_config.py` | frozen config, derived from Zixian's `joint-v2-dpo-2gpu-r2` config |
| `configs/matched-r16a32-lr5e-5-dpo-b0.5.json` | the config that ran |
| `train_dpo_lora_arms.sbatch` | 2 × H200 training |
| `sbatch_merge_lora_arms.sbatch` | LoRA merge with Zixian's `merge_arm_c2_student.py`, 1 MIG instance |
| `sbatch_eval_her_protocol.sh`, `arm_eval_ports.patch` | OM2W eval: Zixian's `arm_eval --mode baseline` @ `299f03f` plus one patch (browser ports via `setdefault`) |

Two of Zixian's commits are used: `fe65405` (training and merge; worktree must be clean) and
`299f03f` (eval; worktree = `299f03f` + `arm_eval_ports.patch`). Both are on her `arm`
branch; this branch starts at `fe65405`. Env: our `zixian-panels` conda env
(`/gpfs/projects/krishna/dan29/conda/envs/zixian-panels`, torch 2.9.1, transformers 4.57.1)
with peft 0.18.1; her runtime venv has the same core versions.

### 1. Data

```bash
python3 tillicum/dan_dpo_b05/filter_joint_eqcount.py --data <Zixian's joint data dir> --out $DATA \
  --drop-for-batch Piotr:ep9__t6
```

Expected: `kept 4078`, `drop_call_count_mismatch 1461`, train sha256 `113615b2…` in
`filter-audit.json`. Your HF `openwebrl-arm-matched-pairs` is the same set.

### 2. Config

```bash
PYTHONPATH=<fe65405 worktree>/scripts python3 tillicum/dan_dpo_b05/make_dpo_config.py \
  --data $DATA --actor <OpenWebRL-4B-SFT> --output $RUN --config $CFG \
  --learning-rate 5e-5 --beta 0.5 --wandb-group rpo-ablation
```

With our paths this reproduces `configs/matched-r16a32-lr5e-5-dpo-b0.5.json` exactly.

### 3. Train (2 × H200, 2 h 34 min for us, job 384195)

```bash
CFG=$CFG sbatch --job-name=dpo-b0.5 tillicum/dan_dpo_b05/train_dpo_lora_arms.sbatch
```

It ends with `ARM OK: …/student/update-000128 updates 128, adapter r 16 alpha 32, lr 5e-05`.
Resubmit with `RESUME=1` if it times out.

### 4. Merge (1 MIG instance, 21 s, job 384200)

```bash
ARMS=matched-r16a32-lr5e-5-dpo-b0.5 sbatch tillicum/dan_dpo_b05/sbatch_merge_lora_arms.sbatch
```

The run root is hard-coded to `/gpfs/scrubbed/dan29/lora_dpo_arms/runs/<ARM>`. Our merged
adapter sha256: `0b8bd484…` (`evaluation/merged-endpoint/merge-manifest.json`).

### 5. OM2W eval (1 × H200, 2.0–2.3 h per run)

```bash
MODEL=$RUN/evaluation/merged-endpoint OUT=<fresh dir> JUDGE_MODEL=gpt-5.6-terra \
  sbatch --time=03:30:00 --exclude=g013 tillicum/dan_dpo_b05/sbatch_eval_her_protocol.sh
```

The script needs `OPENAI_API_KEY` in the environment and checks that the judge answers before starting.
Seed 42, T 0.7, top-p 0.9, 1,024 new tokens, 30 steps, 1 screenshot, full history. Our runs:
384399 (10-23), 389695 and 389696 (10-26). Always use a fresh `OUT`: the eval skips tasks
that already have a result.

## Things to watch

- **Judge outages make tasks silently invalid.** On 10-25 an OpenAI spend limit returned 429 for
  two hours; finished tasks came back with `reward: null` and counted as failures. Count
  `status COMPLETED` with `reward null` in `results/` before quoting a number.
- **Compare against base from the same day.** Run-to-run spread is large (this arm: 87–106),
  and the first run of an arm has tended to be its best.
- **Length caps.** β 0.5 had few 1,024-token cut-offs (6 in run 3), unlike DPO at β 0.1 on
  the same pairs, which ran away.
