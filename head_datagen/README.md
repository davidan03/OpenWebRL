# Head-executed on-policy data generation (lookahead-k3-1380 / piotr-data head)

This is the pipeline that built our RPO training corpus for the lookahead-head arm (`data_h1380_2pass`, 4,736 rows).
Trained with RPO on that corpus, the model scored **-9.00 pp vs on-policy RPO** (69.33 vs 96.33 n_success/300,
OM2W, gpt-5.6-terra judge, n=3 same-job pairs, CI [-12.0, -6.0]). A selection head picks one of 5 actor samples
at every step, and that pick is executed. Pairs come from executed turns of successful episodes: the chosen
candidate is the head's pick, the rejected one is another candidate. `HEAD=piotr-data` runs the same pipeline
with the piotr-data head.

## For an agent running this

Goal: regenerate this data on Tillicum and compare it with ours (`$DATA/reference/`, numbers below).
Everything you need is in this directory plus `$DATA` = `/gpfs/projects/krishna/dan29/share/head_datagen/`
(group `krishna`, readable on Tillicum). Do this, in order:

1. **Get the four local inputs** (login node: it has the fast network; compute nodes are on 1 GbE). Every
   download is checked against a sha256 by `sbatch_collect_head.sh` before anything runs.
   ```bash
   export ENVP=/gpfs/projects/krishna/dan29/conda/envs/zixian-panels   # read-only shared env; or yours with the same versions
   export HEADS=<scratch>/action-selection-heads BASE=<scratch>/OpenWebRL-4B-SFT HF_HOME=<scratch>/hf PW_SRC=<scratch>/ms-playwright
   $ENVP/bin/hf download zixianma/action-selection-heads --revision 97252f7101d8576cd07b7f4dc50fac21099e6421 --local-dir $HEADS
   $ENVP/bin/hf download OpenWebRL/OpenWebRL-4B-SFT --revision 15e777db2ddba2e0e82080ebccd3ad8d215b7f0a --local-dir $BASE
   PLAYWRIGHT_BROWSERS_PATH=$PW_SRC $ENVP/bin/playwright install chromium chromium-headless-shell   # playwright 1.58.0 -> chromium-1208
   sha256sum $HEADS/lookahead-k3-1380/head.pt $HEADS/piotr-data/head.pt   # 11d5eff8ffe1966c... / 755d4cd4913765e2...
   ```
   The heads repo's `main` has moved past 97252f71; use the pinned revision. If `hf download` returns quickly
   with missing files, you hit the HF rate limit (it exits 0); retry later and check every expected file exists.
2. **Build the code trees**: step 0 below. Expect `SETUP_OK`.
3. **Ask the user before every `sbatch`**: GPU jobs are billed. State the GPUs, `--time`, the maximum GPU-hours
   (GPUs x `--time`) and the expected GPU-hours. Pilot: 1 H200 x 1:00 (about 0.3 GPU-h). Each full pass:
   1 H200 x 10:00 (ours took 9:52). The live judge (`OPENAI_API_KEY`) costs about $20 per pass.
4. **Pilot, then check its log.** It passed if the log has `JUDGE_OK gpt-5.6-terra`, `SERVICES_READY ... (capture
   hooks verified)`, `COLLECT_EXIT 0 results=20`, `PORTS_CLEAN` and `PARITY_EXIT 0`. Compare `<OUT>/parity.log`
   with ours: 33/40 same picks; the head card says 34-37/40. `FATAL:` lines name the input that failed its check.
   A `Killed ... sglang.launch_server` line just before `PARITY_EXIT` is expected: the script stops the actor to
   free the GPU for the offline re-score.
5. **Two full passes** (seeds 42 and 43), then pairs, then corpus (steps 2-4 below).
6. **Report**: run the `diagnosis/` scripts on the new data and put each number next to ours (section "Our
   reference outputs"). Lead with valid / successful tasks per pass, then the head's repeat-pick rate vs chance,
   the stop-pick rate in failed episodes, the top-prob mean, the corpus row count and the repeat-teaching pair
   counts. Expect a statistical match, not identical data.

## Provenance

This branch is zixianma/OpenWebRL @1065034 plus this directory. Everything here is a copy of the code we ran.
The only edits are the hardcoded paths, which are now env vars, and one bug fix in `diagnosis/collect_stats.py`
(see the last section).

Tested on the login node with no GPU. `setup_worktree.sh` passes the same checks the sbatch makes. On our 20-task
pilot, the packaged `prepare_head_pairs.py` gives byte-identical pairs to the original (108 train + 5 validation
rows). The packaged `make_corpus.py` runs there without errors. The pilot has no successful judged episodes, so
it builds 0 rows.

## Files

| in git (`head_datagen/`) | |
|---|---|
| `setup_worktree.sh` | builds `$WT` = zixianma/OpenWebRL @1065034 + `arm_c2_head.patch` + `her_files/*`, and `$JOINT_REPO` = @fe65405 |
| `arm_c2_head.patch` | your C2 collector with ports + terra judge, plus your `actor_cache_head` ActionSelector branch, the rid passed to the actor, and a check that the actor exposes capture hooks |
| `her_files/` | your `actor_head_capture.py` / `actor_head_scoring.py` / `selection_head.py` / `selection_head_features.py`, copied from `arm-selection-head` on 10-29 (sha256 prefixes cc6627c95054 / 611a8d2ec9d2 / 6f90d3264fa8 / 1870159ef11f; that branch's current `selection_head.py` is 45ec794daf07) |
| `sbatch_collect_head.sh` | one collection pass on 1 H200 (actor with capture hooks -> head scores 5 candidates -> argmax executed) |
| `prepare_head_pairs.py` | pairs per pass (your `prepare_arm_joint_data.py` @fe65405; label = head argmax, checked against the executed index) |
| `make_corpus.py` | two passes -> RPO corpus |
| `parity_check.py` | `PARITY=1`: re-scores 40 saved turns offline with your scorer |
| `diagnosis/` | stats scripts whose outputs on our data are listed below |

Data files are **not in git** (our no-data rule). They are on Tillicum in `$DATA` =
`/gpfs/projects/krishna/dan29/share/head_datagen/` (group `krishna`):

- `code/tasks_c2pass2_1978.jsonl`: your 2,091 C2 tasks minus the 113 C2 validation task groups of your joint split.
- `code/onpolicy_panels/`: the two validation panels `make_corpus.py` copies.
- `reference/`: our run's outputs (see below).

## Pinned inputs

| input | value |
|---|---|
| actor | OpenWebRL/OpenWebRL-4B-SFT @15e777db (model.safetensors sha256 2b92e7fb...) |
| heads | zixianma/action-selection-heads @97252f71: `lookahead-k3-1380/head.pt` sha256 11d5eff8... (= your sft-actions-20261021/round1/teacher_k3/head.pt), `piotr-data/head.pt` 755d4cd4... |
| python env | `/gpfs/projects/krishna/dan29/conda/envs/zixian-panels` (group krishna can read it; sglang 0.5.6.post2, torch 2.9.1, transformers 4.57.1) |

## Settings (yours = what your crossbench head5 live run uses; ours = where we differ or chose)

- Actor serving: your head5 flags (mem-fraction .45, 40 running, chunked prefill 4096, cuda graphs 48,
  `--random-seed 4200`, forward hooks capture layers 18/27/36).
- Candidates: 5 per step, kept with duplicates, T 0.7 / top_p 0.9 / 1,024 tokens. Hardcoded in `arm_c2.py:303`
  of your C2 collector. Your head card evaluates at T 1.0; your crossbench default is 0.7.
- Ties go to the lowest index. 30 steps max, 16 tasks in parallel, viewport 1280x1000 (OpenWebRL release config).
  The lookahead-k3-1380 head's training states are 1280x720.
- Judge: gpt-5.6-terra (ours; o4-mini was retired 10-23). We ran with `JUDGE_OFFLINE=1` (our OpenAI key had
  been invalidated) and re-judged unscored COMPLETED episodes afterwards with the same model. The simplest route
  is the live judge (`OPENAI_API_KEY` set, about $20/pass). Same judge model, different call path.
- Two passes: seed 42 and seed 43. Pass-2 states that also appear in pass 1 are dropped in the corpus.
- Corpus filters (from `merge_c2pass2.py`): status retained, episode success, equal tool-call counts,
  prompt + longer response <= 27,869 tokens.

## Steps (Tillicum)

```bash
git clone -b dan_head_datagen https://github.com/davidan03/OpenWebRL.git owrl_head_datagen
export PKG=$PWD/owrl_head_datagen/head_datagen DATA=/gpfs/projects/krishna/dan29/share/head_datagen
export WT=<new dir>/owrl_head JOINT_REPO=<new dir>/owrl_joint
export HEADS=<local copy of zixianma/action-selection-heads @97252f71>   # contains lookahead-k3-1380/ and piotr-data/
export BASE=<local OpenWebRL-4B-SFT @15e777db> HF_HOME=<your HF cache> PW_SRC=<your ms-playwright browsers dir>
export ENVP=/gpfs/projects/krishna/dan29/conda/envs/zixian-panels   # or your own env with the same versions
```

0. Code trees (login node): `bash $PKG/setup_worktree.sh` -> prints `SETUP_OK`.
1. Pilot (recommended first: 20 tasks, about 0.3 GPU-h):
   ```bash
   head -20 $DATA/code/tasks_c2pass2_1978.jsonl > pilot20.jsonl
   OUT=<fresh dir> TASKS=$PWD/pilot20.jsonl SEED=42 HEAD=lookahead-k3-1380 PARITY=1 \
     sbatch --qos=debug --time=1:00:00 -o <log>.out $PKG/sbatch_collect_head.sh
   ```
   Our pilot's parity (`$DATA/reference/parity.pilot20_h1380.log`): 33/40 same pick live vs offline. 5 of the 7
   flips are ties between identical candidates. Per-state logit Pearson median 0.996.
2. Full passes, 1 H200 each. Ours took 9:52 per pass (`--time 10:00`). The script stops 8 min before the limit,
   and resubmitting with the same `OUT` resumes:
   ```bash
   OUT=<fresh dir s42> TASKS=$DATA/code/tasks_c2pass2_1978.jsonl SEED=42 HEAD=lookahead-k3-1380 sbatch --time=10:00:00 -o <log> $PKG/sbatch_collect_head.sh
   OUT=<fresh dir s43> ... SEED=43 ...
   ```
   Always use a fresh `OUT`: reusing a finished one skips every task.
3. Pairs per pass (login node, single core; full collections need `--cpu-seconds 1500`):
   ```bash
   PYTHONPATH=$JOINT_REPO/scripts $ENVP/bin/python $PKG/prepare_head_pairs.py --collection <OUT s42> --output <new pairs s42> --cpu-seconds 1500
   ```
   A killed run leaves a partial `--output` dir; delete it before rerunning.
4. Corpus:
   `python3 -I $PKG/make_corpus.py --pairs <pairs s42> <pairs s43> --onpolicy $DATA/code/onpolicy_panels --out <new dir> --head-name lookahead-k3-1380`.
   The validation panels are only used to exclude validation task groups and are copied into the corpus. The
   `images` link it makes points at nothing here; only the validation panels use those images.

## Our reference outputs (`$DATA/reference/`)

- `h1380_full1978_s{42,43}/`: `results/*.json` (one per task, re-judged), `frozen-config.json`,
  `collection-summary.json`. Valid / successful tasks (from `results/`): s42 1,887 / 947, s43 1,887 / 889, of 1,978.
- `h1380_pairs_s{42,43}/audit.json`: pairs s42 10,534 train / 1,251 val, s43 10,317 / 1,438.
- `data_h1380_2pass/corpus-audit.json`: 4,736 rows (sha256 c8fd246d...), stop chosen 616 / continue chosen 716.
- `stats/`: outputs of the `diagnosis/` scripts on our data. Lines to compare:
  - `live_repeat_picks.s42.txt`: on turns whose 5 candidates mix repeats and non-repeats of the previous action,
    the head picks the repeat 49.4 % vs chance 47.0 % (s43 49.5 vs 47.9). After a scroll, 63.5 % vs 51.2 %.
  - `collect_stats.s42.txt`: head top-prob mean 0.49, below 0.5 on 58 % of turns; picks stop in mixed sets
    of failed episodes 7.4 % vs chance 38.6 %.
  - `corpus_repeat.txt`: 587 pairs teach "repeat the previous action" vs 356 teach "don't".

Run the same scripts on your output: `python3 -I $PKG/diagnosis/live_repeat_picks.py <OUT>/selections`,
`python3 -I $PKG/diagnosis/collect_stats.py <OUT>`, `python3 -I $PKG/diagnosis/corpus_stats.py <corpus>/joint_pairs.train.jsonl`,
`python3 -I $PKG/diagnosis/corpus_repeat.py <corpus>/joint_pairs.train.jsonl`,
`python3 -I $PKG/diagnosis/integrity.py <OUT s42> <OUT s43> <corpus>`. For the last one, note that its episode-success
check reads `attempts/*/outcome.json`, which is wrong for judge-offline runs; with the live judge it is fine.

Expect a statistical match, not a byte-identical one. Candidate sampling is seeded, but SGLang batching under 16
parallel tasks and live websites both change trajectories run to run.

## Changes vs the code we ran

- `sbatch_collect_head.sh`: paths come from `PKG` / `WT` / `HEADS` / `BASE` / `HF_HOME` / `PW_SRC` / `ENVP`;
  `--output` is relative (pass `-o`); the action-reward-models commit check is gone (that repo is unused).
- `prepare_head_pairs.py`: `JOINT_REPO` / `WT` / `BASE` come from the environment. We ran with the collector
  repo at 1065034 + our c2pass2 patch. That patch is contained in `arm_c2_head.patch`, and the parser file the
  builder hashes (`openwebrl/base/utils.py`) is unchanged by both.
- `diagnosis/collect_stats.py` (also fixed in our copy): it now takes episode outcomes from re-judged
  `results/` when present. Before, it read the unpatched `outcome.json` and counted 5 successes in s42
  instead of ~940.
