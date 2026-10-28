# WebVoyager / DeepShop evaluation: how we got our base numbers (tillicum)

**Short answer: we don't filter anything.** Our numbers are successes out of all scheduled tasks, i.e.
`success_rate_all_scheduled` from `arm_eval`'s `summary.json` (invalid episodes count as failures). Our eval is
your `arm_eval.py --mode baseline --protocol {webvoyager,deepshop}` at `0b2fc0b`, with your settings. Your 10-27 base
run is lower because sites blocked it, not because of a methodology difference.

| Base OpenWebRL-4B-SFT, WebVoyager (595 tasks, gpt-4o judge) | successes | Coursera | GitHub | Hugging Face |
|---|---|---|---|---|
| your 10-22 run (`passbench1/base`) | 261 | 26/40 | 26/41 | 23/37 |
| **your 10-27 run (`passbench2/base`)** | **205** | **3/40** | 16/41 | 8/37 |
| our 10-27 runs r1 / r2 / r3 | 261 / 261 / 263 | 31 / 32 / 34 of 40 | 25 / 27 / 27 of 41 | 18 / 11 / 14 of 37 |

Some sites block every run equally and don't explain any gap: Cambridge Dictionary (block page in 42–43 of 43
tasks), Google Search (39–40 of 43) and the blank Allrecipes first page (20–27 of 40) look the same in your 10-22
run, your 10-27 run and ours.

- **Same eval:** the `manifest.json` of your `passbench2/base/webvoyager` and our `r1/base/webvoyager` differ only
  in file paths and browser ports. They have the same 595 task ids, task-file sha256, sampling, steps and judge.
- **Your 10-27 base hit blocks:** Coursera's first page was a CloudFront block page ("ERROR: The request could not
  be satisfied") in 26 of 40 tasks, against 0 in our runs that day. That is about half of the 56-task gap. Most of
  the rest is GitHub and Hugging Face, where your run also hit Hugging Face `429` pages and Google's `/sorry/`
  CAPTCHA more often.
- **Likely cause (inferred, not proven):** how many browsers were open at once from the same cluster. During your
  base job (`eval-wv2`, 393104, 19:43–21:40) up to 25 of your GPUs were busy: about 20 WebVoyager/DeepShop model
  evals plus 4 collection shards. Every eval walks the same task file in the same order, so many browsers reach the
  same site at the same time. Our runs never had more than 4 eval jobs (8 models) at once.
- **Check a run in seconds:** `python bench_n3.py --bench webvoyager --sites <pass>/<label>/webvoyager` prints
  per-site successes and the first-page titles. A site whose first page is a block page is the tell.

DeepShop is different: your 10-27 base (38/150) matches our 10-27 r1 (38/150). On DeepShop everyone has been low
since 10-26 because Amazon serves its bot-check page (`/checkpoint/`) and forced sign-in (`/ap/`) far more often than
on 10-22. Our runs hit those in 37–49 of 150 base episodes, against 16–18 in yours from 10-22, and an episode that hits
one almost always fails.

## Our setup

`eval_concurrent_piotr.sbatch` (next to this README, byte-identical to what ran) is your
`tillicum/eval_concurrent.sbatch` at `0b2fc0b` with the eval and SGLang command lines verbatim:

```
arm_eval --mode baseline --protocol {webvoyager,deepshop} --task-file openwebrl/data/eval/{webvoyager_fara,deepshop}.jsonl
         --parallel 12 --seed 42 --temperature 0.7 --top-p 0.9 --max-new-tokens 1024 --max-steps 30 --task-timeout 1800
sglang   --dtype bfloat16 --tp 1 --mem-fraction-static 0.45 --context-length 32768 --max-running-requests 24
         --chunked-prefill-size 4096 --disable-cuda-graph
```

- **Judge:** gpt-4o, chosen by the protocol (`--judge-model` is not passed, as in your script).
- **Pairing:** one H200 per model, with base and the checkpoint in the same job, running at the same time on the
  same tasks (`MODELS=base,onpol`, 2 H200). Each pair is one job, and we ran 3 jobs per benchmark.
- **Infra-only differences from your script:**
  - our own conda env with node-local browser state;
  - SGLang ports `32000+(job%90)*8` and browser ports `30100–30299 / 30400–30599`, with a per-user port lock;
  - the judge key comes from `~/.bashrc`;
  - no merge step (our checkpoints are already merged).

## Metrics

- **full (result of record):** successes / all scheduled tasks, i.e. `summary.json` `success_rate_all_scheduled`.
- **clean:** tasks with `valid = True` in all 6 runs (3 base + 3 checkpoint). `valid` is `arm_eval`'s own flag: the
  episode finished and the judge's verdict could be read. It's a robustness check, not a filter of record.
- **no block page:** tasks where no run of either arm landed on a block page. Block pages are read from the
  observations (active-tab URL and title), never from the agent's text. They cover Cloudflare "Just a moment",
  CloudFront, 403/429/Access Denied/CAPTCHA titles, Google `/sorry/`, and Amazon `/checkpoint/` and `/ap/`. The README
  keyword filter reads the agent's final answer too. On WebVoyager it flagged base on 112–118 tasks against 76–84 for
  the checkpoint, because base writes "blocked" more often, so it is biased between arms.
- **Stats:** per-task mean over the 3 runs, checkpoint minus base. 95% CI from a task bootstrap, p from a sign-flip
  permutation test.

## Valid / invalid, and how our subsets differ from yours

**The `valid` flag is the same for both of us.** It is set per episode by `arm_eval.py` (identical in your `0b2fc0b`
and in Zixian's `299f03f`, which we use for OM2W), and we never change or re-derive it:

```python
record["valid"] = record.get("reward") is not None and "ABORTED" not in record.get("status", "")
if record["metadata"].get("judge_invalid") or record["metadata"].get("judge_timeout"):
    record["valid"] = False
# any exception while evaluating the task -> {"reward": None, "valid": False}
```

| an episode is invalid when | set in | typical `terminate_reason` / sign |
|---|---|---|
| it was aborted | `generate_browser.py` (status ABORTED) | `env_step_error` (browser step failed or hung), `generation_error: ...` (actor server error, e.g. HTTP 400, or the 1,800 s task timeout), `generation_abort` |
| the judge gave no usable verdict (`reward` is None) | `eval/reward_webvoyager.py`, `reward_deepshop.py` | log line `Judge response missing SUCCESS/NOT SUCCESS` (judge wrote free text) |
| the judge call failed or timed out | `reward_browser.py` | `metadata.judge_invalid` / `metadata.judge_timeout` |
| the task crashed before finishing | `arm_eval.py` except branch | result has `error_type`, no trajectory (e.g. the start page never loaded) |

Everything else is valid, including ordinary failures: a wrong final answer, hitting the 30-step limit
(`max_steps_exhausted`), or ending the episode on a block page. Blocked episodes are therefore usually *valid
failures*, as your README notes.

**Where we differ is in how the subsets are built and tested:**

| | your README_BENCHMARKS.md (10-22) | ours (this README) |
|---|---|---|
| runs per model | 1 | 3, each base run in the same job as one checkpoint run |
| "valid" subset | tasks valid in **both** models' single runs (n 565 / 143) | tasks valid in **all 6 runs**, decided jointly: one invalid episode in any run drops the task for both arms (n 539 / 123) |
| "not blocked" subset | keyword match on tool responses **and the final answer** (403, blocked, captcha, access denied, unusual traffic, robot), not blocked in both (n 417 / 128) | block page seen in the **observations only** (active-tab URL + title, list under Metrics), none in any of the 6 runs (n 402 / 15) |
| test | two-sided sign test on discordant tasks | per-task 3-run mean difference: task bootstrap CI + sign-flip permutation p |
| "valid only" rate | `success_rate_valid` per run | not used as a headline |

Two consequences:
- **Keyword blocking depends on wording.** Reading the final answer means the filter depends on how each model talks.
  In our WebVoyager runs it flagged base on 112–118 tasks against 76–84 for the checkpoint, while real block pages
  (observations) were about equal (122–124 vs 131–140 episodes with the full list). So keyword-"unblocked" subsets
  can move a paired difference either way.
- **Our subsets are smaller and stricter.** They are joint over 6 runs, so they are smaller than any one run's
  valid count, and the same tasks are compared in every run.

## Results: on-policy local-ARM RPO vs base, n=3 each, same-job pairs (10-27)

On-policy RPO = RPO β 0.1 on on-policy local-ARM pairs only ([data](https://huggingface.co/datasets/davidan03/openwebrl-arm-onpolicy-pairs)), see [README_DPO_B05.md](../dan_dpo_b05/README_DPO_B05.md).

| WebVoyager (595) | r1 | r2 | r3 | mean | Δ full | p full | Δ clean (n=539) | p clean |
|---|---|---|---|---|---|---|---|---|
| base | 261 | 261 | 263 | 261.7 (44.0%) | | | | |
| on-policy RPO | 290 | 281 | 273 | 281.3 (47.3%) | +3.31 pp [+0.7, +5.9] | 0.015 | +2.84 pp | 0.043 |

| DeepShop (150) | r1 | r2 | r3 | mean | Δ full | p full | Δ clean (n=123) | p clean |
|---|---|---|---|---|---|---|---|---|
| base | 38 | 42 | 48 | 42.7 (28.4%) | | | | |
| on-policy RPO | 49 | 54 | 57 | 53.3 (35.6%) | +7.11 pp [+1.6, +12.7] | 0.016 | +7.59 pp | 0.021 |

For reference, Online-Mind2Web (300 tasks, gpt-5.6-terra judge, Zixian's protocol, same-job pairs, 10-26) gives base
89/86/82 vs on-policy RPO 102/110/104, +6.56 pp [+3.3, +10.0], p 0.0002.

**On WebVoyager the gain sits on blocked tasks.** When an episode hits a block page, on-policy RPO still succeeds
26.6% of the time (109/410), against 13.0% (48/368) for base. Without a block the two are level (53.5% vs 52.0%). On
the 402 tasks no run was ever blocked on, the difference is −0.08 pp. On the 193 tasks where some run was blocked,
it is +10.4 pp. So on WebVoyager, at least, the checkpoint mainly recovers better after a CAPTCHA, rate limit or
forced sign-in. On DeepShop too few tasks are block-free in all 6 runs (15) to split this way.

## Files

| file | what |
|---|---|
| `eval_concurrent_piotr.sbatch` | the eval job as run (paths point at our dirs) |
| `bench_n3.py` | reproduces every number above from the results dirs, plus the `--sites` check |
| `webvoyager_per_task.csv`, `deepshop_per_task.csv` | our 3+3 runs per benchmark: `task_id, arm, run, success, valid, block_page` |

```bash
python bench_n3.py --bench webvoyager --base <r1>/base/webvoyager <r2>/base/webvoyager <r3>/base/webvoyager \
                   --arm <r1>/onpol/webvoyager <r2>/onpol/webvoyager <r3>/onpol/webvoyager
python bench_n3.py --bench webvoyager --sites /gpfs/scrubbed/piotrt/owrl/eval/passbench2/base/webvoyager
```
