#!/bin/bash
#SBATCH --job-name=head-collect
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --output=%x_%j.out
#
# HEAD-EXECUTED ON-POLICY COLLECTION (user 2026-10-29: the winning on-policy recipe with Zixian's selection head picking
# and executing instead of the 4B decoder SelectionARM; implemented the way she ran the head live). Copy of
# iterdpo_r1/sbatch_collect_c2pass2.sh with only these changes:
#  - worktree zixian_owrl_head = 1065034 + head_onpolicy/arm_c2_head.patch (= arm_c2_c2pass2.patch + her ActionSelector
#    'actor_cache_head' branch, copied from evaluations/selectors-samehw-20261012/source/openwebrl/arm_inference.py;
#    rid passed to the actor request as her crossbench MeteredActor does; collector checks the actor's capture hooks
#    instead of a selector /health) + her actor_head_capture.py / actor_head_scoring.py / selection_head.py /
#    selection_head_features.py copied unchanged from her arm-selection-head branch (sha256 pinned below);
#  - no ARM server; the actor is launched with her crossbench head5 flags (mem-fraction .45, 40 running, chunked 4096,
#    cuda graphs 48, --random-seed 4200, --forward-hooks capture layers 18/27/36, ACTORHEAD_* env);
#  - head = zixianma/action-selection-heads @97252f71 piotr-data/head.pt (sha256 755d4cd4..., her live-eval checkpoint);
#    HEAD=lookahead-k3-1380 (user 2026-10-30) swaps in her Luna-lookahead head (@97252f71, sha256 11d5eff8..., == her live-eval
#    file sft-actions-20261021/round1/teacher_k3/head.pt; her live runs of the two heads differ only in head path/sha).
#  - SEED 42 allowed (pass 1 = 42, pass 2 = 43, as C2's two passes); optional PARITY=1 re-scores 40 saved turns with
#    her offline SelectionHeadScorer after the collection (live-vs-offline pick agreement; her card: 34-37/40).
# Original description follows.
#
# C2 SECOND PASS (user 2026-10-23: scale up the RPO pairs, ARM-only, Tillicum). Copy of sbatch_collect.sh (round 1)
# with only these changes, so the data matches Zixian's C2 run (c2-full-282782): actor = OpenWebRL-4B-SFT (C2's actor,
# HF 15e777db) instead of pi1; selector = our SelectionARM copy via her scripts/serve_arm.py DIRECTLY (as in C2; no
# mixed_selector / GPT-5.5); candidate seed = $SEED (must differ from C2's 42: candidate sampling seeds are
# hash(seed, task, turn, candidate), so seed 42 would redraw C2's turn-0 candidates); tasks = her 2,091 minus the 113 C2
# validation task groups of her joint split (tasks_c2pass2_1978.jsonl). Collector = her openwebrl/arm_c2.py collect
# @1065034 + arm_c2_c2pass2.patch (= arm_c2_ports.patch + success judge o4-mini -> gpt-5.6-terra: OpenAI retired o4-mini
# 2026-10-23; C2 pass 1 was judged by o4-mini, terra is stricter, see experiments/rejudge_terra); serving = her C2
# flags. A resubmit into the same OUT resumes.
# Submit: env OUT=<fresh dir> TASKS=<jsonl> SEED=42|43 [HEAD=piotr-data|lookahead-k3-1380] [JUDGE_OFFLINE=1] [PARALLEL=16] [PARITY=1] sbatch --qos=.. --time=.. sbatch_collect_head.sh
set -euo pipefail
: "${OUT:?}" "${TASKS:?}" "${SEED:?}"
PARALLEL=${PARALLEL:-16}
# PACKAGE (2026-11-01): the only edits vs the script we ran are these paths, now env vars (see README.md); the
# action-reward-models commit check was dropped (that repo is not used by this pipeline).
X=${PKG:?set PKG=<package>/code}
WT=${WT:?set WT=<worktree from setup_worktree.sh>}
HEADS=${HEADS:?set HEADS=<local zixianma/action-selection-heads @97252f71>}
HEAD=${HEAD:-piotr-data}
case $HEAD in piotr-data) HEAD_SHA=755d4cd49137;; lookahead-k3-1380) HEAD_SHA=11d5eff8ffe1;; *) echo "FATAL: unknown HEAD=$HEAD"; exit 1;; esac
HEAD_PT=$HEADS/$HEAD/head.pt
BASE=${BASE:?set BASE=<local OpenWebRL/OpenWebRL-4B-SFT @15e777db>}
ACTOR=$BASE
ENVP=${ENVP:-/gpfs/projects/krishna/dan29/conda/envs/zixian-panels}
PY=$ENVP/bin/python
[ "$(git -C $WT rev-parse HEAD)" = 1065034e07d3575bf29fcca68bb35c31ed14ab78 ] || { echo "FATAL: worktree not at 1065034"; exit 1; }
cmp -s <(git -C $WT diff) $X/arm_c2_head.patch || { echo "FATAL: worktree differs from 1065034 + arm_c2_head.patch"; exit 1; }
for f in actor_head_capture:cc6627c95054 actor_head_scoring:611a8d2ec9d2 selection_head_features:1870159ef11f selection_head:6f90d3264fa8; do
  [ "$(sha256sum $WT/openwebrl/${f%%:*}.py | cut -c1-12)" = "${f##*:}" ] || { echo "FATAL: ${f%%:*}.py is not her copy"; exit 1; }
done
[ "$(sha256sum $HEAD_PT | cut -c1-12)" = $HEAD_SHA ] || { echo "FATAL: head.pt is not $HEAD $HEAD_SHA"; exit 1; }
JUDGE=gpt-5.6-terra; grep -q "judge_api_model='$JUDGE'" $WT/openwebrl/arm_c2.py || { echo "FATAL: collector judge is not $JUDGE"; exit 1; }
if [ "${JUDGE_OFFLINE:-0}" = 1 ]; then
  # user 2026-10-30 (OpenAI key invalidated): collect now, judge later. Every COMPLETED episode's live judge call fails
  # (4 tries, ~40 s) and is saved reward None / valid False; afterwards rejudge_terra/rejudge_om2w.py --collection
  # --only-unscored + iterdpo_r1/patch_rejudged_results.py fill the verdicts (as after the 10-25 outage).
  echo "JUDGE_OFFLINE: live judge skipped; re-judge offline before building pairs"
else
: "${OPENAI_API_KEY:?}"
curl -sf --max-time 60 https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" -d "{\"model\":\"$JUDGE\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}" \
  >/dev/null || { echo "FATAL: judge model $JUDGE does not answer"; exit 1; }
echo "JUDGE_OK $JUDGE"
fi
mkdir -p "$OUT"
[ "${JUDGE_OFFLINE:-0}" != 1 ] || echo "live judge skipped (JUDGE_OFFLINE=1, job $SLURM_JOB_ID, $(date -Is)): re-judge unscored COMPLETED episodes offline before building pairs" >> "$OUT/JUDGE_OFFLINE.txt"
[ -n "$(ls -A "$OUT/results" 2>/dev/null)" ] && echo "NOTE: resuming into $OUT ($(ls "$OUT/results" | wc -l) results present)"

unset -f module ml 2>/dev/null || true
unset BASH_ENV
export HF_HOME=${HF_HOME:?} HF_HUB_OFFLINE=1
export CUDA_HOME=/gpfs/software/cuda/12.8.2
export PATH="$ENVP/bin:$CUDA_HOME/bin:$PATH"
PW_LOCAL=/tmp/pw_${SLURM_JOB_ID}; mkdir -p $PW_LOCAL && cp -r ${PW_SRC:?set PW_SRC=<ms-playwright browsers dir>}/. $PW_LOCAL/
export PLAYWRIGHT_BROWSERS_PATH=$PW_LOCAL TMPDIR=/tmp/job_${SLURM_JOB_ID} HOME=/tmp/home_${SLURM_JOB_ID}
export XDG_CONFIG_HOME=$HOME/.config XDG_CACHE_HOME=$HOME/.cache XDG_DATA_HOME=$HOME/.local/share
mkdir -p $TMPDIR $XDG_CONFIG_HOME $XDG_CACHE_HOME $XDG_DATA_HOME
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START=30000 SLIME_BROWSER_LOCAL_PROCESS_PORT_END=31999
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=/tmp/slime_ports_$USER
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=180
export NO_PROXY="127.0.0.1,0.0.0.0,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,0.0.0.0,localhost,${no_proxy:-}"
export SLIME_HF_VISION_ATTN_IMPL=sdpa SLIME_ASYNC_USE_STDLIB_LOOP=1 OMP_NUM_THREADS=4  # her scripts/h200_env.sh
export FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer TRITON_CACHE_DIR=$TMPDIR/triton TORCHINDUCTOR_CACHE_DIR=$TMPDIR/inductor

P0=$((11000 + (SLURM_JOB_ID % 500) * 4)); ACTOR_PORT=$P0; ARM_PORT=$((P0 + 1)); NCCL_PORT=$((P0 + 3))
PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null || true; done; rm -rf "$PW_LOCAL" "$TMPDIR" "$HOME" "${ACTORHEAD_DIR:-/nonexistent}"; }
trap cleanup EXIT
cd $WT
# Her actor_cache_head serving (run_crossbench_selector_eval.py server_stages): capture hooks inside the actor server.
export ACTORHEAD_DIR=/dev/shm/actorhead-${SLURM_JOB_ID} ACTORHEAD_HEAD=$HEAD_PT ACTORHEAD_TOKENIZER=$ACTOR \
  ACTORHEAD_PARAMS=$((4022468096 + 415347712))
mkdir -p $ACTORHEAD_DIR
HOOKS=$(PYTHONPATH=$WT $PY -c "import json; from openwebrl.actor_head_capture import hook_specs; print(json.dumps(hook_specs((18, 27, 36))))")
PYTHONPATH=$WT setsid $PY -m sglang.launch_server --model-path "$ACTOR" --host 127.0.0.1 --port $ACTOR_PORT --nccl-port $NCCL_PORT \
  --dtype bfloat16 --tp 1 --mem-fraction-static .45 --context-length 32768 --max-running-requests 40 \
  --chunked-prefill-size 4096 --cuda-graph-max-bs 48 --random-seed 4200 --forward-hooks "$HOOKS" \
  > "$OUT/actor.$SLURM_JOB_ID.log" 2>&1 &
PIDS+=($!)
wait_url() {  # url name pid
  for t in $(seq 1 120); do
    curl -sf --noproxy '*' "$1" >/dev/null && return 0
    kill -0 $3 2>/dev/null || { echo "SERVICE_DIED $2"; tail -30 "$OUT/$2.$SLURM_JOB_ID.log"; exit 1; }
    sleep 10
  done
  echo "SERVICE_TIMEOUT $2"; exit 1
}
wait_url http://127.0.0.1:$ACTOR_PORT/health_generate actor ${PIDS[0]}
INFO=$(curl -sf --noproxy '*' http://127.0.0.1:$ACTOR_PORT/get_model_info)
$PY -c "import json, os, sys; p = json.loads(sys.argv[1])['model_path']; assert os.path.realpath(p) == os.path.realpath(sys.argv[2]), p" "$INFO" "$ACTOR"
SI=$(curl -sf --noproxy '*' http://127.0.0.1:$ACTOR_PORT/server_info)
$PY -c "import json, sys; assert json.loads(sys.argv[1]).get('forward_hooks') == json.loads(sys.argv[2]), 'actor lacks capture hooks'" "$SI" "$HOOKS"
HEALTH=$(printf '{"mode": "actor_cache_head", "head": "%s", "head_sha256": "%s", "hooks": %s}' "$HEAD_PT" "$(sha256sum $HEAD_PT | cut -d' ' -f1)" "$HOOKS")
LEFT=$(squeue -h -j $SLURM_JOB_ID -o %L)
echo "SERVICES_READY $(date) actor=$ACTOR_PORT (capture hooks verified) left=$LEFT head=$HEAD_PT"

# Her collector's config (+ frozen-config.json, which her build_dataset reads). Stop 8 min before the time limit.
$PY - "$OUT" "$TASKS" "$PARALLEL" "$ACTOR" "$HEALTH" "$LEFT" $ACTOR_PORT $ARM_PORT "$SEED" "$JUDGE" <<'PY'
import datetime as dt, hashlib, json, sys
from pathlib import Path
out, tasks, par, actor, health, left, ap, mp, seed, judge = sys.argv[1:]
parts = left.replace('-', ':').split(':'); parts = [int(x) for x in parts]
secs = sum(v * m for v, m in zip(reversed(parts), [1, 60, 3600, 86400]))
stop = dt.datetime.now(dt.timezone.utc) + dt.timedelta(seconds=secs - 480)
cfg = {"output": out, "task_file": tasks, "task_count": sum(1 for l in open(tasks) if l.strip()),
       "task_file_sha256": hashlib.sha256(Path(tasks).read_bytes()).hexdigest(), "parallel": int(par), "seed": int(seed),
       "stop_utc": stop.isoformat(), "teacher": "actor_cache_head",
       "teacher_manifest": {"mode": "actor_cache_head", "actor": actor, "selector_health": json.loads(health), "judge": judge},
       "training": {"max_tokens": 32768},
       "collection_replicas": [{"actor_port": int(ap), "selector_port": int(mp)}]}
frozen = Path(out) / "frozen-config.json"
if frozen.exists():  # resume: everything that defines the data must be unchanged
    old = json.loads(frozen.read_text())
    for k in ("task_file", "task_count", "task_file_sha256", "seed", "teacher"):
        assert old[k] == cfg[k], (k, old[k], cfg[k])
    o, n = old["teacher_manifest"], cfg["teacher_manifest"]
    assert o == n, (o, n)
else:
    frozen.write_text(json.dumps(cfg, indent=2) + "\n")
(Path(out) / "collect-config.json").write_text(json.dumps(cfg, indent=2) + "\n")
print("CONFIG", json.dumps({k: cfg[k] for k in ("task_count", "parallel", "stop_utc")}))
PY

set +e
PYTHONPATH=$WT setsid $PY -u -m openwebrl.arm_c2 collect --config "$OUT/collect-config.json" &
CPID=$!
PIDS+=($CPID)
# Watcher: a dead service ends the collector through its SIGTERM path.
while kill -0 $CPID 2>/dev/null; do
  why=""
  kill -0 ${PIDS[0]} 2>/dev/null || why="service 0 died"
  if [ -n "$why" ]; then echo "STOPPING_COLLECTOR $why"; kill -TERM $CPID; break; fi
  sleep 20
done
wait $CPID; rc=$?
set -e
echo "COLLECT_EXIT $rc results=$(ls "$OUT/results" 2>/dev/null | wc -l)"; cat "$OUT/collection-summary.json" 2>/dev/null
if grep -lE 'Errno 98|address already in use' "$OUT"/browser_logs/* 2>/dev/null; then
  echo "PORT_COLLISION: browser-server bind failures above -> run INVALID" | tee "$OUT/PORT_COLLISION"; exit 3
fi
echo "PORTS_CLEAN: no browser-server bind failures"
if [ "${PARITY:-0}" = 1 ] && [ $rc -eq 0 ]; then  # pilot gate: free the GPU, then re-score saved turns offline
  kill -- -${PIDS[0]} 2>/dev/null; sleep 20
  PYTHONPATH=$WT $PY -u $X/parity_check.py --collection "$OUT" --heads $HEADS --head-name $HEAD --actor $ACTOR --n 40 > "$OUT/parity.log" 2>&1
  echo "PARITY_EXIT $? $(tail -1 $OUT/parity.log)"
fi
exit $rc
