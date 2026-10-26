#!/bin/bash
#SBATCH --job-name=zx-eval
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=192G
#SBATCH --time=3:30:00
#SBATCH --output=/gpfs/scrubbed/dan29/logs/%x_%j.out
#
# Tillicum copy of ../../../zixian_1a/sbatch_eval_1a.sh = Zixian Ma's OM2W protocol, the one behind the LoRA joint DPO
# 35.33 % (dpo_run{1,2,3}): her code at 299f03f, `python -m openwebrl.arm_eval --mode baseline`, seed 42, T 0.7,
# top_p 0.9, 1,024 new tokens, 30 steps, 1,800 s/task, parallel 6, 1 screenshot, full history, o4-mini
# OSU-AgentTrek judge; SGLang with her flags (ctx 32,768, max-running 24, chunked prefill 4,096, no CUDA graph,
# mem-fraction 0.8 as on klone). Tillicum differences (infra only):
#   - her arm_eval HARD-CODES browser ports 19200-19399; she runs the same code on this cluster's shared nodes, so a
#     co-located job of hers would share ports (cf. our 364597 /reset into her job, 2026-10-16). ONE patch
#     (arm_eval_ports.patch: plain assignment -> setdefault, defaults unchanged) lets us set 30000-31999 + our
#     per-user lock dir, the same block as sbatch_eval_tillicum.sh. Bind failures are checked after the run.
#   - SGLang port 32000+(jobid%180)*4 and --nccl-port = +1 (below the ephemeral range; klone: 19000+)
#   - env zixian-panels (= all 288 pins of klone's openwebrl env), nvcc /gpfs/software/cuda/12.8.2 (klone module
#     cuda/12.8.1), 1 H200 / 8 CPUs (klone A40 / 12 CPUs; parallel 6 unchanged), paths in /gpfs/scrubbed/dan29
# Her arm_eval skips tasks that already have a result, so a resubmit into the same OUT finishes a run.
# JUDGE_MODEL is required (2026-10-23: OpenAI retired o4-mini; a failed judge call silently marks the task invalid).
# gpt-5.6-terra = OpenAI's listed replacement; only compare against runs judged by the same model.
# Submit: env MODEL=<merged model> OUT=<fresh dir> JUDGE_MODEL=<model> [TASK_INDICES=..] [TASK_FILE=<path, default OM2W; another benchmark keeps the OM2W judge>] sbatch --job-name=<name> sbatch_eval_her_protocol.sh
set -euo pipefail
: "${MODEL:?}" "${OUT:?}" "${OPENAI_API_KEY:?}" "${JUDGE_MODEL:?}"
curl -sf --max-time 60 https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" -d "{\"model\":\"$JUDGE_MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}]}" \
  >/dev/null || { echo "FATAL: judge model $JUDGE_MODEL does not answer"; exit 1; }
echo "JUDGE_OK $JUDGE_MODEL"
X=/gpfs/projects/krishna/dan29/web-agent/OpenWebRL/experiments/zixian_joint/full_ft_dpo/tillicum
WT=/gpfs/scrubbed/dan29/src/zixian_owrl_1a
ENVP=/gpfs/projects/krishna/dan29/conda/envs/zixian-panels
PY=$ENVP/bin/python
[ "$(git -C $WT rev-parse HEAD)" = 299f03fa1739bea52eb6b1df9fce2b70d6dbecb4 ] || { echo "FATAL: worktree not at 299f03f"; exit 1; }
cmp -s <(git -C $WT diff) $X/arm_eval_ports.patch || { echo "FATAL: worktree differs from 299f03f + arm_eval_ports.patch"; git -C $WT status --porcelain; exit 1; }
mkdir -p "$OUT"
[ -n "$(ls -A "$OUT/results" 2>/dev/null)" ] && echo "NOTE: resuming into $OUT ($(ls "$OUT/results" | wc -l) results present)"

unset -f module ml 2>/dev/null || true
unset BASH_ENV
export HF_HOME=/gpfs/scrubbed/dan29/hf HF_HUB_OFFLINE=1
export CUDA_HOME=/gpfs/software/cuda/12.8.2
export PATH="$ENVP/bin:$CUDA_HOME/bin:$PATH"
# node-local browser state (GPFS HOME/TMPDIR stall Chromium)
PW_LOCAL=/tmp/pw_${SLURM_JOB_ID}; mkdir -p $PW_LOCAL && cp -r /gpfs/scrubbed/dan29/ms-playwright/. $PW_LOCAL/
export PLAYWRIGHT_BROWSERS_PATH=$PW_LOCAL TMPDIR=/tmp/job_${SLURM_JOB_ID} HOME=/tmp/home_${SLURM_JOB_ID}
export XDG_CONFIG_HOME=$HOME/.config XDG_CACHE_HOME=$HOME/.cache XDG_DATA_HOME=$HOME/.local/share
mkdir -p $TMPDIR $XDG_CONFIG_HOME $XDG_CACHE_HOME $XDG_DATA_HOME
export SLIME_BROWSER_LOCAL_PROCESS_PORT_START=30000 SLIME_BROWSER_LOCAL_PROCESS_PORT_END=31999
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=/tmp/slime_ports_$USER
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=${ENV_STARTUP_TIMEOUT:-180}  # as klone (her default 30 s)
export NO_PROXY="127.0.0.1,0.0.0.0,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,0.0.0.0,localhost,${no_proxy:-}"
export SLIME_HF_VISION_ATTN_IMPL=sdpa SLIME_ASYNC_USE_STDLIB_LOOP=1  # her scripts/h200_env.sh
export FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer TRITON_CACHE_DIR=$TMPDIR/triton
PORT=$((32000 + (SLURM_JOB_ID % 180) * 4))
setsid $PY -m sglang.launch_server --model-path "$MODEL" --host 127.0.0.1 --port $PORT --nccl-port $((PORT + 1)) \
  --dtype bfloat16 --tp 1 --mem-fraction-static 0.8 --context-length 32768 --max-running-requests 24 \
  --chunked-prefill-size 4096 --disable-cuda-graph > "$OUT/actor-server.$SLURM_JOB_ID.log" 2>&1 &
SPID=$!
trap 'kill -- -$SPID 2>/dev/null || true; rm -rf "$PW_LOCAL" "$TMPDIR" "$HOME"' EXIT
for t in $(seq 1 180); do
  curl -sf --noproxy '*' http://127.0.0.1:$PORT/health_generate >/dev/null && break
  kill -0 $SPID 2>/dev/null || { echo "SERVER_DIED"; tail -30 "$OUT/actor-server.$SLURM_JOB_ID.log"; exit 1; }
  sleep 10
done
INFO=$(curl -sf --noproxy '*' http://127.0.0.1:$PORT/get_model_info) || { echo "SERVER_NOT_READY"; exit 1; }
$PY -c "import json, os, sys; p = json.loads(sys.argv[1])['model_path']; assert os.path.realpath(p) == os.path.realpath(sys.argv[2]), p" "$INFO" "$MODEL"
echo "SERVER_READY $(date) model=$MODEL port=$PORT"

cd $WT
set +e
PYTHONPATH=$WT $PY -u -m openwebrl.arm_eval --mode baseline --actor "$MODEL" --actor-port $PORT --output "$OUT" \
  --parallel 6 --task-file "${TASK_FILE:-openwebrl/data/eval/online-mind2web.jsonl}" --seed 42 --temperature 0.7 --top-p 0.9 \
  --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --judge-model "$JUDGE_MODEL" --env-file "" \
  ${TASK_INDICES:+--task-indices "$TASK_INDICES"}
rc=$?
set -e
echo "EVAL_EXIT $rc results=$(ls "$OUT/results" 2>/dev/null | wc -l)"; cat "$OUT/summary.json" 2>/dev/null
# a bind failure means some task may have talked to another process's server: the run is not trustworthy
if grep -lE 'Errno 98|address already in use' "$OUT"/browser_logs/* 2>/dev/null; then
  echo "PORT_COLLISION: browser-server bind failures above -> run INVALID" | tee "$OUT/PORT_COLLISION"; exit 3
fi
echo "PORTS_CLEAN: no browser-server bind failures"
exit $rc
