#!/usr/bin/env bash
#SBATCH --job-name=bench-piotr
#SBATCH --account=zixianma
#SBATCH --partition=gpu-h200
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=240G
#SBATCH --time=03:00:00
#SBATCH --output=/gpfs/scrubbed/dan29/logs/%x_%j.out
#
# WebVoyager / DeepShop eval = Piotr Teterwak's tillicum/eval_concurrent.sbatch (github.com/piotr-teterwak/OpenWebRL
# branch piotr_arm @0b2fc0b, README_BENCHMARKS.md), run from OUR pinned clone of that branch: one H200 per model, all
# models concurrent, base always in the same job (site blocking drifts over hours). His arm_eval.py --mode baseline
# --protocol {webvoyager,deepshop} picks the task file's judge (gpt-4o; reward_webvoyager.py / reward_deepshop.py);
# --judge-model is NOT passed, as in his script. Eval and SGLang command lines are his, verbatim.
# Differences from his script (infra only; user rule: jobs read only our own copies):
#   - env: our zixian-panels conda + node-local browser state (= sbatch_eval_her_protocol.sh, whose OM2W evals hang
#     ~0.05 %/step), not Zixian's venvs / her runtime root (his scripts/h200_env.sh is not sourced)
#   - ports: SGLang 32000+(job%90)*8 (+4 per model, nccl +1), browsers 30000+300*i+100..+299 (below the ephemeral
#     range; his 20xxx); per-user port-lock dir; bind failures checked after the run
#   - judge key from ~/.bashrc (his .env); no merge step (our models are already merged)
# Submit: BENCH=webvoyager|deepshop MODELS=base,sft-noarm PASS=<name> sbatch [--time=..] eval_concurrent_piotr.sbatch
# Results: /gpfs/scrubbed/dan29/evals/bench_piotr/pass<PASS>/<label>/<bench>/ (resumes on resubmission).
set -euo pipefail
PASS="${PASS:?set PASS}"
BENCH="${BENCH:?set BENCH=webvoyager or deepshop}"
case "$BENCH" in
  webvoyager) TASK_FILE=openwebrl/data/eval/webvoyager_fara.jsonl ;;
  deepshop) TASK_FILE=openwebrl/data/eval/deepshop.jsonl ;;
  *) echo "bad BENCH=$BENCH"; exit 1 ;;
esac
REPO=/gpfs/scrubbed/dan29/src/piotr_owrl_piotr_arm
[ "$(git -C $REPO rev-parse HEAD)" = 0b2fc0b8d3f1620267fd74876ad5e4ab6f4b46e4 ] || { echo "FATAL: clone not at 0b2fc0b"; exit 1; }
[ -z "$(git -C $REPO status --porcelain --untracked-files=no)" ] || { echo "FATAL: clone has local changes"; exit 1; }
ENVP=/gpfs/projects/krishna/dan29/conda/envs/zixian-panels
PY=$ENVP/bin/python
OUT=/gpfs/scrubbed/dan29/evals/bench_piotr/pass$PASS
source <(grep "^export OPENAI_API_KEY" ~/.bashrc)
[ ${#OPENAI_API_KEY} -gt 40 ] || { echo "FATAL OPENAI_API_KEY did not resolve"; exit 1; }
curl -sf --max-time 60 https://api.openai.com/v1/chat/completions -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "Content-Type: application/json" -d '{"model":"gpt-4o","messages":[{"role":"user","content":"Say OK"}]}' \
  >/dev/null || { echo "FATAL: judge model gpt-4o does not answer"; exit 1; }
echo "JUDGE_OK gpt-4o"

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
export SLIME_BROWSER_LOCAL_PROCESS_PORT_LOCK_DIR=/tmp/slime_ports_$USER
export SLIME_BROWSER_LOCAL_PROCESS_STARTUP_TIMEOUT_SECS=180
export NO_PROXY="127.0.0.1,0.0.0.0,localhost,${NO_PROXY:-}" no_proxy="127.0.0.1,0.0.0.0,localhost,${no_proxy:-}"
export SLIME_HF_VISION_ATTN_IMPL=sdpa SLIME_ASYNC_USE_STDLIB_LOOP=1 OMP_NUM_THREADS=4  # his h200_env.sh values
export FLASHINFER_WORKSPACE_BASE=$TMPDIR/flashinfer TRITON_CACHE_DIR=$TMPDIR/triton
export PYTHONPATH=$REPO
cd $REPO
mkdir -p $OUT

# label | model path
SPECS=(
  "base|/gpfs/scrubbed/dan29/hf/OpenWebRL-4B-SFT"
  "sft-noarm|/gpfs/scrubbed/dan29/iterdpo_r1/noarm_full1978_s42/evaluation/merged-endpoint"
  "onpol|/gpfs/scrubbed/dan29/lora_dpo_arms/runs/matched-r16a32-lr5e-5-rpo1-onpolicy/evaluation/merged-endpoint"
)
if [ -n "${MODELS:-}" ]; then
  KEEP=()
  for spec in "${SPECS[@]}"; do [[ ",$MODELS," == *",${spec%%|*},"* ]] && KEEP+=("$spec"); done
  [ ${#KEEP[@]} -eq $(tr ',' '\n' <<<"$MODELS" | wc -l) ] || { echo "FATAL unknown label in MODELS=$MODELS"; exit 1; }
  SPECS=("${KEEP[@]}")
fi
N=${#SPECS[@]}
mapfile -t GPUS < <(nvidia-smi --query-gpu=uuid --format=csv,noheader)
[ ${#GPUS[@]} -eq $N ] || { echo "FATAL need $N GPUs, got ${#GPUS[@]}"; exit 1; }

PIDS=()
cleanup() { for p in "${PIDS[@]}"; do kill -- -$p 2>/dev/null || kill $p 2>/dev/null || true; done; rm -rf "$PW_LOCAL" "$TMPDIR" "$HOME"; }
trap cleanup EXIT

# 1. One sglang server per GPU, his flags.
END=$(date -d "$(squeue -h -j $SLURM_JOB_ID -o %e)" +%s)   # %e is local time
SBASE=$((32000 + (SLURM_JOB_ID % 90) * 8))
i=0
for spec in "${SPECS[@]}"; do
  IFS='|' read label model <<<"$spec"
  mkdir -p $OUT/$label
  CUDA_VISIBLE_DEVICES=${GPUS[$i]} setsid $PY -m sglang.launch_server --model-path "$model" \
    --host 127.0.0.1 --port $((SBASE + 4*i)) --nccl-port $((SBASE + 4*i + 1)) --dtype bfloat16 --tp 1 --mem-fraction-static 0.45 \
    --context-length 32768 --max-running-requests 24 --chunked-prefill-size 4096 --disable-cuda-graph \
    > $OUT/$label/actor-server-$BENCH.log 2>&1 &
  PIDS+=($!); i=$((i+1))
done
for i in $(seq 0 $((N-1))); do
  port=$((SBASE + 4*i)); t0=$(date +%s)
  until curl -sf --noproxy '*' http://127.0.0.1:$port/health_generate >/dev/null; do
    kill -0 ${PIDS[$i]} 2>/dev/null || { echo "FATAL server $i died"; tail -20 $OUT/*/actor-server-$BENCH.log; exit 1; }
    [ $(( $(date +%s) - t0 )) -lt 900 ] || { echo "FATAL server $i not healthy after 15 min"; exit 1; }
    sleep 5
  done
done
echo "[eval] $N servers healthy"

# 2. All evals at once; stop 5 min before the limit (results so far are kept and a resubmission skips them).
REMAIN=$(( END - $(date +%s) - 300 ))
[ $REMAIN -gt 1800 ] || { echo "FATAL only ${REMAIN}s left; check the deadline math"; exit 1; }
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader -l 300 > $OUT/gpu-util-$BENCH.$SLURM_JOB_ID.csv &
PIDS+=($!)
i=0; EVAL_PIDS=()
for spec in "${SPECS[@]}"; do
  IFS='|' read label model <<<"$spec"
  b=$((30000 + 300*i))
  timeout --signal=TERM --kill-after=120 $REMAIN $PY -m openwebrl.arm_eval --mode baseline \
    --actor "$model" --actor-port $((SBASE + 4*i)) --output $OUT/$label/$BENCH \
    --parallel 12 --protocol ${BENCH//-/_} --task-file $TASK_FILE --seed 42 --temperature 0.7 --top-p 0.9 \
    --max-new-tokens 1024 --max-steps 30 --task-timeout 1800 --env-file "" \
    --browser-port-start $((b + 100)) --browser-port-end $((b + 299)) > $OUT/$label/eval-$BENCH.log 2>&1 &
  EVAL_PIDS+=($!); i=$((i+1))
done
echo "[eval] pass $PASS $BENCH running: ${SPECS[*]%%|*}"
rc=0; for p in "${EVAL_PIDS[@]}"; do wait $p || rc=$?; done

for spec in "${SPECS[@]}"; do
  label=${spec%%|*}; s=$OUT/$label/$BENCH/summary.json
  n=$(ls $OUT/$label/$BENCH/results 2>/dev/null | wc -l)
  [ -f $s ] && echo "[eval] $label: $(tr -d '\n ' < $s)" || echo "[eval] $label: incomplete ($n results)"
  if grep -lE 'Errno 98|address already in use' $OUT/$label/$BENCH/browser_logs/* 2>/dev/null; then
    echo "PORT_COLLISION $label: browser-server bind failures -> run INVALID" | tee $OUT/$label/$BENCH/PORT_COLLISION; rc=3
  fi
done
exit $rc
