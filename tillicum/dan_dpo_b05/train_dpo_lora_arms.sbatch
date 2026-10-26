#!/bin/bash
#SBATCH --job-name=lora-dpo-arm
#SBATCH --partition=gpu-h200
#SBATCH --account=zixianma
#SBATCH --qos=normal
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH --time=4:30:00
#SBATCH --output=/gpfs/scrubbed/dan29/logs/%x_%j.out
# LoRA DPO RANK / LR ARMS (2026-10-20, user go): copy of train_dpo_r1.sbatch for Zixian's JOINT data (klone-mirror copy =
# HF release, the data our 10-14 reproduction trained on; pair order == hers), one arm per CFG (make_dpo_config.py
# --lora-rank/--lora-alpha/--learning-rate; trainer edit 5 reads them). Changes: CFG required; cd into the data dir
# (joint rows use image paths relative to it); --time 4:30 (her 5,540-pair run took 3:22 on 2 H200); end gate: the final
# adapter's r / lora_alpha == the config. Usage: CFG=<config.json> sbatch --job-name=<arm> train_dpo_lora_arms.sbatch
# Original description follows.
# Round-1 fresh LoRA DPO from pi0 (OpenWebRL-4B-SFT, reference = pi0 with adapters disabled), user go 2026-10-18:
# Zixian's joint-v2 DPO recipe unchanged (LoRA r16, beta 0.1, lr 1e-5, 32 pairs/update, 1 epoch) on the round-1 pairs
# filtered to successful episodes + equal tool-call counts (filter_dpo_pairs.py: 2,778 train / 346 validation).
# Trainer = train_dpo_r1_ddp.py (her train_arm_joint_ddp.py + 3 listed edits); config = make_dpo_config.py output.
# Her own run (287447, 2x H200, 5,540 pairs) took 3:22 incl. reference cache + gates; this one ~half the pairs.
# The trainer saves and pauses 20 min before the limit; resume with RESUME=1 (same config/output).
set -euo pipefail
EXP=/gpfs/projects/krishna/dan29/web-agent/OpenWebRL/experiments/iterdpo_r1
WT=/gpfs/scrubbed/dan29/src/zixian_owrl_joint
: "${CFG:?set CFG=<config.json>}"
ENV=/gpfs/projects/krishna/dan29/conda/envs/zixian-panels
[ "$(git -C $WT rev-parse HEAD)" = "$(git -C $WT rev-parse fe65405)" ] || { echo "worktree not at fe65405"; exit 1; }
[ -z "$(git -C $WT status --porcelain)" ] || { echo "worktree modified"; git -C $WT status --porcelain; exit 1; }
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPUs not free"; exit 1; }
WANT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['trainer_copy_sha256'])" $CFG)
[ "$(sha256sum $EXP/train_dpo_r1_ddp.py | cut -d' ' -f1)" = "$WANT" ] || { echo "trainer copy differs from the frozen config"; exit 1; }
OUT=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['output'])" $CFG)
if [ -z "${RESUME:-}" ] && [ -e "$OUT/student" ]; then echo "$OUT/student exists; set RESUME=1 or use a fresh output"; exit 1; fi
export HF_HOME=/gpfs/scrubbed/dan29/hf HF_HUB_OFFLINE=1 SLIME_HF_VISION_ATTN_IMPL=sdpa WANDB_MODE=offline
# CUDA_HOME: her native-parser gate imports sglang -> deep_gemm, which asserts a CUDA home at import (as the
# zixian-panels eval/collection sbatches set it)
export CUDA_HOME=/gpfs/software/cuda/12.8.2
export PYTHONPATH=$WT/scripts:$WT:/gpfs/scrubbed/dan29/pydeps/peft_0181 PATH=$ENV/bin:$CUDA_HOME/bin:$PATH
export TMPDIR=/tmp/job_${SLURM_JOB_ID}; mkdir -p $TMPDIR; trap 'rm -rf "$TMPDIR"' EXIT
python -c "import peft, torch, transformers; assert peft.__version__ == '0.18.1', peft.__version__; print('peft', peft.__version__, 'torch', torch.__version__, 'transformers', transformers.__version__)"
END=$(squeue -h -j $SLURM_JOB_ID -o %e); STOP=$(( $(date -d "$END" +%s) - 1200 ))
DATA=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['data'])" $CFG); cd "$DATA"
echo "CONFIG cfg=$CFG out=$OUT stop_unix=$STOP resume=${RESUME:-0} $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd,)"
torchrun --nproc_per_node 2 --master_port $((20000 + SLURM_JOB_ID % 10000)) $EXP/train_dpo_r1_ddp.py \
  --config $CFG --stop-unix $STOP ${RESUME:+--resume}
echo "TRAIN_EXIT 0"; cat $OUT/training-status.json
python3 - "$CFG" "$OUT" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); out = sys.argv[2]
done = json.load(open(f"{out}/student/complete.json"))
a = json.load(open(f"{done['checkpoint']}/adapter_config.json"))
assert done["updates"] == c["total_updates"], (done["updates"], c["total_updates"])
assert (a["r"], a["lora_alpha"]) == (c["lora_rank"], c["lora_alpha"]), (a["r"], a["lora_alpha"])
print(f"ARM OK: {done['checkpoint']} updates {done['updates']}, adapter r {a['r']} alpha {a['lora_alpha']}, lr {c['learning_rate']}")
PY
