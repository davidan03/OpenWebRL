#!/bin/bash
#SBATCH --job-name=lora-dpo-merge
#SBATCH --partition=gpu-h200-mig
#SBATCH --account=zixianma
#SBATCH --qos=normal
#SBATCH --gpus=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=34G
#SBATCH --time=0:40:00
#SBATCH --output=/gpfs/scrubbed/dan29/logs/%x_%j.out
# Tillicum copy of experiments/zixian_joint/sbatch_merge_joint.sh for the LoRA DPO dose arms (2026-10-20): merge each
# arm's update-174 endpoint into OpenWebRL-4B-SFT with her merge_arm_c2_student.py (fe65405, sha256 ed82948f…), exactly as
# her run_arm_joint_pipeline.py: --checkpoint student/update-000174 --output evaluation/merged-endpoint.
# Changes: Tillicum paths (zixian-panels env, scrubbed HF_HOME/peft), one MIG instance (her merge loads the 4B base on
# cuda:0, ~9 GB bf16; the CPU-only version failed on klone 41816800), several arms per job (ARMS = run dir names under
# /gpfs/scrubbed/dan29/lora_dpo_arms/runs; an arm whose merge-manifest already validates is a no-op in her script).
# Usage: ARMS="joint-r64a128-lr1e-5 joint-r16a32-lr3e-5" sbatch sbatch_merge_lora_arms.sbatch
# 2026-10-22 (RPO ablation, user: screen intermediate checkpoints too): optional UPDATES="32 64" merges those saved
# checkpoints (student/update-0000NN, progress.json updates == NN) into evaluation/merged-u0000NN; unset = the final
# update into evaluation/merged-endpoint as before (training-complete gates unchanged for that case).
set -euo pipefail
: "${ARMS:?space-separated arm dirs}"
WT=/gpfs/scrubbed/dan29/src/zixian_owrl_joint
ENV=/gpfs/projects/krishna/dan29/conda/envs/zixian-panels
[ "$(git -C $WT rev-parse HEAD)" = "$(git -C $WT rev-parse fe65405)" ] || { echo "worktree not at fe65405"; exit 1; }
[ -z "$(git -C $WT status --porcelain)" ] || { echo "worktree modified"; exit 1; }
[ "$(sha256sum $WT/scripts/merge_arm_c2_student.py | cut -c1-8)" = ed82948f ] || { echo "merge script hash differs"; exit 1; }
export HF_HOME=/gpfs/scrubbed/dan29/hf HF_HUB_OFFLINE=1 PYTHONPATH=$WT:/gpfs/scrubbed/dan29/pydeps/peft_0181
for A in $ARMS; do
  ROOT=/gpfs/scrubbed/dan29/lora_dpo_arms/runs/$A
  $ENV/bin/python -c "import peft; assert peft.__version__ == '0.18.1', peft.__version__"
  if [ -z "${UPDATES:-}" ]; then
    N=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['total_updates'])" /gpfs/scrubbed/dan29/lora_dpo_arms/configs/$A.json)
    grep -q '"phase": "complete"' $ROOT/training-status.json || { echo "$A: training not complete"; cat $ROOT/training-status.json; exit 1; }
    grep -q "\"updates\": $N" $ROOT/student/complete.json || { echo "$A: complete.json is not update $N"; cat $ROOT/student/complete.json; exit 1; }
    $ENV/bin/python $WT/scripts/merge_arm_c2_student.py --run-root $ROOT \
      --checkpoint $ROOT/student/update-$(printf %06d $N) --output $ROOT/evaluation/merged-endpoint
    ls -la $ROOT/evaluation/merged-endpoint
  else
    for U in $UPDATES; do
      C=$ROOT/student/update-$(printf %06d $U)
      python3 -c "import json,sys; p=json.load(open(sys.argv[1])); assert p['updates']==int(sys.argv[2]), p" $C/progress.json $U || { echo "$A: $C is not update $U"; exit 1; }
      $ENV/bin/python $WT/scripts/merge_arm_c2_student.py --run-root $ROOT --checkpoint $C --output $ROOT/evaluation/merged-u$(printf %06d $U)
      ls -la $ROOT/evaluation/merged-u$(printf %06d $U)
    done
  fi
  echo "MERGE_DONE $A"
done
