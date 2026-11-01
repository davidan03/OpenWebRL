#!/bin/bash
# Build the two code trees the pipeline runs from (login node, no GPU):
#   $WT         = zixianma/OpenWebRL @1065034 + arm_c2_head.patch + her 4 selection-head files (pinned copies in her_files/)
#   $JOINT_REPO = zixianma/OpenWebRL @fe65405 (scripts/prepare_arm_joint_data.py, used by prepare_head_pairs.py)
# Usage: PKG=<package>/code WT=<new dir> JOINT_REPO=<new dir> [REPO_URL=...] bash setup_worktree.sh
set -euo pipefail
: "${PKG:?}" "${WT:?}" "${JOINT_REPO:?}"
REPO_URL=${REPO_URL:-https://github.com/zixianma/OpenWebRL.git}
git clone -q "$REPO_URL" "$WT"
git -C "$WT" checkout -q 1065034e07d3575bf29fcca68bb35c31ed14ab78
git -C "$WT" apply "$PKG/arm_c2_head.patch"
cp "$PKG"/her_files/{actor_head_capture,actor_head_scoring,selection_head,selection_head_features}.py "$WT/openwebrl/"
git clone -q "$REPO_URL" "$JOINT_REPO"
git -C "$JOINT_REPO" checkout -q fe65405ea6e94ce4adfc9b48e44bfc572c88d1a8
# the same checks sbatch_collect_head.sh makes before it starts
cmp -s <(git -C "$WT" diff) "$PKG/arm_c2_head.patch" || { echo "FATAL: worktree differs from 1065034 + arm_c2_head.patch"; exit 1; }
for f in actor_head_capture:cc6627c95054 actor_head_scoring:611a8d2ec9d2 selection_head_features:1870159ef11f selection_head:6f90d3264fa8; do
  [ "$(sha256sum "$WT/openwebrl/${f%%:*}.py" | cut -c1-12)" = "${f##*:}" ] || { echo "FATAL: ${f%%:*}.py is not the pinned copy"; exit 1; }
done
grep -q "judge_api_model='gpt-5.6-terra'" "$WT/openwebrl/arm_c2.py" || { echo "FATAL: collector judge is not gpt-5.6-terra"; exit 1; }
echo "SETUP_OK WT=$WT JOINT_REPO=$JOINT_REPO"
