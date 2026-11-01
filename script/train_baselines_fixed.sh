#!/usr/bin/env bash
# Fresh sequential runs of the released baselines with connected image gradients.
# Usage: conda activate lgd; GPU=0 bash script/train_baselines_fixed.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."

gpu="${GPU:-0}"
epochs="${EPOCHS:-50}"
python_bin="${PYTHON_BIN:-python}"
run_tag="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
mkdir -p logs/reruns

for net in lgrconvnet3 lggcnn lragt; do
  channels=32
  if [[ "$net" == lragt ]]; then
    channels=18
  fi
  description="${net}_fixed_${run_tag}"
  CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u train_network.py \
    --dataset grasp-anywhere \
    --dataset-path data/grasp-anything \
    --add-file-path data/grasp-anything++/seen \
    --split-file split/grasp-anything++/train/seen.obj \
    --use-depth 0 --use-rgb 1 --seen 1 \
    --network "$net" --channel-size "$channels" \
    --epochs "$epochs" --batch-size 8 --batches-per-epoch 1000 \
    --split 0.9 --random-seed 123 \
    --description "$description" \
    2>&1 | tee "logs/reruns/${description}.log"
done
