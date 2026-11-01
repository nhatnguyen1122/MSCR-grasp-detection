#!/usr/bin/env bash
# Train the MSCR grasp network and its ablation variants, several in parallel, round-robin over GPUs.
#
#   MASK_DIR=data/grasp-anything++/seen/part_mask GPUS="0 1 2" bash script/train_mscr.sh
#
# VARIANTS selects networks (default: all of Table 2):
#   mscr                    grasp network (FiLM before BatchNorm + mask loss)
#   mscr_no_mask            - mask loss
#   mscr_no_film            - FiLM (text added as in the baseline)
#   mscr_film_post          FiLM after BatchNorm, unbounded (earlier network)
#   mscr_film_post_no_mask  the same without the mask loss (collapses)
# Every variant trains on the same samples (those with a part mask). SEEDS="123 124" adds seeds;
# PER_GPU = concurrent runs per GPU (default 2). Checkpoints: logs/<time>_<variant>_s<seed>/best_model.pt
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${MASK_DIR:?set MASK_DIR to the Grasp-Anything++ part-mask directory}"
variants="${VARIANTS:-mscr mscr_no_mask mscr_no_film mscr_film_post mscr_film_post_no_mask}"
seeds="${SEEDS:-123}"
gpus=(${GPUS:-0})
per_gpu="${PER_GPU:-2}"
epochs="${EPOCHS:-50}"
python_bin="${PYTHON_BIN:-python}"
mkdir -p logs/train

job=0
pids=()
for seed in $seeds; do
  for variant in $variants; do
    gpu="${gpus[$(( job % ${#gpus[@]} ))]}"
    desc="${variant}_s${seed}"
    echo "[gpu $gpu] $desc"
    CUDA_VISIBLE_DEVICES="$gpu" "$python_bin" -u train_network.py \
      --dataset grasp-anywhere --dataset-path data/grasp-anything \
      --add-file-path data/grasp-anything++/seen \
      --split-file split/grasp-anything++/train/seen.obj \
      --use-depth 0 --use-rgb 1 --seen 1 --channel-size 32 \
      --epochs "$epochs" --batch-size 8 --batches-per-epoch 1000 \
      --split 0.9 --random-seed "$seed" \
      --mask-dir "$MASK_DIR" --network "$variant" \
      --description "$desc" > "logs/train/${desc}.log" 2>&1 &
    pids+=($!)
    job=$((job + 1))
    # keep at most PER_GPU runs per GPU
    if (( ${#pids[@]} >= per_gpu * ${#gpus[@]} )); then
      wait "${pids[0]}"; pids=("${pids[@]:1}")
    fi
  done
done
wait
echo "Done. Best checkpoints: logs/*_<variant>_s<seed>/best_model.pt"
