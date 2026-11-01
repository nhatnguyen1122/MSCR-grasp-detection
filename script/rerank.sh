#!/usr/bin/env bash
# Re-ranking for one trained grasp network: top-1 evaluation, candidate mining, scorer training, ablations, cost.
#
#   NET=logs/<run>/best_model.pt MASK_DIR=data/grasp-anything++/seen/part_mask \
#   NEG_DIR=data/grasp-anything++/seen/negative_grasp NAME=mscr GPUS="0 1 2" bash script/rerank.sh
#
# The training-set mining (the slow step, ~25k samples) is sharded over GPUS; everything else runs on the first GPU.
# STEPS selects a subset of: top1 mine scorer candidates ablations time (default: all).
# Outputs: cache/$NAME/*.npz, scorers/$NAME/*.pt, results/$NAME/*.json (file names as in docs/EXPERIMENTS.md)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

: "${NET:?set NET to the grasp network checkpoint}"
NAME="${NAME:-$(basename "$(dirname "$NET")")}"
steps="${STEPS:-top1 mine scorer candidates ablations time}"
py="${PYTHON_BIN:-python}"
gpus=(${GPUS:-0})
export CUDA_VISIBLE_DEVICES="${gpus[0]}"
C=cache/$NAME; S=scorers/$NAME; R=results/$NAME
mkdir -p "$C" "$S" "$R"
TR=split/grasp-anything++/train/seen.obj
SEEN=split/grasp-anything++/test/seen.obj
UNSEEN=split/grasp-anything++/test/unseen.obj
mask=(); [[ -n "${MASK_DIR:-}" ]] && mask=(--mask-dir "$MASK_DIR")
neg=(); [[ -n "${NEG_DIR:-}" ]] && neg=(--neg-dir "$NEG_DIR")
has() { [[ " $steps " == *" $1 "* ]]; }

# --- top-1 decoding, with and without mask gating; grounding and error breakdown
if has top1; then
  $py evaluate_top1.py --network "$NET" --split-files "$SEEN" "$UNSEEN" "${mask[@]}" --alphas 0 0.5 1 2 \
      --out "$R/top1.json"
fi

# --- candidates: scorer training data (training part, hybrid + extras), validation, test caches per strategy
mine_test() {  # strategy name, extra args...
  local name=$1; shift
  for split in seen unseen; do
    local sf=$SEEN; [[ $split == unseen ]] && sf=$UNSEEN
    $py mine_candidates.py --network "$NET" --split-file "$sf" --k 20 "${mask[@]}" "$@" --out "$C/${split}_$name.npz"
  done
}
if has mine; then
  n=${#gpus[@]}; shards=()
  for k in $(seq 0 $((n - 1))); do
    CUDA_VISIBLE_DEVICES="${gpus[$k]}" $py mine_candidates.py --network "$NET" --split-file $TR --part train \
        --strategy hybrid --k 20 --extras "${neg[@]}" --shard "$k/$n" --out "$C/train_hybrid.shard$k.npz" &
    shards+=("$C/train_hybrid.shard$k.npz")
  done
  wait
  $py mine_candidates.py --network "$NET" --split-file $TR --merge "${shards[@]}" --out "$C/train_hybrid.npz"
  $py mine_candidates.py --network "$NET" --split-file $TR --part val --strategy hybrid --k 20 \
      --out "$C/val_hybrid.npz"
  for strat in hybrid topk perturb tta mcdropout; do mine_test $strat --strategy $strat; done
  for tau in 0.05 0.1 0.2; do mine_test sample_tau$tau --strategy sample --tau $tau; done
fi

# --- scorers: the full one, then one ablation per changed input / negative set / loss
train_scorer() {  # name, extra args...
  local name=$1; shift
  $py reranker.py train --train "$C/train_hybrid.npz" --val "$C/val_hybrid.npz" --out "$S/$name.pt" "$@"
}
if has scorer; then
  train_scorer full
  train_scorer no_instruction           --inputs crop image geom
  train_scorer crop_instruction         --inputs crop prompt
  train_scorer no_geometry              --inputs crop prompt image
  train_scorer crop_only                --inputs crop
  train_scorer dataset_negatives        --negatives data
  train_scorer dataset_hard_negatives   --negatives data hard
  train_scorer dataset_hard_lang_negatives --negatives data hard lang
  train_scorer listwise                 --loss listwise
fi

eval_scorer() {  # scorer name, candidate strategy, output name
  $py reranker.py eval --scorer "$S/$1.pt" --tune "$C/val_hybrid.npz" \
      --test "$C/seen_$2.npz" "$C/unseen_$2.npz" --out "$R/$3.json"
}
# --- the full scorer on every candidate strategy (rerank_full.json is the paper's re-ranking row)
if has candidates; then
  eval_scorer full hybrid rerank_full
  for strat in topk perturb tta mcdropout sample_tau0.05 sample_tau0.1 sample_tau0.2; do
    eval_scorer full "$strat" "candidates_$strat"
  done
  $py reranker.py eval --test "$C/seen_hybrid.npz" "$C/unseen_hybrid.npz" --out "$R/oracle_only.json"
fi
# --- scorer ablations on hybrid candidates
if has ablations; then
  for name in no_instruction crop_instruction no_geometry crop_only dataset_negatives dataset_hard_negatives \
              dataset_hard_lang_negatives listwise; do
    eval_scorer "$name" hybrid "rerank_$name"
  done
fi
# --- cost per image for K = 1, 5, 10, 20
if has time; then
  $py mine_candidates.py --network "$NET" --split-file "$SEEN" --strategy hybrid --time 100 --time-k 1 5 10 20 \
      --scorer "$S/full.pt" --out "$R/timing.json"
fi
echo "Results in $R"
