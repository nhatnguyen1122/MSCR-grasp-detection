# Experiments: paper numbers → commands and result files

Run from the repository root in the `lgd` environment, with the data unzipped into `data/` (see the README).
`M=data/grasp-anything++/seen/part_mask`, `N=data/grasp-anything++/seen/negative_grasp`.

## Commands

| Step | Command |
| --- | --- |
| Train the baselines (GR-ConvNet / GG-CNN / RAGT + CLIP) | `bash script/train_baselines_fixed.sh` |
| Train MSCR and its variants (~3 h each, two per GPU) | `MASK_DIR=$M GPUS="0 1 2" bash script/train_mscr.sh` |
| Top-1 evaluation of any network | `python evaluate_top1.py --network logs/<run>/best_model.pt --mask-dir $M --alphas 0 --out results/<name>.json` |
| Re-ranking: mining, scorers, ablations, cost | `NET=logs/<run>/best_model.pt MASK_DIR=$M NEG_DIR=$N NAME=<name> GPUS="0 1 2" bash script/rerank.sh` |

`evaluate_top1.py` evaluates `test/seen.obj` and `test/unseen.obj` by default. `rerank.sh` writes
`results/<name>/` with the file names below; its steps can also be run one by one (see the script, or the
docstrings of `mine_candidates.py` and `reranker.py`).

## Where each number comes from

The result JSONs in `results/` are the ones behind the paper. Their keys are the original run folders, which use
the project's working names: `gtv_a3s` = `mscr`, `gtv_a2s` = `mscr_no_mask`, `gtv_a3d` = `mscr_no_film`,
`gtv_a3` = `mscr_film_post`, `gtv_a2` = `mscr_film_post_no_mask`. Old checkpoints load under the new names
(`mscr/common.py:load_network`). Top-1 rows use α = 0 (no mask gating); re-ranking rows use the `main` field
(α and K tuned on validation: K = 20, α = 0).

**Table 1**

| Row | Network | File |
| --- | --- | --- |
| GG-CNN + CLIP | `lggcnn` | `baselines.json` (`lggcnn_fixed`) |
| GR-ConvNet + CLIP | `lgrconvnet3` | `baselines.json` (`lgrconvnet3_fixed`) |
| MSCR | `mscr` | `mscr.json` (evaluated on a second machine, transcribed from its log) |
| MSCR† + re-ranking | `mscr_film_post` | `rerank_full.json` |

`baselines.json` also holds the released baselines (`*_original`, image features detached before fusion).

**Table 2**

| Row | File |
| --- | --- |
| Grasp network | `mscr.json` |
| − mask loss | `top1_no_mask_no_film.json` (`gtv_a2s`) |
| − FiLM | `top1_no_mask_no_film.json` (`gtv_a3d`) |
| FiLM after BatchNorm, unbounded | `top1_film_post.json` (`gtv_a3`); collapsed run without the mask loss: `gtv_a2` (epoch 10) |
| Re-ranking (K = 20) | `rerank_full.json` |
| − re-ranking (first candidate) | `rerank_full.json`, field `top1` |
| − instruction input | `rerank_no_instruction.json` (± over `_seed1`, `_seed2` and `rerank_full_seed*.json`) |
| − image and geometry inputs | `rerank_crop_instruction.json` |
| − perturbed candidates (top-K peaks) | `candidates_topk.json` |
| − extra negatives (dataset only) | `rerank_dataset_negatives.json` |

Also used in the text: oracle H 0.572 (`rerank_full.json`, `H` → `K=20` → `oracle`) and 167 / 17 ms per image
(`timing.json`). Other results not in the paper: the remaining candidate strategies (`candidates_*.json`), crop-only
and no-geometry scorers, hard and language negatives, the listwise loss, and the grasp network alone on the candidate
lists (`oracle_only.json`).
