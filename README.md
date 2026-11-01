# Mask Supervision and Candidate Re-ranking for Language-Conditioned Grasp Detection

Language-driven grasp detection: given an image and an instruction such as *"Grip sunglasses on its frame."*,
predict a grasp rectangle on the named object part. About three quarters of the failures of lightweight
CLIP-conditioned grasp networks land on the wrong object or part. MSCR (Mask Supervision with Candidate Re-ranking)
changes a GR-ConvNet in three ways:

1. **FiLM conditioning**: the instruction scales and shifts each feature channel at the bottleneck and both
   upsampling layers, applied before BatchNorm with the scale bounded to (0, 2).
2. **Mask loss**: an extra head predicts the instructed part's mask (BCE + Dice), so the network learns which
   pixels the instruction refers to.
3. **Candidate re-ranking**: the 5 best peaks and their perturbations give K candidates. A small MLP over frozen
   CLIP features of the grasp crop, the image and the instruction, plus the grasp geometry, picks one.

Built on the [LGD codebase](docs/LGD_README.md). LGD's diffusion model is unchanged, and the baselines differ from
the release only by the fixes listed below.

## Results (Grasp-Anything++, top-1 success)

| Method | Seen | Unseen | H |
| --- | --- | --- | --- |
| GG-CNN + CLIP (retrained) | 0.257 | 0.263 | 0.260 |
| GR-ConvNet + CLIP (retrained) | 0.291 | **0.388** | 0.333 |
| MSCR | 0.454 | 0.341 | **0.390** |
| MSCR† + re-ranking | **0.478** | 0.301 | 0.370 |

H is the harmonic mean of seen and unseen. A grasp is correct if IoU > 0.25 and the angle error is < 30°.
†Re-ranking has so far only been run on an earlier version of the network (FiLM after BatchNorm, unbounded).
Every number, including the ablations, maps to a JSON in `results/`: see [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).
All results are single training runs.

## Code

| Path | Content |
| --- | --- |
| `inference/models/mscr.py` | The MSCR grasp network and its ablation variants (`mscr`, `mscr_no_mask`, `mscr_no_film`, …) |
| `mscr/decode.py` | Mask gating, candidate strategies (hybrid, top-K, perturbation, TTA, MC dropout, sampling), fast success test |
| `mscr/scorer.py` | Grasp crops, frozen CLIP features, the scorer MLP |
| `mscr/common.py` | Dataset, checkpoint and forward-pass helpers shared by the scripts |
| `train_network.py` | Training for MSCR and the baselines (`--mask-dir`, `--split-file`, …) |
| `evaluate_top1.py` | Top-1 success, grounding accuracy, error breakdown and mask IoU on seen / unseen |
| `mine_candidates.py` | Generates candidates, labels them and caches CLIP features; `--time` measures cost |
| `reranker.py` | Trains / evaluates the scorer from the caches; tunes α and K on validation |
| `script/` | `train_mscr.sh`, `rerank.sh`, `train_baselines_fixed.sh` |
| `tests/` | `python -m unittest tests/test_mscr.py tests/test_baseline_gradients.py` |

**Fixes to the released code** (applied to every baseline here):
- the encoders of `lgrconvnet3`, `lggcnn` and `lragt` were detached before text fusion, so they never trained;
- training read the test split list; it now takes `--split-file` with the official train list;
- validation used augmented images;
- the models were fed object names instead of the part instructions.

## Setup

```bash
conda create -n lgd python=3.9 && conda activate lgd
pip install torch==2.1.2 torchvision==0.16.2   # any CUDA build
pip install -r requirements.txt
```

**Data.** Download our exact subset: [Google Drive](https://drive.google.com/drive/folders/1OVzbPBT-o6wIfGOrJoRBVnVCDRsXID_k?usp=sharing). Then unzip it in the repository root:
```bash
unzip mscr_data.zip   # -> data/grasp-anything/{image,prompt}, data/grasp-anything++/seen/{positive_grasp,
                      #    grasp_instructions,part_mask,negative_grasp}
```
The subset contains the Grasp-Anything++ images, part instructions, positive and negative grasps and part masks for
LGD's official split lists in `split/grasp-anything++/`. That is 28,621 training samples, used as 25,758 for
training and 2,863 for validation, plus 1,376 seen and 601 unseen test samples. To rebuild it from the
Hugging Face release (`airvlab/Grasp-Anything`, `airvlab/Grasp-Anything-pp`), run `extract_subset.py`.

## Reproduce

```bash
M=data/grasp-anything++/seen/part_mask
# 1. Grasp network (~3 h on one A5000)
python train_network.py --dataset grasp-anywhere --dataset-path data/grasp-anything \
  --add-file-path data/grasp-anything++/seen --split-file split/grasp-anything++/train/seen.obj \
  --use-depth 0 --seen 1 --channel-size 32 --epochs 50 --batch-size 8 --batches-per-epoch 1000 --split 0.9 \
  --random-seed 123 --mask-dir $M --network mscr --description mscr_s123

# 2. Top-1 success on test seen / unseen
python evaluate_top1.py --network logs/*_mscr_s123/best_model.pt --mask-dir $M --alphas 0 --out results/mscr.json

# 3. Re-ranking: candidate mining, scorer training, evaluation, ablations and cost
NET=$(ls logs/*_mscr_s123/best_model.pt) MASK_DIR=$M NEG_DIR=data/grasp-anything++/seen/negative_grasp \
  NAME=mscr GPUS="0 1 2" bash script/rerank.sh
```

The ablation variants and the baselines: `bash script/train_mscr.sh` and `bash script/train_baselines_fixed.sh`
(details in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)).
