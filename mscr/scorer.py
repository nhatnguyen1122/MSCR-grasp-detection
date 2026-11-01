"""Re-ranking scorer.

  make_crop      rotated square crop around a candidate (gripper axis horizontal), rectangle outline drawn on it
  ClipFeaturizer frozen CLIP ViT-B/32 features of crops, full images and prompts (L2-normalised)
  build_inputs   assemble [e_c, e_I, e_t, e_c * e_t, geometry, network signals Q~ and sigma(M)] with switchable input groups
  Scorer         MLP -> one logit
  listwise_loss  -log( sum_pos e^s / sum_all e^s ) per (image, instruction) group (alternative to BCE)
"""
import clip
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mscr.decode import to_grasp

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

# Candidate kinds stored in the caches
KIND_GEN, KIND_GT, KIND_DATA_NEG, KIND_LANG_NEG, KIND_RAND_NEG = 0, 1, 2, 3, 4
KIND_NAMES = {KIND_GEN: 'network', KIND_GT: 'gt_positive', KIND_DATA_NEG: 'dataset_negative',
              KIND_LANG_NEG: 'language_negative', KIND_RAND_NEG: 'random_negative'}


# ---------------------------------------------------------------------------------------------- crops
def crop_transform(cand, out_size=224, context=2.0, min_side=32.0):
    """2 x 3 affine mapping the image to a crop centred on the grasp with the gripper axis horizontal."""
    r, c, a, length, width = cand[:5]
    side = max(context * max(length, width), min_side)
    # The gripper axis has image direction (dx, dy) = (cos a, -sin a), i.e. it is a anticlockwise from horizontal
    # on screen; cv2 rotates anticlockwise for positive angles, so rotate by -a to make it horizontal.
    m = cv2.getRotationMatrix2D((float(c), float(r)), -np.rad2deg(a), out_size / side)
    m[0, 2] += out_size / 2 - c
    m[1, 2] += out_size / 2 - r
    return m


def make_crop(rgb, cand, out_size=224, context=2.0, draw=True):
    """rgb: H x W x 3 uint8 (the 224 network-frame image)."""
    m = crop_transform(cand, out_size, context)
    crop = cv2.warpAffine(rgb, m, (out_size, out_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if draw:
        pts = to_grasp(cand).as_gr.points[:, ::-1].astype(np.float64)  # (row, col) -> (x, y)
        pts = (m[:, :2] @ pts.T + m[:, 2:]).T
        cv2.polylines(crop, [np.round(pts).astype(np.int32)], True, (0, 255, 0), 2)
    return crop


def geometry_features(cands, size=224.0, width_norm=150.0):
    """Normalised x, y, length, width, cos 2a, sin 2a."""
    c = np.asarray(cands, dtype=np.float32).reshape(-1, 5)
    return np.stack([c[:, 1] / size, c[:, 0] / size, c[:, 3] / width_norm, c[:, 4] / width_norm,
                     np.cos(2 * c[:, 2]), np.sin(2 * c[:, 2])], 1).astype(np.float32)


# ---------------------------------------------------------------------------------------------- CLIP
class ClipFeaturizer:
    def __init__(self, device, version='ViT-B/32'):
        self.device = device
        self.model, _ = clip.load(version, device=device, jit=False)
        self.model.eval()
        self.dtype = next(self.model.parameters()).dtype

    @torch.no_grad()
    def images(self, imgs, batch=256):
        """imgs: list / array of 224 x 224 x 3 uint8."""
        feats = []
        for i in range(0, len(imgs), batch):
            x = np.stack(imgs[i:i + batch]).astype(np.float32) / 255.0
            x = (x - CLIP_MEAN) / CLIP_STD
            x = torch.from_numpy(x).permute(0, 3, 1, 2).to(self.device, self.dtype)
            f = self.model.encode_image(x).float()
            feats.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(feats).numpy().astype(np.float16) if feats else np.zeros((0, 512), np.float16)

    @torch.no_grad()
    def texts(self, texts, batch=256):
        feats = []
        for i in range(0, len(texts), batch):
            tok = clip.tokenize(list(texts[i:i + batch]), truncate=True).to(self.device)
            f = self.model.encode_text(tok).float()
            feats.append(F.normalize(f, dim=-1).cpu())
        return torch.cat(feats).numpy().astype(np.float16)


# ---------------------------------------------------------------------------------------------- inputs
INPUT_GROUPS = ('crop', 'prompt', 'image', 'geom')  # switchable for the input ablations


def build_inputs(cache, cand_idx, groups):
    """Feature matrix for the candidates `cand_idx` of a cache (see mine_candidates.py for the cache layout)."""
    s = cache['cand_sample'][cand_idx]
    parts = []
    e_c = torch.from_numpy(cache['crop_feat'][cand_idx].astype(np.float32))
    if 'crop' in groups:
        parts.append(e_c)
    if 'image' in groups:
        parts.append(torch.from_numpy(cache['image_feat'][s].astype(np.float32)))
    if 'prompt' in groups:
        e_t = torch.from_numpy(cache['text_feat'][s].astype(np.float32))
        parts.append(e_t)
        if 'crop' in groups:
            parts.append(e_c * e_t * 16.0)  # "does the crop match the prompt"; x16 keeps it on the scale of the rest
    if 'geom' in groups:
        parts.append(torch.from_numpy(geometry_features(cache['cand'][cand_idx])))
        parts.append(torch.from_numpy(np.stack([cache['cand_q'][cand_idx], cache['cand_m'][cand_idx]], 1)
                                      .astype(np.float32)))
    return torch.cat(parts, 1)


def input_dim(groups):
    d = 0
    d += 512 if 'crop' in groups else 0
    d += 512 if 'image' in groups else 0
    d += 512 if 'prompt' in groups else 0
    d += 512 if ('prompt' in groups and 'crop' in groups) else 0
    d += 8 if 'geom' in groups else 0
    return d


class Scorer(nn.Module):
    def __init__(self, groups, hidden=512, dropout=0.1):
        super().__init__()
        self.groups = tuple(groups)
        self.mlp = nn.Sequential(nn.LayerNorm(input_dim(groups)), nn.Linear(input_dim(groups), hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden // 2), nn.GELU(),
                                 nn.Linear(hidden // 2, 1))

    def forward(self, x):
        return self.mlp(x).squeeze(-1)


def listwise_loss(scores, labels, valid):
    """scores, labels, valid: G x L (padded groups). Groups without a positive are skipped."""
    neg_inf = torch.finfo(scores.dtype).min
    s_all = scores.masked_fill(~valid, neg_inf)
    s_pos = scores.masked_fill(~(valid & (labels > 0.5)), neg_inf)
    has_pos = (valid & (labels > 0.5)).any(1)
    if not has_pos.any():
        return scores.sum() * 0
    loss = torch.logsumexp(s_all, 1) - torch.logsumexp(s_pos, 1)
    return loss[has_pos].mean()
