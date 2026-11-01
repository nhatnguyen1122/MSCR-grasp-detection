"""Decoding grasp maps into grasps: mask gating and candidate generation.

A candidate is a row [row, col, angle, length, width] in the 224 x 224 network frame, the same parametrisation
as utils.dataset_processing.grasp.Grasp (angle in radians, anticlockwise from horizontal; `length` is the gripper
opening read from the width map, `width` = length / 2 as in the repo's detect_grasps).
"""
import numpy as np
from skimage.feature import peak_local_max

from utils.dataset_processing.grasp import Grasp

REPO_MIN_DISTANCE = 20   # detect_grasps() defaults: keep top-1 decoding identical to the repo protocol
REPO_THRESHOLD = 0.2


# ---------------------------------------------------------------------------------------------- gating
def gate_quality(q, mask_prob, alpha=1.0):
    """Q~ = Q * sigmoid(M)^alpha. alpha = 0 (or no mask head) returns Q unchanged."""
    if mask_prob is None or alpha == 0:
        return q
    return q * np.power(np.clip(mask_prob, 0.0, 1.0), alpha)


def find_peaks(q, q_gated, k, min_distance=REPO_MIN_DISTANCE, threshold=REPO_THRESHOLD):
    """Local maxima of the gated map, ranked by Q~, keeping only pixels whose *ungated* Q >= threshold.

    Applying the threshold to Q (not Q~) keeps the repo's "confident grasps only" rule while letting the mask
    re-rank peaks; with alpha = 0 this is exactly detect_grasps().
    """
    m = np.where(q >= threshold, q_gated, 0.0)
    if m.max() <= 0:
        return np.zeros((0, 2), dtype=int)
    return peak_local_max(m, min_distance=min_distance, threshold_abs=1e-8, num_peaks=k)


def grasp_at(maps, r, c):
    """Read angle and width at a pixel, as detect_grasps() does."""
    length = float(maps['width'][r, c])
    return [float(r), float(c), float(maps['ang'][r, c]), length, length / 2]


def to_grasp(cand):
    r, c, a, length, width = cand[:5]
    return Grasp(np.array([r, c]), a, length, width)


class GTIndex:
    """Ground-truth rectangles of one sample, with angles and bounding boxes precomputed.

    `success(cand)` is exactly `to_grasp(cand).max_iou(gt) > 0.25` (the repo criterion: raster IoU with the
    30-degree angle test) but only rasterises pairs that pass the angle test and whose bounding boxes overlap,
    and stops at the first match.
    """

    def __init__(self, gtbb):
        self.grs = list(gtbb.grs)
        pts = np.array([g.points for g in self.grs], dtype=np.float64).reshape(-1, 4, 2)
        self.lo, self.hi = pts.min(1), pts.max(1)
        self.ang = np.array([g.angle for g in self.grs])

    def _candidates(self, gr, angle_threshold):
        lo, hi = gr.points.min(0), gr.points.max(0)
        d = np.abs((gr.angle - self.ang + np.pi / 2) % np.pi - np.pi / 2)
        ok = (d <= angle_threshold) & np.all(self.lo <= hi + 1, 1) & np.all(self.hi + 1 >= lo, 1)
        return np.nonzero(ok)[0]

    def success(self, cand, threshold=0.25):
        gr = to_grasp(cand).as_gr
        return any(gr.iou(self.grs[j]) > threshold for j in self._candidates(gr, np.pi / 6))

    def max_overlap(self, gr):
        """Max IoU ignoring the angle (used to keep language negatives off the prompted part)."""
        js = self._candidates(gr, np.pi)
        return max((gr.iou(self.grs[j], angle_threshold=np.pi) for j in js), default=0.0)


def score_at(maps, q_gated, cand):
    """Network signals at the candidate centre: Q~ and sigmoid(M) (1 when there is no mask head)."""
    h, w = q_gated.shape
    r = int(np.clip(round(cand[0]), 0, h - 1))
    c = int(np.clip(round(cand[1]), 0, w - 1))
    m = 1.0 if maps.get('mask') is None else float(maps['mask'][r, c])
    return float(q_gated[r, c]), m


# ---------------------------------------------------------------------------------------------- strategies
def top1(maps, alpha=1.0):
    """Top-1 decoding with the repo's peak settings; returns [] when no confident peak exists."""
    qg = gate_quality(maps['q'], maps.get('mask'), alpha)
    pk = find_peaks(maps['q'], qg, 1)
    return [grasp_at(maps, r, c) for r, c in pk]


def topk_peaks(maps, k, alpha=1.0, min_distance=8, threshold=0.0):
    """Top-K: the k highest local maxima of Q~, at least min_distance px apart."""
    qg = gate_quality(maps['q'], maps.get('mask'), alpha)
    pk = find_peaks(maps['q'], qg, k, min_distance=min_distance, threshold=threshold)
    return [grasp_at(maps, r, c) for r, c in pk]


# Mildest variants first, so the first K of the list is a sensible K-budget.
PERTURBATIONS = [('angle', 15), ('angle', -15), ('width', 0.8), ('width', 1.2),
                 ('along', 1), ('along', -1), ('across', 1), ('across', -1),
                 ('angle', 30), ('angle', -30)]


def perturb(cand, kind, value, shift_px=4.0):
    r, c, a, length, width = cand[:5]
    if kind == 'angle':
        a = a + np.deg2rad(value)
        a = (a + np.pi / 2) % np.pi - np.pi / 2
    elif kind == 'width':
        length = length * value
        width = length / 2
    elif kind == 'along':   # along the gripper (closing) axis; direction (d_row, d_col) = (-sin a, cos a)
        r, c = r - value * shift_px * np.sin(a), c + value * shift_px * np.cos(a)
    elif kind == 'across':
        r, c = r - value * shift_px * np.cos(a), c - value * shift_px * np.sin(a)
    return [r, c, a, length, width]


def perturbations(cand, shift_px=4.0):
    return [perturb(cand, k, v, shift_px) for k, v in PERTURBATIONS]


def perturb_top(maps, k, alpha=1.0, shift_px=4.0):
    """Perturbation: the top peak followed by its angle / width / centre variants."""
    base = topk_peaks(maps, 1, alpha)
    if not base:
        return []
    return ([base[0]] + perturbations(base[0], shift_px))[:k]


def hybrid(maps, k, alpha=1.0, n_peaks=5, min_distance=8, shift_px=4.0):
    """Hybrid (the paper's candidates): top peaks first (location diversity), then their perturbations round-robin (geometry diversity)."""
    peaks = topk_peaks(maps, n_peaks, alpha, min_distance=min_distance)
    out = list(peaks)
    variants = [perturbations(p, shift_px) for p in peaks]
    for j in range(len(PERTURBATIONS)):
        for v in variants:
            out.append(v[j])
    return out[:k]


def sample_quality(maps, k, alpha=1.0, tau=0.1, rng=None):
    """Sampling: the top peak, then centres drawn without replacement from p(i, j) ~ exp(Q~_ij / tau)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    qg = gate_quality(maps['q'], maps.get('mask'), alpha)
    out = topk_peaks(maps, 1, alpha)
    logits = qg.ravel() / tau
    p = np.exp(logits - logits.max())
    p /= p.sum()
    n = min(k - len(out), int((p > 0).sum()))
    if n > 0:
        idx = rng.choice(p.size, size=n, replace=False, p=p)
        h, w = qg.shape
        out += [grasp_at(maps, i // w, i % w) for i in idx]
    return out[:k]


# ---------------------------------------------------------------------------------------------- test-time augmentation
# Dihedral views: (number of anticlockwise 90-degree rotations, horizontal flip). View 0 is the identity.
TTA_VIEWS = [(0, False), (1, False), (2, False), (3, False), (0, True), (1, True), (2, True), (3, True)]


def tta_forward_transform(x, k, flip):
    """Apply a view to a B x C x H x W tensor: rotate k times by 90 deg anticlockwise, then flip columns."""
    import torch
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def tta_map_back(cand, k, flip, size):
    """Map a candidate found in a transformed view back to the original frame."""
    r, c, a, length, width = cand[:5]
    dr, dc = -np.sin(a), np.cos(a)
    if flip:
        c, dc = size - 1 - c, -dc
    for _ in range(k):  # invert one anticlockwise rotation: new (r', c') = (W-1-c, r)  ->  (r, c) = (c', W-1-r')
        r, c = c, size - 1 - r
        dr, dc = dc, -dr
    a = np.arctan2(-dr, dc)
    a = (a + np.pi / 2) % np.pi - np.pi / 2
    return [r, c, a, length, width]
