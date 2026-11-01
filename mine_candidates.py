"""Re-ranking data: run a trained grasp network, produce candidates, label them, cache frozen-CLIP features.

One cache (.npz) per (checkpoint, split list, strategy). The scorer is trained and evaluated purely from caches,
so every scorer ablation takes minutes, and a candidate strategy is evaluated by mining the test lists with it.

Strategies:  top1 | topk (peaks) | perturb (best peak perturbed) | tta (flip / rotation views) | mcdropout |
             sample (p ~ exp(Q~ / tau)) | hybrid (5 best peaks, then their perturbations: the paper's)
Candidates are stored in network order (cand_rank), so oracle / reranked success at any K <= --k is the
first K of the list.

--extras adds the scorer's training examples: ground-truth positives, language negatives (positive grasps of
other parts / objects of the same image), dataset negatives (--neg-dir, Grasp-Anything++ grasp_label_negative)
and random negatives (random rectangles that fail the success test).

Examples:
  # scorer training data: hybrid candidates on the network's training part, plus all extras
  python mine_candidates.py --network logs/<run>/best_model.pt --split-file split/grasp-anything++/train/seen.obj \
      --part train --strategy hybrid --k 20 --extras --neg-dir <negative_grasp> --out cache/mscr/train_hybrid.npz
  # validation (tunes alpha, K) and test caches
  python mine_candidates.py ... --part val --strategy hybrid --k 20 --out cache/mscr/val_hybrid.npz
  python mine_candidates.py ... --split-file split/grasp-anything++/test/seen.obj --strategy hybrid --k 20 \
      --mask-dir <part_mask> --out cache/mscr/seen_hybrid.npz
  # inference cost per image for K = 1, 5, 10, 20
  python mine_candidates.py ... --split-file split/grasp-anything++/test/seen.obj --strategy hybrid --time 100 \
      --scorer scorers/mscr/full.pt
"""
import argparse
import glob
import json
import logging
import math
import os
import time
from collections import defaultdict

import numpy as np
import torch

from mscr import decode
from mscr.common import add_dataset_args, build_dataset, forward_maps, gt_mask, in_mask, load_network, make_loader, \
    set_mc_dropout, sync_partition
from mscr.scorer import KIND_DATA_NEG, KIND_GEN, KIND_GT, KIND_LANG_NEG, KIND_RAND_NEG, ClipFeaturizer, make_crop
from hardware.device import get_device
from utils.dataset_processing.grasp import GraspRectangles

logging.basicConfig(level=logging.INFO, format='%(message)s')
STRATEGIES = ['top1', 'topk', 'perturb', 'tta', 'mcdropout', 'sample', 'hybrid']


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--network', required=True)
    add_dataset_args(p)
    p.add_argument('--strategy', default='hybrid', choices=STRATEGIES)
    p.add_argument('--k', type=int, default=20, help='Candidates per (image, prompt)')
    p.add_argument('--alpha', type=float, default=1.0, help='Mask-gating exponent for Q~ (0 = off)')
    p.add_argument('--min-distance', type=int, default=8, help='NMS distance for top-K peaks (px)')
    p.add_argument('--n-peaks', type=int, default=5, help='Hybrid: number of peaks to perturb')
    p.add_argument('--shift-px', type=float, default=4.0, help='Perturbation: centre shift (px)')
    p.add_argument('--tau', type=float, default=0.1, help='Sampling temperature (strategy sample)')
    p.add_argument('--extras', action='store_true', help='Add GT positives and dataset/language/random negatives')
    p.add_argument('--n-extra', type=int, default=5, help='Max examples of each extra kind per sample')
    p.add_argument('--neg-dir', default=None, help='Grasp-Anything++ negative grasps (<hash>_<obj>_<part>.pt)')
    p.add_argument('--context', type=float, default=2.0, help='Crop side = context x max(length, width)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default=None)
    p.add_argument('--shard', default=None, help="'k/n': mine every n-th sample starting at k (run n in parallel)")
    p.add_argument('--merge', nargs='+', default=None, help='Merge shard caches into --out and exit')
    p.add_argument('--time', type=int, default=0, help='Only time the pipeline on N images (batch 1)')
    p.add_argument('--time-k', type=int, nargs='+', default=[1, 5, 10, 20])
    p.add_argument('--scorer', default=None, help='Scorer checkpoint, included in --time measurements')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


# ---------------------------------------------------------------------------------------------- candidates
def generate(strategy, net, x, query, maps, args, rng):
    """Candidates for every sample of a batch, in network order."""
    k, a = args.k, args.alpha
    if strategy == 'top1':
        return [decode.top1(m, a)[:1] or decode.topk_peaks(m, 1, a) for m in maps]
    if strategy == 'topk':
        return [decode.topk_peaks(m, k, a, args.min_distance) for m in maps]
    if strategy == 'perturb':
        return [decode.perturb_top(m, k, a, args.shift_px) for m in maps]
    if strategy == 'hybrid':
        return [decode.hybrid(m, k, a, args.n_peaks, args.min_distance, args.shift_px) for m in maps]
    if strategy == 'sample':
        return [decode.sample_quality(m, k, a, args.tau, rng) for m in maps]
    if strategy == 'tta':
        views = decode.TTA_VIEWS
        per_view = max(1, math.ceil(k / len(views)))
        per = [[[] for _ in views] for _ in maps]
        size = x.shape[-1]
        for v, (rot, flip) in enumerate(views):
            vm = maps if v == 0 else forward_maps(net, decode.tta_forward_transform(x, rot, flip), query)
            for b, m in enumerate(vm):
                per[b][v] = [decode.tta_map_back(c, rot, flip, size)
                             for c in decode.topk_peaks(m, per_view, a, args.min_distance)]
        out = []
        for b in range(len(maps)):  # round-robin over views: the top-1 of every view first
            lst = [per[b][v][j] for j in range(per_view) for v in range(len(views)) if j < len(per[b][v])]
            out.append(lst[:k])
        return out
    if strategy == 'mcdropout':
        out = [decode.topk_peaks(m, 1, a) for m in maps]  # deterministic pass first
        set_mc_dropout(net, True)
        try:
            for _ in range(k - 1):
                for b, m in enumerate(forward_maps(net, x, query)):
                    out[b] += decode.topk_peaks(m, 1, a)
        finally:
            set_mc_dropout(net, False)
        return out
    raise ValueError(strategy)


# ---------------------------------------------------------------------------------------------- extras
class Extras:
    def __init__(self, ds, args, rng):
        self.ds, self.args, self.rng = ds, args, rng
        self.scale = args.input_size / 416.0
        self.by_image = defaultdict(list)
        for f in glob.glob(os.path.join(args.add_file_path, 'positive_grasp', '*.pt')):
            self.by_image[os.path.basename(f).split('_')[0]].append(f)
        self.n_data_neg_missing = 0

    @staticmethod
    def rect_to_cand(gr):
        g = gr.as_grasp
        return [float(g.center[0]), float(g.center[1]), float(g.angle), float(g.length), float(g.width)]

    def _pick(self, items):
        items = list(items)
        if len(items) > self.args.n_extra:
            sel = self.rng.choice(len(items), self.args.n_extra, replace=False)
            items = [items[i] for i in sel]
        return items

    def __call__(self, i, gtbb, gti):
        out = []  # (cand, kind)
        n = self.args.n_extra
        out += [(self.rect_to_cand(g), KIND_GT) for g in self._pick(gtbb.grs)]

        # Language negatives: positives of the other parts / objects in the same image, if they are elsewhere
        own = self.ds.grasp_files[i]
        others = [f for f in self.by_image[os.path.basename(own).split('_')[0]] if f != own]
        lang = []
        for f in others:
            for g in GraspRectangles.load_from_grasp_anything_file(f, scale=self.scale).grs:
                if gti.max_overlap(g) < 0.1:
                    lang.append(g)
        out += [(self.rect_to_cand(g), KIND_LANG_NEG) for g in self._pick(lang)]

        # Dataset negatives
        if self.args.neg_dir:
            nf = os.path.join(self.args.neg_dir, os.path.basename(own))
            if os.path.exists(nf):
                negs = GraspRectangles.load_from_grasp_anything_file(nf, scale=self.scale).grs
                out += [(self.rect_to_cand(g), KIND_DATA_NEG) for g in self._pick(negs)]
            else:
                self.n_data_neg_missing += 1

        # Random negatives: random rectangles anywhere in the image that fail the success test
        lengths = [g.length for g in gtbb.grs] or [40.0]
        tries = 0
        while sum(1 for _, kd in out if kd == KIND_RAND_NEG) < n and tries < 10 * n:
            tries += 1
            s = self.args.input_size
            ln = float(self.rng.choice(lengths) * self.rng.uniform(0.6, 1.4))
            c = [float(self.rng.uniform(0, s - 1)), float(self.rng.uniform(0, s - 1)),
                 float(self.rng.uniform(-np.pi / 2, np.pi / 2)), ln, ln / 2]
            if not gti.success(c):
                out.append((c, KIND_RAND_NEG))
        return out


# ---------------------------------------------------------------------------------------------- mining
def mine(args):
    device = get_device(args.cpu)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    ds, idx = build_dataset(args)
    logging.info('%d samples from %s (part=%s)', len(idx), args.split_file, args.part)
    net = load_network(args.network, device)
    clipf = ClipFeaturizer(device)
    extras = Extras(ds, args, rng) if args.extras else None

    S = defaultdict(list)   # per sample
    C = defaultdict(list)   # per candidate
    t0 = time.time()
    for bi, (x, _, didx, _, _, _, query, rgbs) in enumerate(make_loader(ds, idx, args, with_rgb=True)):
        x = x.to(device)
        maps = forward_maps(net, x, query)
        cands = generate(args.strategy, net, x, query, maps, args, rng)
        crops, full = [], []
        for b, m in enumerate(maps):
            i = int(didx[b])
            s = len(S['index'])
            gtbb = ds.get_gtbb(i)
            gti = decode.GTIndex(gtbb)
            gm = gt_mask(ds, i)
            rgb = rgbs[b].numpy()
            qg = decode.gate_quality(m['q'], m.get('mask'), args.alpha)
            items = [(c, KIND_GEN) for c in cands[b]]
            if extras is not None:
                items += extras(i, gtbb, gti)
            rank = 0
            for c, kind in items:
                q, mp = decode.score_at(m, qg, c)
                if kind == KIND_GEN:
                    label = int(gti.success(c))
                    C['rank'].append(rank)
                    rank += 1
                else:
                    label = int(kind == KIND_GT)
                    C['rank'].append(-1)
                C['cand'].append(c)
                C['sample'].append(s)
                C['kind'].append(kind)
                C['label'].append(label)
                C['q'].append(q)
                C['m'].append(mp)
                C['inmask'].append(in_mask(gm, c))
                crops.append(make_crop(rgb, c, context=args.context))
            S['index'].append(i)
            S['name'].append(os.path.basename(ds.grasp_files[i])[:-3])
            S['prompt'].append(query[b])
            full.append(rgb)
        C['crop_feat'].append(clipf.images(crops))
        S['image_feat'].append(clipf.images(full))
        S['text_feat'].append(clipf.texts(list(query)))
        if bi % 20 == 0:
            logging.info('batch %d: %d samples, %d candidates, %.1fs', bi, len(S['index']), len(C['sample']),
                         time.time() - t0)

    cache = {
        'sample_index': np.array(S['index'], np.int32), 'sample_name': np.array(S['name']),
        'prompt': np.array(S['prompt']),
        'image_feat': np.concatenate(S['image_feat']), 'text_feat': np.concatenate(S['text_feat']),
        'cand': np.array(C['cand'], np.float32).reshape(-1, 5), 'cand_sample': np.array(C['sample'], np.int32),
        'cand_rank': np.array(C['rank'], np.int16), 'cand_kind': np.array(C['kind'], np.int8),
        'cand_label': np.array(C['label'], np.int8), 'cand_q': np.array(C['q'], np.float32),
        'cand_m': np.array(C['m'], np.float32), 'cand_inmask': np.array(C['inmask'], np.int8),
        'crop_feat': np.concatenate(C['crop_feat']),
        'meta': np.array(json.dumps({k: v for k, v in vars(args).items()})),
    }
    gen = cache['cand_kind'] == KIND_GEN
    n_s = len(cache['sample_index'])
    solved = np.zeros(n_s, bool)
    solved[cache['cand_sample'][gen & (cache['cand_label'] == 1)]] = True
    top1 = np.zeros(n_s, bool)
    top1[cache['cand_sample'][gen & (cache['cand_rank'] == 0) & (cache['cand_label'] == 1)]] = True
    logging.info('Mined %d samples, %d candidates (%d from the network) in %.1fs | rank-0 success %.4f | oracle@%d %.4f',
                 n_s, len(gen), gen.sum(), time.time() - t0, top1.mean(), args.k, solved.mean())
    for kind in np.unique(cache['cand_kind']):
        sel = cache['cand_kind'] == kind
        logging.info('  kind %d: %d candidates, %.3f positive', kind, sel.sum(), cache['cand_label'][sel].mean())
    if extras is not None and args.neg_dir:
        logging.info('  dataset negatives missing for %d samples', extras.n_data_neg_missing)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    np.savez(args.out, **cache)
    logging.info('Saved %s', args.out)


def merge(paths, out):
    """Concatenate shard caches (re-indexing candidates to the merged sample list)."""
    parts = [dict(np.load(p)) for p in paths]
    per_sample = ['sample_index', 'sample_name', 'prompt', 'image_feat', 'text_feat']
    per_cand = ['cand', 'cand_rank', 'cand_kind', 'cand_label', 'cand_q', 'cand_m', 'cand_inmask', 'crop_feat']
    merged, offset, samples = {k: [] for k in per_sample + per_cand + ['cand_sample']}, 0, 0
    for z in parts:
        for k in per_sample + per_cand:
            merged[k].append(z[k])
        merged['cand_sample'].append(z['cand_sample'] + offset)
        offset += len(z['sample_index'])
    merged = {k: np.concatenate(v) for k, v in merged.items()}
    merged['meta'] = parts[0]['meta']
    np.savez(out, **merged)
    logging.info('Merged %d shards: %d samples, %d candidates -> %s', len(parts), offset, len(merged['cand']), out)


# ---------------------------------------------------------------------------------------------- timing
def time_pipeline(args):
    """Wall time per image (batch 1): grasp network + candidates + crops + CLIP + scorer, for each K."""
    from mscr.scorer import Scorer, build_inputs  # noqa: F401
    device = get_device(args.cpu)
    rng = np.random.default_rng(args.seed)
    ds, idx = build_dataset(args)
    idx = idx[:args.time]
    net = load_network(args.network, device)
    clipf = ClipFeaturizer(device)
    scorer = None
    if args.scorer:
        ck = torch.load(args.scorer, map_location=device)
        scorer = Scorer(ck['groups']).to(device).eval()
        scorer.load_state_dict(ck['state_dict'])
    args.batch_size = 1
    sync = (lambda: torch.cuda.synchronize()) if device.type == 'cuda' else (lambda: None)
    results = {}
    for k in args.time_k:
        args.k = k
        times, warm = [], 3
        for j, (x, _, didx, _, _, _, query) in enumerate(make_loader(ds, idx, args)):
            x = x.to(device)
            sync()
            t0 = time.time()
            maps = forward_maps(net, x, query)
            strategy = 'top1' if k == 1 else args.strategy
            cands = generate(strategy, net, x, query, maps, args, rng)[0]
            if k > 1 and cands:
                rgb = np.ascontiguousarray(ds.get_rgb(int(didx[0]), normalise=False)).astype(np.uint8)
                e_c = clipf.images([make_crop(rgb, c, context=args.context) for c in cands])
                e_i = clipf.images([rgb])
                e_t = clipf.texts(list(query))
                if scorer is not None:
                    qg = decode.gate_quality(maps[0]['q'], maps[0].get('mask'), args.alpha)
                    qm = np.array([decode.score_at(maps[0], qg, c) for c in cands], np.float32)
                    cache = {'crop_feat': e_c, 'image_feat': e_i, 'text_feat': e_t,
                             'cand_sample': np.zeros(len(cands), np.int32), 'cand': np.array(cands, np.float32),
                             'cand_q': qm[:, 0], 'cand_m': qm[:, 1]}
                    with torch.no_grad():
                        scorer(build_inputs(cache, np.arange(len(cands)), scorer.groups).to(device))
            sync()
            if j >= warm:
                times.append(time.time() - t0)
        results[k] = 1000 * float(np.mean(times))
        logging.info('K=%2d (%s): %.1f ms / image over %d images', k, 'top1' if k == 1 else args.strategy,
                     results[k], len(times))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'strategy': args.strategy, 'ms_per_image': results, 'scorer': args.scorer}, f, indent=2)


if __name__ == '__main__':
    a = parse_args()
    if a.merge:
        merge(a.merge, a.out)
        raise SystemExit
    sync_partition(a, a.network)
    if a.time:
        time_pipeline(a)
    else:
        assert a.out, '--out is required'
        mine(a)
