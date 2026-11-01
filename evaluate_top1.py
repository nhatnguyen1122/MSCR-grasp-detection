"""Evaluate a grasp network with top-1 decoding (no re-ranking): MSCR variants and the baselines.

For each split list it reports, per mask-gating exponent alpha (alpha = 0 is ungated decoding):
  success      IoU > 0.25 and angle error < 30 deg against the instructed part's ground truth (repo criterion)
  grounding    share of top-1 grasps whose centre lies inside the instructed part's mask (needs --mask-dir)
  errors       failures split into no_peak / grounding (centre outside the part) / geometry (inside, wrong IoU/angle)
plus the predicted mask IoU (networks with a mask head) and inference time per image.
With both a seen and an unseen list, H = 2 S U / (S + U) is added.

Example:
  python evaluate_top1.py --network logs/<run>/best_model.pt \
      --split-files split/grasp-anything++/test/seen.obj split/grasp-anything++/test/unseen.obj \
      --mask-dir data/grasp-anything++/seen/part_mask --alphas 0 --out results/mscr.json
Works for the baselines too (lgrconvnet3, lggcnn, clipfusion checkpoints): alpha is then ignored.
"""
import argparse
import json
import logging
import os
import time
from types import SimpleNamespace

import numpy as np
import torch

from mscr.common import build_dataset, forward_maps, gt_mask, harmonic, in_mask, load_network, make_loader, \
    sync_partition
from mscr.decode import top1, to_grasp
from hardware.device import get_device

logging.basicConfig(level=logging.INFO, format='%(message)s')


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--network', nargs='+', required=True, help='Checkpoint(s) saved by train_network.py')
    p.add_argument('--split-files', nargs='+', default=['split/grasp-anything++/test/seen.obj',
                                                        'split/grasp-anything++/test/unseen.obj'])
    p.add_argument('--part', default='all', choices=['all', 'train', 'val'])
    p.add_argument('--split', type=float, default=0.9)
    p.add_argument('--alphas', type=float, nargs='+', default=[0.0, 1.0], help='Mask-gating exponents')
    p.add_argument('--dataset', default='grasp-anywhere')
    p.add_argument('--dataset-path', default='data/grasp-anything')
    p.add_argument('--add-file-path', default='data/grasp-anything++/seen')
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--use-depth', type=int, default=0)
    p.add_argument('--use-rgb', type=int, default=1)
    p.add_argument('--max-samples', type=int, default=None)
    p.add_argument('--mask-dir', default=None)
    p.add_argument('--mask-type', default='part', choices=['part', 'object'])
    p.add_argument('--require-mask', type=int, default=None)
    p.add_argument('--limit', type=int, default=None)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--out', default=None, help='JSON file for the results')
    p.add_argument('--cpu', action='store_true')
    return p.parse_args()


def split_name(path):
    return os.path.splitext(os.path.basename(path))[0]


def evaluate_split(net, ds, idx, args, device):
    loader = make_loader(ds, idx, args)
    has_mask = None
    per_alpha = {a: {'correct': 0, 'no_peak': 0, 'grounding_err': 0, 'geometry_err': 0, 'in_mask': 0,
                     'with_gt_mask': 0} for a in args.alphas}
    mask_ious, n, t_net = [], 0, 0.0
    for x, _, didx, _, _, _, query in loader:
        x = x.to(device)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.time()
        maps = forward_maps(net, x, query)
        t_net += time.time() - t0
        for b, m in enumerate(maps):
            i = int(didx[b])
            n += 1
            has_mask = m['mask'] is not None
            gtbb = ds.get_gtbb(i)
            gm = gt_mask(ds, i)
            if has_mask and gm is not None:
                pm = m['mask'] > 0.5
                union = (pm | gm).sum()
                mask_ious.append(float((pm & gm).sum() / union) if union else 1.0)
            for a in args.alphas:
                r = per_alpha[a]
                cands = top1(m, alpha=a if has_mask else 0.0)
                inside = in_mask(gm, cands[0]) if cands else -1
                if gm is not None:
                    r['with_gt_mask'] += 1
                    r['in_mask'] += int(inside == 1)
                ok = bool(cands) and to_grasp(cands[0]).max_iou(gtbb) > 0.25
                if ok:
                    r['correct'] += 1
                elif not cands:
                    r['no_peak'] += 1
                elif inside == 0:
                    r['grounding_err'] += 1
                elif inside == 1:
                    r['geometry_err'] += 1
    out = {'n': n, 'has_mask_head': bool(has_mask), 'net_ms_per_image': 1000 * t_net / max(n, 1),
           'mask_iou': float(np.mean(mask_ious)) if mask_ious else None, 'alphas': {}}
    for a, r in per_alpha.items():
        if not has_mask and a != args.alphas[0]:
            continue  # gating is meaningless without a mask head
        failed = n - r['correct']
        out['alphas'][str(a)] = {
            'success': r['correct'] / max(n, 1), 'correct': r['correct'], 'failed': failed,
            'grounding_acc': r['in_mask'] / r['with_gt_mask'] if r['with_gt_mask'] else None,
            'errors': {'no_peak': r['no_peak'], 'grounding': r['grounding_err'], 'geometry': r['geometry_err'],
                       'unknown_no_gt_mask': failed - r['no_peak'] - r['grounding_err'] - r['geometry_err']},
        }
    return out


def main():
    args = parse_args()
    device = get_device(args.cpu)
    all_results = {}
    datasets = {}
    for sf in args.split_files:
        a = SimpleNamespace(**vars(args))
        a.split_file = sf
        if args.part != 'all':  # validation ids of the (first) checkpoint's training run
            sync_partition(a, args.network[0])
        datasets[a.split_file] = build_dataset(a)
    for ckpt in args.network:
        net = load_network(ckpt, device)
        res = {}
        for sf, (ds, idx) in datasets.items():
            name = split_name(sf)
            res[name] = evaluate_split(net, ds, idx, args, device)
            for a, r in res[name]['alphas'].items():
                logging.info('%s | %-7s alpha=%s  success %.4f (%d/%d)  grounding %s  errors %s  mask IoU %s  %.1f ms/img',
                             os.path.basename(os.path.dirname(ckpt)) or ckpt, name, a, r['success'], r['correct'],
                             res[name]['n'], r['grounding_acc'], r['errors'], res[name]['mask_iou'],
                             res[name]['net_ms_per_image'])
        if 'seen' in res and 'unseen' in res:
            res['H'] = {a: harmonic(res['seen']['alphas'][a]['success'], res['unseen']['alphas'][a]['success'])
                        for a in res['seen']['alphas'] if a in res['unseen']['alphas']}
            logging.info('H: %s', res['H'])
        all_results[ckpt] = res
        del net
        torch.cuda.empty_cache()
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(all_results, f, indent=2)
        logging.info('Saved %s', args.out)


if __name__ == '__main__':
    main()
