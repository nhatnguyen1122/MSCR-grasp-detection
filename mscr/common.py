"""Shared helpers: datasets with the training partition, checkpoint loading, network forward -> maps."""
import json
import logging
import os
import pickle
import types

import numpy as np
import torch
import torch.nn as nn

from inference.post_process import post_process_output
from utils.data import get_dataset


def add_dataset_args(parser):
    parser.add_argument('--dataset', type=str, default='grasp-anywhere')
    parser.add_argument('--dataset-path', type=str, default='data/grasp-anything')
    parser.add_argument('--add-file-path', type=str, default='data/grasp-anything++/seen')
    parser.add_argument('--split-file', type=str, required=True,
                        help='Pickled id list (train/seen.obj, test/seen.obj or test/unseen.obj)')
    parser.add_argument('--part', type=str, default='all', choices=['all', 'train', 'val'],
                        help="For the training list: 'train' = first --split fraction, 'val' = the rest "
                             "(the same partition train_network.py uses); 'all' for test lists")
    parser.add_argument('--split', type=float, default=0.9)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--use-depth', type=int, default=0)
    parser.add_argument('--use-rgb', type=int, default=1)
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Same option as in train_network.py; must match it to reproduce the partition')
    parser.add_argument('--mask-dir', type=str, default=None,
                        help='Part masks: needed for mask IoU / grounding metrics')
    parser.add_argument('--mask-type', type=str, default='part', choices=['part', 'object'])
    parser.add_argument('--require-mask', type=int, default=None,
                        help='Filter to samples with masks. Default: 1 for the training list (matches '
                             'train_network.py), 0 for test lists (every test id is scored)')
    parser.add_argument('--limit', type=int, default=None, help='Only the first N samples (debugging)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=16)


PARTITION_KEYS = ('dataset', 'dataset_path', 'add_file_path', 'split_file', 'split', 'max_samples', 'mask_dir',
                  'mask_type', 'require_mask', 'use_depth', 'use_rgb', 'input_size')


def sync_partition(args, checkpoint):
    """For --part train/val, copy the data-partition options from the run's commandline_args.json, so the
    train / val ids are exactly the ones the grasp network was trained / validated on (no leakage)."""
    if args.part == 'all':
        return
    path = os.path.join(os.path.dirname(checkpoint), 'commandline_args.json')
    if not os.path.exists(path):
        logging.warning('No %s: make sure the dataset options match the training run', path)
        return
    with open(path) as f:
        run = json.load(f)
    for k in PARTITION_KEYS:
        if k in run and getattr(args, k, None) != run[k]:
            if getattr(args, k, None) is not None:
                logging.warning('--%s: using the training run value %r instead of %r',
                                k.replace('_', '-'), run[k], getattr(args, k))
            setattr(args, k, run[k])
    if run.get('ds_shuffle') or run.get('ds_rotate'):
        raise ValueError('Training runs with --ds-shuffle / --ds-rotate are not supported for --part train/val')


def build_dataset(args):
    require_mask = args.require_mask
    if require_mask is None:
        require_mask = int(bool(args.mask_dir) and args.part != 'all')
    Dataset = get_dataset(args.dataset)
    ds = Dataset(args.dataset_path, output_size=args.input_size, ds_rotate=0, random_rotate=False,
                 random_zoom=False, include_depth=args.use_depth, include_rgb=args.use_rgb, seen=1,
                 add_file_path=args.add_file_path, split_file=args.split_file, max_samples=args.max_samples,
                 mask_dir=args.mask_dir, mask_type=args.mask_type, return_mask=False,
                 require_mask=bool(require_mask))
    idx = list(range(ds.length))
    cut = int(np.floor(args.split * ds.length))
    if args.part == 'train':
        idx = idx[:cut]
    elif args.part == 'val':
        idx = idx[cut:]
    if args.limit:
        idx = idx[:args.limit]
    shard = getattr(args, 'shard', None)
    if shard:
        k, n = (int(v) for v in shard.split('/'))
        idx = idx[k::n]
    return ds, idx


class WithRGB(torch.utils.data.Dataset):
    """Subset that also returns the un-normalised 224 x 224 uint8 image (loaded in the worker processes)."""

    def __init__(self, ds, idx):
        self.ds, self.idx = ds, idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, j):
        i = self.idx[j]
        rgb = np.ascontiguousarray(self.ds.get_rgb(i, normalise=False)).astype(np.uint8)
        return self.ds[i] + (rgb,)


def make_loader(ds, idx, args, with_rgb=False):
    sub = WithRGB(ds, idx) if with_rgb else torch.utils.data.Subset(ds, idx)
    return torch.utils.data.DataLoader(sub, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)


class _LegacyUnpickler(pickle.Unpickler):
    """Maps checkpoints saved under the project's working name (inference.models.gtv) to inference.models.mscr."""

    def find_class(self, module, name):
        if module == 'inference.models.gtv':
            from inference.models.mscr import LEGACY_CLASSES
            module, name = 'inference.models.mscr', LEGACY_CLASSES.get(name, name)
        return super().find_class(module, name)


_legacy_pickle = types.ModuleType('legacy_pickle')
_legacy_pickle.__dict__.update(pickle.__dict__)
_legacy_pickle.Unpickler = _LegacyUnpickler


def load_network(path, device):
    """Load a whole-model checkpoint saved by train_network.py."""
    net = torch.load(path, map_location=device, pickle_module=_legacy_pickle)
    net.eval()
    return net.to(device)


def set_mc_dropout(net, on=True):
    """Monte Carlo dropout: dropout layers sample in eval mode (BatchNorm stays in eval)."""
    if hasattr(net, 'set_mc_dropout'):
        net.set_mc_dropout(on)
        return
    for m in net.modules():
        if isinstance(m, nn.Dropout):
            m.train(on)
    if on and hasattr(net, 'dropout'):
        net.dropout = True


@torch.no_grad()
def forward_maps(net, x, query):
    """Run the network and post-process every sample like the repo (Gaussian smoothing, width x 150).

    Returns a list of dicts with numpy maps: q, ang, width, and mask = sigmoid(M) (None without a mask head).
    """
    if type(net).__name__ == 'CLIPFusion':  # non-standard forward: go through compute_loss with dummy targets
        z = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], device=x.device)
        p = net.compute_loss(x, [z, z, z, z], list(query), list(query))['pred']
        outs = (p['pos'], p['cos'], p['sin'], p['width'])
    else:
        outs = net(x, list(query), list(query))
    pos, cos, sin, width = outs[:4]
    mask = torch.sigmoid(outs[4]) if len(outs) > 4 else None
    maps = []
    for b in range(x.shape[0]):
        q_img, ang_img, w_img = post_process_output(pos[b:b + 1], cos[b:b + 1], sin[b:b + 1], width[b:b + 1])
        maps.append({'q': q_img, 'ang': ang_img, 'width': w_img,
                     'mask': None if mask is None else mask[b, 0].cpu().numpy()})
    return maps


def gt_mask(ds, idx):
    """Ground-truth mask at 224 x 224 (no augmentation), or None if unavailable."""
    if not getattr(ds, 'mask_dir', None):
        return None
    try:
        return ds.get_mask(idx) > 0.5
    except FileNotFoundError:
        return None


def in_mask(mask, cand):
    if mask is None:
        return -1
    h, w = mask.shape
    r = int(np.clip(round(cand[0]), 0, h - 1))
    c = int(np.clip(round(cand[1]), 0, w - 1))
    return int(mask[r, c])


def harmonic(seen, unseen):
    return 2 * seen * unseen / (seen + unseen) if (seen + unseen) > 0 else 0.0
