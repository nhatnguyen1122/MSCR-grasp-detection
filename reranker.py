"""Train and evaluate the re-ranking scorer from caches written by mine_candidates.py.

train:  python reranker.py train --train cache/mscr/train_hybrid.npz --val cache/mscr/val_hybrid.npz \
            --out scorers/mscr/full.pt
          Ablations: --inputs crop | crop prompt | crop prompt image | crop image geom (no instruction)
                     --negatives data | data hard | data hard lang (default: data random hard lang;
                     data = Grasp-Anything++ negatives, if mined with --neg-dir; random = random failing
                     rectangles; hard = failed candidates; lang = grasps of other parts of the image)
                     --loss bce | listwise

eval:   python reranker.py eval --scorer scorers/mscr/full.pt --tune cache/mscr/val_hybrid.npz \
            --test cache/mscr/seen_hybrid.npz cache/mscr/unseen_hybrid.npz --out results/rerank_full.json
          Tunes alpha and K on the validation cache, then reports for each test cache and K:
            top1          first candidate of the grasp network (no re-ranking)
            oracle        any of the first K candidates correct (ceiling)
            reranked      argmax_g alpha Q~(g) + (1 - alpha) sigmoid(s(g)) over the first K
            scorer_only   the same with alpha = 0
          plus grounding accuracy and the grounding / geometry error split of the chosen grasp.
          Caches named *seen* and *unseen* give H; `main` is the row tuned on validation.
"""
import argparse
import json
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from mscr.common import harmonic
from mscr.scorer import (INPUT_GROUPS, KIND_DATA_NEG, KIND_GEN, KIND_GT, KIND_LANG_NEG, KIND_RAND_NEG, Scorer,
                        build_inputs, listwise_loss)

logging.basicConfig(level=logging.INFO, format='%(message)s')
NEG_KINDS = {'data': KIND_DATA_NEG, 'random': KIND_RAND_NEG, 'lang': KIND_LANG_NEG}


def load_cache(path):
    z = np.load(path, allow_pickle=False)
    return {k: z[k] for k in z.files}


# ---------------------------------------------------------------------------------------------- selection
def training_candidates(cache, negatives):
    kind, label = cache['cand_kind'], cache['cand_label']
    keep = (kind == KIND_GT) | ((kind == KIND_GEN) & (label == 1))
    if 'hard' in negatives:
        keep |= (kind == KIND_GEN) & (label == 0)
    for n in negatives:
        if n in NEG_KINDS:
            keep |= kind == NEG_KINDS[n]
    return np.nonzero(keep)[0]


def network_candidates(cache, k):
    sel = (cache['cand_kind'] == KIND_GEN) & (cache['cand_rank'] < k)
    return np.nonzero(sel)[0]


@torch.no_grad()
def score_all(scorer, cache, idx, device, batch=8192):
    out = []
    for i in range(0, len(idx), batch):
        out.append(torch.sigmoid(scorer(build_inputs(cache, idx[i:i + batch], scorer.groups).to(device))).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def select(cache, k, alpha, probs_by_cand):
    """Per sample: (chosen candidate index or -1, oracle solved) among the first k candidates of the grasp network."""
    n = len(cache['sample_index'])
    idx = network_candidates(cache, k)
    comb = alpha * cache['cand_q'][idx] + (1 - alpha) * probs_by_cand[idx]
    chosen = -np.ones(n, np.int64)
    best = np.full(n, -np.inf)
    oracle = np.zeros(n, bool)
    for j, c in enumerate(idx):
        s = cache['cand_sample'][c]
        oracle[s] |= cache['cand_label'][c] == 1
        if comb[j] > best[s]:
            best[s], chosen[s] = comb[j], c
    return chosen, oracle


def summarize(cache, chosen):
    ok = np.array([c >= 0 and cache['cand_label'][c] == 1 for c in chosen])
    inside = np.array([cache['cand_inmask'][c] if c >= 0 else -1 for c in chosen])
    known = inside >= 0
    fail = ~ok
    return {
        'success': float(ok.mean()),
        'grounding_acc': float((inside[known] == 1).mean()) if known.any() else None,
        'errors': {'no_candidate': int((chosen < 0).sum()),
                   'grounding': int((fail & (inside == 0)).sum()),
                   'geometry': int((fail & (inside == 1)).sum()),
                   'unknown_no_gt_mask': int((fail & (chosen >= 0) & ~known).sum())},
    }


def evaluate(scorer, cache, ks, alpha, device):
    probs = np.zeros(len(cache['cand_kind']), np.float32)
    gen = np.nonzero(cache['cand_kind'] == KIND_GEN)[0]
    if scorer is not None:
        probs[gen] = score_all(scorer, cache, gen, device)
    res = {'n': int(len(cache['sample_index']))}
    top1, _ = select(cache, 1, 1.0, probs)
    res['top1'] = summarize(cache, top1)
    for k in ks:
        ch, oracle = select(cache, k, alpha, probs)
        ch0, _ = select(cache, k, 0.0, probs)
        res['K=%d' % k] = {'oracle': float(oracle.mean()), 'reranked': summarize(cache, ch),
                           'scorer_only': summarize(cache, ch0)}
    return res, probs


# ---------------------------------------------------------------------------------------------- training
def make_groups(cache, idx):
    """Pad candidates per sample into G x L index matrix (-1 = padding)."""
    s = cache['cand_sample'][idx]
    order = np.argsort(s, kind='stable')
    idx, s = idx[order], s[order]
    uniq, start, counts = np.unique(s, return_index=True, return_counts=True)
    G = np.full((len(uniq), counts.max()), -1, np.int64)
    for g, (st, ct) in enumerate(zip(start, counts)):
        G[g, :ct] = idx[st:st + ct]
    return G


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr, va = load_cache(args.train), load_cache(args.val)
    groups = tuple(g for g in INPUT_GROUPS if g in args.inputs)
    idx = training_candidates(tr, args.negatives)
    y_all = torch.from_numpy(tr['cand_label'].astype(np.float32))
    pos, neg = idx[tr['cand_label'][idx] == 1], idx[tr['cand_label'][idx] == 0]
    logging.info('Inputs %s | negatives %s | loss %s', groups, args.negatives, args.loss)
    for kind in np.unique(tr['cand_kind'][idx]):
        sel = tr['cand_kind'][idx] == kind
        logging.info('  kind %d: %d examples (%.3f positive)', kind, sel.sum(), tr['cand_label'][idx][sel].mean())
    if len(neg) == 0 or len(pos) == 0:
        raise ValueError('Need both positives and negatives; check --negatives against the mined kinds')

    # Features of the selected candidates only; below, `pos`, `neg` and the groups index rows of X
    X = build_inputs(tr, idx, groups)
    y_all = y_all[idx]
    pos, neg = np.nonzero(tr['cand_label'][idx] == 1)[0], np.nonzero(tr['cand_label'][idx] == 0)[0]
    scorer = Scorer(groups, hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(scorer.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    G = None
    if args.loss == 'listwise':
        rows = {c: r for r, c in enumerate(idx)}
        G = np.vectorize(lambda c: rows.get(c, -1))(make_groups(tr, idx))

    best, best_state, history = -1.0, None, []
    for epoch in range(args.epochs):
        scorer.train()
        tot, nb = 0.0, 0
        if args.loss == 'bce':
            n_half = max(len(pos), len(neg)) if args.epoch_size == 0 else args.epoch_size // 2
            p = rng.choice(pos, n_half, replace=len(pos) < n_half)
            q = rng.choice(neg, n_half, replace=len(neg) < n_half)
            ep = rng.permutation(np.concatenate([p, q]))  # balanced: half positives, half negatives
            for i in range(0, len(ep), args.batch_size):
                b = ep[i:i + args.batch_size]
                logit = scorer(X[b].to(device))
                loss = F.binary_cross_entropy_with_logits(logit, y_all[b].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot, nb = tot + loss.item(), nb + 1
        else:
            gb = max(1, args.batch_size // G.shape[1])
            perm = rng.permutation(len(G))
            for i in range(0, len(perm), gb):
                Gi = torch.from_numpy(G[perm[i:i + gb]])
                valid = Gi >= 0
                flat = Gi.clamp(min=0).reshape(-1)
                s = scorer(X[flat].to(device)).view(Gi.shape)
                loss = listwise_loss(s, y_all[flat].view(Gi.shape).to(device), valid.to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot, nb = tot + loss.item(), nb + 1
        scorer.eval()
        val, _ = evaluate(scorer, va, [args.val_k], 0.0, device)
        v = val['K=%d' % args.val_k]['scorer_only']['success']
        history.append({'epoch': epoch, 'loss': tot / max(nb, 1), 'val_scorer_only': v,
                        'val_oracle': val['K=%d' % args.val_k]['oracle'], 'val_top1': val['top1']['success']})
        logging.info('epoch %2d loss %.4f | val@K=%d scorer-only %.4f (top1 %.4f, oracle %.4f)', epoch,
                     tot / max(nb, 1), args.val_k, v, val['top1']['success'], val['K=%d' % args.val_k]['oracle'])
        if v > best:
            best, best_state = v, {k: t.detach().cpu().clone() for k, t in scorer.state_dict().items()}
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.save({'state_dict': best_state, 'groups': groups, 'hidden': args.hidden, 'dropout': args.dropout,
                'args': vars(args), 'history': history, 'best_val': best}, args.out)
    logging.info('Saved %s (best val scorer-only success %.4f)', args.out, best)


def load_scorer(path, device):
    ck = torch.load(path, map_location=device)
    s = Scorer(ck['groups'], hidden=ck.get('hidden', 512), dropout=ck.get('dropout', 0.1)).to(device)
    s.load_state_dict(ck['state_dict'])
    return s.eval()


def run_eval(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    scorer = load_scorer(args.scorer, device) if args.scorer else None
    out = {'scorer': args.scorer}
    alpha, k_best = args.alpha, None
    if args.tune:
        va = load_cache(args.tune)
        grid = {}
        for a in args.alpha_grid:
            r, _ = evaluate(scorer, va, args.ks, a, device)
            for k in args.ks:
                grid[(a, k)] = r['K=%d' % k]['reranked']['success']
        (alpha, k_best), v = max(grid.items(), key=lambda kv: (kv[1], -kv[0][1]))  # ties: smaller K
        out['tuning'] = {'alpha': alpha, 'K': k_best, 'val_success': v,
                         'grid': {'alpha=%g,K=%d' % ak: s for ak, s in grid.items()}}
        logging.info('Tuned on %s: alpha=%g K=%d (val reranked %.4f)', args.tune, alpha, k_best, v)
    out['alpha'] = alpha
    succ = {}
    for path in args.test:
        name = os.path.splitext(os.path.basename(path))[0]
        r, _ = evaluate(scorer, load_cache(path), args.ks, alpha, device)
        out[name] = r
        logging.info('%s (n=%d): top1 %.4f', name, r['n'], r['top1']['success'])
        for k in args.ks:
            e = r['K=%d' % k]
            logging.info('  K=%2d oracle %.4f | reranked %.4f | scorer-only %.4f | grounding %s | errors %s', k,
                         e['oracle'], e['reranked']['success'], e['scorer_only']['success'],
                         e['reranked']['grounding_acc'], e['reranked']['errors'])
        succ[name] = r
    seen = [n for n in succ if 'unseen' not in n and 'seen' in n]
    unseen = [n for n in succ if 'unseen' in n]
    if len(seen) == 1 and len(unseen) == 1:
        s, u = succ[seen[0]], succ[unseen[0]]
        out['H'] = {'top1': harmonic(s['top1']['success'], u['top1']['success'])}
        for k in args.ks:
            key = 'K=%d' % k
            out['H'][key] = {m: harmonic(s[key][m]['success'], u[key][m]['success'])
                             for m in ('reranked', 'scorer_only')}
            out['H'][key]['oracle'] = harmonic(s[key]['oracle'], u[key]['oracle'])
        logging.info('H: %s', json.dumps(out['H']))
        if k_best is not None:
            key = 'K=%d' % k_best
            out['main'] = {'K': k_best, 'alpha': alpha, 'seen': s[key]['reranked']['success'],
                           'unseen': u[key]['reranked']['success'], 'H': out['H'][key]['reranked']}
            logging.info('Main row (alpha=%g, K=%d): seen %.4f unseen %.4f H %.4f', alpha, k_best,
                         out['main']['seen'], out['main']['unseen'], out['main']['H'])
    if args.out:
        os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        logging.info('Saved %s', args.out)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    t = sub.add_parser('train')
    t.add_argument('--train', required=True)
    t.add_argument('--val', required=True)
    t.add_argument('--inputs', nargs='+', default=list(INPUT_GROUPS), choices=INPUT_GROUPS)
    t.add_argument('--negatives', nargs='+', default=['data', 'random', 'hard', 'lang'],
                   choices=['data', 'random', 'hard', 'lang'])
    t.add_argument('--loss', default='bce', choices=['bce', 'listwise'])
    t.add_argument('--epochs', type=int, default=20)
    t.add_argument('--epoch-size', type=int, default=0, help='BCE examples per epoch (0 = 2 x larger class)')
    t.add_argument('--batch-size', type=int, default=1024)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--weight-decay', type=float, default=1e-4)
    t.add_argument('--hidden', type=int, default=512)
    t.add_argument('--dropout', type=float, default=0.1)
    t.add_argument('--val-k', type=int, default=10, help='K used for checkpoint selection on --val')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--out', required=True)
    t.add_argument('--cpu', action='store_true')
    e = sub.add_parser('eval')
    e.add_argument('--scorer', default=None, help='Omit to evaluate the grasp network alone (top1 / oracle)')
    e.add_argument('--test', nargs='+', required=True)
    e.add_argument('--tune', default=None, help='Validation cache: tune alpha and K on it')
    e.add_argument('--alpha', type=float, default=0.0, help='Used when --tune is not given')
    e.add_argument('--alpha-grid', type=float, nargs='+', default=[round(0.1 * i, 1) for i in range(11)])
    e.add_argument('--ks', type=int, nargs='+', default=[1, 5, 10, 20])
    e.add_argument('--out', default=None)
    e.add_argument('--cpu', action='store_true')
    return p.parse_args()


if __name__ == '__main__':
    a = parse_args()
    train(a) if a.cmd == 'train' else run_eval(a)
