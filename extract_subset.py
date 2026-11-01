"""Extract only the Grasp-Anything / Grasp-Anything++ files referenced by LGD's official split lists.

Avoids concatenating the two image parts (reads them as one virtual file) and avoids unzipping
~1M files onto a network drive. Output is laid out exactly as LGD's GraspAnywhereDataset expects:

    OUT/grasp-anything/image/<hash>.jpg
    OUT/grasp-anything/prompt/<hash>.pkl
    OUT/grasp-anything++/seen/positive_grasp/<hash>_<obj>_<part>.pt
    OUT/grasp-anything++/seen/grasp_instructions/<hash>_<obj>_<part>.pkl

Usage:
    python extract_subset.py --lgd ~/LGD --ga ~/datasets/ga --gapp ~/datasets/gapp --out /path/to/lgd_data
"""
import argparse
import io
import os
import pickle
import shutil
import sys
import time
import zipfile


class ConcatReader(io.RawIOBase):
    """Read several files (e.g. image_part_aa, image_part_ab made by `split`) as one seekable file."""

    def __init__(self, paths):
        self.files = [open(p, 'rb') for p in paths]
        self.sizes = [os.path.getsize(p) for p in paths]
        self.starts = []
        total = 0
        for s in self.sizes:
            self.starts.append(total)
            total += s
        self.total = total
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=io.SEEK_SET):
        if whence == io.SEEK_SET:
            self.pos = offset
        elif whence == io.SEEK_CUR:
            self.pos += offset
        elif whence == io.SEEK_END:
            self.pos = self.total + offset
        return self.pos

    def readinto(self, b):
        want = len(b)
        got = 0
        while got < want and self.pos < self.total:
            i = max(k for k, st in enumerate(self.starts) if st <= self.pos)
            f = self.files[i]
            f.seek(self.pos - self.starts[i])
            chunk = f.read(min(want - got, self.starts[i] + self.sizes[i] - self.pos))
            if not chunk:
                break
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self.pos += len(chunk)
        return got

    def close(self):
        for f in self.files:
            f.close()
        super().close()


def load_keys(split_dir):
    keys = {}
    for rel in ['train/seen.obj', 'test/seen.obj', 'test/unseen.obj']:
        path = os.path.join(split_dir, rel)
        with open(path, 'rb') as f:
            keys[rel] = set(pickle.load(f))
        print(f'{rel}: {len(keys[rel])} keys')
    return keys


def extract(zf, names, dest_dir, label):
    os.makedirs(dest_dir, exist_ok=True)
    t0 = time.time()
    for i, info in enumerate(names, 1):
        target = os.path.join(dest_dir, os.path.basename(info.filename))
        if not os.path.exists(target):
            with zf.open(info) as src, open(target, 'wb') as dst:
                shutil.copyfileobj(src, dst)
        if i % 2000 == 0 or i == len(names):
            print(f'  {label}: {i}/{len(names)} ({time.time() - t0:.0f}s)', flush=True)


def stem(name):
    base = os.path.basename(name)
    return base.rsplit('.', 1)[0] if '.' in base else base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lgd', required=True, help='LGD repo root (for split/grasp-anything++)')
    ap.add_argument('--ga', required=True, help='folder with image_part_aa, image_part_ab, scene_description.zip')
    ap.add_argument('--gapp', required=True, help='folder with grasp_label_positive.zip, grasp_instructions.zip')
    ap.add_argument('--out', required=True, help='output data root (prefer a local disk)')
    args = ap.parse_args()

    keys = load_keys(os.path.join(args.lgd, 'split', 'grasp-anything++'))
    all_keys = set().union(*keys.values())                    # <hash>_<obj>
    hashes = {k.rsplit('_', 1)[0] for k in all_keys}          # <hash>
    print(f'total keys: {len(all_keys)}, images: {len(hashes)}')

    ga_out = os.path.join(args.out, 'grasp-anything')
    pp_out = os.path.join(args.out, 'grasp-anything++', 'seen')

    # Grasp-Anything++: labels and instructions for the needed <hash>_<obj> keys
    # MSCR also needs part masks and negative grasps: any zip in --gapp whose name contains
    # 'mask' -> part_mask/, 'negative' -> negative_grasp/ (skipped when not downloaded)
    optional = []
    for z in sorted(os.listdir(args.gapp)):
        if z.endswith('.zip') and 'mask' in z.lower():
            optional.append((z, 'part_mask'))
        elif z.endswith('.zip') and 'negative' in z.lower():
            optional.append((z, 'negative_grasp'))
    print('optional zips:', optional or 'none (part masks / negatives not found in --gapp)')
    for zname, sub in [('grasp_label_positive.zip', 'positive_grasp'),
                       ('grasp_instructions.zip', 'grasp_instructions')] + optional:
        with zipfile.ZipFile(os.path.join(args.gapp, zname)) as zf:
            names = [i for i in zf.infolist() if not i.is_dir()
                     and stem(i.filename).rsplit('_', 1)[0] in all_keys]
            print(f'{zname}: {len(names)} matching files')
            extract(zf, names, os.path.join(pp_out, sub), sub)

    # Grasp-Anything: scene descriptions (LGD's "prompt" folder) for the needed images
    with zipfile.ZipFile(os.path.join(args.ga, 'scene_description.zip')) as zf:
        names = [i for i in zf.infolist() if not i.is_dir() and stem(i.filename) in hashes]
        print(f'scene_description.zip: {len(names)} matching files')
        extract(zf, names, os.path.join(ga_out, 'prompt'), 'prompt')

    # Grasp-Anything images: read the two split parts as one zip, no concatenated copy
    parts = sorted(os.path.join(args.ga, p) for p in os.listdir(args.ga) if p.startswith('image_part_'))
    print('image parts:', parts)
    reader = io.BufferedReader(ConcatReader(parts), buffer_size=1 << 20)
    with zipfile.ZipFile(reader) as zf:
        names = [i for i in zf.infolist() if not i.is_dir()
                 and i.filename.endswith('.jpg') and stem(i.filename) in hashes]
        print(f'images: {len(names)} matching files')
        extract(zf, names, os.path.join(ga_out, 'image'), 'image')

    missing = hashes - {stem(n) for n in os.listdir(os.path.join(ga_out, 'image'))}
    print(f'done. images missing: {len(missing)}')
    if missing:
        print('example missing hashes:', list(missing)[:5], file=sys.stderr)


if __name__ == '__main__':
    main()
