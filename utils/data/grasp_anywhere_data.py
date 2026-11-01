import glob
import os
import random
import re

import pickle

import numpy as np
import torch
from imageio import imread
from skimage.transform import resize

from utils.dataset_processing import grasp, image, mask
from .language_grasp_data import LanguageGraspDatasetBase


class GraspAnywhereDataset(LanguageGraspDatasetBase):
    """
    Dataset wrapper for the Grasp-Anything dataset.
    """

    def __init__(self, file_path, ds_rotate=0, **kwargs):
        """
        :param file_path: Grasp-Anything Dataset directory.
        :param ds_rotate: If splitting the dataset, rotate the list of items by this fraction first
        :param kwargs: kwargs for GraspDatasetBase
        """
        super(GraspAnywhereDataset, self).__init__(**kwargs)

        addition_file_path = kwargs["add_file_path"]

        # Grasp-Anything++ part-level positive grasps: <hash>_<obj>_<part>.pt
        self.grasp_files = glob.glob(os.path.join(addition_file_path, 'positive_grasp', '*.pt'))
        self.prompt_dir = os.path.join(file_path, 'prompt')
        self.instruction_dir = os.path.join(addition_file_path, 'grasp_instructions')
        self.rgb_dir = os.path.join(file_path, 'image')

        # Split file: explicit --split-file, otherwise the official test lists
        split_file = kwargs.get("split_file")
        if not split_file:
            split_file = os.path.join('split/grasp-anything++/test',
                                      'seen.obj' if kwargs["seen"] else 'unseen.obj')
        with open(split_file, 'rb') as f:
            idxs = set(pickle.load(f))  # keys are <hash>_<obj>; a set makes filtering fast
        self.grasp_files = [x for x in self.grasp_files
                            if os.path.basename(x)[:-3].rsplit('_', 1)[0] in idxs]
        self.grasp_files.sort()

        # Part masks: keep only samples that have one (identical data for every ablation row)
        self._mask_index = None
        if self.mask_dir:
            self._mask_index = self._index_masks(self.mask_dir)
            if self.require_mask:
                before = len(self.grasp_files)
                self.grasp_files = [x for x in self.grasp_files if self._mask_paths(x)]
                print('[GraspAnywhereDataset] {}/{} samples have a {} mask'.format(
                    len(self.grasp_files), before, self.mask_type))

        # Optional fixed random subset (same seed for every method)
        max_samples = kwargs.get("max_samples")
        if max_samples and len(self.grasp_files) > max_samples:
            self.grasp_files = sorted(random.Random(0).sample(self.grasp_files, max_samples))

        self.length = len(self.grasp_files)

        if self.length == 0:
            raise FileNotFoundError('No dataset files found. Check path: {}'.format(file_path))

        if ds_rotate:
            self.grasp_files = self.grasp_files[int(self.length * ds_rotate):] + self.grasp_files[
                                                                                 :int(self.length * ds_rotate)]
            

    @staticmethod
    def _index_masks(mask_dir):
        """Map '<hash>_<obj>_<part>' -> path for every mask file in mask_dir (any extension)."""
        return {os.path.splitext(f)[0]: os.path.join(mask_dir, f) for f in os.listdir(mask_dir)}

    def _mask_paths(self, grasp_file):
        """Mask file(s) for a sample: the prompted part, or every part of its object."""
        name = os.path.basename(grasp_file)[:-3]  # <hash>_<obj>_<part>
        if self.mask_type == 'part':
            return [self._mask_index[name]] if name in self._mask_index else []
        prefix = name.rsplit('_', 1)[0] + '_'
        return [p for k, p in self._mask_index.items() if k.startswith(prefix)]

    @staticmethod
    def _read_mask(path):
        ext = os.path.splitext(path)[1]
        if ext == '.npy':
            m = np.load(path)
        elif ext in ('.pt', '.pth'):
            m = torch.load(path)
            m = m.numpy() if torch.is_tensor(m) else np.asarray(m)
        else:
            m = imread(path)
        m = np.squeeze(np.asarray(m))
        if m.ndim == 3:  # collapse the (small) channel axis
            m = m.max(axis=int(np.argmin(m.shape)))
        return (m > 0).astype(np.float32)

    def get_mask_native(self, idx):
        """Binary mask at native resolution, no augmentation."""
        paths = self._mask_paths(self.grasp_files[idx])
        if not paths:
            raise FileNotFoundError('No {} mask for {}'.format(self.mask_type, self.grasp_files[idx]))
        m = None
        for p in paths:
            cur = self._read_mask(p)
            if m is None:
                m = cur
            else:
                if cur.shape != m.shape:
                    cur = resize(cur, m.shape, order=0, preserve_range=True)
                m = np.maximum(m, cur)
        return m

    def get_mask(self, idx, rot=0, zoom=1.0):
        """Mask transformed exactly like get_rgb (rotate, zoom, resize), returned as a 224x224 {0,1} array."""
        m = image.Image(self.get_mask_native(idx))
        m.rotate(rot)
        m.zoom(zoom)
        m.resize((self.output_size, self.output_size))
        return (m.img > 0.5).astype(np.float32)

    def _get_crop_attrs(self, idx):
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        center = gtbbs.center
        left = max(0, min(center[1] - self.output_size // 2, 416 - self.output_size))
        top = max(0, min(center[0] - self.output_size // 2, 416 - self.output_size))
        return center, left, top

    def get_gtbb(self, idx, rot=0, zoom=1.0):       
        # Jacquard try
        gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx], scale=self.output_size / 416.0)

        c = self.output_size // 2
        gtbbs.rotate(rot, (c, c))
        gtbbs.zoom(zoom, (c, c))

        # Cornell try
        # gtbbs = grasp.GraspRectangles.load_from_grasp_anything_file(self.grasp_files[idx])
        # center, left, top = self._get_crop_attrs(idx)
        # gtbbs.rotate(rot, center)
        # gtbbs.offset((-top, -left))
        # gtbbs.zoom(zoom, (self.output_size // 2, self.output_size // 2))
        return gtbbs

    def get_depth(self, idx, rot=0, zoom=1.0):
        depth_img = image.DepthImage.from_tiff(self.depth_files[idx])
        center, left, top = self._get_crop_attrs(idx)
        depth_img.rotate(rot, center)
        depth_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        depth_img.normalise()
        depth_img.zoom(zoom)
        depth_img.resize((self.output_size, self.output_size))
        return depth_img.img

    def get_rgb(self, idx, rot=0, zoom=1.0, normalise=True):
        # image name = hash before the first underscore (works for multi-digit obj/part ids)
        rgb_file = os.path.basename(self.grasp_files[idx]).split('_')[0] + '.jpg'
        rgb_file = os.path.join(self.rgb_dir, rgb_file)
        rgb_img = image.Image.from_file(rgb_file)

        # Jacquard try
        rgb_img.rotate(rot)
        rgb_img.zoom(zoom)
        rgb_img.resize((self.output_size, self.output_size))
        if normalise:
            rgb_img.normalise()
            rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        return rgb_img.img

        # Cornell try
        # center, left, top = self._get_crop_attrs(idx)
        # rgb_img.rotate(rot, center)
        # rgb_img.crop((top, left), (min(480, top + self.output_size), min(640, left + self.output_size)))
        # rgb_img.zoom(zoom)
        # rgb_img.resize((self.output_size, self.output_size))
        # if normalise:
        #     rgb_img.normalise()
        #     rgb_img.img = rgb_img.img.transpose((2, 0, 1))
        # return rgb_img.img

    @staticmethod
    def _to_text(obj):
        """Grasp-Anything++ instruction pickles may hold a str, list or dict; return one string."""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, (list, tuple)) and len(obj) > 0:
            return GraspAnywhereDataset._to_text(obj[0])
        if isinstance(obj, dict):
            for k in ('instruction', 'prompt', 'text'):
                if k in obj:
                    return GraspAnywhereDataset._to_text(obj[k])
        return str(obj)

    def get_prompts(self, idx):
        """Return (prompt, query). Models read `query`: the Grasp-Anything++ grasping instruction."""
        name = os.path.basename(self.grasp_files[idx])[:-3]  # <hash>_<obj>_<part>
        with open(os.path.join(self.instruction_dir, name + '.pkl'), 'rb') as f:
            instruction = self._to_text(pickle.load(f))
        return instruction, instruction
