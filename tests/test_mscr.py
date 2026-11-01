"""Checks for the MSCR grasp network, decoding and scorer geometry.

Run from the repository root:  python -m unittest tests/test_mscr.py
Needs the lgd environment (CLIP weights are loaded once; a GPU is used if available).
"""
import glob
import io
import os
import unittest

import numpy as np
import torch

from inference.models import get_network
from inference.models.mscr import MSCRNet, MSCRNoMask
from mscr import decode
from mscr.common import _LegacyUnpickler
from mscr.scorer import crop_transform, listwise_loss
from utils.dataset_processing.grasp import Grasp

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
TEXT = ['grasp the mug by its handle', 'pick up the knife by the blade']


def make(name, **kw):
    torch.manual_seed(0)
    return get_network(name)(input_channels=3, dropout=False, prob=0.0, channel_size=32, **kw).to(DEV)


class TestGraspNetwork(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x = torch.randn(2, 3, 224, 224, device=DEV)

    def test_no_film_is_the_baseline(self):
        """'- FiLM' without the mask head is lgrconvnet3: same parameters, same outputs."""
        ref = make('lgrconvnet3').eval()
        net = make('mscr_no_film', mask_head=False).eval()
        missing, unexpected = net.load_state_dict(ref.state_dict(), strict=False)
        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        with torch.no_grad():
            for o_ref, o in zip(ref(self.x, TEXT, TEXT), net(self.x, TEXT, TEXT)):
                self.assertTrue(torch.allclose(o_ref, o, atol=1e-5))

    def _encoder_grad(self, net):
        y = [torch.rand(2, 1, 224, 224, device=DEV) for _ in range(4)]
        net.train()
        net.compute_loss(self.x, y, TEXT, TEXT)['loss'].backward()
        g = net.conv1.weight.grad
        return 0.0 if g is None else g.abs().sum().item()

    def test_encoder_is_trained(self):
        for name in ('mscr', 'mscr_no_film', 'mscr_film_post'):
            self.assertGreater(self._encoder_grad(make(name)), 0.0, name)

    def test_film_starts_as_identity(self):
        net = make('mscr').eval()
        f = torch.randn(2, 128, 56, 56, device=DEV)
        t = torch.randn(2, 512, device=DEV)
        self.assertTrue(torch.equal(net.film_bottleneck(f, t), f))

    def test_film_placement(self):
        """FiLM before / after BatchNorm give the same network at init; the 'pre' scale stays in (0, 2)."""
        for post, pre in (('mscr_film_post_no_mask', 'mscr_no_mask'), ('mscr_film_post', 'mscr')):
            a, b = make(post).eval(), make(pre).eval()
            self.assertEqual((a.film_mode, b.film_mode), ('post', 'pre'))
            self.assertEqual(set(a.state_dict()), set(b.state_dict()))
            with torch.no_grad():
                for oa, ob in zip(a(self.x, TEXT, TEXT), b(self.x, TEXT, TEXT)):
                    self.assertTrue(torch.allclose(oa, ob, atol=1e-5))
        film = make('mscr').film4
        with torch.no_grad():
            film.mlp[-1].bias.fill_(50.0)  # huge gamma and beta
            f = torch.ones(2, 64, 4, 4, device=DEV)
            self.assertLessEqual(film(f, torch.randn(2, 512, device=DEV)).max().item(), 2.0 + 50.0 + 1e-4)

    def test_legacy_checkpoint_names(self):
        """Checkpoints saved as inference.models.gtv.GTV_* load as the matching MSCR variant."""
        unpickler = _LegacyUnpickler(io.BytesIO())
        self.assertIs(unpickler.find_class('inference.models.gtv', 'GTV_A3s'), MSCRNet)
        self.assertIs(unpickler.find_class('inference.models.gtv', 'GTV_A2s'), MSCRNoMask)

    def test_mask_head_outputs_and_loss(self):
        net = make('mscr')
        outs = net(self.x, TEXT, TEXT)
        self.assertEqual(len(outs), 5)
        self.assertEqual(tuple(outs[4].shape), (2, 1, 224, 224))
        y = [torch.rand(2, 1, 224, 224, device=DEV) for _ in range(4)] + \
            [(torch.rand(2, 1, 224, 224, device=DEV) > 0.8).float()]
        d = net.compute_loss(self.x, y, TEXT, TEXT)
        self.assertIn('mask_bce', d['losses'])
        self.assertIn('mask', d['pred'])
        # Without the mask target the mask term is skipped (e.g. evaluation without masks)
        self.assertNotIn('mask_bce', net.compute_loss(self.x, y[:4], TEXT, TEXT)['losses'])

    def test_overrides(self):
        net = make('mscr', fusion='add', mask_head=False, lambda_m=1.0)
        self.assertEqual((net.fusion, net.mask_head, net.lambda_m), ('add', False, 1.0))
        with self.assertRaises(ValueError):
            make('mscr', bogus=1)


class TestDecoding(unittest.TestCase):
    def test_tta_roundtrip(self):
        rng = np.random.default_rng(0)
        for k, flip in decode.TTA_VIEWS:
            r, c, a = 70.0, 150.0, 0.4
            # forward-map the grasp by transforming its two jaw points, then map it back
            g = Grasp(np.array([r, c]), a, 60, 30).as_gr.points
            p0, p1 = (g[0] + g[3]) / 2, (g[1] + g[2]) / 2
            img = np.zeros((224, 224))
            img[int(round(p0[0])), int(round(p0[1]))] = 1
            img[int(round(p1[0])), int(round(p1[1]))] = 2
            t = decode.tta_forward_transform(torch.tensor(img)[None, None], k, flip)[0, 0].numpy()
            q0, q1 = np.argwhere(t == 1)[0], np.argwhere(t == 2)[0]
            cen, d = (q0 + q1) / 2, q1 - q0
            back = decode.tta_map_back([cen[0], cen[1], np.arctan2(-d[0], d[1]), 60, 30], k, flip, 224)
            self.assertLess(abs(back[0] - r) + abs(back[1] - c), 2.0)
            self.assertLess(abs((back[2] - a + np.pi / 2) % np.pi - np.pi / 2), 0.05)

    def test_gating_reranks_peaks(self):
        q = np.zeros((224, 224))
        q[50, 50], q[150, 150] = 0.9, 0.6
        mask = np.zeros((224, 224))
        mask[140:160, 140:160] = 1.0
        maps = {'q': q, 'ang': np.zeros_like(q), 'width': np.full_like(q, 40.0), 'mask': mask}
        self.assertEqual(decode.top1(maps, alpha=0)[0][:2], [50.0, 50.0])
        self.assertEqual(decode.top1(maps, alpha=1)[0][:2], [150.0, 150.0])
        self.assertEqual(len(decode.hybrid(maps, 20, alpha=0)), 20)  # 2 peaks + 2 x 10 variants, truncated

    def test_crop_axis_horizontal(self):
        for a in np.linspace(-1.5, 1.5, 7):
            m = crop_transform([100.0, 120.0, a, 50.0, 25.0])
            p = Grasp(np.array([100.0, 120.0]), a, 50, 25).as_gr.points[:, ::-1]
            q = (m[:, :2] @ p.T + m[:, 2:]).T
            axis = q[1] - q[0]
            self.assertLess(abs(axis[1]), 1e-6)
            self.assertTrue(np.allclose(q.mean(0), [112, 112], atol=1e-4))

    def test_listwise_loss(self):
        s = torch.tensor([[2.0, 0.0, -1.0], [0.0, 0.0, 0.0]])
        y = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        v = torch.tensor([[True, True, False], [True, True, True]])
        expected = torch.logsumexp(torch.tensor([2.0, 0.0]), 0) - 2.0  # second group has no positive
        self.assertTrue(torch.allclose(listwise_loss(s, y, v), expected))


class TestMaskAugmentation(unittest.TestCase):
    """The part mask must get the same rotation / zoom as the grasp maps (needs data + a mask dir)."""

    def test_mask_follows_grasps(self):
        mask_dir = os.environ.get('MSCR_TEST_MASK_DIR')
        if not mask_dir or not glob.glob(os.path.join(mask_dir, '*')):
            self.skipTest('set MSCR_TEST_MASK_DIR to a mask directory (e.g. masks made from the GT rectangles)')
        from utils.data import get_dataset
        ds = get_dataset('grasp-anywhere')('data/grasp-anything', output_size=224, random_rotate=True,
                                           random_zoom=True, include_depth=0, include_rgb=1, seen=1,
                                           add_file_path='data/grasp-anything++/seen',
                                           split_file='split/grasp-anything++/train/seen.obj',
                                           mask_dir=mask_dir, return_mask=True, require_mask=True)
        from skimage.draw import polygon
        from utils.dataset_processing.grasp import GraspRectangles

        def overlap(rot, zoom, i):
            pos = ds.get_gtbb(i, rot, zoom).draw((224, 224))[0] > 0
            mask = ds.get_mask(i, rot, zoom) > 0.5
            return (pos & mask).sum() / max(pos.sum(), 1), pos.sum()

        samples = list(range(0, len(ds), max(1, len(ds) // 20)))
        rng = np.random.default_rng(0)

        # 1. Real masks: rotations are exact (zoom crops parts at the image border differently from the
        #    rectangles, so it is tested in 2.)
        for i in samples:
            base, _ = overlap(0.0, 1.0, i)
            self.assertAlmostEqual(overlap(float(rng.choice([np.pi / 2, np.pi, 3 * np.pi / 2])), 1.0, i)[0],
                                   base, delta=0.05)

        # 2. A mask rasterised from the sample's own rectangles must stay on the grasp maps under rotation + zoom.
        native = ds.get_mask_native
        try:
            for i in samples:
                fake = np.zeros((416, 416), np.float32)
                for gr in GraspRectangles.load_from_grasp_anything_file(ds.grasp_files[i]).grs:
                    rr, cc = polygon(gr.points[:, 0], gr.points[:, 1], (416, 416))
                    fake[rr, cc] = 1
                ds.get_mask_native = lambda idx, m=fake: m
                rot, zoom = float(rng.choice([0, np.pi / 2, np.pi, 3 * np.pi / 2])), rng.uniform(0.5, 1.0)
                ov, n = overlap(rot, zoom, i)
                if n > 50:  # the grasps are still in the frame
                    self.assertGreater(ov, 0.9)
        finally:
            ds.get_mask_native = native


if __name__ == '__main__':
    unittest.main()
