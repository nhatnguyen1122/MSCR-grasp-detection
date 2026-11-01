"""Check that the actual baseline loss trains each model's image backbone."""
import unittest
from unittest import mock

import torch
from torch import nn

from inference.models.lgrconvnet3 import GenerativeResnet
from inference.models.lggcnn import LGGCNN
from inference.models.ragt.ragt import LRAGT


class BaselineGradientTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def check_backbone_update(self, model_class, channel_size, backbone_name):
        torch.manual_seed(123)
        # The text encoder is frozen in production. Stub its embedding to avoid
        # downloads/GPU initialization while retaining the real fusion and loss.
        with mock.patch.object(model_class, '_load_and_freeze_clip',
                               return_value=nn.Identity()):
            net = model_class(input_channels=3, channel_size=channel_size,
                              dropout=False)
        features = torch.randn(2, 512)
        with mock.patch.object(net, '_encode_text', return_value=features):
            net.train()
            inputs = torch.randn(2, 3, 224, 224)
            targets = [torch.randn(2, 1, 224, 224) for _ in range(4)]
            weight = dict(net.named_parameters())[backbone_name]
            before = weight.detach().clone()
            optimizer = torch.optim.SGD(net.parameters(), lr=.01)
            loss = net.compute_loss(inputs, targets, ['probe'] * 2,
                                    ['probe'] * 2)['loss']
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(weight.grad,
                                 'Grasp loss must reach the image backbone')
            self.assertTrue(torch.isfinite(weight.grad).all())
            self.assertGreater(weight.grad.abs().sum().item(), 0)
            optimizer.step()
            self.assertFalse(torch.equal(before, weight),
                             'An optimizer step must update image weights')

    def test_lgrconvnet3_image_backbone_learns(self):
        self.check_backbone_update(GenerativeResnet, 32, 'conv1.weight')

    def test_lggcnn_image_backbone_learns(self):
        self.check_backbone_update(LGGCNN, 32, 'conv1.weight')

    def test_lragt_image_backbone_learns(self):
        with mock.patch.object(LRAGT, '_load_and_freeze_clip',
                               return_value=nn.Identity()):
            net = LRAGT(input_channels=3, channel_size=18, dropout=False)
        backbone_name = next('mobile_vit.' + n
                             for n, _ in net.mobile_vit.named_parameters())
        del net
        self.check_backbone_update(LRAGT, 18, backbone_name)


if __name__ == '__main__':
    unittest.main()
