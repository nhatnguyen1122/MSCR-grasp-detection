"""MSCR grasp network: GR-ConvNet (`lgrconvnet3`) with FiLM text conditioning and a mask head.

Options (class attributes, overridable from train_network.py with --fusion, --film-mode, --mask-head, --lambda-m):
  fusion     'film' : at the bottleneck and both upsampling layers, F <- (1 + gamma(t)) F + beta(t)
             'add'  : the baseline's broadcast add of a 56-d text pattern at the bottleneck
  film_mode  'pre'  : conv -> FiLM -> BN -> ReLU, scale 1 + tanh(gamma) in (0, 2). BN re-centres every channel
                      after FiLM, so beta(t) cannot push a whole channel below the ReLU.
             'post' : conv -> BN -> FiLM -> ReLU, scale 1 + gamma (earlier version). Without the mask loss every
                      channel of the last layer died and training collapsed.
  mask_head  5th output M: logits of the instructed part's mask, trained with lambda_m * (BCE + Dice)

FiLM is zero-initialised, so every variant starts as the same network. At inference, Q~ = Q * sigmoid(M)^alpha
re-ranks the peaks of Q (mscr/decode.py).

Registered names (paper, Table 2):
  mscr                    grasp network: FiLM ('pre') + mask loss
  mscr_no_mask            - mask loss
  mscr_no_film            - FiLM (text added as in the baseline), with the mask loss
  mscr_film_post          FiLM after BatchNorm, unbounded, with the mask loss (earlier network, used for re-ranking)
  mscr_film_post_no_mask  the same without the mask loss (collapsed)
"""
import clip
import torch
import torch.nn as nn
import torch.nn.functional as F

from inference.models.grasp_model import LanguageGraspModel, ResidualBlock


class FiLM(nn.Module):
    """Small MLP t -> (gamma, beta), one value per channel; last layer zero-initialised (identity at start).

    bounded: scale = 1 + tanh(gamma), in (0, 2), instead of 1 + gamma.
    """

    def __init__(self, text_dim, channels, hidden=256, bounded=False):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(text_dim, hidden), nn.GELU(), nn.Linear(hidden, 2 * channels))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)
        self.bounded = bounded

    def forward(self, x, t):
        gamma, beta = self.mlp(t).chunk(2, dim=1)
        if getattr(self, 'bounded', False):  # getattr: checkpoints saved before this option
            gamma = torch.tanh(gamma)
        return (1 + gamma[:, :, None, None]) * x + beta[:, :, None, None]


def dice_loss(logits, target, eps=1.0):
    """1 - 2 sum(sigma(M) M*) / (sum sigma(M) + sum M*), per sample, averaged over the batch."""
    p = torch.sigmoid(logits).flatten(1)
    t = target.flatten(1)
    inter = (p * t).sum(1)
    return (1 - (2 * inter + eps) / (p.sum(1) + t.sum(1) + eps)).mean()


class MSCRNet(LanguageGraspModel):
    OPTIONS = ('fusion', 'film_mode', 'mask_head', 'lambda_m')
    # Defaults = the paper's grasp network; the variants below override them.
    fusion = 'film'
    film_mode = 'pre'
    mask_head = True
    lambda_m = 0.5

    def __init__(self, input_channels=3, output_channels=1, channel_size=32, dropout=False, prob=0.0,
                 clip_version='ViT-B/32', **overrides):
        super().__init__()
        for k, v in overrides.items():
            if v is None:
                continue
            if k not in self.OPTIONS:
                raise ValueError('Unknown MSCRNet option {}'.format(k))
            setattr(self, k, v)
        assert self.fusion in ('add', 'film') and self.film_mode in ('pre', 'post')

        c = channel_size
        # --- encoder / decoder: identical to lgrconvnet3 ---
        self.conv1 = nn.Conv2d(input_channels, c, kernel_size=9, stride=1, padding=4)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, c * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(c * 2, c * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.res1 = ResidualBlock(c * 4, c * 4)
        self.res2 = ResidualBlock(c * 4, c * 4)
        self.res3 = ResidualBlock(c * 4, c * 4)
        self.res4 = ResidualBlock(c * 4, c * 4)
        self.res5 = ResidualBlock(c * 4, c * 4)
        self.conv4 = nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(c * 2)
        self.conv5 = nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=2, output_padding=1)
        self.bn5 = nn.BatchNorm2d(c)
        self.conv6 = nn.ConvTranspose2d(c, c, kernel_size=9, stride=1, padding=4)

        self.pos_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.cos_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.sin_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.width_output = nn.Conv2d(c, output_channels, kernel_size=2)
        self.mask_output = nn.Conv2d(c, 1, kernel_size=2) if self.mask_head else None

        self.dropout = dropout
        self.dropout_pos = nn.Dropout(p=prob)
        self.dropout_cos = nn.Dropout(p=prob)
        self.dropout_sin = nn.Dropout(p=prob)
        self.dropout_wid = nn.Dropout(p=prob)
        self.dropout_mask = nn.Dropout(p=prob)

        # --- language: frozen CLIP text encoder ---
        self.clip_version = clip_version
        self.lang_model = self._load_and_freeze_clip(clip_version)
        if self.fusion == 'add':
            # Baseline fusion: 512 -> 56-d pattern broadcast over all 128 bottleneck channels
            self.y_flatten = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128), nn.GELU(),
                                           nn.Linear(128, 56), nn.GELU())
        else:
            bounded = self.film_mode == 'pre'
            self.film_bottleneck = FiLM(512, c * 4, bounded=bounded)
            self.film4 = FiLM(512, c * 2, bounded=bounded)
            self.film5 = FiLM(512, c, bounded=bounded)

        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.weight.requires_grad:
                nn.init.xavier_uniform_(m.weight, gain=1)

    # ------------------------------------------------------------------ language (as in lgrconvnet3)
    def _load_and_freeze_clip(self, clip_version):
        model, _ = clip.load(clip_version, device='cpu', jit=False)
        clip.model.convert_weights(model)
        model.eval()
        for p in model.parameters():
            p.requires_grad = False
        return model

    def _encode_text(self, raw_text, device):
        context_length = 22  # start + 20 tokens + end, as in lgrconvnet3
        tokens = clip.tokenize(list(raw_text), context_length=context_length, truncate=True).to(device)
        tokens = torch.cat([tokens, torch.zeros(tokens.shape[0], 77 - context_length,
                                                dtype=tokens.dtype, device=device)], dim=1)
        if next(self.lang_model.parameters()).device != device:
            self.lang_model.to(device)
        with torch.no_grad():
            return self.lang_model.encode_text(tokens).float()

    def train(self, mode=True):
        super().train(mode)
        self.lang_model.eval()
        return self

    def set_mc_dropout(self, on=True):
        """Monte Carlo dropout: keep only the head dropouts stochastic in eval mode."""
        for d in (self.dropout_pos, self.dropout_cos, self.dropout_sin, self.dropout_wid, self.dropout_mask):
            d.train(on)
        self.dropout = self.dropout or on

    # ------------------------------------------------------------------ forward
    def forward(self, x_in, prompt, query):
        t = self._encode_text(query, x_in.device)

        x = F.relu(self.bn1(self.conv1(x_in)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.res5(self.res4(self.res3(self.res2(self.res1(x)))))

        if self.fusion == 'add':
            y = self.y_flatten(t)
            x = x + y[:, None, :, None].expand(-1, x.shape[1], -1, x.shape[3])  # same as lgrconvnet3
            x = F.relu(self.bn4(self.conv4(x)))
            x = F.relu(self.bn5(self.conv5(x)))
        elif self.film_mode == 'pre':
            x = self.film_bottleneck(x, t)
            x = F.relu(self.bn4(self.film4(self.conv4(x), t)))
            x = F.relu(self.bn5(self.film5(self.conv5(x), t)))
        else:
            x = self.film_bottleneck(x, t)
            x = F.relu(self.film4(self.bn4(self.conv4(x)), t))
            x = F.relu(self.film5(self.bn5(self.conv5(x)), t))
        x = self.conv6(x)

        if self.dropout:
            outs = [self.pos_output(self.dropout_pos(x)), self.cos_output(self.dropout_cos(x)),
                    self.sin_output(self.dropout_sin(x)), self.width_output(self.dropout_wid(x))]
            if self.mask_output is not None:
                outs.append(self.mask_output(self.dropout_mask(x)))
        else:
            outs = [self.pos_output(x), self.cos_output(x), self.sin_output(x), self.width_output(x)]
            if self.mask_output is not None:
                outs.append(self.mask_output(x))
        return tuple(outs)

    # ------------------------------------------------------------------ loss
    def compute_loss(self, xc, yc, prompt, query):
        y_pos, y_cos, y_sin, y_width = yc[:4]
        outs = self(xc, prompt, query)
        pos_pred, cos_pred, sin_pred, width_pred = outs[:4]

        p_loss = F.smooth_l1_loss(pos_pred, y_pos)
        cos_loss = F.smooth_l1_loss(cos_pred, y_cos)
        sin_loss = F.smooth_l1_loss(sin_pred, y_sin)
        width_loss = F.smooth_l1_loss(width_pred, y_width)
        loss = p_loss + cos_loss + sin_loss + width_loss
        losses = {'p_loss': p_loss, 'cos_loss': cos_loss, 'sin_loss': sin_loss, 'width_loss': width_loss}
        pred = {'pos': pos_pred, 'cos': cos_pred, 'sin': sin_pred, 'width': width_pred}

        if self.mask_output is not None:
            pred['mask'] = outs[4]
            if len(yc) > 4 and self.lambda_m > 0:
                y_mask = yc[4]
                bce = F.binary_cross_entropy_with_logits(outs[4], y_mask)
                dice = dice_loss(outs[4], y_mask)
                loss = loss + self.lambda_m * (bce + dice)
                losses['mask_bce'] = bce
                losses['mask_dice'] = dice

        return {'loss': loss, 'losses': losses, 'pred': pred}


# ---------------------------------------------------------------------- ablation variants (Table 2)
class MSCRNoMask(MSCRNet):
    """- mask loss."""
    mask_head = False


class MSCRNoFiLM(MSCRNet):
    """- FiLM: text added as in the baseline, with the mask loss."""
    fusion = 'add'


class MSCRFiLMPost(MSCRNet):
    """FiLM after BatchNorm with an unbounded scale (earlier network), with the mask loss."""
    film_mode = 'post'


class MSCRFiLMPostNoMask(MSCRNet):
    """FiLM after BatchNorm, unbounded, without the mask loss (collapsed at epoch 11)."""
    film_mode, mask_head = 'post', False


VARIANTS = {'mscr': MSCRNet, 'mscr_no_mask': MSCRNoMask, 'mscr_no_film': MSCRNoFiLM,
            'mscr_film_post': MSCRFiLMPost, 'mscr_film_post_no_mask': MSCRFiLMPostNoMask}

# Class names of checkpoints saved under the working name of the project (inference/models/gtv.py)
LEGACY_CLASSES = {'GroundedGenerator': 'MSCRFiLMPost', 'GTV_A2': 'MSCRFiLMPostNoMask', 'GTV_A3': 'MSCRFiLMPost',
                  'GTV_A3d': 'MSCRNoFiLM', 'GTV_A2s': 'MSCRNoMask', 'GTV_A3s': 'MSCRNet'}
