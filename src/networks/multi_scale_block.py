import torch
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union
from basicsr.archs.rrdbnet_arch import RRDB
import torch.nn.functional as F
class Refine_block(nn.Module):
    """
    Quoting from https://arxiv.org/abs/2302.05543: "Stable Diffusion uses a pre-processing method similar to VQ-GAN
    [11] to convert the entire dataset of 512 × 512 images into smaller 64 × 64 “latent images” for stabilized
    training. This requires ControlNets to convert image-based conditions to 64 × 64 feature space to match the
    convolution size. We use a tiny network E(·) of four convolution layers with 4 × 4 kernels and 2 × 2 strides
    (activated by ReLU, channels are 16, 32, 64, 128, initialized with Gaussian weights, trained jointly with the full
    model) to encode image-space conditions ... into feature maps ..."
    """

    def __init__(
        self,
        conditioning_embedding_channels: int=3,
        conditioning_channels: int = 3,
        block_out_channels: Tuple[int] = (16, 32, 64, 128),
        return_rgbs: bool = True,
        use_rrdb: bool = True,
    ):
        super().__init__()

        self.return_rgbs = return_rgbs
        self.use_rrdb = use_rrdb

        self.conv_in = nn.Conv2d(conditioning_channels, block_out_channels[0], kernel_size=3, padding=1)

        if self.use_rrdb:
            num_rrdb_block = 2
            layers = (RRDB(block_out_channels[0], block_out_channels[0]) for i in range(num_rrdb_block))
            self.preprocesser = nn.Sequential(*layers)

        self.blocks = nn.ModuleList([])
        self.to_rgbs = nn.ModuleList([])

        for i in range(len(block_out_channels) - 1):
            channel_in = block_out_channels[i]
            channel_out = block_out_channels[i + 1]
            self.blocks.append(nn.Conv2d(channel_in, channel_in, kernel_size=3, padding=1))
            self.blocks.append(nn.Conv2d(channel_in, channel_out, kernel_size=3, padding=1, stride=2))

            if return_rgbs:
                self.to_rgbs.append(nn.Conv2d(channel_out, 3, kernel_size=3, padding=1)) # channel_in

        self.conv_out = zero_module(
            nn.Conv2d(block_out_channels[-1], conditioning_embedding_channels, kernel_size=3, padding=1)
        )

    def forward(self, conditioning):

        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        if self.use_rrdb:
            embedding = self.preprocesser(embedding)

        out_rgbs = []
        for i, block in enumerate(self.blocks):
            embedding = block(embedding)
            embedding = F.silu(embedding)

            if i%2 and self.return_rgbs: # 0
                out_rgbs.append(self.to_rgbs[i//2](embedding))

        embedding = self.conv_out(embedding)
        return out_rgbs
        # return [embedding, out_rgbs] if self.return_rgbs else embedding

def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module

