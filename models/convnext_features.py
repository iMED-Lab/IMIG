"""
ConvNeXt backbone feature extractor for the multi-modal retinal diagnosis model.
Provides separate factory functions for CFP and FFA branches to allow independent
pretrained weight loading if needed.
"""

from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
from torchvision.ops.misc import Conv2dNormActivation, Permute
from torchvision.ops.stochastic_depth import StochasticDepth


model_urls = {
    "convnext_base": "https://download.pytorch.org/models/convnext_base-6075fbad.pth",
    "convnext_large": "https://download.pytorch.org/models/convnext_large-ea097f82.pth",
}


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm applied on channels-first (B, C, H, W) tensors."""
    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        x = x.permute(0, 3, 1, 2)
        return x


class CNBlock(nn.Module):
    """ConvNeXt block: depthwise conv -> LayerNorm -> pointwise expand -> GELU -> pointwise shrink."""
    def __init__(self, dim, layer_scale, stochastic_depth_prob, norm_layer=None):
        super().__init__()
        if norm_layer is None:
            norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.block = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim, bias=True),
            Permute([0, 2, 3, 1]),
            norm_layer(dim),
            nn.Linear(in_features=dim, out_features=4 * dim, bias=True),
            nn.GELU(),
            nn.Linear(in_features=4 * dim, out_features=dim, bias=True),
            Permute([0, 3, 1, 2]),
        )
        self.layer_scale = nn.Parameter(torch.ones(dim, 1, 1) * layer_scale)
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob, "row")

        self.kernel_sizes = [7]
        self.strides = [1]
        self.paddings = [3]

    def forward(self, input):
        result = self.layer_scale * self.block(input)
        result = self.stochastic_depth(result)
        result += input
        return result

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings


class CNBlockConfig:
    """Configuration for a single ConvNeXt stage."""
    def __init__(self, input_channels, out_channels, num_layers):
        self.input_channels = input_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

    def __repr__(self):
        return (f"CNBlockConfig(input_channels={self.input_channels}, "
                f"out_channels={self.out_channels}, num_layers={self.num_layers})")


class ConvNeXt_features(nn.Module):
    """ConvNeXt feature extractor (without classification head).
    
    Outputs spatial feature maps from the last stage, suitable for
    prototype-based matching.
    """
    def __init__(self, block_setting, stochastic_depth_prob=0.0,
                 layer_scale=1e-6, block=None, norm_layer=None, **kwargs):
        super(ConvNeXt_features, self).__init__()

        self.kernel_sizes = [4]
        self.strides = [4]
        self.paddings = [0]

        if block is None:
            block = CNBlock
        if norm_layer is None:
            norm_layer = partial(LayerNorm2d, eps=1e-6)

        layers: List[nn.Module] = []

        # Stem: patchify with 4x4 non-overlapping convolution
        firstconv_output_channels = block_setting[0].input_channels
        layers.append(
            Conv2dNormActivation(
                3, firstconv_output_channels,
                kernel_size=4, stride=4, padding=0,
                norm_layer=norm_layer, activation_layer=None, bias=True,
            )
        )

        total_stage_blocks = sum(cnf.num_layers for cnf in block_setting)
        stage_block_id = 0
        for cnf in block_setting:
            stage: List[nn.Module] = []
            for _ in range(cnf.num_layers):
                sd_prob = stochastic_depth_prob * stage_block_id / (total_stage_blocks - 1.0)
                stage.append(block(cnf.input_channels, layer_scale, sd_prob))
                stage_block_id += 1

                block_kernel_sizes, block_strides, block_paddings = stage[0].conv_info()
                self.kernel_sizes.extend(block_kernel_sizes)
                self.strides.extend(block_strides)
                self.paddings.extend(block_paddings)

            layers.append(nn.Sequential(*stage))
            if cnf.out_channels is not None:
                # Downsampling between stages
                layers.append(nn.Sequential(
                    norm_layer(cnf.input_channels),
                    nn.Conv2d(cnf.input_channels, cnf.out_channels, kernel_size=2, stride=2),
                ))
                self.kernel_sizes.extend([2])
                self.strides.extend([2])
                self.paddings.extend([0])

        self.features = nn.Sequential(*layers)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.features(x)

    def conv_info(self):
        return self.kernel_sizes, self.strides, self.paddings


def _convnext_base_config():
    """Return the ConvNeXt-Base block configuration."""
    return [
        CNBlockConfig(128, 256, 3),
        CNBlockConfig(256, 512, 3),
        CNBlockConfig(512, 1024, 27),
        CNBlockConfig(1024, None, 3),
    ]


def convnext_base_featuresCFP(pretrained=False, **kwargs):
    """Construct ConvNeXt-Base feature extractor for the CFP branch."""
    model = ConvNeXt_features(_convnext_base_config())
    return model


def convnext_base_featuresFFA(pretrained=False, **kwargs):
    """Construct ConvNeXt-Base feature extractor for the FFA branch."""
    model = ConvNeXt_features(_convnext_base_config())
    return model


def convnext_large_features(pretrained=False, **kwargs):
    """Construct ConvNeXt-Large feature extractor."""
    block_setting = [
        CNBlockConfig(192, 384, 3),
        CNBlockConfig(384, 768, 3),
        CNBlockConfig(768, 1536, 27),
        CNBlockConfig(1536, None, 3),
    ]
    model = ConvNeXt_features(block_setting)
    if pretrained:
        my_dict = load_state_dict_from_url(model_urls["convnext_large"])
        keys_to_remove = {k for k in my_dict if k.startswith('classifier')}
        for key in keys_to_remove:
            del my_dict[key]
        model.load_state_dict(my_dict)
    return model
