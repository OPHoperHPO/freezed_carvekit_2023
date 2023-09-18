"""
Source url: https://github.com/Karel911/TRACER
Author: Min Seok Lee and Wooseok Shin
Modified by Nikita Selin [OPHoperHPO](https://github.com/OPHoperHPO).
License: Apache License 2.0
Changes:
    - Refactored code
    - Removed unused code
    - Added comments
"""
from typing import List, Optional, Mapping, Any

import loguru
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from carvekit import version
from carvekit.ml.arch.tracerb7.att_modules import (
    RFB_Block,
    aggregation,
    ObjectAttention,
)
from carvekit.ml.arch.tracerb7.efficientnet import EfficientEncoderB7
from carvekit.ml.files import checkpoints_dir
from carvekit.utils.models_utils import save_optimized_model, get_optimized_model


class TracerJitTraced(nn.Module):
    def __init__(self,
                 encoder: EfficientEncoderB7,
                 features_channels: Optional[List[int]] = None,
                 rfb_channel: Optional[List[int]] = None, ):
        super().__init__()
        path = checkpoints_dir.joinpath("optimized_models").joinpath(f"tracer-b7-{version}.pt")
        if path.exists():
            try:
                self.net = get_optimized_model(path)
            except BaseException as e:
                loguru.logger.warning(f"Failed to load optimized model: {e}")
                path.unlink()
        if not path.exists():
            loguru.logger.info("Optimizing Tracer model! This runs only once. Please wait...")
            with torch.jit.optimized_execution(True):
                net = TracerDecoder(encoder=encoder,
                                    features_channels=features_channels,
                                    rfb_channel=rfb_channel)
                net.eval()
                self.net = torch.jit.trace(net, (
                    torch.rand(*[1, 3, 960, 960])))
            loguru.logger.info("Optimized Tracer model!")
            if not path.parent.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
            save_optimized_model(self.net, path)
        else:
            self.net = get_optimized_model(path)
        self.is_optimized = False

    def load_state_dict(self, state_dict: Mapping[str, Any],
                        strict: bool = True):
        self.net.load_state_dict(state_dict, strict=strict)

    def forward(self, *args, **kwargs):
        return self.net(*args, **kwargs)


class TracerDecoder(nn.Module):
    """Tracer Decoder"""

    def __init__(
            self,
            encoder: EfficientEncoderB7,
            features_channels: Optional[List[int]] = None,
            rfb_channel: Optional[List[int]] = None,
    ):
        """
        Initialize the tracer decoder.

        Args:
            encoder: The encoder to use.
            features_channels: The channels of the backbone features at different stages. default: [48, 80, 224, 640]
            rfb_channel: The channels of the RFB features. default: [32, 64, 128]
        """
        super().__init__()
        if rfb_channel is None:
            rfb_channel = [32, 64, 128]
        if features_channels is None:
            features_channels = [48, 80, 224, 640]
        self.encoder = encoder
        self.features_channels = features_channels

        # Receptive Field Blocks
        features_channels = rfb_channel
        self.rfb2 = RFB_Block(self.features_channels[1], features_channels[0])
        self.rfb3 = RFB_Block(self.features_channels[2], features_channels[1])
        self.rfb4 = RFB_Block(self.features_channels[3], features_channels[2])

        # Multi-level aggregation
        self.agg = aggregation(features_channels)

        # Object Attention
        self.ObjectAttention2 = ObjectAttention(
            channel=self.features_channels[1], kernel_size=3
        )
        self.ObjectAttention1 = ObjectAttention(
            channel=self.features_channels[0], kernel_size=3
        )

    def forward(self, inputs: torch.Tensor) -> Tensor:
        """
        Forward pass of the tracer decoder.

        Args:
            inputs: Preprocessed images.

        Returns:
            Tensors of segmentation masks and mask of object edges.
        """
        features = self.encoder(inputs)
        x3_rfb = self.rfb2(features[1])
        x4_rfb = self.rfb3(features[2])
        x5_rfb = self.rfb4(features[3])

        D_0 = self.agg(x5_rfb, x4_rfb, x3_rfb)

        ds_map0 = F.interpolate(D_0, scale_factor=8, mode="bilinear")

        D_1 = self.ObjectAttention2(D_0, features[1])
        ds_map1 = F.interpolate(D_1, scale_factor=8, mode="bilinear")

        ds_map = F.interpolate(D_1, scale_factor=2, mode="bilinear")
        D_2 = self.ObjectAttention1(ds_map, features[0])
        ds_map2 = F.interpolate(D_2, scale_factor=4, mode="bilinear")

        final_map = (ds_map2 + ds_map1 + ds_map0) / 3

        return torch.sigmoid(final_map)
