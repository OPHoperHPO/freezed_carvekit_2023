"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

from carvekit.api.high import HiInterface


def test_init():
    HiInterface(batch_size_seg=1, batch_size_matting=4,
                device='cpu',
                seg_mask_size=160, matting_mask_size=1024,
                trimap_prob_threshold=1,
                trimap_dilation=2,
                trimap_erosion_iters=3,
                fp16=False)
    HiInterface(batch_size_seg=0, batch_size_matting=0,
                device='cpu',
                seg_mask_size=0, matting_mask_size=0,
                trimap_prob_threshold=0,
                trimap_dilation=0,
                trimap_erosion_iters=0,
                fp16=True
                )
