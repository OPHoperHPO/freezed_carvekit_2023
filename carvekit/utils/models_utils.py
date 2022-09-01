"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""

import random
import warnings
from typing import Union

import torch
from torch import autocast


class EmptyAutocast(object):
    """
    Empty class for disable any autocasting.
    """

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        return

    def __call__(self, func):
        return


def get_precision_autocast(device="cpu", fp16=True, override_dtype=None) -> Union[EmptyAutocast, autocast]:
    """
    Returns precision and autocast settings for given device and fp16 settings.
    Args:
        device: Device to get precision and autocast settings for.
        fp16: Whether to use fp16 precision.
        override_dtype: Override dtype for autocast.

    Returns:
        Autocast object
    """
    dtype = torch.float32
    if device == "cpu" and fp16:
        warnings.warn("Accuracy BFP16 has experimental support on the CPU. "
                      "This may result in an unexpected reduction in quality.")
        dtype = torch.bfloat16  # Using bfloat16 for CPU, since autocast is not supported for float16
    if "cuda" in device and fp16:
        dtype = torch.float16

    if override_dtype is not None:
        dtype = override_dtype

    if dtype == torch.float32 and device == "cpu":
        return EmptyAutocast()

    return torch.autocast(
        device_type=device,
        dtype=dtype,
        enabled=True)


def fix_seed(seed=42):
    """Sets fixed random seed

    Args:
        seed: Random seed to be set
    """
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.benchmark = False
    return True


def suppress_warnings():
    # Suppress PyTorch 1.11.0 warning associated with changing order of args in nn.MaxPool2d layer,
    # since source code is not affected by this issue and there aren't any other correct way to hide this message.
    warnings.filterwarnings("ignore",
                            category=UserWarning,
                            message="Note that order of the arguments: ceil_mode and "
                                    "return_indices will changeto match the args list "
                                    "in nn.MaxPool2d in a future release.",
                            module="torch")
