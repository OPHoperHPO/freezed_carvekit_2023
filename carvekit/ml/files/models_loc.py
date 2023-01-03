"""
Source url: https://github.com/OPHoperHPO/image-background-remove-tool
Author: Nikita Selin (OPHoperHPO)[https://github.com/OPHoperHPO].
License: Apache License 2.0
"""
import pathlib
from carvekit.ml.files import checkpoints_dir
from carvekit.utils.download_models import downloader


def u2net_full_pretrained() -> pathlib.Path:
    """Returns u2net pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("u2net.pth")


def basnet_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("basnet.pth")


def deeplab_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("deeplab.pth")


def fba_pretrained() -> pathlib.Path:
    """Returns basnet pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("fba_matting.pth")


def tracer_b7_pretrained() -> pathlib.Path:
    """Returns TRACER with EfficientNet v1 b7 encoder pretrained model location

    Returns:
        pathlib.Path to model location
    """
    return downloader("tracer_b7.pth")


def scene_classifier_pretrained() -> pathlib.Path:
    """Returns scene classifier pretrained model location
    This model is used to classify scenes into 3 categories: hard, soft, digital

    hard - scenes with hard edges, such as objects, buildings, etc.
    soft - scenes with soft edges, such as portraits, hairs, animal, etc.
    digital - digital scenes, such as screenshots, graphics, etc.

    more info: https://huggingface.co/Carve/scene_classifier

    Returns:
        pathlib.Path to model location
    """
    return downloader("scene_classifier.pth")


def download_all():
    u2net_full_pretrained()
    fba_pretrained()
    deeplab_pretrained()
    basnet_pretrained()
    tracer_b7_pretrained()
