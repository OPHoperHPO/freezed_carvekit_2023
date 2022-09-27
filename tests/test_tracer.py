import pytest
import torch
from PIL import Image

from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7

def test_init():
    TracerUniversalB7(input_image_size=[640, 640], load_pretrained=True)
    TracerUniversalB7(input_image_size=640, load_pretrained=True)
    TracerUniversalB7(load_pretrained=False)
    TracerUniversalB7(fp16=True)


def test_preprocessing(tracer_model, converted_pil_image, black_image_pil):
    tracer_model = tracer_model(False)
    assert isinstance(tracer_model.data_preprocessing(converted_pil_image), torch.FloatTensor) is True
    assert isinstance(tracer_model.data_preprocessing(black_image_pil), torch.FloatTensor) is True



def test_postprocessing(tracer_model, converted_pil_image, black_image_pil):
    tracer_model = tracer_model(False)
    assert isinstance(tracer_model.data_postprocessing(torch.ones((1, 640, 640), dtype=torch.float64),
                                                      converted_pil_image), Image.Image)


def test_seg(tracer_model, image_pil, image_str, image_path, black_image_pil):
    tracer_model = tracer_model(False)
    tracer_model([image_pil])
    tracer_model([image_pil, image_str, image_path, black_image_pil])


def test_seg_with_fp12(tracer_model, image_pil, image_str, image_path, black_image_pil):
    tracer_model = tracer_model(True)
    tracer_model([image_pil])
    tracer_model([image_pil, image_str, image_path, black_image_pil])

