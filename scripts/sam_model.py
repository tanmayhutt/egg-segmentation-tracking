"""Loading SAM ViT-B on the best available device."""
import numpy as np
import torch
from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry

CHECKPOINT = "weights/sam_vit_b_01ec64.pth"


def load_sam():
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    return sam_model_registry["vit_b"](checkpoint=CHECKPOINT).to(device)


def _float32_coords(transform):
    # Apple's GPU (MPS) has no 64-bit floats, but SAM builds click coordinates as 64-bit
    # numpy arrays. Convert them to 32-bit before they reach the GPU.
    to_coords = transform.apply_coords
    transform.apply_coords = lambda c, size: to_coords(c, size).astype(np.float32)


def automatic_generator(sam):
    """SAM's automatic mode: clicks a 32x32 grid of points and returns every object found."""
    generator = SamAutomaticMaskGenerator(sam)
    _float32_coords(generator.predictor.transform)
    return generator


def point_predictor(sam):
    """SAM's prompted mode: give it clicks on an object, get that object's mask."""
    predictor = SamPredictor(sam)
    _float32_coords(predictor.transform)
    return predictor
