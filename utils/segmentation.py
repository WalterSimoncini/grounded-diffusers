import torch
import numpy as np


def calculate_iou(mask1, mask2):
    assert mask1.shape == mask2.shape, "Masks must have the same shape."

    # Calculate intersection and union.
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Count the number of pixels in the intersection and union.
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    # Avoid division by zero.
    if union_count == 0:
        return 0

    iou = intersection_count / union_count

    return iou


def preprocess_mask(mask: torch.Tensor):
    probabilities = torch.sigmoid(mask)

    mask = torch.zeros_like(probabilities, dtype=torch.float32)
    mask[probabilities > 0.5] = 1

    return mask[0].cpu().numpy()
