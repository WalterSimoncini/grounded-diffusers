import torch
import numpy as np

from enum import Enum


class TrainingType(Enum):
  SINGLE = "single"
  TWO = "two"
  RANDOM = "random"


def plot_mask(img, masks, colors=None, alpha=0.8,indexlist=[0,1]) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    H, W = masks.shape[0], masks.shape[1]
    color_list=[[255,97,0],[128,42,42],[220,220,220],[255,153,18],[56,94,15],[127,255,212],[210,180,140],[221,160,221],[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],[0,255,128],[0,255,255],[0,128,255],[0,0,255],[128,0,255],[255,0,255],[255,0,128]]*6
    final_color_list=[np.array([[i]*512]*512) for i in color_list]
    
    background=np.ones(img.shape)*255
    count=0
    colors=final_color_list[indexlist[count]]

    for mask, color in zip(masks, colors):
        color=final_color_list[indexlist[count]]
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha,background*0.4+img*0.6 )
        count+=1

    return img.astype(np.uint8)


def calculate_iou(mask1, mask2, threshold=180):
    assert mask1.shape == mask2.shape, "Masks must have the same shape."
    
    # Apply thresholding to the masks to convert them to binary format.
    binary_mask1 = np.where(mask1 >= threshold, 1, 0)
    binary_mask2 = np.where(mask2 >= threshold, 1, 0)

    # Calculate intersection and union.
    intersection = np.logical_and(binary_mask1, binary_mask2)
    union = np.logical_or(binary_mask1, binary_mask2)

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
