import torch
import numpy as np

from typing import Dict, List


class SegmentationSample:
    """
        Class used for a segmentation sample
        with one semantic class
    """
    def __init__(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        unet_features: Dict[str, List[torch.Tensor]],
        label: str
    ) -> None:
        self.image = image
        self.mask = mask
        self.unet_features = unet_features
        self.label = label


class MultiClassSegmentationSample:
    """
        Class used for a segmentation sample
        with multiple semantic classes
    """
    def __init__(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        unet_features: Dict[str, List[torch.Tensor]],
        labels: str
    ) -> None:
        self.image = image
        self.masks = masks
        self.unet_features = unet_features
        self.labels = labels
