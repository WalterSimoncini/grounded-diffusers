import torch
import numpy as np

from typing import Dict, List


class SegmentationSample:
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
