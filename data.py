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
        labels: List[str]
    ) -> None:
        self.image = image
        self.masks = masks
        self.unet_features = unet_features
        self.labels = labels


class PromptsMultiClassSegmentationSample(MultiClassSegmentationSample):
    """
        Class used for a segmentation sample
        with multiple semantic classes and
        prompts with visual adjectives
    """
    def __init__(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        unet_features: Dict[str, List[torch.Tensor]],
        labels: List[str],
        visual_labels: List[str],
        camera_parameters: List[str]
    ) -> None:
        super().__init__(
            image=image,
            masks=masks,
            unet_features=unet_features,
            labels=labels
        )

        self.visual_labels = visual_labels
        self.camera_parameters = camera_parameters
 

class DINOPromptsMultiClassSegmentationSample(PromptsMultiClassSegmentationSample):
    """
        Class used for a segmentation sample with multiple
        semantic classes and prompts with visual adjectives,
        plus the DINO sakliency map extracted from the image
    """
    def __init__(
        self,
        image: np.ndarray,
        masks: np.ndarray,
        unet_features: Dict[str, List[torch.Tensor]],
        labels: List[str],
        visual_labels: List[str],
        camera_parameters: List[str],
        dino_saliency_map: torch.Tensor
    ) -> None:
        super().__init__(
            image=image,
            masks=masks,
            unet_features=unet_features,
            labels=labels,
            visual_labels=visual_labels,
            camera_parameters=camera_parameters
        )

        self.dino_saliency_map = dino_saliency_map
