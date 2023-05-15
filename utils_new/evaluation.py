import torch
import pickle
import numpy as np

from tqdm import tqdm
from typing import List
from .prompts import get_embeddings
from .segmentation import preprocess_mask, calculate_iou


def evaluate_seg_model(
    model,
    tokenizer,
    embedder,
    device: torch.device,
    tokenizer_inverted_vocab: dict,
    samples_paths: List[str]
):
    iou_scores = []

    for _, file_path in enumerate(tqdm(samples_paths)):
        with open(file_path, "rb") as sample_file:
            sample = pickle.load(sample_file)

        # Unpack the sample data
        labels = sample.labels
        segmentations = sample.masks
        unet_features = sample.unet_features

        # Move the UNet features to cpu
        for key in unet_features.keys():
            unet_features[key] = [x.to(device) for x in unet_features[key]]

        sample_iou = []

        prompt = " and a ".join(labels)
        label_embeddings = get_embeddings(
            tokenizer=tokenizer,
            embedder=embedder,
            device=device,
            prompt=prompt,
            labels=labels,
            inverted_vocab=tokenizer_inverted_vocab
        )
    
        for label, segmentation in zip(labels, segmentations):
            fusion_segmentation = model(unet_features, label_embeddings[label])
            fusion_segmentation_pred = fusion_segmentation[0, 0, :, :]
            fusion_mask = preprocess_mask(mask=fusion_segmentation_pred.unsqueeze(0))

            sample_iou.append(calculate_iou(segmentation, fusion_mask))

        iou_scores.append(np.array(sample_iou).mean())

    return np.array(iou_scores).mean()