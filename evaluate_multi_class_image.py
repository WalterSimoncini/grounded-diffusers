"""
    This script calculates the mean IoU for a grounding
    module checkpoint for a dataset.

    This is a work in progress!
"""
import os
import glob
import torch
import pickle
import numpy as np

from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from seg_module import Segmodule
from utils import preprocess_mask, get_embeddings, calculate_iou, seed_everything


seed = 42
use_sd2 = False
batch_size = 1
grounding_checkpoint = "checkpoints/run-May13_11-16-27/checkpoint_9_1000.pth"
device = torch.device("cuda")
model_name = "stabilityai/stable-diffusion-2" if use_sd2 else "runwayml/stable-diffusion-v1-5"
data_path = "dataset-unseen/samples/"

seed_everything(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the segmentation module
seg_module = Segmodule(
    use_sd2=use_sd2,
    output_image_dim=768 if use_sd2 else 512
).to(device)

seg_module.load_state_dict(
    torch.load(grounding_checkpoint, map_location=device), strict=True
)

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)
pipeline_components = pipeline.components

# Setup tokenizer and the CLIP embedder
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]

# Compute the token_id -> token text mapping
tokenizer_inverted_vocab = {
    v: k for k, v in tokenizer.get_vocab().items()
}

test_samples = glob.glob(data_path + "*.pk")
total_steps = len(test_samples)

if total_steps == 0:
    raise ValueError(
        f"{data_path} does not contain any data. "
        "make sure you added a trailing / to the data_path"
    )

seg_module.eval()

with torch.no_grad():
    # Do a single pass over all the data sample
    iou_scores = []

    for step, file_path in enumerate(tqdm(test_samples)):
        image_path = test_samples[step]

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
            fusion_segmentation = seg_module(unet_features, label_embeddings[label])
            fusion_segmentation_pred = fusion_segmentation[0, 0, :, :]
            fusion_mask = preprocess_mask(mask=fusion_segmentation_pred.unsqueeze(0))

            sample_iou.append(calculate_iou(segmentation, fusion_mask))

        iou_scores.append(np.array(sample_iou).mean())

    mean_iou = np.array(iou_scores).mean()

    print(f"the mean IoU across the dataset at {data_path} using the checkpoint {grounding_checkpoint} is {mean_iou}")
