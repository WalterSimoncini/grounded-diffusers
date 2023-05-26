"""
    This script calculates the mean IoU for a grounding
    module checkpoint for a given dataset.
"""
import os
import glob
import torch
import pickle
import argparse
import numpy as np

from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from seg_module import Segmodule
from utils import preprocess_mask, get_embeddings, calculate_iou, get_default_device


parser = argparse.ArgumentParser(prog="grounding  training")

parser.add_argument("--use-sd2", action="store_true")
parser.add_argument("--grounding-ckpt", type=str, default="checkpoints/mock-May25_20-45-58/checkpoint_3_0.pth")
parser.add_argument("--model-name", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument("--samples-path", type=str, default="val/samples/")

args = parser.parse_args()
device = get_default_device()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the segmentation module
seg_module = Segmodule(
    use_sd2=args.use_sd2,
    output_image_dim=768 if args.use_sd2 else 512
).to(device)

seg_module.load_state_dict(
    torch.load(args.grounding_ckpt, map_location=device), strict=True
)

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(args.model_name).to(device)
pipeline_components = pipeline.components

# Setup tokenizer and the CLIP embedder
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]

# Compute the token_id -> token text mapping
tokenizer_inverted_vocab = {
    v: k for k, v in tokenizer.get_vocab().items()
}

test_samples = glob.glob(args.samples_path + "*.pk")
total_steps = len(test_samples)

if total_steps == 0:
    raise ValueError(
        f"{args.samples_path} does not contain any data. "
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

    print(f"the mean IoU across the dataset at {args.samples_path} using the checkpoint {args.grounding_ckpt} is {mean_iou}")
