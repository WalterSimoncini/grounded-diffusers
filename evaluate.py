"""
    This script calculates the mean IoU for a grounding
    module checkpoint for a dataset.

    This is a work in progress!
"""
import os
import glob
import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from datetime import datetime
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from torch.utils.tensorboard import SummaryWriter

from seg_module import Segmodule
from utils import preprocess_mask, get_embeddings, plot_mask, calculate_iou


batch_size = 1
# grounding_checkpoint = "checkpoints/run-May04_21-07-32/checkpoint_100.pth"
grounding_checkpoint = "/home/lcur0899/grounded-diffusers1/saved_models/checkpoint_1000.pth"
device = torch.device("cuda")
model_name = "runwayml/stable-diffusion-v1-5"
# data_path = "dataset/samples/"
data_path = "/home/lcur0899/grounded-diffusers1/dataset/single_class/samples/"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load the segmentation module
seg_module = Segmodule().to(device)
seg_module.load_state_dict(torch.load(grounding_checkpoint, map_location=device), strict=True)

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)
pipeline_components = pipeline.components

# Setup tokenizer and the CLIP embedder
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]

test_samples = glob.glob(data_path + "*.pk")
total_steps = len(test_samples)

if total_steps == 0:
    print(
        f"{data_path} does not contain any data. "
        "make sure you added a trailing / to the data_path"
    )

seg_module.eval()

with torch.no_grad():
    # Do a single pass over all the data sample
    for step, file_path in enumerate(tqdm(test_samples)):
        image_path = test_samples[step]

        with open(file_path, "rb") as sample_file:
            sample = pickle.load(sample_file)

        # Unpack the sample data
        label = sample.label
        segmentation = sample.mask
        unet_features = sample.unet_features

        # Move the UNet features to cpu
        for key in unet_features.keys():
            unet_features[key] = [x.to(device) for x in unet_features[key]]

        # FIXME: We could precompute these
        prompt_embeddings = get_embeddings(
            tokenizer=tokenizer,
            embedder=embedder,
            device=device,
            prompt=f"a photograph of a {label}",
            batch_size=batch_size
        )

        # Predict the mask using the fusion module
        fusion_segmentation = seg_module(unet_features, prompt_embeddings)
        fusion_segmentation_pred = fusion_segmentation[0, 0, :, :]

        fusion_mask = preprocess_mask(mask=fusion_segmentation_pred.unsqueeze(0))

        iou = calculate_iou(segmentation, fusion_mask)

        print(iou)
    