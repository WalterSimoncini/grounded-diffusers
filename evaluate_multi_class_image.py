"""
    This script calculates the mean IoU for a grounding
    module checkpoint for a dataset.

    This is a work in progress!
"""
import os
import glob
import torch
import pickle
from tqdm import tqdm
from diffusers import StableDiffusionPipeline
from seg_module import Segmodule
from utils import preprocess_mask, get_embeddings, calculate_iou


batch_size = 1
grounding_checkpoint = "checkpoints/run-May04_21-07-32/checkpoint_100.pth"
device = torch.device("cuda")
model_name = "runwayml/stable-diffusion-v1-5"
data_path = "dataset/samples/"

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Load the segmentation module
seg_module = Segmodule().to(device)
seg_module.load_state_dict(
    torch.load(grounding_checkpoint, map_location=device), strict=True
)

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
        labels = sample.labels
        segmentation = sample.masks
        unet_features = sample.unet_features

        # Move the UNet features to cpu
        for key in unet_features.keys():
            unet_features[key] = [x.to(device) for x in unet_features[key]]

        total_iou = 0
        for label in range(len(labels)):
            prompt_embeddings = get_embeddings(
                tokenizer=tokenizer,
                embedder=embedder,
                device=device,
                prompt=f"a photograph of a {label}",
                batch_size=batch_size,
            )

            # Predict the mask using the fusion module
            fusion_segmentation = seg_module(unet_features, prompt_embeddings)
            fusion_segmentation_pred = fusion_segmentation[0, 0, :, :]
            fusion_mask = preprocess_mask(mask=fusion_segmentation_pred.unsqueeze(0))
            iou = calculate_iou(segmentation[label], fusion_mask)
            total_iou += iou

        print(iou)
