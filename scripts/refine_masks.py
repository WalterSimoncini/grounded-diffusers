"""
    This script updates MultiClassSegmentationSample instances
    to PromptsMultiClassSegmentationSample ones
"""
import os
import glob
import pickle
import segmentation_refinement as refine

from PIL import Image
from tqdm import tqdm
from data import PromptsMultiClassSegmentationSample


dataset_folder = "dataset-old/seen_unseen/samples"
output_folder = "dataset-old/seen_unseen/refined-updated-samples"

refiner = refine.Refiner(device="cuda") 

os.makedirs(output_folder, exist_ok=True)

for path in tqdm(glob.glob(os.path.join(dataset_folder, "*.pk"))):
    sample = pickle.load(open(path, "rb"))
    sample_filename = path.split("/")[-1]
    sample_image = Image.fromarray(sample.image)

    refined_masks = [
        refiner.refine(sample.image, mask * 255., fast=False, L=900) / 255.
        for mask in sample.masks
    ]

    updated_sample = PromptsMultiClassSegmentationSample(
        image=sample.image,
        masks=refined_masks,
        unet_features=sample.unet_features,
        labels=sample.labels,
        visual_labels=[],
        camera_parameters=[]
    )

    with open(os.path.join(output_folder, sample_filename), "wb") as sample_out_file:
        pickle.dump(updated_sample, sample_out_file)
