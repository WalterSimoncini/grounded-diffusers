"""
    This script updates MultiClassSegmentationSample instances
    to PromptsMultiClassSegmentationSample ones
"""
import os
import glob
import pickle

from data import PromptsMultiClassSegmentationSample


dataset_folder = "dataset/samples"
output_folder = "dataset/updated-samples"

os.makedirs(output_folder, exist_ok=True)

for path in glob.glob(os.path.join(dataset_folder, "*.pk")):
    sample = pickle.load(open(path, "rb"))
    sample_filename = path.split("/")[-1]
    
    updated_sample = PromptsMultiClassSegmentationSample(
        image=sample.image,
        masks=sample.masks,
        unet_features=sample.unet_features,
        labels=sample.labels,
        visual_labels=[],
        camera_parameters=[]
    )

    with open(os.path.join(output_folder, sample_filename), "wb") as sample_out_file:
        pickle.dump(updated_sample, sample_out_file)
