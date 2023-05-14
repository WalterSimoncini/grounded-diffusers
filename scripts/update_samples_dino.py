"""
    This script updates PromptsMultiClassSegmentationSample instances
    to DINOPromptsMultiClassSegmentationSample ones, extracting the DINO
    saliency map in the process
"""
import os
import glob
import torch
import pickle
import torchvision.transforms as transforms

from PIL import Image
from tqdm import tqdm
from seg_utils.dino_extractor import ViTExtractor
from data import DINOPromptsMultiClassSegmentationSample


device = torch.device("cuda")
dataset_folder = "dataset-unseen/samples"
output_folder = "dataset-unseen/updated-samples"

os.makedirs(output_folder, exist_ok=True)

dino_image_size = 512
transform = transforms.ToTensor()
extractor = ViTExtractor("dino_vits8", 8, device=device)

with torch.no_grad():
    for path in tqdm(glob.glob(os.path.join(dataset_folder, "*.pk"))):
        sample = pickle.load(open(path, "rb"))
        sample_filename = path.split("/")[-1]

        image = extractor.preprocess(transform(Image.fromarray(sample.image)), dino_image_size)
        saliency_map = extractor.extract_saliency_maps(image.to(device)).cpu()

        updated_sample = DINOPromptsMultiClassSegmentationSample(
            image=sample.image,
            masks=sample.masks,
            unet_features=sample.unet_features,
            labels=sample.labels,
            visual_labels=getattr(sample, "visual_labels", None),
            camera_parameters=getattr(sample, "camera_parameters", None),
            dino_saliency_map=saliency_map
        )

        with open(os.path.join(output_folder, sample_filename), "wb") as sample_out_file:
            pickle.dump(updated_sample, sample_out_file)
