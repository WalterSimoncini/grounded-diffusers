"""
    This script exports the segmentation masks from
    a list of pickled MultiClassSegmentationSample
    instances
"""
import os
import pickle
import numpy as np

from PIL import Image
from seg_utils import plot_mask
from data import MultiClassSegmentationSample


root_dir = "/Users/walter/Desktop/picked-ronny/"
filenames = [
    ("1683282633_204.pk", "1683282633_204.png"),
    ("1683234291_421.pk", "1683234291_421.png"),
    ("1683274714_432.pk", "1683274714_432.png"),
    ("1683282633_25.pk", "1683282633_25.png"),
    ("1683282633_347.pk", "1683282633_347.png"),
    ("1683282633_489.pk", "1683282633_489.png")
]


for sample_filename, image_filename in filenames:
    sample = pickle.load(open(os.path.join(root_dir, sample_filename), "rb"))
    image = Image.open(os.path.join(root_dir, image_filename))

    mask_base_filename = image_filename.split(".png")[0]

    # For each filename extract the masks and the labels and
    # save the to the root directory
    for mask, label in zip(sample.masks, sample.labels):
        masked_image = Image.fromarray(plot_mask(np.array(image), np.expand_dims(mask, 0)))
        masked_image.save(os.path.join(root_dir, f"{mask_base_filename}_{label}_mask_rcnn.png"))
