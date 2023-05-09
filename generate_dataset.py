import os
import json
import time
import torch
import pickle
import random
import numpy as np

from data import PromptsMultiClassSegmentationSample
from mmdet.apis import init_detector, inference_detector
from utils import load_stable_diffusion, has_mask_for_classes
from utils.prompts import visual_adjectives_prompt


# the number of classes in a single sample
n_classes = 2
total_samples = 500
output_dir = "dataset_test"
pascal_class_split = 1
device = torch.device("cuda")
model_name = "runwayml/stable-diffusion-v1-5"
model_type = model_name.split("/")[-1]

images_dir = os.path.join(output_dir, "images")
samples_dir = os.path.join(output_dir, "samples")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mask_rnn_config = {
  "config": "mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
  "checkpoint": "mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
}

os.makedirs(output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# Load COCO and Pascal-VOC classes
coco_classes = open("mmdetection/demo/coco_80_class.txt").read().split("\n")
coco_classes = dict([(c, i) for i, c in enumerate(coco_classes)])

pascal_classes = open(f"VOC/class_split{pascal_class_split}.csv").read().split("\n")
pascal_classes = [c.split(",")[0] for c in pascal_classes]

train_classes, test_classes = pascal_classes[:15], pascal_classes[15:]

# Load visual adjectives for the classes and the possible camera configurations
visual_adjectives = json.loads(open("config/visual_adjectives.json").read())
camera_parameters = json.loads(open("config/camera.json").read())

# Load Mask R-CNN
pretrain_detector = init_detector(
  mask_rnn_config["config"],
  mask_rnn_config["checkpoint"],
  device=device
)

execution_time = int(time.time())
pipeline, grounded_unet = load_stable_diffusion(model_name=model_name, device=device)

for i in range(total_samples):
    # Pick classes
    picked_classes = random.sample(train_classes, n_classes)
    class_indices = [coco_classes[x] for x in picked_classes]

    # Add visual adjectives to classes
    visual_picked_classes = [
        visual_adjectives_prompt(
            label=label,
            visual_adjectives=visual_adjectives
        ) for label in picked_classes
    ]

    camera_angle = random.choice(camera_parameters["camera_angle"])
    camera_position = random.choice(camera_parameters["camera_position"])

    # Either pick a camera angle or a camera position
    camera_parameter = random.choice([camera_angle, camera_position])

    # Build a prompt
    prompt_classes = " and a ".join(visual_picked_classes)
    prompt = f"a photograph of a {prompt_classes} {camera_parameter}"

    print(f"generating sample {i} for classes {picked_classes}")
    print(f"the prompt is: {prompt}")

    grounded_unet.clear_grounding_features()

    # Sample an image
    image = pipeline(prompt).images[0]
    array_image = np.array(image)

    # Get the UNet features
    unet_features = grounded_unet.get_grounding_features()

    # Move the UNet features to cpu
    for key in unet_features.keys():
        unet_features[key] = [x.to("cpu") for x in unet_features[key]]

    # Get the segmentation
    _, segmentation = inference_detector(
        pretrain_detector,
        [array_image]
    ).pop()

    has_masks = has_mask_for_classes(
        masks=segmentation,
        class_indices=class_indices
    )

    if not has_masks:
        print(f"sample {i} is missing one or more masks")

        continue

    segmented_classes = [
        segmentation[class_index][0].astype(int)
        for class_index in class_indices
    ]

    # For each sample we want to save
    #
    # the generated image
    # The mask R-CNN masks
    # the UNet features dict
    # The class names
    sample = PromptsMultiClassSegmentationSample(
        image=array_image,
        masks=segmented_classes,
        unet_features=unet_features,
        labels=picked_classes,
        visual_labels=visual_picked_classes,
        camera_parameters=[camera_parameter]
    )

    image.save(os.path.join(images_dir, f"{execution_time}_{i}.png"))

    pickle.dump(
        sample,
        open(os.path.join(samples_dir, f"{execution_time}_{i}.pk"), "wb")
    )
