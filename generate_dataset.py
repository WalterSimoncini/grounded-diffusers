import os
import json
import time
import pickle
import random
import argparse
import numpy as np

from data import PromptsMultiClassSegmentationSample
from mmdet.apis import init_detector, inference_detector

from utils import (
    get_default_device,
    load_stable_diffusion,
    has_mask_for_classes,
    DatasetGenerationType
)


parser = argparse.ArgumentParser(prog="dataset generation")

parser.add_argument("--output-dir", type=str, default="generated_dataset")
parser.add_argument("--n-classes", type=int, default=2)
parser.add_argument("--total-samples", type=int, default=500)
parser.add_argument("--pascal-class-split", type=int, default=1)
parser.add_argument("--model-name", type=str, default="runwayml/stable-diffusion-v1-5")
parser.add_argument(
    "--dataset-type",
    type=str,
    help="the type of dataset to be generated ['seen', 'seen_unseen', 'unseen']",
    choices=list([x.value for x in DatasetGenerationType])
)

args = parser.parse_args()

args.dataset_type = DatasetGenerationType(args.dataset_type)

device = get_default_device()
model_type = args.model_name.split("/")[-1]

images_dir = os.path.join(args.output_dir, "images")
samples_dir = os.path.join(args.output_dir, "samples")

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mask_rnn_config = {
  "config": "mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
  "checkpoint": "mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
}

os.makedirs(args.output_dir, exist_ok=True)
os.makedirs(images_dir, exist_ok=True)
os.makedirs(samples_dir, exist_ok=True)

# Load COCO and Pascal-VOC classes
coco_classes = open("mmdetection/demo/coco_80_class.txt").read().split("\n")
coco_classes = dict([(c, i) for i, c in enumerate(coco_classes)])

pascal_classes = open(f"VOC/class_split{args.pascal_class_split}.csv").read().split("\n")
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
pipeline, grounded_unet = load_stable_diffusion(
    model_name=args.model_name,
    device=device
)

for i in range(args.total_samples):
    # Pick classes
    if args.dataset_type == DatasetGenerationType.SEEN:
        picked_classes = random.sample(train_classes, args.n_classes)
    elif args.dataset_type == DatasetGenerationType.UNSEEN:
        picked_classes = random.sample(test_classes, args.n_classes)
    else:
        assert args.n_classes % 2 == 0, "the number of objects must be even for seen/unseen datasets"

        picked_classes = random.sample(train_classes, int(args.n_classes / 2))
        picked_classes += random.sample(test_classes, int(args.n_classes / 2))

    class_indices = [coco_classes[x] for x in picked_classes]

    # Build a prompt
    prompt_classes = " and a ".join(picked_classes)
    prompt = f"a photograph of a {prompt_classes}"

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
        visual_labels=None,
        camera_parameters=None
    )

    image.save(os.path.join(images_dir, f"{execution_time}_{i}.png"))

    pickle.dump(
        sample,
        open(os.path.join(samples_dir, f"{execution_time}_{i}.pk"), "wb")
    )
