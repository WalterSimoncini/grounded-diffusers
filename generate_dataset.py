import os
import time
import torch
import pickle
import random
import numpy as np

from data import MultiClassSegmentationSample
from mmdet.apis import init_detector, inference_detector
from utils import load_stable_diffusion, has_mask_for_classes, calculate_iou


# the number of classes in a single sample
n_classes = 2
total_samples = 5
output_dir = "/scratch/data_sd1/"
# output_dir = "/home/lcur0899/save_samples/test5/"
pascal_class_split = 1
device = torch.device("cuda")
model_name = "runwayml/stable-diffusion-v1-5"
model_type = model_name.split("/")[-1]

images_dir = os.path.join(output_dir, "images")
samples_dir = os.path.join(output_dir, "samples")

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

mask_rnn_config = {
    "config": "mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
    "checkpoint": "mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth",
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

# Load Mask R-CNN
pretrain_detector = init_detector(
    mask_rnn_config["config"], mask_rnn_config["checkpoint"], device=device
)

execution_time = int(time.time())
pipeline, grounded_unet = load_stable_diffusion(model_name=model_name, device=device)

for i in range(total_samples):
    # Pick classes
    picked_classes = random.sample(train_classes, n_classes)
    class_indices = [coco_classes[x] for x in picked_classes]

    # Build a prompt
    prompt_classes = " and a ".join(picked_classes)
    prompt = f"a photograph of a {prompt_classes}"

    print(f"generating sample {i} for classes {picked_classes}")

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
    _, segmentation = inference_detector(pretrain_detector, [array_image]).pop()

    has_masks = has_mask_for_classes(masks=segmentation, class_indices=class_indices)

    if not has_masks:
        continue

    segmented_classes = [
        segmentation[class_index][0].astype(int) for class_index in class_indices
    ]

    checkiou = calculate_iou(segmented_classes[0], segmented_classes[1])
    if checkiou > 0.95:
        continue
    print("check iou : ", checkiou)
    # For each sample we want to save
    #
    # the generated image
    # The mask R-CNN masks
    # the UNet features dict
    # The class names
    sample = MultiClassSegmentationSample(
        image=array_image,
        masks=segmented_classes,
        unet_features=unet_features,
        labels=picked_classes,
    )

    image.save(os.path.join(images_dir, f"{execution_time}_{i}.png"))

    pickle.dump(
        sample, open(os.path.join(samples_dir, f"{execution_time}_{i}.pk"), "wb")
    )
