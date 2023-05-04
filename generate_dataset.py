import os
import time
import torch
import pickle
import random
import numpy as np

from data import SegmentationSample
from utils import load_stable_diffusion
from mmdet.apis import init_detector, inference_detector


total_samples = 500
output_dir = "dataset"
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

# Load Mask R-CNN
pretrain_detector = init_detector(
  mask_rnn_config["config"],
  mask_rnn_config["checkpoint"],
  device=device
)

execution_time = int(time.time())
pipeline, grounded_unet = load_stable_diffusion(model_name=model_name, device=device)

for i in range(total_samples):
    print(f"generating sample {i}")

    # Pick a class
    # FIXME: Use uniform sampling
    picked_class = random.choice(train_classes)
    class_index = coco_classes[picked_class]

    prompt = f"a photograph of a {picked_class}"

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

    # Get Mask R-CNN mask tensor
    if len(segmentation[class_index]) == 0:
      print(f"no mask detected for class {picked_class}. skipping")

      continue

    segmented_class = segmentation[class_index][0].astype(int)

    # For each sample we want to save
    #
    # the generated image
    # the UNet features dict
    # The class name
    # The mask R-CNN mask
    sample = SegmentationSample(
        image=array_image,
        mask=segmented_class,
        unet_features=unet_features,
        label=picked_class
    )
    
    image.save(os.path.join(images_dir, f"{execution_time}_{i}.png"))
    pickle.dump(
        sample,
        open(os.path.join(samples_dir, f"{execution_time}_{i}.pk"), "wb")
    )
