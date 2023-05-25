"""
    This script generates a sample given a prompt and segments
    the given classes using both Mask R-CNN and the Segmentation
    Module
"""
import os
import time
import torch
import argparse
import torchvision
import numpy as np

from PIL import Image
from seg_module import Segmodule
from mmdet.apis import init_detector, inference_detector
from utils import preprocess_mask, get_embeddings, plot_mask, load_stable_diffusion


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(prog="grounded generation")

parser.add_argument("--use-sd2", action="store_true")
parser.add_argument("--output-dir", type=str, default="generations")
parser.add_argument("--prompt", type=str, default="a photograph of a cat and a dog")
parser.add_argument("--classes", type=str, default="cat,dog")
parser.add_argument("--grounding_ckpt", type=str, default="checkpoints/normal_arch_checkpoint.pth")
parser.add_argument("--seed", type=int, default=2147483647)

args = parser.parse_args()

rand_generator = torch.Generator()
rand_generator.manual_seed(args.seed)

device = torch.device("cuda")
model_name = "stabilityai/stable-diffusion-2" if args.use_sd2 else "runwayml/stable-diffusion-v1-5"

mask_rnn_config = {
  "config": "mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
  "checkpoint": "mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
}

# Load COCO classes
coco_classes = open("mmdetection/demo/coco_80_class.txt").read().split("\n")
coco_classes = dict([(i, c) for i, c in enumerate(coco_classes)])

# Add the current timestamp to the output folder
args.output_dir = os.path.join(args.output_dir, str(int(time.time())))

os.makedirs(args.output_dir, exist_ok=True)

# Load the segmentation module
seg_module = Segmodule(
    use_sd2=args.use_sd2,
    output_image_dim=768 if args.use_sd2 else 512
).to(device)

seg_module.load_state_dict(
    torch.load(args.grounding_ckpt, map_location=device), strict=True
)

# Load Mask R-CNN
pretrain_detector = init_detector(
  mask_rnn_config["config"],
  mask_rnn_config["checkpoint"],
  device=device
)

# Load the stable diffusion pipeline
pipeline, grounded_unet = load_stable_diffusion(model_name=model_name, device=device)
pipeline_components = pipeline.components

# Setup tokenizer and the CLIP embedder
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]

# Compute the token_id -> token text mapping
tokenizer_inverted_vocab = {
    v: k for k, v in tokenizer.get_vocab().items()
}

seg_module.eval()

with torch.no_grad():
    classes = [c.strip() for c in args.classes.split(",")]

    print(f"Generating sample using prompt: {args.prompt}")
    print(f"The target classes are: {classes}")

    grounded_unet.clear_grounding_features()

    # Sample an image
    image = pipeline(args.prompt, generator=rand_generator).images[0]
    array_image = np.array(image)

    # Get the Mask R-CNN segmentation
    _, mask_rcnn_segmentations = inference_detector(
        pretrain_detector,
        [array_image]
    ).pop()

    # Save all masks from Mask R-CNN
    for i, masks in enumerate(mask_rcnn_segmentations):
        if len(masks) == 0:
            continue

        masked_image = Image.fromarray(plot_mask(
            np.array(image),
            np.expand_dims(masks[0], 0)
        ))

        masked_image.save(
            os.path.join(
                args.output_dir,
                f"masked_image_{coco_classes[i]}_mask_rcnn.png"
            )
        )

    # Save the genearted image
    image.save(os.path.join(args.output_dir, f"sd_image.png"))

    # Get the UNet features
    unet_features = grounded_unet.get_grounding_features()

    # Extract embeddings for each individual class,
    # using the prompt "a photograph of a {x}"
    single_class_embeddings = {}

    for label in classes:
        single_class_embeddings[label] = get_embeddings(
            tokenizer=tokenizer,
            embedder=embedder,
            device=device,
            prompt=f"a photograph of a {label}",
            labels=[label],
            inverted_vocab=tokenizer_inverted_vocab
        )[label]

    all_fusion_masks = []

    for label in classes:
        embedding = single_class_embeddings[label]

        # Subtract the embeddings from all other classes
        for other_label in set(classes) - set([label]):
            embedding -= single_class_embeddings[other_label]

        fusion_segmentation = seg_module(unet_features, embedding)
        fusion_segmentation_pred = fusion_segmentation[0, 0, :, :]
        fusion_mask = preprocess_mask(mask=fusion_segmentation_pred.unsqueeze(0))

        # Save the fusion mask
        torchvision.utils.save_image(
            torch.from_numpy(fusion_mask),
            os.path.join(args.output_dir, f"mask_{label}_segmodule.png"),
            normalize=True,
            scale_each=True,
        )

        # Also plot the mask over the image
        masked_image = Image.fromarray(plot_mask(np.array(image), np.expand_dims(fusion_mask, 0)))
        masked_image.save(os.path.join(args.output_dir, f"masked_image_{label}_segmodule.png"))

        all_fusion_masks.append(fusion_mask)

        # Mask the original image and save the cutted out portion
        expanded_mask = np.stack([fusion_mask.astype(int)] * 3, axis=-1)

        extracted_image = np.array(image)
        extracted_image[expanded_mask == 0] = 0

        Image.fromarray(extracted_image).save(
            os.path.join(args.output_dir, f"extracted_{label}_segmodule.png")
        )

    all_fusion_image = Image.fromarray(plot_mask(np.array(image), all_fusion_masks))
    all_fusion_image.save(os.path.join(args.output_dir, f"segmodule_all_masks.png"))
