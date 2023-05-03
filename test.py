import os
import pdb
import torch
import torchvision
import numpy as np

from PIL import Image
from diffusers import StableDiffusionPipeline

from seg_module import Segmodule
from grounded_unet import GroundedUNet2DConditionModel
from utils import plot_mask, calculate_iou, preprocess_mask

from mmdet.apis import init_detector, inference_detector


class_name = "lion"
temp_unet_dir = "temp"
outputs_dir = "outputs"
# FIXME: This is not fully integrated
batch_size = 1
device = torch.device("cuda")
model_name = "runwayml/stable-diffusion-v1-5"
prompt = f"a picture of a {class_name} and a dog on a field"
grounding_module_checkpoint = "mmdetection/checkpoint/grounding_module.pth"


os.environ["CUDA_VISIBLE_DEVICES"] = "3"

mask_rnn_config = {
  "config": "mmdetection/configs/swin/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco.py",
  "checkpoint": "mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth"
}

os.makedirs(temp_unet_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)

# Load Mask R-CNN
pretrain_detector = init_detector(
  mask_rnn_config["config"],
  mask_rnn_config["checkpoint"],
  device=device
)

# Load the segmentation module
seg_module = Segmodule().to(device)
# seg_module.load_state_dict(torch.load(grounding_module_checkpoint, map_location="cpu"), strict=True)

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)

# Save the pretrained UNet to disk
pipeline_components = pipeline.components
pipeline_components["unet"].save_pretrained(os.path.join(temp_unet_dir, "unet_model"))

# Reload the UNet as the grounded subclass
grounded_unet = GroundedUNet2DConditionModel.from_pretrained(
    os.path.join(temp_unet_dir, "unet_model")
).to(device)

pipeline_components["unet"] = grounded_unet

pipeline = StableDiffusionPipeline(**pipeline_components)

# Generate an image
image = pipeline(prompt).images[0]

# Save the generated image
image.save(os.path.join(outputs_dir, "sd_image.png"))

# Obtain the feature maps from the UNet
unet_features = grounded_unet.get_grounding_features()

# NOTE: Print out UNet features
for key in unet_features.keys():
  values = unet_features[key]
  unet_features[key] = [x.to(device) for x in values]

  print(f"{len(values)} for {key}. shapes: {[x.shape for x in values]}")

# Run Mask R-CNN to obtain the segmentation mask
array_image = np.array(image)

_, segmentation = inference_detector(pretrain_detector, [array_image]).pop()
segmented_classes = [(i, x) for i, x in enumerate(segmentation) if len(x) > 0]

for i, masks in segmented_classes:
  # FIXME: We may have to preprocess the data here
  masked_image = Image.fromarray(plot_mask(array_image, np.array(masks)))
  masked_image.save(os.path.join(outputs_dir, f"class_{i}_masks.png"))

# Tokenize the class prompt
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]

# FIXME: We may want to customize the query
tokens = tokenizer(prompt, return_tensors="pt")

tokens["input_ids"] = tokens["input_ids"].to("cuda")
tokens["attention_mask"] = tokens["attention_mask"].to("cuda")

token_embeddings = embedder(**tokens).last_hidden_state

# FIXME: Not sure about this, but we want the embedding for the last token
# FIXME: How do we handle multiple objects in the prompt?
token_embeddings = token_embeddings[:, len(tokens["input_ids"]), :].to(device)

# Repeat embeddings for a batch as needed
token_embeddings = token_embeddings.repeat(batch_size, 1, 1)

fusion_segmentation = seg_module(unet_features, token_embeddings)

# Pick the last mask as the testing target
test_mask = torch.from_numpy(np.array(masks.pop()).astype(int))
test_mask = test_mask.float().unsqueeze(0).unsqueeze(0)

# Preprocess the debug Mask R-CNN mask and save it to disk
debug_test_mask = preprocess_mask(mask=test_mask)
debug_fusion_mask = preprocess_mask(mask=fusion_segmentation)

# Save masked images
rcnn_image = Image.fromarray(plot_mask(array_image, debug_test_mask, alpha=0.9, indexlist=[0]))
rcnn_image.save(os.path.join(outputs_dir, f"debug_maskrcnn.png"))

fusion_image = Image.fromarray(plot_mask(array_image, debug_fusion_mask, alpha=0.9, indexlist=[0]))
fusion_image.save(os.path.join(outputs_dir, f"debug_fusion.png"))

# Calculate the IoU score
iou = calculate_iou(debug_test_mask, debug_fusion_mask)

print(f"IoU score: {iou}")

# Save masks
torchvision.utils.save_image(
  torch.from_numpy(debug_test_mask),
  os.path.join(outputs_dir, f"debug_maskrcnn_mask.png"),
  normalize=True,
  scale_each=True
)

torchvision.utils.save_image(
  torch.from_numpy(debug_fusion_mask),
  os.path.join(outputs_dir, f"debug_fusion_mask.png"),
  normalize=True,
  scale_each=True
)
