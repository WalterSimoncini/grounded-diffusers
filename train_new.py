import os
import torch
import random
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim
import glob
import pickle
from datetime import datetime
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from torch.utils.tensorboard import SummaryWriter
from grounded_unet import GroundedUNet2DConditionModel
from torchvision import transforms
from PIL import Image
from seg_module import Segmodule
from utils import TrainingType, preprocess_mask
from mmdet.apis import init_detector, inference_detector


seed = 42
temp_dir = "temp"
outputs_dir = "outputs"
checkpoints_dir = "checkpoints"
device = torch.device("cuda")
pascal_class_split = 1
model_name = "runwayml/stable-diffusion-v1-5"
model_type = model_name.split("/")[-1]
images_path = "/home/lcur0899/grounded-diffusers/dataset/single_class/images/"
features_path = "/home/lcur0899/grounded-diffusers/dataset/single_class/samples/"

# FIXME: Not fully implemented
batch_size = 1
learning_rate = 1e-5

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

seed_everything(seed)

os.makedirs(temp_dir, exist_ok=True)
os.makedirs(outputs_dir, exist_ok=True)
os.makedirs(checkpoints_dir, exist_ok=True)

# Load COCO and Pascal-VOC classes
coco_classes = open("mmdetection/demo/coco_80_class.txt").read().split("\n")
coco_classes = dict([(c, i) for i, c in enumerate(coco_classes)])

pascal_classes = open(f"VOC/class_split{pascal_class_split}.csv").read().split("\n")
pascal_classes = [c.split(",")[0] for c in pascal_classes]

train_classes, test_classes = pascal_classes[:15], pascal_classes[15:]

images_list = glob.glob(images_path + "*.png")
convert_tensor = transforms.ToTensor()

# Load the segmentation module
seg_module = Segmodule().to(device)

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)
pipeline_components = pipeline.components

# Setup tokenizer and the CLIP embedder
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]


def get_embeddings(prompt: str):
    tokens = tokenizer(prompt, return_tensors="pt")
    tokens["input_ids"] = tokens["input_ids"].to("cuda")
    tokens["attention_mask"] = tokens["attention_mask"].to("cuda")
    token_embeddings = embedder(**tokens).last_hidden_state
    token_embeddings = token_embeddings[:, len(tokens["input_ids"]), :].to(device)
    return token_embeddings.repeat(batch_size, 1, 1)


# Start training
print(f"starting training")

current_time = datetime.now().strftime("%b%d_%H-%M-%S")

# Create folders to store checkpoints, training data, etc.
run_dir = os.path.join(checkpoints_dir, f"run-{current_time}")

run_logs_dir = os.path.join(run_dir, "logs")
training_dir = os.path.join(run_dir, "training")

os.makedirs(run_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

# Setup logger, optimizer and loss
torch_writer = SummaryWriter(log_dir=run_logs_dir)

loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(params=seg_module.parameters(), lr=learning_rate)

for i in range(len(images_list)):
    image_path = images_list[i]
    # image = Image.open(image_path)
    # image_tensor = convert_tensor(image)
    feature_file = os.path.basename(image_path).replace(".png", ".pk")
    full_feature_path = features_path + str(feature_file)
    objects = []
    with open(full_feature_path, "rb") as openfile:
        objects.append(pickle.load(openfile))

    for obj in objects:
        unet_features = obj.unet_features
        label = obj.label
        segmentation = obj.mask

    # Move the UNet features to cpu
    for key in unet_features.keys():
        unet_features[key] = [x.to(device) for x in unet_features[key]]

    prompt = f"a photograph of a {label}"
    prompt_embeddings = get_embeddings(prompt=prompt)
    fusion_segmentation = seg_module(unet_features, prompt_embeddings)
    class_index = coco_classes[label]
    fusion_segmentation_pred = torch.unsqueeze(
        fusion_segmentation[0, 0, :, :], 0
    ).unsqueeze(0)
    fusion_mask = preprocess_mask(mask=fusion_segmentation_pred)

    torchvision.utils.save_image(
        torch.from_numpy(fusion_mask),
        os.path.join(training_dir, f"vis_sample_{i}_{label}_pred_seg.png"),
        normalize=True,
        scale_each=True,
    )

    segmentation_new = (
        torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0).to(device)
    )
    loss = loss_fn(fusion_segmentation_pred, segmentation_new.float())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    torch_writer.add_scalar("train/loss", loss.item(), global_step=i)
    print(f"training step: {i}/{101}, loss: {loss}")

    if i % 50 == 0:
        print(f"saving checkpoint...")
        torch.save(
            seg_module.state_dict(), os.path.join(run_dir, f"checkpoint_{i}.pth")
        )
