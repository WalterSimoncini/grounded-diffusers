import os
import glob
import torch
import pickle
import torchvision
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from PIL import Image
from datetime import datetime
from pytorch_lightning import seed_everything
from diffusers import StableDiffusionPipeline
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from seg_module import Segmodule
from seg_utils.evaluation import evaluate_seg_model
from seg_utils import preprocess_mask, get_embeddings, plot_mask
from loss_fn import BCEDiceLoss, DiceLoss, BCELogCoshDiceLoss

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from seg_utils.dino_extractor import ViTExtractor
import pyiqa


seed = 42
n_epochs = 10
use_sd2 = False
visualize_examples = True
pascal_class_split = 1
loss_name = "log_cosh"
checkpoints_dir = "/home/lcur0899/save_samples/trial_checkpoint/"
checkpoints_dir = "checkpoint_dino_scoring/"
device = torch.device("cuda")
model_name = "stabilityai/stable-diffusion-2" if use_sd2 else "runwayml/stable-diffusion-v1-5"
train_data_path = "/nfs/scratch/data_sd15/samples/"
train_images_path = "/nfs/scratch/data_sd15/images/"
validation_data_path = "/nfs/scratch/data_sd15_val/samples/"
validation_images_path = "/nfs/scratch/data_sd15_val/images/"
learning_rate = 1e-5

seed_everything(seed)

os.makedirs(checkpoints_dir, exist_ok=True)

# Load COCO and Pascal-VOC classes
coco_classes = open("/home/lcur0899/grounded-diffusers1/mmdetection/demo/coco_80_class.txt").read().split("\n")
coco_classes = dict([(c, i) for i, c in enumerate(coco_classes)])

pascal_classes = open(f"/home/lcur0899/grounded-diffusers1/VOC/class_split{pascal_class_split}.csv").read().split("\n")
pascal_classes = [c.split(",")[0] for c in pascal_classes]

train_classes, test_classes = pascal_classes[:15], pascal_classes[15:]

# Load the segmentation module
seg_module = Segmodule(
    use_sd2=use_sd2,
    output_image_dim=768 if use_sd2 else 512,
    dropout_rate=0.1
).to(device)

# Load the stable diffusion pipeline
pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)
pipeline_components = pipeline.components

# Setup tokenizer and the CLIP embedder
tokenizer = pipeline_components["tokenizer"]
embedder = pipeline_components["text_encoder"]

# Compute the token_id -> token text mapping
tokenizer_inverted_vocab = {
    v: k for k, v in tokenizer.get_vocab().items()
}

# Create folders to store checkpoints, training data, etc.
current_time = datetime.now().strftime("%b%d_%H-%M-%S")
run_dir = os.path.join(checkpoints_dir, f"run-{current_time}")

run_logs_dir = os.path.join(run_dir, "logs")
training_dir = os.path.join(run_dir, "training")

os.makedirs(run_dir, exist_ok=True)
os.makedirs(training_dir, exist_ok=True)

# Setup logger, optimizer and loss
torch_writer = SummaryWriter(log_dir=run_logs_dir)

loss_fn = {
    "bce": nn.BCEWithLogitsLoss(),
    "dice": DiceLoss(),
    "bce_dice": BCEDiceLoss(),
    "log_cosh": BCELogCoshDiceLoss()
}[loss_name]

optimizer = optim.Adam(params=seg_module.parameters(), lr=learning_rate)

best_val_miou, best_epoch = 0, 0

train_path = glob.glob(train_images_path + "*.png")
validation_path = glob.glob(validation_images_path + "*.png")

train_path = train_path[0:15]
validation_path = validation_path[0:6]
metric = pyiqa.create_metric("clipiqa+", device=device)

# Start training
print(f"starting scoring")

scored_samples = []
transform = transforms.ToTensor()
for file_path in tqdm(train_path):
    image_tensor = transform(Image.open(file_path))
    scored_samples.append((
        file_path,
        metric(image_tensor.unsqueeze(dim=0))
    ))

sorted_samples = sorted(scored_samples, key=lambda x: x[-1], reverse=True)
scores, paths = zip(*[(x[-1], x[0]) for x in sorted_samples])
scores, paths = list(scores), list(paths)

# training_samples = paths[0:8001]
train_path_new = paths[0:11]

scored_samples = []
transform = transforms.ToTensor()
for file_path in tqdm(validation_path):
    image_tensor = transform(Image.open(file_path))
    scored_samples.append((
        file_path,
        metric(image_tensor.unsqueeze(dim=0))
    ))

sorted_samples = sorted(scored_samples, key=lambda x: x[-1], reverse=True)
scores, paths = zip(*[(x[-1], x[0]) for x in sorted_samples])
scores, paths = list(scores), list(paths)

# val_samples = paths[0:801]
validation_path_new = paths[0:4]

training_samples = []
val_samples = []

for i in range(len(train_path_new)):
    feature_file_train = os.path.basename(train_path_new[i]).replace(".png", ".pk")
    full_feature_train = train_data_path + str(feature_file_train)
    training_samples.append(full_feature_train)

for i in range(len(validation_path_new)):
    feature_file_val = os.path.basename(validation_path_new[i]).replace(".png", ".pk")
    full_feature_val = validation_data_path + str(feature_file_val)
    val_samples.append(full_feature_val)

total_steps = len(training_samples)

if total_steps == 0:
    print(f"{train_data_path} does not contain any data. make sure you added a trailing / to the path")

if len(val_samples) == 0:
    print(f"{validation_data_path} does not contain any data. make sure you added a trailing / to the path")

miou_train_list = []
miou_val_list = []
extractor = ViTExtractor("dino_vits8", 8, device=device)
transform = transforms.ToTensor()

# Start training
print(f"starting training")

for epoch in range(n_epochs):
    print(f"starting epoch {epoch}")

    seg_module.train()

    # Do a single pass over all the data sample
    for step, file_path in enumerate(tqdm(training_samples)):
        image_path = training_samples[step]
        image_file = file_path.split("/")[-1].replace(".pk", ".png")
        full_image_path = os.path.join(train_images_path, image_file)
        open_image = Image.open(full_image_path)

        with torch.no_grad():
            image = extractor.preprocess(transform(open_image), 512)
            saliency_map = extractor.extract_saliency_maps(image.to(device))

        with open(file_path, "rb") as sample_file:
            sample = pickle.load(sample_file)

        # Unpack the sample data
        labels = sample.labels
        segmentations = sample.masks
        unet_features = sample.unet_features

        # Move the UNet features to cpu
        for key in unet_features.keys():
            unet_features[key] = [x.to(device) for x in unet_features[key]]

        step_loss = 0

        # FIXME: We could precompute these
        prompt = " and a ".join(labels)
        label_embeddings = get_embeddings(
            tokenizer=tokenizer,
            embedder=embedder,
            device=device,
            prompt=prompt,
            labels=labels,
            inverted_vocab=tokenizer_inverted_vocab
        )

        for label, segmentation in zip(labels, segmentations):
            # Predict the mask using the fusion module
            fusion_segmentation = seg_module(unet_features, label_embeddings[label], saliency_map.to(device))
            fusion_segmentation_pred = torch.unsqueeze(
                fusion_segmentation[0, 0, :, :], 0
            ).unsqueeze(0)

            if step % 1000 == 0 and visualize_examples:
                # FIXME: We should move these to Tensorboard
                # Save the fusion module mask every 25 steps
                fusion_mask = preprocess_mask(mask=fusion_segmentation_pred)

                torchvision.utils.save_image(
                    torch.from_numpy(fusion_mask),
                    os.path.join(training_dir, f"vis_sample_{epoch}_{step}_{label}_pred_seg.png"),
                    normalize=True,
                    scale_each=True,
                )

                # Also plot the mask over the image
                filename = file_path.split("/")[-1].replace(".pk", ".png")
                image = Image.open(os.path.join(train_images_path, filename))

                masked_image = Image.fromarray(plot_mask(np.array(image), fusion_mask))
                masked_image.save(os.path.join(training_dir, f"vis_image_{epoch}_{step}_{label}_masked.png"))

            segmentation = (
                torch.from_numpy(segmentation).unsqueeze(0).unsqueeze(0).to(device)
            ).float()

            # Calculate the loss and run one training step
            # FIXME: Try averaging the loss here
            step_loss += loss_fn(fusion_segmentation_pred, segmentation)

        optimizer.zero_grad()
        step_loss.backward()
        optimizer.step()

        torch_writer.add_scalar("train/loss", step_loss.item(), global_step=step)

        if step % 1000 == 0 and step > 0:
            # Save a checkpoint every 500 steps
            torch.save(
                seg_module.state_dict(),
                os.path.join(run_dir, f"checkpoint_{epoch}_{step}.pth")
            )

    seg_module.eval()

    with torch.no_grad():
        # Evaluate on the training and validation sets
        print(f"evaluating the model for epoch {epoch}")

        train_miou = evaluate_seg_model(
            model=seg_module,
            extractor=extractor,
            tokenizer=tokenizer,
            embedder=embedder,
            phase_images_path=train_images_path,
            device=device,
            tokenizer_inverted_vocab=tokenizer_inverted_vocab,
            samples_paths=training_samples
        )

        print(f"training mIoU: {train_miou}")

        val_miou = evaluate_seg_model(
            model=seg_module,
            extractor=extractor,
            tokenizer=tokenizer,
            embedder=embedder,
            phase_images_path=validation_images_path,
            device=device,
            tokenizer_inverted_vocab=tokenizer_inverted_vocab,
            samples_paths=val_samples
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch

        print(f"validation mIoU: {val_miou}")
        print(f"epoch {best_epoch} has the best validation mIoU ({best_val_miou})")

        miou_train_list.append(train_miou)
        miou_val_list.append(val_miou)

        torch_writer.add_scalar("train/miou", train_miou, epoch)
        torch_writer.add_scalar("val/miou", val_miou, epoch)

n_epochs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
data = {'Epoch': n_epochs, 'Training mIOU': miou_train_list, 'Validation mIOU': miou_val_list}
df = pd.DataFrame(data)
sns.set_style('whitegrid')
sns.lineplot(data=df, x='Epoch', y='Training mIOU', label='Training mIOU', marker='o', markeredgewidth=2)
sns.lineplot(data=df, x='Epoch', y='Validation mIOU', label='Validation mIOU', marker='o', markeredgewidth=2)
plt.title('Training and Validation mIOU')
plt.xticks(n_epochs)
plt.xlabel('Epoch')
plt.ylabel('mIOU')
plt.savefig('dino_scoring_training_validation_miou.png')
