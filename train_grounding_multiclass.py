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

from seg_module import Segmodule
from utils.evaluation import evaluate_seg_model
from utils import preprocess_mask, get_embeddings, plot_mask
from loss_fn import BCEDiceLoss, DiceLoss, BCELogCoshDiceLoss


seed = 42
n_epochs = 10
use_sd2 = False
visualize_examples = False
pascal_class_split = 1
loss_name = "log_cosh"
checkpoints_dir = "checkpoints"
device = torch.device("cuda")
model_name = "stabilityai/stable-diffusion-2" if use_sd2 else "runwayml/stable-diffusion-v1-5"
train_data_path = "dataset/samples/"
train_images_path = "dataset/images/"
validation_data_path = "val_dataset/samples/"
learning_rate = 1e-5

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

seed_everything(seed)

os.makedirs(checkpoints_dir, exist_ok=True)

# Load COCO and Pascal-VOC classes
coco_classes = open("mmdetection/demo/coco_80_class.txt").read().split("\n")
coco_classes = dict([(c, i) for i, c in enumerate(coco_classes)])

pascal_classes = open(f"VOC/class_split{pascal_class_split}.csv").read().split("\n")
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

# Start training
print(f"starting training")

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

training_samples = glob.glob(train_data_path + "*.pk")
val_samples = glob.glob(validation_data_path + "*.pk")

total_steps = len(training_samples)

if total_steps == 0:
    print(f"{train_data_path} does not contain any data. make sure you added a trailing / to the path")

if len(val_samples) == 0:
    print(f"{validation_data_path} does not contain any data. make sure you added a trailing / to the path")

for epoch in range(n_epochs):
    print(f"starting epoch {epoch}")

    seg_module.train()

    # Do a single pass over all the data sample
    for step, file_path in enumerate(tqdm(training_samples)):
        image_path = training_samples[step]

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
            fusion_segmentation = seg_module(unet_features, label_embeddings[label])
            fusion_segmentation_pred = torch.unsqueeze(
                fusion_segmentation[0, 0, :, :], 0
            ).unsqueeze(0)

            if step % 25 == 0 and visualize_examples:
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

        if step % 500 == 0 and step > 0:
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
            tokenizer=tokenizer,
            embedder=embedder,
            device=device,
            tokenizer_inverted_vocab=tokenizer_inverted_vocab,
            samples_paths=training_samples
        )

        print(f"training mIoU: {train_miou}")

        val_miou = evaluate_seg_model(
            model=seg_module,
            tokenizer=tokenizer,
            embedder=embedder,
            device=device,
            tokenizer_inverted_vocab=tokenizer_inverted_vocab,
            samples_paths=val_samples
        )

        if val_miou > best_val_miou:
            best_val_miou = val_miou
            best_epoch = epoch

        print(f"validation mIoU: {val_miou}")
        print(f"epoch {best_epoch} has the best validation mIoU ({best_val_miou})")

        torch_writer.add_scalar("train/miou", train_miou, epoch)
        torch_writer.add_scalar("val/miou", val_miou, epoch)
