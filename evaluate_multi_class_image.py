"""
    This script calculates the mean IoU for a grounding
    module checkpoint for a dataset.

    This is a work in progress!
"""
import os
import glob
import torch

from seg_module import Segmodule
from seg_utils import seed_everything
from diffusers import StableDiffusionPipeline
from seg_utils.evaluation import evaluate_seg_model


seed = 42
use_sd2 = False
batch_size = 1
grounding_checkpoint = "checkpoints/dino-10-epochs-01-dropout/checkpoint_9_1000.pth"
device = torch.device("cuda")
model_name = "stabilityai/stable-diffusion-2" if use_sd2 else "runwayml/stable-diffusion-v1-5"
data_path = "dataset-unseen/updated-samples/"
images_path = "dataset-unseen/images/"

seed_everything(seed)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Load the segmentation module
seg_module = Segmodule(
    use_sd2=use_sd2,
    output_image_dim=768 if use_sd2 else 512
).to(device)

seg_module.load_state_dict(
    torch.load(grounding_checkpoint, map_location=device), strict=True
)

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

test_samples = glob.glob(data_path + "*.pk")
total_steps = len(test_samples)

if total_steps == 0:
    raise ValueError(
        f"{data_path} does not contain any data. "
        "make sure you added a trailing / to the data_path"
    )

seg_module.eval()

with torch.no_grad():
    # Do a single pass over all the data sample
    miou_score = evaluate_seg_model(
        model=seg_module,
        tokenizer=tokenizer,
        embedder=embedder,
        device=device,
        tokenizer_inverted_vocab=tokenizer_inverted_vocab,
        samples_paths=test_samples
    )

    print(f"the mean IoU across the dataset at {data_path} using the checkpoint {grounding_checkpoint} is {miou_score}")
