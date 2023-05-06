import os
import glob
import pyiqa
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.utils import make_grid

from utils.visualization import plot_grid


dataset_path = "dataset/images"
grid_visualization_path = "iqa_grid.png"
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
metric = pyiqa.create_metric("clipiqa+", device=device)

scored_samples = []
transform = transforms.ToTensor() 

# Load images and calculate metrics
target_images = glob.glob(os.path.join(dataset_path, "*.png"))

for file_path in tqdm(target_images):
    image_tensor = transform(Image.open(file_path))

    scored_samples.append((
        file_path,
        image_tensor,
        metric(image_tensor.unsqueeze(dim=0))
    ))

sorted_samples = sorted(scored_samples, key=lambda x: x[-1], reverse=True)
scores, images = zip(*[(x[-1], x[1]) for x in sorted_samples])
scores, images = list(scores), list(images)

# Create a grid of images and save it to disk
fig = plot_grid(make_grid(images))
fig.savefig(grid_visualization_path)

# FIXME: Pick the elements to be deleted/kept