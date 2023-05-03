import os

from diffusers import StableDiffusionPipeline
from .grounded_unet import GroundedUNet2DConditionModel


temp_unet_dir = "temp"
model_name = "runwayml/stable-diffusion-v1-5"
prompt = "a bouquet of yellow and orange tulips"

os.makedirs(temp_unet_dir, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained(model_name).to("cuda")

# Save the pretrained UNet to disk
pipeline_components = pipeline.components
pipeline_components["unet"].save_pretrained(os.path.join(temp_unet_dir, "unet_model"))

# Reload the UNet as the grounded subclass
grounded_unet = GroundedUNet2DConditionModel.from_pretrained(
    os.path.join(temp_unet_dir, "unet_model")
).to("cuda")

pipeline_components["unet"] = grounded_unet

pipeline = StableDiffusionPipeline(**pipeline_components)

# Generate an image
image = pipeline(prompt).images[0]

# Obtain the feature maps from the UNet
unet_features = grounded_unet.get_grounding_features()

for key in unet_features.keys():
  values = unet_features[key]

  print(f"{len(values)} for {key}")
  print(f"shapes: {[x.shape for x in values]}")
