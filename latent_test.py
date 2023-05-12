import os
import json
import time
import torch
import pickle
import random
import numpy as np

from PIL import Image
from data import PromptsMultiClassSegmentationSample
from mmdet.apis import init_detector, inference_detector
from utils import load_stable_diffusion, has_mask_for_classes
from utils.prompts import visual_adjectives_prompt
from utils import get_embeddings, preprocess_mask, plot_mask
from seg_module import Segmodule


n_classes = 2
total_samples = 500
output_dir = "dataset_test"
pascal_class_split = 1
device = torch.device("cuda")
model_name = "runwayml/stable-diffusion-v1-5"
model_type = model_name.split("/")[-1]
# sample_image = Image.open("voc_data/VOCdevkit/VOC2012/JPEGImages/2007_000033.jpg")
sample_image = Image.open("dataset_test/images/1683873061_0.png")
grounding_checkpoint = "checkpoint_1000.pth"

pipeline, grounded_unet = load_stable_diffusion(model_name=model_name, device=device)

# Load the segmentation module
seg_module = Segmodule(
    use_sd2=False,
    output_image_dim=512
).to(device)

seg_module.load_state_dict(
    torch.load(grounding_checkpoint, map_location=device), strict=True
)

timestep = 1

with torch.no_grad():
    label = "cat"

    test_encode = torch.from_numpy(np.array(sample_image))
    test_encode = test_encode.permute(2, 0, 1).unsqueeze(dim=0)
    test_encode = test_encode.to(device).float()

    sampled_latent = pipeline.vae.encode(test_encode).latent_dist.sample()
    sampled_latent = torch.cat([sampled_latent, sampled_latent], dim=0)
    sampled_latent = pipeline.scheduler.scale_model_input(sampled_latent, timestep)

    print(torch.cuda.memory_allocated(device="cuda"))

    prompt = f"a photograph of a {label}"

    tokens = pipeline.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipeline.tokenizer.model_max_length,
        return_tensors="pt",
        truncation=True
    )

    tokens["input_ids"] = tokens["input_ids"].to(device)
    tokens["attention_mask"] = tokens["attention_mask"].to(device)

    # size: 473088
    token_embeddings = pipeline.text_encoder(**tokens).last_hidden_state
    token_embeddings = torch.cat([
        token_embeddings,
        token_embeddings
    ], dim=0).to(device)

    # Setup tokenizer and the CLIP embedder
    tokenizer = pipeline.tokenizer
    embedder = pipeline.text_encoder

    # Compute the token_id -> token text mapping
    tokenizer_inverted_vocab = {
        v: k for k, v in tokenizer.get_vocab().items()
    }

    print(prompt)

    grounded_unet.clear_grounding_features()

    test_unet = grounded_unet(
        # should be [2, 4, 64, 64]
        sampled_latent,
        # timestep
        timestep,
        # should be [2, 77, 768]
        encoder_hidden_states=token_embeddings,
        cross_attention_kwargs=None
    ).sample

    unet_features = grounded_unet.get_grounding_features()
    label_embeddings = get_embeddings(
        tokenizer=tokenizer,
        embedder=embedder,
        device=device,
        prompt=prompt,
        labels=[label],
        inverted_vocab=tokenizer_inverted_vocab
    )

    fusion_segmentation = seg_module(unet_features, label_embeddings[label])
    fusion_segmentation_pred = fusion_segmentation[0, 0, :, :]

    # decoded_image = pipeline.decode_latents(test_unet)

    fusion_mask = preprocess_mask(mask=fusion_segmentation_pred.unsqueeze(dim=0))

    masked_image = Image.fromarray(plot_mask(np.array(sample_image), np.expand_dims(fusion_mask, 0)))
    masked_image.save("sam_test.png")

    # pipeline.numpy_to_pil(decoded_image)[0].save("ciao.png")

    print("lorem")