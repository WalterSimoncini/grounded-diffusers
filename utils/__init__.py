import os
import torch
import numpy as np

from enum import Enum
from typing import Tuple, List, Dict

from diffusers import StableDiffusionPipeline
from grounded_unet import GroundedUNet2DConditionModel


class TrainingType(Enum):
  SINGLE = "single"
  TWO = "two"
  RANDOM = "random"


def plot_mask(img, masks, colors=None, alpha=0.8,indexlist=[0,1]) -> np.ndarray:
    """Visualize segmentation mask.

    Parameters
    ----------
    img: numpy.ndarray
        Image with shape `(H, W, 3)`.
    masks: numpy.ndarray
        Binary images with shape `(N, H, W)`.
    colors: numpy.ndarray
        corlor for mask, shape `(N, 3)`.
        if None, generate random color for mask
    alpha: float, optional, default 0.5
        Transparency of plotted mask

    Returns
    -------
    numpy.ndarray
        The image plotted with segmentation masks, shape `(H, W, 3)`

    """
    H, W = masks.shape[0], masks.shape[1]
    color_list=[[255,97,0],[128,42,42],[220,220,220],[255,153,18],[56,94,15],[127,255,212],[210,180,140],[221,160,221],[255,0,0],[255,128,0],[255,255,0],[128,255,0],[0,255,0],[0,255,128],[0,255,255],[0,128,255],[0,0,255],[128,0,255],[255,0,255],[255,0,128]]*6
    final_color_list=[np.array([[i]*512]*512) for i in color_list]
    
    background=np.ones(img.shape)*255
    count=0
    colors=final_color_list[indexlist[count]]

    for mask, color in zip(masks, colors):
        color=final_color_list[indexlist[count]]
        mask = np.stack([mask, mask, mask], -1)
        img = np.where(mask, img * (1 - alpha) + color * alpha,background*0.4+img*0.6 )
        count+=1

    return img.astype(np.uint8)


def calculate_iou(mask1, mask2):
    assert mask1.shape == mask2.shape, "Masks must have the same shape."

    # Calculate intersection and union.
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)

    # Count the number of pixels in the intersection and union.
    intersection_count = np.sum(intersection)
    union_count = np.sum(union)

    # Avoid division by zero.
    if union_count == 0:
        return 0

    iou = intersection_count / union_count

    return iou


def preprocess_mask(mask: torch.Tensor):
    probabilities = torch.sigmoid(mask)

    mask = torch.zeros_like(probabilities, dtype=torch.float32)
    mask[probabilities > 0.5] = 1

    return mask[0].cpu().numpy()


def load_stable_diffusion(
    model_name: str,
    device: torch.device,
    temp_dir="temp"
) -> Tuple[StableDiffusionPipeline, GroundedUNet2DConditionModel]:
    # Load the stable diffusion pipeline
    pipeline = StableDiffusionPipeline.from_pretrained(model_name).to(device)

    # Save the pretrained UNet to disk
    model_type = model_name.split("/")[-1]

    unet_model_dir = os.path.join("unet_model", model_type)
    pretrained_unet_dir = os.path.join(temp_dir, unet_model_dir)

    pipeline_components = pipeline.components

    if not os.path.isdir(pretrained_unet_dir):
        pipeline_components["unet"].save_pretrained(pretrained_unet_dir)

    # Reload the UNet as the grounded subclass
    grounded_unet = GroundedUNet2DConditionModel.from_pretrained(
        pretrained_unet_dir
    ).to(device)

    pipeline_components["unet"] = grounded_unet

    return StableDiffusionPipeline(**pipeline_components), grounded_unet


def token_indices_for_labels(prompt_tokens: Dict[str, torch.Tensor], labels: List[str], inverted_vocab: dict):
    """
        Given the tokens of a prompt and a list of labels it returns
        a mapping from each label to the list of associated token ids,
        for example:
        
        {
            "pottedplant": [3547, 4841, 3912]
        }

        This is because the tokenizer splits `pottedplant` into three
        tokens: pot, ted, plant
    """
    # Map the token ids to tuples in the form (text_token, input_id)
    text_tokens = [
        (inverted_vocab[input_id].replace("</w>", ""), input_id)
        for input_id in prompt_tokens["input_ids"].squeeze().tolist()
    ]

    # Remove non-relevant tokens
    ignored_tokens = {"<|startoftext|>", "a", "photograph", "of", "a", "<|endoftext|>", "and"}
    text_tokens = [token for token in text_tokens if token[0] not in ignored_tokens]

    # Create a mapping label -> input_ids for each label
    token_ids_mapping = {}

    for label in labels:
        token_ids_mapping[label] = [
            token[1] for token in text_tokens if token[0] in label
        ]

    return token_ids_mapping


def get_embeddings(
    tokenizer,
    embedder,
    device: torch.device,
    prompt: str,
    labels: List[str],
    inverted_vocab: dict
):
    tokens = tokenizer(prompt, return_tensors="pt")

    # Then in the other function we get the embeddings for those words and average
    # them together if they use multiple tokens
    token_ids_mapping = token_indices_for_labels(
        prompt_tokens=tokens,
        labels=labels,
        inverted_vocab=inverted_vocab
    )

    tokens["input_ids"] = tokens["input_ids"].to(device)
    tokens["attention_mask"] = tokens["attention_mask"].to(device)

    input_ids_list = tokens["input_ids"]
    token_embeddings = embedder(**tokens).last_hidden_state
    label_embeddings = {}

    for label in token_ids_mapping.keys():
        label_token_indices = torch.Tensor([
            (input_ids_list == token_id).nonzero(as_tuple=True)[1]
            for token_id in token_ids_mapping[label]
        ]).long()

        # Take the mean of all the embedding related to this label.
        # for example pottedplant will have three tokens (pot, ted,
        # plant)
        label_embeddings[label] = token_embeddings[
            :, label_token_indices, :
        ].mean(dim=1, keepdim=True).to(device)

    return label_embeddings


def has_mask_for_classes(masks: List[List], class_indices: List[int]) -> bool:
    """
        Returns whether the Mask R-CNN segmentation has
        a mask for all the given classes. masks is an array
        of masks (one per available class) as returned by
        Mask R-CNN.
    """
    for class_index in class_indices:
      # Get Mask R-CNN mask tensor
      if len(masks[class_index]) == 0:
        return False

    return True