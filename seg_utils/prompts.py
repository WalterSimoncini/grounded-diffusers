import torch
import random

from typing import Dict, List


def visual_adjectives_prompt(
    label: str,
    visual_adjectives: Dict[str, Dict[str, List[str]]],
    max_adjectives=2
):
    picked_adjectives = []
    class_adjectives = visual_adjectives[label]

    for adjective_group in class_adjectives.keys():
        picked_adjectives.append(
            random.choice(class_adjectives[adjective_group])
        )

    # Pick at most max_adjectives
    picked_adjectives = random.sample(
        picked_adjectives,
        k=min(max_adjectives, len(picked_adjectives))
    )

    picked_adjectives = " ".join(picked_adjectives)

    return f"{picked_adjectives} {label}"


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
