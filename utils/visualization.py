import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F


def plot_grid(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]

    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False, figsize=(24, 16))

    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)

        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    return fig
