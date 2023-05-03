### Environment setup

```sh
module load 2022
module load Anaconda3/2022.05
nvidia-smi

conda create --name gdiff python==3.10
source activate diffusers-gdiff
sh setup_env.sh
```

### Notes

```python
# Ok this Unet has 9 blocks, the other has 25.

# Out UNet
# 2 for high
# shapes: [torch.Size([1, 640, 32, 32]), torch.Size([1, 2560, 32, 32])]
# 2 for mid
# shapes: [torch.Size([1, 1280, 16, 16]), torch.Size([1, 2560, 16, 16])]
# 3 for low
# shapes: [torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8])]
# 2 for highest
# shapes: [torch.Size([1, 1280, 64, 64]), torch.Size([1, 640, 64, 64])]

# Original UNet
# 6 for low
# shapes: [torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8]), torch.Size([1, 2560, 8, 8])]
# 6 for mid
# shapes: [torch.Size([1, 1280, 16, 16]), torch.Size([1, 2560, 16, 16]), torch.Size([1, 2560, 16, 16]), torch.Size([1, 2560, 16, 16]), torch.Size([1, 2560, 16, 16]), torch.Size([1, 2560, 16, 16])]
# 6 for high
# shapes: [torch.Size([1, 640, 32, 32]), torch.Size([1, 1280, 32, 32]), torch.Size([1, 1280, 32, 32]), torch.Size([1, 2560, 32, 32]), torch.Size([1, 1280, 32, 32]), torch.Size([1, 1280, 32, 32])]
# 7 for highest
# shapes: [torch.Size([1, 640, 64, 64]), torch.Size([1, 640, 64, 64]), torch.Size([1, 640, 64, 64]), torch.Size([1, 1280, 64, 64]), torch.Size([1, 640, 64, 64]), torch.Size([1, 640, 64, 64]), torch.Size([1, 640, 64, 64])]
```