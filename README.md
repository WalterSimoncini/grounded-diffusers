### Environment setup

```sh
module load 2022
module load Anaconda3/2022.05
nvidia-smi

conda create --name gdiff python==3.10
source activate diffusers-gdiff
sh setup_env.sh
```
