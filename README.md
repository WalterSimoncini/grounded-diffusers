### Environment setup

```sh
module load 2022
module load Anaconda3/2022.05
nvidia-smi

conda create --name gdiff python==3.10
source activate diffusers-gdiff
sh setup_env.sh
```

### Prompts to generate the poster images

```json
[
    {
        "prompt": "a photograph of a basketball with a cat standing on top of it on a field long shot",
        "classes": "cat,basketball"
    },
    {
        "prompt": "a photograph of a mug and a pottedplant side by side sitting on top of a table in a kitchen with strong natural light. In the background a white wall can be seen",
        "classes": "mug,pottedplant"
    },
    {
        "prompt": "a photograph of a monkey sitting side by side with a bunch of bananas. there is a strong natural warm light in the scene. high quality photography 8k",
        "classes": "monkey,banana"
    },
    {
        "prompt": "a photograph of a red car and an eagle flying in the sky",
        "classes": "eagle,car"
    },
    {
        "prompt": "a photograph of a dog sitting on top of a boat sailing on a river, lit by a warm natural light long shot",
        "classes": "dog,boat"
    }
]
```