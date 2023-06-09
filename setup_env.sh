# conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html
pip install einops
pip install kornia
pip install pytorch-lightning==1.6.5
pip install openmim
pip install mmdet==2.22.0
pip install transformers
pip install accelerate
pip install diffusers
pip install taming-transformers-rom1504 
pip install black
pip install gdown
pip install pyiqa

mkdir -p ./mmdetection/checkpoint

gdown "https://drive.google.com/uc?id=1JbJ7tWB15DzCB9pfLKnUHglckumOdUio" -O ./mmdetection/checkpoint/mask_rcnn_swin-s-p4-w7_fpn_fp16_ms-crop-3x_coco_20210903_104808-b92c91f1.pth
gdown "https://drive.google.com/uc?id=1duStqgGKMtgakOqdj29AGcNqnoj5ADoZ" -O ./mmdetection/checkpoint/grounding_module.pth
