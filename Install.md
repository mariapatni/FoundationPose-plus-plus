Setting the environment for foundation pose
```
# Pull the following docker image and launch it:
docker pull shingarey/foundationpose_custom_cuda121:latest

# Enter the container, git clone this repo:
git clone https://github.com/Psi-Robot/dataset-postprocess.git

# Build FoundationPose:
bash /path/to/dataset-postprocess/src/FoundationPose/build_all.sh

# additional requirements
pip install Hydra

```

Utilities
```
# For SAM-HQ
pip install segment-anything-hq
cd dataset-postprocess/src/sam-hq
pip install -e .

# For Qwen-VL-2-Instruct
pip install git+https://github.com/huggingface/transformers@21fac7abba2a37fae86106f87fcf9974fd1e3830 accelerate
pip install qwen-vl-util

# For Cutie
cd dataset-postprocess/src/Cutie
pip install -e .
python cutie/utils/download_models.py

```

weights
```
1. Download Qwen2-VL weights to Qwen2-VL/weights
2. Download Sam-HQ weights sam_hq_vit_h.pth to sam-hq/pretrained_checkpoints
```