#!/bin/bash

# Show running container
# sudo /usr/bin/docker ps -a

# Test by running a base CUDA container
# sudo /usr/bin/docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# build docker
# sudo /usr/bin/docker build . -t nvidia_rn50

# map one dataset
sudo /usr/bin/docker run --rm -it \
-v /PATH/tiny-imagenet-200:/tiny-imagenet-200 \
-v /PATH/ConvNets:/workspace/rn50 \
--ipc=host \
--gpus all \
nvidia_rn50
