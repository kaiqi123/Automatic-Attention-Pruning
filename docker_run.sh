#!/bin/bash

# Show running container
docker ps -a

# Test by running a base CUDA container
docker run --rm --gpus all nvidia/cuda:11.0-base nvidia-smi

# build docker
docker build . -t nvidia_rn50

# run docker
docker run --rm -it \
-v /home/users/kzhao27/tiny-imagenet-200:/tiny-imagenet-200 \
-v /home/users/kzhao27/DeepLearningExamples/PyTorch/Classification/ConvNets:/workspace/rn50 \
--ipc=host \
--gpus all \
nvidia_rn50

