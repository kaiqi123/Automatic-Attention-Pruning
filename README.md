# Automatic Attention Pruning: Improving and Automating Model Pruning using Attentions (AAP)

This repo implements the paper published in **AISTATS 2023**:

**Automatic Attention Pruning: Improving and Automating Model Pruning using Attentions** (termed AAP)

The link of the paper is: https://arxiv.org/pdf/2303.08595.pdf.


## Setup
We implemented AAP on PyTorch version 1.6.0 and CUDA 11.2, and conducted experiments on 4 Nvidia RTX 2080 Ti GPUs.
We include the Dockerfile that extends the PyTorch NGC container and encapsulates some dependencies. 
About how to build and run the docker, please refer to the commands in the "docker_run.sh".

## Run the code
1. For the results on CIFAR-10, follow the commands in: 
```
scripts/cifar10_mobilenetV2.sh
scripts/cifar10_resnet56.sh
scripts/cifar10_shufflenetV2.sh
scripts/cifar10_vgg19.sh
```

2. For the results on Tiny-ImageNet, follow the commands in: 
```
scripts/tinyImageNet_resnet101.sh
scripts/tinyImageNet_vgg19.sh
```

About preprocessing the Tiny-ImageNet dataset, please follow the commands in scripts/preprocess_tiny_imagenet.sh.


## Citation

If you think this repo is helpful for your research, please consider citing the paper:
```
@article{zhao2023automatic,
  title={Automatic Attention Pruning: Improving and Automating Model Pruning using Attentions},
  author={Zhao, Kaiqi and Jain, Animesh and Zhao, Ming},
  journal={arXiv preprint arXiv:2303.08595},
  year={2023}
}
```

## Reference
https://github.com/NVIDIA/DeepLearningExamples

