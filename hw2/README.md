# Digit Detection with DETR (DEtection TRansformer)

This repository contains the implementation of a Digit Detection model using the DETR architecture with a ResNet-50 backbone. The objective is to accurately predict bounding boxes and class labels for digits in RGB images.

## 1. Introduction
The model leverages the official DETR architecture, replacing hand-crafted anchors and NMS with a transformer-based encoder-decoder and bipartite matching loss. To adapt to the digit detection task, several architectural and loss-level modifications were implemented:
* [Insert your architectural modification, e.g., Adjusted Object Queries from 100 to 50]
* [Insert your loss modification, e.g., Replaced standard CE with Focal Loss]

## 2. Environment Setup
Developed on TWCC (Taiwan Computing Cloud).
```bash
conda create -n detr_env python=3.13 -y
conda activate detr_env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install transformers pycocotools pandas tqdm