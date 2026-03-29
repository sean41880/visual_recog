# Image Classification Project

This is a similar project structure to the reference project, designed for image classification tasks using deep learning.

## Installation

It is recommended to run this code in an isolated Python environment via conda/mamba.

```sh
conda env create -f environment.yml
conda activate hw1_env
```

## Repository Structure

Before running the project, create the required directories:

```sh
mkdir result
mkdir weights
```

The data should be downloaded by the user and extracted:

```sh
# Download your data and extract it
tar xvf your-data.tar.gz
```

Project structure:

```sh
.
├── data
│   ├── test
│   ├── train
│   └── val
├── result
├── weights
├── src
│   ├── dataset
│   │   ├── dataset.py
│   │   └── transform.py
│   ├── models
│   │   └── pretrain.py
│   ├── train
│   │   ├── train.py
│   │   └── utils.py
│   ├── utils
│   │   ├── logger.py
│   │   ├── parser.py
│   │   └── utils.py
│   ├── main.py
│   └── predict.py
├── environment.yml
└── README.md
```

## Usage

To train the model:

```sh
cd src
python main.py --model resnet18 --batch_size 64 --epochs 100 --lr 1e-5
```

### Command Line Arguments

- `--seed`: Random seed (default: 42)
- `--batch_size`: Batch size for training (default: 64)
- `--lr`: Learning rate (default: 1e-5)
- `--device`: Device to use, 'cpu' or 'cuda' (default: 'cpu')
- `--patient`: Patience for early stopping (default: 20)
- `--epochs`: Number of training epochs (default: 100)
- `--loss_function`: Loss function to use, 'CrossEntropy' or 'Focal' (default: 'CrossEntropy')
- `--optimizer`: Optimizer to use, 'Adam', 'AdamW', 'SGD', or 'Adafactor' (default: 'Adam')
- `--model`: Model architecture to use (default: 'resnet18')
- `--transform`: Data augmentation strategy, '', 'autoAug', 'customAug', or 'advanceAug' (default: '')
- `--pretrain_model_weight`: Pretrained model weights, 'DEFAULT' or None (default: 'DEFAULT')
- `--freeze_layer`: Layer freeze strategy, None, 'conv', or 'half' (default: None)
- `--enable_wandb`: Enable Weights & Biases logging (default: False)

### Supported Models

- resnet18
- resnet34
- resnet50
- resnet101
- resnet152
- resnext50_32x4d
- resnext101_32x8d
- resnext101_64x4d

## Project Details

This project is designed for multiclass image classification using deep learning. It leverages pretrained models from PyTorch and applies various augmentation strategies and training techniques.

Key Features:
- Flexible model selection (ResNet variants, ResNeXt variants)
- Multiple data augmentation strategies (AutoAugment, Custom, Advanced)
- Support for different loss functions (CrossEntropy, Focal Loss)
- Layer freezing for transfer learning
- Early stopping and model checkpointing
- Weights & Biases integration for experiment tracking
