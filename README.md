# CIFAR-10 Image Classification Model

This repository contains a Convolutional Neural Network (CNN) model built using TensorFlow/Keras for classifying images from the CIFAR-10 dataset. The CIFAR-10 dataset is a popular benchmark dataset in computer vision, consisting of 60,000 32x32 color images in 10 different classes, with 6,000 images per class.

## Project Overview
This project is a small, hands-on study of the tradeoff between:

- **A research-grade, speed-optimized CIFAR-10 training recipe** (based on the *CIFAR-10 Airbench* work)
- **A simple “vanilla” CNN baseline** (implemented by me in TensorFlow/Keras)

I implemented both so I could understand *why* Airbench reaches ~94% accuracy so quickly, and how far a straightforward CNN gets in accuracy when trained normally.

## Key Features:
- **Dataset**: CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Two implementations**:
  - **PyTorch (Airbench-style)**: fast training pipeline + small CNN architecture designed for speed
  - **TensorFlow/Keras (vanilla CNN)**: a simple baseline CNN trained in a conventional way
- **What is being compared**:
  - **Accuracy** achieved on CIFAR-10 test set
  - **Training speed** and practical GPU constraints on consumer hardware (RTX 4050)

## Repository Contents
- **`airbench_94_code.py`**: Airbench-style PyTorch training script (saves `airbench94.pth`)
- **`test.py`**: PyTorch inference / evaluation script for `airbench94.pth`
- **`Cifar_cnn_model.py`**: TensorFlow/Keras vanilla CNN training script (saves `cifar10_model.h5`)
- **`airbench_best_94.pth`**: pretrained weights (included in this repo)
- **`cifar10_good_86.56.pt`**: a saved checkpoint from my own run (naming reflects the achieved accuracy)

## Results (My Runs)
- **Vanilla CNN (TensorFlow/Keras)**: **~86.65%** test accuracy
- **Airbench-style (PyTorch)**: **~94%** is the reference target reported by Airbench (and achievable with the recipe)

### Hardware note (RTX 4050)
Airbench’s published timings are on an NVIDIA A100. On my RTX 4050, the same style of code took **~40 seconds** (previously **~56 seconds**) after reducing VRAM pressure.

Because GPUs differ significantly (VRAM bandwidth, tensor cores, cache sizes), **the point of this repo is not to beat A100 timings**—it’s to understand the methods and reproduce the behavior on accessible hardware.

## Setup
There is no `requirements.txt` in this repo, so install dependencies manually.

### Option A: Run the Airbench-style PyTorch code
- **Python**: 3.9+ recommended
- Install:

```bash
pip install torch torchvision
```

If you want maximum speed on NVIDIA GPUs, install a CUDA-enabled PyTorch build that matches your driver/CUDA version (follow the official PyTorch install selector).

### Option B: Run the vanilla TensorFlow/Keras CNN
- **Python**: 3.9+ recommended
- Install:

```bash
pip install tensorflow
```

## How to Run

### 1) Train Airbench-style model (PyTorch)
This script downloads CIFAR-10 automatically (into `./cifar10/`) and writes `airbench94.pth`.

```bash
python airbench_94_code.py
```

### 2) Evaluate / predict with the Airbench-style model (PyTorch)

- **Evaluate on full CIFAR-10 test set**:

```bash
python test.py
```

- **Predict on a single image** (it will be resized to 32x32 and normalized to CIFAR-10 stats):

```bash
python test.py path/to/image.jpg
```

### 3) Train vanilla CNN (TensorFlow/Keras)
This uses `tf.keras.datasets.cifar10` (downloads automatically) and saves `cifar10_model.h5`.

```bash
python Cifar_cnn_model.py
```

## Why this implementation?
- **To reproduce a strong research baseline**: Airbench shows that with the right pipeline choices (input pipeline, FP16, augmentation strategy, learning-rate schedule, and a small but well-tuned CNN), CIFAR-10 can be trained to high accuracy extremely quickly.
- **To keep a simple baseline**: a vanilla CNN provides a “normal” reference point. It’s easier to understand, but typically needs more training time and still tends to underperform compared to optimized recipes.
- **To learn by measuring**: I compared accuracy and runtime on my own GPU (RTX 4050), since many published results are reported on datacenter GPUs like the A100.

## References
- **CIFAR-10 Airbench repo** (paper + scripts):
  - https://github.com/KellerJordan/cifar10-airbench

If you use this project for coursework or a report, please cite the original Airbench work appropriately and treat this repo as a reproduction/comparison project.

## License
This project is licensed under the MIT License - see the LICENSE file for details.
