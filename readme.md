# URLOST: Unsupervised Representation Learning without Stationarity or Topology

## Table of Contents
- [Overview](#overview)
- [Installation](#installation)
- [Usage](#usage)
  - [Datasets](#datasets)
  - [Training](#training)

## Overview

We introduce a novel framework that learns from high-dimensional data without prior knowledge of its stationarity and topology. Our model, dubbed URLOST, combines a learnable self-organizing layer, density-adjusted spectral clustering, and masked autoencoders. We evaluate its effectiveness on three diverse data modalities including simulated biological vision, neural recordings from the primary visual cortex, and gene expressions. Compared to state-of-the-art unsupervised learning methods like SimCLR and MAE, our model excels at learning meaningful representations across diverse modalities without knowing their stationarity or topology. It also outperforms other methods not dependent on these factors, setting a new benchmark in the field. We position this work as a step toward unsupervised learning methods capable of generalizing across diverse high-dimensional data modalities.

## Installation

To get started with this project, clone the repository and install the required dependencies:

```bash
cd Magic_TV
conda env create -f environment.yml
conda activate magic_tv
```
## Usage

#datasets
The base dataset is CIFAR10, we create non-stationary version of CIFAR10 using functions in utils. We provide two example in this repo. Permutated CIFAR10 and Foveated CIFAR10.

#training
For Permutated CIFAR10, 
```
python run_mae.py --patch_size 4 --scale_lo 0.3 --scale_high 1.0 --ratio_lo 0.75 --ratio_high 1.33 --flip 0.5 --max_device_batch_size 2000 --total_epoch 10001 --num_groups 1
```

For Foveated CIFAR10, 
```
python run_mae.py --patch_size 4 --scale_lo 0.3 --scale_high 1.0 --ratio_lo 0.75 --ratio_high 1.33 --flip 0.5 --max_device_batch_size 2000 --total_epoch 10000 --n_clusters 64 --padding 1 --K 20 --oracle 0.0 --mask_ratio 0.75 --w_factor 1 --retina
```

