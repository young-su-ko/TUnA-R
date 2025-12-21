# TUnA: Refactored, Reimplemented, Rebooted, Reloaded (under construction)

## Purpose of this repository

[TUnA](https://academic.oup.com/bib/article/25/5/bbae359/7720609), my first PhD project, was published a little over a year ago now. In the original [TUnA repository](https://github.com/Wang-lab-UCSD/TUnA), my main goal was providing a codebase so others could reproduce the results in the paper. However, I didn't consider how someone might interact with the code--whether this means running inference, modifying the architecture, or just understanding the components. In addition, I didn't know much about the best practices in writing code (modularity, type-hinting, single-responsibility, etc.).

While the project is over a year old now, I realized this is a great opportunity to apply what I learned about engineering, get familar with some of the popular packages, and hopefully make TUnA easier to understand for you, the reader. Thus, TUnA-R is my re-implementation of the original codebase with the following goals in mind:

### **1. Better engineering practices**

- Basic stuff: writing re-usuable code, making it easier for myself and others to understand. Trying to make my code more tasteful.
- Writing tests.
- lightning/hydra/wandb stack (I personally find that this combination allows me to get a model up and running really fast, but I know opinions can be mixed about lightning).
- Package management with uv.

### **2. Stay true to the original**

- The architecture, training configuration, evaluation metrics, and core experimental setup are preserved. This is the same model/logic from the manuscript, just expressed more cleanly.
- *However*, results are not one-to-one to those in the manuscript. **If for some reason, you need to exact values in the manuscript, please refer to the original repository.**

### **3. Better usability**

- Making it easier to run TUnA (e.g. CLI + PyPI package). Will also look into making it easy to fine-tune.
- Weights on HuggingFace.

## Overview of Codebase Structure

### Models

To streamline the two different model architectures utilized in the manuscript (MLP-based vs Transformer-based), we defined a `PPIPredictor` class which can run either backbones. It is a `nn.Module` that will later passed to the PyTorch Lightning module for training, but also used directly for inference.

The `PPIPredictor`'s responsibility is to output an interaction prediction given the inputs.

### Lightning Module

The `LitPPI` is the PyTorch Lightning module we will use to streamline training. It inherits from `BaseModule`, which just defines some helper and setup functions such as initializing the `torchmetrics` objects. `LitPPI`'s reponsibilities include: Initialization of model weights, optimizer configuration, and train/val loops.

### Configs

The configs are currently set up to work with Lightning and Hydra's `instantiate` function.

### Tests

Here are some basic tests and smoke tests to make sure the code is working as intended.

## Results

Again, there are differences from the results presented in the manuscript. I am not aiming for one-to-one reproduction of results, but rather making the code as streamlined and readable as possible, while keeping the original logic. If I had to guess why 

## Installation

...

## Usage

TODO

- the batch size=1 during validation/test kills speed
- re-run training to check for bugs
- add tests
- add uncertainty calculations at test step or validation.
