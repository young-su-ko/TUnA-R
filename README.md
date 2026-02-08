# TUnA: Refactored (Uncertainty-aware sequence-based PPI prediction)

## Installation
For now, installation is limited to cloning this repo and installing the necessary dependencies with `uv`. 

> pip install tuna-r coming soon.


<!-- 
Please install `uv` beforehand.

```bash
git clone https://github.com/young-su-ko/TUnA-R.git
cd TUnA-R
uv init
uv sync
```

> We are working on an accompanying PyPI package as well -->

## Usage (Inference & Training)

### Inference

#### CLI

```
tuna predict $seqA $seqB --model $tuna --device $device
```
The CLI is intentionally minimal for a single‑pair prediction. For batch inference and embedding reuse, use the API below.

Pretrained shortcuts: `tuna`, `tfc`, `esm_mlp`, `esm_gp` (loads weights from HuggingFace). You can also pass a full HF repo id or a local path if you trained your own models on your own datasets.

#### API

<details>
<summary><strong>Embedding Management (Inference)</strong></summary>

For batch inference, embeddings are managed explicitly:
- `EmbeddingStore.load(path)` to reuse saved embeddings
- `EmbeddingStore.save(path)` to persist updated embeddings
- `InferencePipeline.predict_pairs(...)` to score pairs using the store

If you pass a store, the pipeline will only compute missing embeddings and reuse existing ones.

</details>

Batch inference utilities live in `tuna.inference.pipeline` and `tuna.inference.embeddings`.

```python
from tuna.inference.embeddings import EmbeddingStore
from tuna.inference.pipeline import InferencePipeline
from tuna.inference.predictor import Predictor

pairs = [
    ("MKPPPW", "MKDDW"),
    ("MKWA", "MKWDE"),
]

predictor = Predictor.from_pretrained("tuna", device="cpu") # change to "cuda" if you have a GPU
pipeline = InferencePipeline(predictor)

predictions = pipeline.predict_pairs(pairs, batch_size=32)
print(predictions)
```

Embedding cache example:

```python
store = EmbeddingStore.load("data/embeddings.pt")
pipeline = InferencePipeline(predictor, store=store)
scores = pipeline.predict_pairs(pairs, batch_size=32)
store.save("data/embeddings.pt") # save embeddings for re-use
```
> [!WARNING]  
> Embeddings are saved as a Python dictionary and fully loaded into memory. While this works fine for a small to medium number of embeddings, can be problematic as the # embeddings grows.


### Training

#### CLI

```
tuna train --model tuna --dataset gold-standard --max-epochs 14
```
Pass additional Hydra overrides with repeated `--override` flags:
```
tuna train --model tuna --dataset gold-standard --override trainer.precision=16 --override trainer.devices=1
```
Notes:
- This is a thin wrapper around `python train.py` with Hydra. Any override that works with Hydra can be passed via `--override`.
- Model can also be trained by running `train.py` directly if you want more control, for example adding callbacks.
- `--config-name` and `--config-path` let you point to a custom Hydra config entry.
- Example with a custom config path:
  ```
  tuna train --config-path configs --config-name config --model tuna --dataset gold-standard --max-epochs 10
  ```

For training any of the four models (TUnA, T-FC, ESM-MLP, ESM-GP) on new datasets, please adjust the `configs/config.yaml` and the `configs/model/{model_name}.yaml` files accordingly. 

> T-FC, ESM-MLP, and ESM-GP are all ablated versions of TUnA. See original paper for details.


<details>
<summary><strong>Embedding Generation &amp; Management (Read if using custom dataset!)</strong></summary>

During training, the datamodule requires a protein embedding dictionary mapping `protein_id -> embedding`. You can provide this in your dataset config as `paths.embeddings`. If `embeddings` is missing or null, the code will generate embeddings from a FASTA file and save them as a `.pt` file.

Expected dataset config fields:
- `paths.train`, `paths.val`, `paths.test`: TSV files where the first two columns are protein IDs
- `paths.embeddings`: path to a `.pt` embedding dictionary (optional)
- `paths.fasta`: path to a FASTA file containing all protein IDs used in train/val/test (required if embeddings are missing)

Safety checks:
- If `paths.embeddings` exists, it is loaded and validated to contain all IDs from train/val/test.
- If `paths.embeddings` is missing, the FASTA file is required.
- If any IDs are missing from the FASTA, an error is raised. Please make sure all IDs are include din the FASTA file.

Default output:
- If `paths.embeddings` is not set, embeddings are saved to `embeddings/<dataset_name>_embeddings.pt`.

Example (generate embeddings from FASTA):
```
paths:
  train: "/path/to/train.tsv"
  val: "/path/to/val.tsv"
  test: "/path/to/test.tsv"
  embeddings: null
  fasta: "/path/to/all_sequences.fasta"
```

</details>


## Purpose of this repository

<details><summary><strong>Why is there a new repo?</strong></summary>

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
</details>

## Overview of Codebase Structure

### Models

To streamline the two different model architectures utilized for ablations in the manuscript (MLP-based vs Transformer-based), we defined a `PPIPredictor` class which can run either backbones. It is a `nn.Module` that will later passed to the PyTorch Lightning module for training, but also used without Lightning for inference.


### Lightning Module

The `LitPPI` is the PyTorch Lightning module used to streamline training. It inherits from `BaseModule`, which just defines some helper and setup functions such as initializing the `torchmetrics` objects. `LitPPI`'s reponsibilities include: Initialization of model weights, optimizer configuration, and train/val loops.

### Configs

The configs are currently set up to work with Lightning and Hydra's `instantiate` function.

### Tests

Currently, tests are primarily smoke tests to make sure the code is working as intended.

## Results

Again, there are differences from the results presented in the manuscript. I am not aiming for one-to-one reproduction of results, but rather making the code as streamlined and readable as possible, while keeping the original logic. 

>**For now, I have only trained the models on the Bernett gold-standard dataset.**

| Model | AUROC |MCC |Accuracy |AUPRC | Precision|
| --- | --- | --- | --- | --- | --- |
| TUnA | 0.70 |0.30 |0.65 |0.69 |0.65 |
| T-FC | 0.69 |0.27 |0.63 |0.68 |0.62 |
| ESM-MLP | 0.69 |0.27 |0.64 |0.68 |0.63 |
| ESM-GP | 0.70 |0.27 |0.63 |0.69 |0.65 |


## TODO

- Setting batch size=1 during inference is very slow
- Adding more robust testing
- Train on additional datasets 
