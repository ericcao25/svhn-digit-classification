# SVHN Digit Classification

Classifying house-number digits from the [Street View House Numbers (SVHN)](http://ufldl.stanford.edu/housenumbers/) dataset, comparing an MLP (on raw and PCA-reduced pixel features), a voting ensemble of MLPs, and a custom PyTorch CNN.

## Repo structure

| File | Description |
|---|---|
| [`main.ipynb`](./main.ipynb) | Main notebook: preprocessing, MLP, PCA, ensembling, final CNN. Start here. |
| [`appendix.ipynb`](./appendix.ipynb) | Supporting hyperparameter sweeps and alternative approaches referenced from `main.ipynb`. Self-contained (doesn't require `main.ipynb` to be run first). |
| [`report.pdf`](./report.pdf) | Full write-up: methodology, results, and discussion. |
| `data/` | Not included in this repo (see Setup below). |

## Setup

1. Clone the repo and install dependencies:
   ```
   pip install -r requirements.txt
   ```
2. Download `train_32x32.mat` and `test_32x32.mat` (the "Format 2" cropped-digit files) from the [SVHN site](http://ufldl.stanford.edu/housenumbers/) and place them in `data/`.
3. Open `main.ipynb` in Jupyter and run top to bottom. `appendix.ipynb` can be run independently the same way.

**GPU:** the CNN sections support training on a GPU via `device='cuda:N'` (see `train_cnn`'s docstring in `main.ipynb`). On a shared multi-GPU machine, pass an explicit device index rather than leaving `device=None` (which defaults to `cuda:0`) so multiple runs don't contend for the same GPU.

## Results

| Model | Best test accuracy |
|---|---|
| MLP, raw pixels (no normalization, 512-unit hidden layer) | 0.8079 |
| MLP, PCA features (Otsu normalization, (128, 64)) | 0.8389 |
| MLP ensemble (architecture + seed diversity) | 0.8534 |
| CNN, default architecture (border/interior normalization) | 0.8869 |
| **CNN, tuned architecture + dropout (final model)** | **0.9265** |

See `report.pdf` for the full discussion, including two counterintuitive findings: normalization choice affects the MLP and CNN in opposite directions, and dropout only helps on specific layers rather than uniformly across the network.
