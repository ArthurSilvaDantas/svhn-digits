<div align="center">

# From MLP to CNN on SVHN

**A controlled ablation study of what actually improves digit classification on real-world images**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Model on HF](https://img.shields.io/badge/%F0%9F%A4%97%20model-arthursilvadantas%2FSVHN-yellow)](https://huggingface.co/arthursilvadantas/SVHN)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

[Experiments](docs/EXPERIMENTS.md) · [Report](docs/report/) · [Notebooks](notebooks/) · [Model](https://huggingface.co/arthursilvadantas/SVHN)

</div>

## Overview

This project trains **17 models** on [SVHN](http://ufldl.stanford.edu/housenumbers/) (Street View House Numbers),
from a plain MLP to a regularized VGG-style CNN. Each experiment changes **a single factor** with respect to a
previous one, so every gain or loss can be attributed to that factor. The final model is then examined beyond
accuracy: calibration, Grad-CAM attention, sensitivity to horizontal flips and label noise in the dataset.

## Results

| Final model | Parameters | Test accuracy | Calibration (ECE) |
|:---:|:---:|:---:|:---:|
| VGG-style CNN (E16) | 2.2M | **97.13%** | **0.49%** |

![Ablation study](figures/ablation.png)

| Change | Δ validation accuracy | Verdict |
|---|---:|:---:|
| MLP → CNN | **+6.6 p.p.** | ✅ |
| LeNet-style → VGG-style (stacked 3×3 convolutions) | **+3.4 p.p.** | ✅ |
| Dropout | **+2.0 p.p.** | ✅ |
| Data augmentation | +0.7 p.p. (train/val gap 4.1 → 1.7) | ✅ |
| Wider filters (64-128-256) | +0.3 p.p. at 2.7× the parameters | ➖ |
| Optimizer, learning-rate schedule, batch size | within ±0.4 p.p. | ➖ |
| SGD without momentum | −0.8 p.p., unstable training | ❌ |

**Takeaways.** Architecture and regularization account for almost all of the improvement; optimization settings
barely matter. Part of the remaining error comes from the dataset itself: 136 test labels (0.5%) are flagged as
wrong by an ensemble of 14 independently trained CNNs, and excluding them raises test accuracy to 97.64%.

| Training curves | Calibration |
|:---:|:---:|
| ![Training curves](figures/curves.png) | ![Calibration](figures/calibration.png) |
| Regularization removes overfitting | Confidence matches accuracy |
| **Grad-CAM** | **Horizontal flip test** |
| ![Grad-CAM](figures/gradcam.png) | ![Flip test](figures/flip.png) |
| 83.9% of the attention on the labeled (central) digit | 40.8% accuracy on mirrored digits: no flip augmentation |

The per-experiment analysis is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Methodology

| | |
|---|---|
| **Data split** | official `train` split 90/10, stratified (65,931 / 7,326); official `test` (26,032) for reporting only |
| **Preprocessing** | per-channel standardization with training-set statistics |
| **Augmentation** | rotation ±10°, translation ±10%, scale 0.9-1.1, brightness/contrast ±20%, no flip |
| **Training** | cross-entropy, He initialization, Adam (lr 10⁻³), batch 128, early stopping on validation accuracy |
| **Model selection** | highest validation accuracy; the test set never influenced any decision |
| **Final model** | 3 × [conv 3×3 → BN → ReLU] × 2 blocks (64-128-256), Dropout 0.1/0.5, OneCycle, 40 epochs |

## Getting started

```bash
git clone https://github.com/ArthurSilvaDantas/svhn-digits && cd svhn-digits
pip install -r requirements.txt
cd notebooks && jupyter lab
```

The notebooks (in Portuguese) are already executed, so all results render on GitHub. SVHN is downloaded
automatically on the first run.

| # | Notebook | Content |
|---|---|---|
| 1 | [Data exploration](notebooks/01_data_exploration.ipynb) | loading, stratified split, class distribution |
| 2 | [Data quality](notebooks/02_data_quality.ipynb) | integrity, duplicates, train/test leakage, contrast |
| 3 | [Preprocessing](notebooks/03_preprocessing.ipynb) | normalization and GPU data augmentation |
| 4 | [Models](notebooks/04_models.ipynb) | MLP, LeNet-style and VGG-style CNNs |
| 5 | [Training](notebooks/05_training.ipynb) | training loop and the 17 experiments |
| 6 | [Training curves](notebooks/06_training_curves.ipynb) | learning curves and overfitting diagnostics |
| 7 | [Final model](notebooks/07_final_model.ipynb) | calibration, Grad-CAM, flip test, computational cost |
| 8 | [Label noise audit](notebooks/08_label_noise_audit.ipynb) | wrong labels in SVHN |
| 9 | [Batch inference](notebooks/09_unlabeled_set_predictions.ipynb) | predictions on an unlabeled image set |

## Pretrained models

The final model and the checkpoints of all 17 experiments are available on the
[Hugging Face Hub](https://huggingface.co/arthursilvadantas/SVHN).

```python
# inside notebooks/
from svhn_utils import load_model

model, norm, ckpt = load_model("best_model", "cpu")   # or "E1" ... "E17"
probs = model(norm(x)).softmax(1)                     # x: uint8 tensor [N, 3, 32, 32]
```

## Dataset

SVHN *cropped digits*: 32×32 RGB images, 10 classes. Available from
[Stanford](http://ufldl.stanford.edu/housenumbers/),
[Kaggle](https://www.kaggle.com/datasets/stanfordu/street-view-house-numbers),
[Hugging Face](https://huggingface.co/datasets/ufldl-stanford/svhn) and
[`torchvision`](https://pytorch.org/vision/stable/generated/torchvision.datasets.SVHN.html) (used here).
For non-commercial use only.

## Limitations

- Each experiment was run with a single seed; differences below ~0.4 p.p. may be noise.
- Factors were changed along one path rather than in a full factorial design.
- The 531k-image `extra` split was not used, so results are not comparable to state-of-the-art SVHN models.

## Repository structure

```
├── notebooks/         executed notebooks, shared code (svhn_utils.py) and saved metrics
├── docs/              experiments write-up and report
├── figures/           figures used in this README
└── requirements.txt   dependencies
```

## License

Released under the [MIT License](LICENSE). SVHN: Netzer et al., *Reading Digits in Natural Images with
Unsupervised Feature Learning*, NIPS Workshop 2011.
