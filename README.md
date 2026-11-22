# SAFL-Net

Training and evaluation workflows for joint breast ultrasound lesion segmentation
and benign–malignant classification.

[简体中文](README.zh-CN.md)

## Overview

SAFL-Net uses paired ultrasound images and segmentation masks together with binary
classification labels. The command-line workflow supports explicit fold manifests,
paired data augmentation, point and box prompts, joint loss optimization,
checkpoint selection and per-case evaluation exports.

## Requirements

- Python 3.9
- PyTorch 2.4.1 and torchvision 0.19.1
- Dependencies listed in [requirements.txt](requirements.txt)
- An importable model backend, configured as described below
- CUDA for GPU training; CPU is supported by the workflow utilities

## Installation

```bash
git clone https://github.com/zhl05/SAFL-Net.git
cd SAFL-Net
python -m venv .venv
```

Activate the environment with `source .venv/bin/activate` on Linux/macOS or
`.venv\Scripts\Activate.ps1` in Windows PowerShell. Install the compatible
PyTorch/torchvision pair for your platform using the
[PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/#v241),
then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Project structure

```text
SAFL-Net/
├── safl/
│   ├── backend.py          # Model integration
│   ├── data.py             # Dataset, paired transforms and prompts
│   ├── engine.py           # Training and evaluation routines
│   ├── losses.py           # Segmentation and classification losses
│   ├── metrics.py          # Segmentation and classification metrics
│   └── io.py               # Checkpoint and report output
├── examples/               # Example manifest format
├── tests/                  # Synthetic utility checks
├── train.py                # Single-fold training
├── test.py                 # Checkpoint evaluation
├── evaluate.py             # Alias for test.py
├── requirements.txt
└── README.zh-CN.md
```

## Data preparation

Keep data outside the source directory. Images and masks are paired grayscale
PNGs with identical original dimensions; nonzero mask pixels represent foreground.

```text
data_root/
└── BUSBRA/
    ├── Image/
    │   └── example_case.png
    └── Label/
        └── example_case.png
```

Each UTF-8 manifest row has the following format, without the `.png` extension:

```text
foreground_id/dataset_subdirectory/case_name/class_label
1/BUSBRA/example_case/0
```

`foreground_id` is `1`; classification labels are `0` for benign and `1` for
malignant. Confirm this mapping against your annotations. Blank lines and `#`
comments are accepted. Duplicate case identifiers and overlapping training and
validation cases are rejected. Prepare patient-disjoint splits independently;
the loader does not infer patient identity. The manifests in `examples/` use
fictional identifiers to illustrate the format.

Validate image/mask pairs before training:

```bash
python -m safl.data --data-root /path/to/data_root --manifest /path/to/fold1_val.txt
```

## Model configuration

The environment must provide a model module named `safl_model`, or another
importable module selected with the `SAFL_BACKEND` environment variable.
The adapter in [safl/backend.py](safl/backend.py) expects two functions:

- `build_model(image_size=256, low_image_size=128)` returns a PyTorch module.
- `build_optimizer(model)` returns `(optimizer, scheduler)`. A scheduler may be
  `None`; otherwise it steps after each successful optimizer update.

The model accepts `model(image, points, boxes, use_auto_prompt=False)` and returns
`(segmentation, class_logits, auxiliary_1, auxiliary_2)`. The segmentation dictionary
contains full-resolution `masks` logits of shape `B×1×256×256` and `low_res_logits`
of shape `B×1×128×128`; classification logits have shape `B×2`.

## Training

Run one fold per invocation with its corresponding training and validation manifests:

```bash
python train.py --data-root /path/to/data_root --train-manifest /path/to/fold1_train.txt --val-manifest /path/to/fold1_val.txt --fold 1 --output outputs/fold1
```

Defaults are 100 epochs, batch size 8 and seed 1234. Select `--device cpu` for CPU
execution. Use `python train.py --help` for all arguments. Choose a new or empty
output directory. Repeat with independent manifests and output directories for
folds 1–5.

Training saves `foldN_best_dice.pth`, `foldN_best_auc.pth`, per-selector validation
predictions and `history.json`. The secondary metric breaks selection ties. AUC
checkpoint selection requires both reference classes. These checkpoints contain
model weights and selection metadata; they are not interruption/resume snapshots.

## Testing

Evaluate a matching checkpoint on a separately prepared evaluation manifest:

```bash
python test.py --data-root /path/to/data_root --manifest /path/to/test.txt --checkpoint /path/to/fold1_best_auc.pth --output outputs/test_fold1 --save-masks
```

`evaluate.py` accepts the same arguments. Use `python test.py --help` for all options.
Checkpoint loading uses strict parameter matching. Select a new or empty output
directory. Outputs include `metrics.json` and `predictions.csv`; `--save-masks`
also writes binary PNG masks and computes HD95/ASSD.

## Evaluation protocol

- Images and full masks use `256×256` resolution; loss masks use `128×128`.
- Both training and evaluation use mask-derived point and box prompts. Evaluation
  requires reference masks and uses a deterministic point and tight bounding box.
- Training uses AMP on CUDA. Evaluation uses FP32.
- Segmentation uses sigmoid probability `> 0.5`; classification uses class-1
  softmax probability `>= 0.5`. AUC uses continuous scores and is `null` for a
  single-class evaluation set.
- Dice and IoU are averaged per case with `1e-5` smoothing. Classification exports
  include AUC, accuracy, F1, precision, sensitivity and specificity.
- HD95/ASSD use bidirectional boundary distances in resized-image pixels.
  Both-empty masks have distance zero; one-empty cases are counted separately
  and excluded from the distance means.
- Validation metrics used for checkpoint selection should be reported as validation
  results. Fix evaluation manifests independently of checkpoint selection.

## Verification

```bash
python -m unittest discover -s tests -v
```

These checks use synthetic data to validate utilities and command-line behavior.
They do not measure segmentation or classification performance.

## Acknowledgements and rights

See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for upstream acknowledgements
and [RIGHTS.md](RIGHTS.md) for usage terms. Dataset use is subject to the relevant
dataset access conditions.
