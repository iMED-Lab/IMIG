# Angiography-free Diagnosis of Retinal Diseases via Interpretable Multi-modal Learning
This repository contains the official implementation of **"Angiography-free Diagnosis of Retinal Diseases via Interpretable Multi-modal Learning"**.

## Overview

We propose an interpretable incomplete multi-modal learning framework that enables accurate retinal disease diagnosis using only **Color Fundus Photography (CFP)** at inference time, while leveraging paired CFP and **Fluorescein Fundus Angiography (FFA)** data during training. Instead of synthesizing FFA images, our model learns to disentangle shared and modality-specific features from paired inputs and stores them in disease-specific prototype libraries. At inference, the model completes missing FFA information by indexing CFP features against the pre-built FFA library through a shared projection space.


### Key Features

- **Angiography-free inference**: Only CFP images are required at test time — no FFA acquisition needed.
- **Feature-library-based completion**: Missing FFA information is recovered via cross-modal prototype indexing rather than image synthesis, avoiding generative artifacts.
- **Interpretable decisions**: Predictions are traced back to matched typical disease features, providing clinically aligned visual explanations.
- **Multi-disease coverage**: Supports 7 retinal diseases — DR, AMD, RVO, ME, VH, CSC, and High Myopia.

## Repository Structure

```
├── main.py                      # Main training script (three-stage training)
├── train_and_test.py            # Training/evaluation loops and loss functions
├── conf.py                      # Dataset and dataloader configuration
├── datasets.py                  # Paired CFP-FFA retinal dataset class
├── run.sh                       # Example launch script (DDP, 2 GPUs)
│
├── models/
│   ├── model.py                 # PPNet (single branch) and MultiModel (dual branch)
│   └── convnext_features.py     # ConvNeXt-Base backbone feature extractor
│
└── utils/
    ├── settings.py              # Hyperparameters and training schedule
    ├── receptive_field.py       # Receptive field computation for visualization
    ├── preprocess.py            # ImageNet normalization utilities
    ├── helpers.py               # Directory creation, activation crop utilities
    ├── log.py                   # Simple file + console logger
    └── save.py                  # Conditional model checkpointing
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- torchvision
- scikit-learn
- pandas, openpyxl
- opencv-python
- matplotlib
- numpy

Install dependencies:

```bash
pip install torch torchvision scikit-learn pandas openpyxl opencv-python matplotlib numpy
```

## Data Preparation

### Internal Dataset

The model expects paired CFP-FFA data organized as:

```
dataset/large/
├── dataAll/
│   └── <patient_name>/
│           └── <OS_or_OD>/
│               ├── CFP_enhanced/      # Color fundus photographs
│               │   └── *.jpg
│               └── FFA_select/        # FFA frames (early/mid/late phases)
│                   └── *.jpg
├── label/
│   └── paired/
│           ├── train.xlsx
│           ├── validation.xlsx
│           └── test.xlsx
```


### External Datasets

For external validation, we used 9 publicly available CFP datasets. See the paper's Data Availability section for download links.

## Usage

### Training

Configure your data path in `conf.py`:

```python
datapath = '/path/to/your/dataset/large'
resultPath = '/path/to/save/results'
```

Launch training with Distributed Data Parallel:

```bash
bash run.sh
```

Or manually:

```bash
python -m torch.distributed.launch \
    --nproc_per_node=2 \
    --master_port=21676 \
    --use_env main.py \
    -gpuid='0,1' \
    -num_prototypes=70 \
    -m=0.1 \
    -last_layer_fixed=False \
    -subtractive_margin=False \
    -using_deform=False \
    -topk_k=1 \
    -incorrect_class_connection=-0.5 \
    -rand_seed=1
```

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `-gpuid` | `'0'` | GPU device IDs |
| `-num_prototypes` | `70` | Total number of typical features (divided evenly across classes) |
| `-m` | `1` | Subtractive margin value |
| `-last_layer_fixed` | `True` | Whether to fix classifier weights during warm-up |
| `-topk_k` | `1` | Top-k activations per typical feature during training |
| `-rand_seed` | `20` | Random seed for reproducibility |

### Hyperparameters

Additional hyperparameters (learning rates, training schedule, loss coefficients) are configured in `utils/settings.py`.


## Citation

If you find this work useful, please cite:

```bibtex
@article{hao2026angiography,
  title={Angiography-free Diagnosis of Retinal Diseases via Interpretable Multi-modal Learning},
  author={Hao, Jinkui and others},
  journal={NPJ digital medicine},
  year={2026}
}
```

## License

This project is for research purposes. Please refer to the license terms of the individual external datasets used for validation.
