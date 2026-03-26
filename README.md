# Hybrid DE–BO Hyperparameter Optimisation for YOLO11 Object Detection

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Supplementary code** for the manuscript submitted to *Scientific Reports*:  
> *[Your Full Paper Title Here]*  
> [Author 1], [Author 2], … — [Year]

---

## Overview

This repository contains the complete implementation of the hybrid
**Differential Evolution (DE) + Bayesian Optimisation (BO)** pipeline used to
tune the hyperparameters of a YOLO11-based object detector.

The algorithm proceeds in three phases:

```
Phase 1 – Differential Evolution (low-fidelity screening)
          ↓  best candidate
Phase 2 – High-fidelity evaluation of the DE winner
          ↓  warm-start point
Phase 3 – Bayesian Optimisation (Gaussian Process, high-fidelity)
          ↓
       Final best hyperparameter set
```

---

## Repository Structure

```
.
├── hyperparameter_optimization.py   # Main script (this paper's contribution)
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
└── LICENSE                          # MIT licence
```

---

## Requirements

| Package | Tested version |
|---------|----------------|
| Python  | ≥ 3.9 |
| PyTorch | ≥ 2.0 |
| ultralytics | ≥ 8.3 |
| scikit-optimize | ≥ 0.9 |
| numpy | ≥ 1.24 |
| matplotlib | ≥ 3.7 |
| PyYAML | ≥ 6.0 |

Install all dependencies at once:

```bash
pip install -r requirements.txt
```

---

## Dataset

Experiments in the paper were conducted on the
**[VisDrone2019](https://github.com/VisDrone/VisDrone-Dataset)** dataset.  
Download the dataset and prepare a `data.yaml` file in the
[Ultralytics format](https://docs.ultralytics.com/datasets/detect/):

```yaml
# data.yaml – example
path: /path/to/VisDrone          # root directory
train: images/train
val:   images/val
nc: 10
names: [pedestrian, people, bicycle, car, van, truck, tricycle,
        awning-tricycle, bus, motor]
```

---

## Usage

```bash
python hyperparameter_optimization.py \
    --data_yaml          /path/to/dataset/data.yaml \
    --project_dir        /path/to/output/directory \
    --model_yaml         /path/to/ultralytics/cfg/models/11/yolo11n.yaml \
    --pretrained_weights /path/to/yolo11n.pt \
    --generations        10 \
    --population_size    5 \
    --bo_iterations      15 \
    --optimizers         Adam
```

### Key Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `--data_yaml` | Path to dataset YAML file (**required**) | — |
| `--project_dir` | Root output directory (**required**) | — |
| `--model_yaml` | Path to YOLO11 architecture YAML (**required**) | — |
| `--pretrained_weights` | Path to pretrained `.pt` weights (optional) | `None` |
| `--generations` | DE generations | `10` |
| `--population_size` | DE population size | `5` |
| `--bo_iterations` | BO evaluations | `15` |
| `--optimizers` | `SGD`, `Adam`, or `both` | `Adam` |

Run `python hyperparameter_optimization.py --help` for the full list.

---

## Outputs

All outputs are written to `--project_dir/<optimizer>_optimizer/`:

| File / Folder | Contents |
|---------------|----------|
| `iteration_log.txt` | Per-individual fitness log (CSV-style) |
| `checkpoint.pkl` | Latest resumable checkpoint |
| `gen_*/` | YOLO training artefacts per evaluation |
| `high_fidelity_evaluation/` | Full-epoch training of the DE best |
| `optimization_history.png` | Fitness curve across DE generations |
| `training.log` | Timestamped event log |

---

## Resuming an Interrupted Run

The script automatically saves a `checkpoint.pkl` after every generation.
Re-run the exact same command and it will pick up where it left off.

---

## Hyperparameter Search Space

| Hyperparameter | Type | Range / Choices |
|----------------|------|-----------------|
| Learning rate | Continuous | [1 × 10⁻⁴, 1 × 10⁻²] |
| Batch size | Integer | [8, 16] |
| Momentum | Continuous | [0.85, 0.95] |
| Weight decay | Continuous | [1 × 10⁻⁵, 1 × 10⁻³] |
| Optimizer | Categorical | {SGD, Adam} |

---

## Reproducibility

A global random seed of **42** is applied to Python's `random`, NumPy, and
PyTorch at module load time.

---

## Citation

If you use this code, please cite:

```bibtex
@article{YourLastName_YYYY,
  title   = {Your Full Paper Title},
  author  = {Author One and Author Two},
  journal = {Scientific Reports},
  year    = {YYYY},
  doi     = {10.XXXX/XXXXXXXX}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
