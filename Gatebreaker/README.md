# GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs

Implementation of the GateBreaker paper ([arXiv:2512.21008v2](https://arxiv.org/abs/2512.21008)).

> **Disclaimer**: This implementation is for academic research and reproducibility purposes only. 
> It targets only openly released models and uses publicly available datasets.

## Paper Overview

GateBreaker is a training-free, inference-time framework that analyzes safety alignment in 
Mixture-of-Experts (MoE) LLMs through a three-stage pipeline:

1. **Gate-level Profiling** — Identifies safety experts via gate activation analysis
2. **Expert-level Localization** — Finds safety neurons within those experts
3. **Targeted Safety Removal** — Clamps safety neurons to zero at inference time

## Project Structure

```
try-gatebreaker/
├── src/
│   ├── gatebreaker/
│   │   ├── __init__.py           # Package init
│   │   ├── config.py             # Configuration dataclass
│   │   ├── datasets.py           # Dataset loading & content-token masking
│   │   ├── model_patcher.py      # Runtime MoE compute graph patching
│   │   ├── gate_profiler.py      # Stage 1: Gate-level profiling
│   │   ├── expert_localizer.py   # Stage 2: Expert-level localization
│   │   ├── safety_remover.py     # Stage 3: Targeted safety removal
│   │   └── metrics.py            # ASR, Safety Neuron Ratio evaluation
│   └── modal_app.py              # Modal cloud GPU deployment
├── gatebreaker_demo.ipynb        # Jupyter notebook (main entry point)
├── requirements.txt              # Python dependencies
├── setup.bat                     # Windows setup script
└── README.md                     # This file
```

## Quick Start

### 1. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install jupyter ipykernel

# Linux/Mac
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install jupyter ipykernel
```

### 2. Set Up Modal

```bash
pip install modal
modal setup  # Follow the prompts to authenticate
```

### 3. Run via Modal Cloud GPU

```bash
cd src
modal run modal_app.py
```

### 4. Or Use the Notebook

```bash
jupyter notebook gatebreaker_demo.ipynb
```

## Target Model

The default target is **Qwen/Qwen1.5-MoE-A2.7B-Chat**:
- Mixture MoE architecture (60 sparse + 4 shared experts)
- 2.7B active / 14.3B total parameters
- Top-4 routing

## Key Hyperparameters

| Parameter | Value | Reference |
|-----------|-------|-----------|
| Safety expert selection | top-3k | Section 4.2 |
| z-threshold (τ) | 2.0 | Section 4.2 |
| Aggregation function | element-wise max | Eq. 8-9 |
| Target sublayers | gate_proj + up_proj | Section 7.2 |

## References

```bibtex
@article{wu2025gatebreaker,
  title={GateBreaker: Gate-Guided Attacks on Mixture-of-Expert LLMs},
  author={Wu, Lichao and Behrouzi, Sasha and Rostami, Mohamadreza and Picek, Stjepan and Sadeghi, Ahmad-Reza},
  journal={arXiv preprint arXiv:2512.21008},
  year={2025}
}
```
