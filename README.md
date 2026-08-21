<div align="center">

<img src="static/images/muse.jpg" width="110" alt="VT-MUSE Logo"><br/>

# VT-MUSE

**Multimodal Unified Sequential Visuotactile Representation Learning for Manipulation**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![GitHub Stars](https://img.shields.io/github/stars/VT-SERIES/VT-MUSE?style=social)](https://github.com/VT-SERIES/VT-MUSE/stargazers)

<video src="static/videos/teaser.mp4" autoplay loop muted playsinline width="80%"></video>

</div>

---

VT-MUSE is a temporal visuotactile representation framework that jointly models visual and tactile dynamics through a unified cVAE-based encoder-decoder. A two-stage training scheme first aligns the two modalities temporally, then optimizes the full model under multi-task supervision. The prior latent from `VTMUSE.encode_window` plugs directly into any downstream imitation-learning policy (e.g. ACT).

## Quick Links

| | |
|---|---|
| 📦 [Installation](docs/installation.md) | Environment setup and dependencies |
| 🚀 [Training](docs/training.md) | Data preparation, Stage 1 & 2 training, downstream interface |
| 📂 [Data Format](docs/data_format.md) | HDF5 schema and compatibility rules |

## Repository Structure

```text
vt_muse/
  data.py       HDF5 loading, temporal sampling, masking, depth-flow targets
  tactile.py    tactile temporal and flow preprocessing
  model.py      unified VT-MUSE architecture
  losses.py     Stage 1 and Stage 2 objectives
  train.py      single-GPU and DDP training entry point
scripts/
  train_stage1.sh
  train_stage2.sh
docs/
  installation.md
  training.md
  data_format.md
```

## Citation

Citation information will be added after the anonymous review period.

## License

This project is released under the [MIT License](LICENSE).
