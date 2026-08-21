# Installation

## Requirements

- Python 3.10+
- A CUDA-enabled PyTorch installation (2.0+ recommended)

## Steps

```bash
git clone https://github.com/VT-SERIES/VT-MUSE.git
cd VT-MUSE
python -m pip install -e .
```

This installs the `vt_muse` package along with all required dependencies:

| Package | Version |
|---|---|
| `torch` | ≥ 2.0 |
| `torchvision` | ≥ 0.15 |
| `transformers` | ≥ 4.30 |
| `h5py` | ≥ 3.8 |
| `numpy` | ≥ 1.24 |
| `Pillow` | ≥ 9.5 |
| `tensorboard` | ≥ 2.13 |
| `tqdm` | ≥ 4.65 |
| `matplotlib` | ≥ 3.7 |

### Optional: WandB logging

```bash
pip install -e ".[logging]"
```

## Pretrained ViT checkpoint

The visual and tactile encoders initialize from
[`google/vit-base-patch16-224`](https://huggingface.co/google/vit-base-patch16-224).
When training **offline**, download the checkpoint and set the environment variable:

```bash
export VT_MUSE_VIT_CHECKPOINT=/path/to/local/vit-base-patch16-224
```
