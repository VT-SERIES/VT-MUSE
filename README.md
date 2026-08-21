# VT-MUSE

**Multimodal Unified Sequential Visuotactile Representation Learning for Manipulation**

This repository provides the compact, self-contained training implementation
of VT-MUSE for anonymous review. It focuses on the proposed method: temporal
visual-tactile tokenization, temporal-memory encoding, masked latent
reconstruction, and the representation interface used by downstream
manipulation policies.

The release intentionally contains only method-critical training components.
Evaluation utilities, deployment code, simulator environments, ablation-only
branches, checkpoint sweeps, machine-specific paths, datasets, checkpoints,
and generated artifacts are excluded.

## Method overview

VT-MUSE uses a shared temporal window of visual and bilateral tactile
observations. Stage 1 aligns visual and tactile tokens with cross-modal,
temporal, and consistency objectives. Stage 2 masks the final observations,
uses a temporal-memory transformer and conditional latent model, and jointly
reconstructs RGB observations and tactile depth-flow. The prior latent returned
by `VTMUSE.encode_window` is the representation consumed by a downstream
policy.

## Repository structure

```text
vt_muse/
  data.py           HDF5 loading, temporal sampling, masking, depth-flow targets
  tactile.py        tactile temporal and flow preprocessing
  model.py          unified VT-MUSE architecture
  losses.py         Stage 1 and Stage 2 objectives
  train.py          single-GPU and DDP training entry point
scripts/
  train_stage1.sh
  train_stage2.sh
docs/data_format.md
```

`vt_muse.model.VTMUSE` is the canonical implementation.

## Installation

```bash
git clone https://github.com/VT-SERIES/VT-MUSE.git
cd VT-MUSE
python -m pip install -e .
```

Python 3.10+ and a CUDA-enabled PyTorch installation are recommended. The
visual and tactile encoders initialize from `google/vit-base-patch16-224`.
Set `VT_MUSE_VIT_CHECKPOINT` to a local Hugging Face checkpoint directory
when training offline.

## Data

Prepare separate directories containing training and validation `.hdf5` files:

```text
dataset/
  train/
    insert_HDMI__episode_000000.hdf5
    ...
  val/
    insert_HDMI__episode_000450.hdf5
    ...
```

The required HDF5 keys and compatibility rules are documented in
[`docs/data_format.md`](docs/data_format.md).

The loader validates modality lengths before constructing temporal windows.
RGB observations, bilateral tactile RGB images, bilateral tactile depth maps,
and the eight-dimensional robot state must be aligned within each trajectory.

## Stage 1: visual-tactile alignment

```bash
NPROC_PER_NODE=4 BATCH_SIZE=160 \
  bash scripts/train_stage1.sh dataset/train dataset/val checkpoints/stage1
```

The formal default uses a five-frame window, stride five, a 512-dimensional
latent space, and trains for 30 epochs. The best checkpoint is written to
`checkpoints/stage1/stage1_best.pth`.

## Stage 2: masked multimodal reconstruction

```bash
NPROC_PER_NODE=4 BATCH_SIZE=160 \
  bash scripts/train_stage2.sh \
  dataset/train dataset/val \
  checkpoints/stage1/stage1_best.pth \
  checkpoints/stage2
```

Stage 2 trains for 20 epochs and reconstructs both RGB observations and dense
tactile depth-flow. The best checkpoint is written to
`checkpoints/stage2/stage2_best.pth`.

Batch size and worker count should be adjusted for the available hardware. All
training options are available through:

```bash
python -m vt_muse.train --help
```

## Downstream policy representation

After loading a Stage 2 checkpoint, the policy-facing representation is:

```python
features = model.encode_window(
    visual_seq=visual_window,
    tactile_seq=tactile_window,
    action_seq=None,
    visual_mask=tail_mask,
    delta_steps=delta_steps,
    task_id=task_id,
)
```

The policy architecture itself is deliberately not duplicated here; VT-MUSE
provides the learned multimodal representation that can be connected to ACT or
another imitation-learning policy.

## Citation

Citation information will be added after the anonymous review period.

## License

This project is released under the [MIT License](LICENSE).
