# Training

## Data Preparation

Prepare separate directories of `.hdf5` trajectory files for training and
validation:

```text
dataset/
  train/
    insert_HDMI__episode_000000.hdf5
    ...
  val/
    insert_HDMI__episode_000450.hdf5
    ...
```

See [`data_format.md`](data_format.md) for the required HDF5 schema and
compatibility rules. The loader validates modality lengths before constructing
temporal windows — RGB observations, bilateral tactile RGB, bilateral tactile
depth maps, and the eight-dimensional robot state must be aligned within each
trajectory.

---

## Stage 1 — Visual-Tactile Alignment

Stage 1 trains cross-modal, temporal, and consistency objectives to align
visual and tactile tokens in a shared latent space.

```bash
NPROC_PER_NODE=4 BATCH_SIZE=160 \
  bash scripts/train_stage1.sh dataset/train dataset/val checkpoints/stage1
```

**Defaults:** 5-frame window · stride 5 · 512-dim latent · 30 epochs.
Best checkpoint → `checkpoints/stage1/stage1_best.pth`

---

## Stage 2 — Masked Multimodal Reconstruction

Stage 2 masks the final observations and jointly reconstructs RGB observations
and tactile depth-flow under a temporal-memory transformer with a conditional
latent model.

```bash
NPROC_PER_NODE=4 BATCH_SIZE=160 \
  bash scripts/train_stage2.sh \
    dataset/train dataset/val \
    checkpoints/stage1/stage1_best.pth \
    checkpoints/stage2
```

**Defaults:** 20 epochs.
Best checkpoint → `checkpoints/stage2/stage2_best.pth`

Adjust `NPROC_PER_NODE` and `BATCH_SIZE` for available hardware. All options:

```bash
python -m vt_muse.train --help
```

---

## Downstream Policy Interface

After loading a Stage 2 checkpoint, call `VTMUSE.encode_window` to obtain
the representation consumed by a downstream policy (e.g. ACT):

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

`VTMUSE.encode_window` returns the prior latent `z` — a fixed-size vector
ready to be fed into any imitation-learning policy head.
