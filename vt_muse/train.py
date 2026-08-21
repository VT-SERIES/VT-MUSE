"""Distributed Stage 1 and Stage 2 training entry point for VT-MUSE."""

import argparse
import json
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.optim as optim
from PIL import Image, ImageDraw
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vt_muse.data import VTMUSEDataset
from vt_muse.losses import (
    MaskedTemporalReconstructionLoss,
    compute_stage1_loss,
    compute_stage2_loss,
)
from vt_muse.model import VTMUSE
from vt_muse.visualization import TrainingCurvePlotter

try:
    import wandb
except ImportError:
    wandb = None


PREFERRED_TASK_ORDER = [
    "insert_HDMI",
    "insert_hole",
    "lift_bottle",
    "pull_out_key",
]


def order_task_names(task_names):
    task_name_set = set(task_names)
    ordered = [task_name for task_name in PREFERRED_TASK_ORDER if task_name in task_name_set]
    ordered.extend(sorted(task_name_set.difference(PREFERRED_TASK_ORDER)))
    return ordered


def is_dist_enabled() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_dist_enabled() or dist.get_rank() == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def reduce_scalar(value: float, device: str) -> float:
    if not is_dist_enabled():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    tensor /= dist.get_world_size()
    return float(tensor.item())


def unique_trainable_params(*modules):
    params = []
    seen = set()
    for module in modules:
        for param in module.parameters():
            if not param.requires_grad:
                continue
            param_id = id(param)
            if param_id in seen:
                continue
            seen.add(param_id)
            params.append(param)
    return params


def set_modality_encoder_trainable_vit_layers(modality_encoder, trainable_layers: int) -> dict:
    """Freeze a ViT backbone except the final N transformer blocks and final norm."""
    total_params = sum(param.numel() for param in modality_encoder.parameters())
    vit = modality_encoder.vit
    if hasattr(vit, "encoder") and hasattr(vit.encoder, "layer"):
        layers = vit.encoder.layer
    elif hasattr(vit, "layers"):
        layers = vit.layers
    else:
        raise AttributeError(
            "Cannot locate ViT transformer blocks; expected `vit.encoder.layer` or `vit.layers`."
        )
    total_layers = len(layers)

    if trainable_layers < 0:
        for param in modality_encoder.parameters():
            param.requires_grad = True
        return {
            "total_layers": total_layers,
            "trainable_layers": "all",
            "trainable_params": total_params,
            "total_params": total_params,
        }

    if trainable_layers > total_layers:
        raise ValueError(
            f"Requested {trainable_layers} trainable ViT layers, but backbone only has {total_layers} layers."
        )

    for param in modality_encoder.parameters():
        param.requires_grad = False

    for layer in layers[total_layers - trainable_layers :]:
        for param in layer.parameters():
            param.requires_grad = True

    if trainable_layers > 0 and hasattr(modality_encoder.vit, "layernorm"):
        for param in modality_encoder.vit.layernorm.parameters():
            param.requires_grad = True

    trainable_params = sum(param.numel() for param in modality_encoder.parameters() if param.requires_grad)
    return {
        "total_layers": total_layers,
        "trainable_layers": trainable_layers,
        "trainable_params": trainable_params,
        "total_params": total_params,
    }


def configure_stage1_vit_trainability(model, trainable_layers: int) -> dict:
    base_model = unwrap_model(model)
    return {
        "visual_encoder": set_modality_encoder_trainable_vit_layers(
            base_model.visual_encoder,
            trainable_layers,
        ),
        "tactile_encoder": set_modality_encoder_trainable_vit_layers(
            base_model.tactile_encoder,
            trainable_layers,
        ),
    }


def configure_stage2_frozen_encoders(base_model, args):
    base_model.reuse_frozen_encoder_tokens = bool(args.stage2_reuse_frozen_encoder_tokens)
    for param in base_model.visual_encoder.parameters():
        param.requires_grad = False
    for param in base_model.tactile_encoder.parameters():
        param.requires_grad = False
    if args.stage2_frozen_encoders_eval:
        base_model.visual_encoder.eval()
        base_model.tactile_encoder.eval()


def setup_distributed(args):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        args.distributed = False
        args.rank = 0
        args.local_rank = 0
        return

    args.distributed = True
    args.rank = int(os.environ["RANK"])
    args.local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend="nccl")
    args.device = f"cuda:{args.local_rank}"


def cleanup_distributed():
    if is_dist_enabled():
        dist.destroy_process_group()


def maybe_init_wandb(args, output_dir: Path):
    if not is_main_process():
        return None
    if args.wandb_mode == "disabled":
        return None
    if wandb is None:
        raise ImportError("wandb is not installed, but wandb logging was requested.")

    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        job_type=f"stage{args.stage}",
        dir=str(output_dir),
        config=vars(args),
        mode=args.wandb_mode,
        sync_tensorboard=False,
        save_code=False,
    )
    run.define_metric("global_step")
    run.define_metric("epoch")
    run.define_metric("stage1/train/*", step_metric="global_step")
    run.define_metric("stage2/train/*", step_metric="global_step")
    run.define_metric("stage1/epoch/*", step_metric="epoch")
    run.define_metric("stage2/epoch/*", step_metric="epoch")
    run.define_metric("stage2/val/*", step_metric="epoch")
    run.define_metric("stage2/val_task/*", step_metric="epoch")
    run.define_metric("stage1_train_*", step_metric="global_step")
    run.define_metric("stage2_train_*", step_metric="global_step")
    run.define_metric("training/loss_curves", step_metric="global_step")
    run.define_metric("stage2/reconstructions", step_metric="epoch")
    print(f"W&B initialized in {args.wandb_mode} mode at {output_dir / 'wandb'}")
    return run


def configure_precision(args):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
        torch.backends.cudnn.allow_tf32 = bool(args.tf32)
        torch.set_float32_matmul_precision("high" if args.tf32 else "highest")
    if is_main_process():
        print(
            "Precision config: "
            f"tf32={args.tf32}, amp_dtype={args.amp_dtype}, "
            f"matmul_allow_tf32={torch.backends.cuda.matmul.allow_tf32}, "
            f"float32_matmul_precision={torch.get_float32_matmul_precision()}"
        )


def autocast_context(args):
    if not str(args.device).startswith("cuda") or args.amp_dtype == "none":
        return nullcontext()
    dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def create_grad_scaler(args):
    enabled = str(args.device).startswith("cuda") and args.amp_dtype == "fp16"
    return torch.cuda.amp.GradScaler(enabled=enabled)


def scalar_value(value) -> float:
    if torch.is_tensor(value):
        return float(value.detach().item())
    return float(value)


def log_step_metrics(
    prefix: str,
    losses: dict,
    step: int,
    writer: SummaryWriter | None,
    wandb_run,
    curve_plotter: TrainingCurvePlotter | None = None,
):
    metrics = {}
    plot_metrics = {}
    alias_prefix = prefix.replace("/", "_")
    for key, value in losses.items():
        tag = f"{prefix}/{key}"
        metrics[tag] = scalar_value(value)
        plot_metrics[tag] = metrics[tag]
        metrics[f"{alias_prefix}_{key}"] = metrics[tag]
        if writer is not None:
            writer.add_scalar(tag, metrics[tag], step)

    plot_path = None
    if curve_plotter is not None:
        plot_path = curve_plotter.update(plot_metrics, step)
        if plot_path is not None and writer is not None:
            writer.add_image(
                "training/loss_curves",
                curve_plotter.read_image(),
                step,
                dataformats="HWC",
            )

    if wandb_run is not None and metrics:
        payload = {**metrics, "global_step": step}
        if plot_path is not None and wandb is not None:
            payload["training/loss_curves"] = wandb.Image(str(plot_path))
        wandb_run.log(payload)


def log_epoch_metrics(
    prefix: str,
    metrics: dict,
    epoch: int,
    global_step: int,
    writer: SummaryWriter | None,
    wandb_run,
):
    clean_metrics = {key: scalar_value(value) for key, value in metrics.items()}
    if writer is not None:
        for key, value in clean_metrics.items():
            writer.add_scalar(f"{prefix}/{key}", value, epoch)
    if wandb_run is not None:
        wandb_run.log(
            {
                **{f"{prefix}/{key}": value for key, value in clean_metrics.items()},
                "epoch": epoch + 1,
                "global_step": global_step,
            }
        )


STAGE2_VAL_LOSS_KEYS = (
    "total_loss",
    "recon_loss",
    "tactile_flow_loss",
    "kl_loss",
    "perceptual_loss",
)

STAGE2_FRAME_METRIC_KEYS = ("mse", "l1", "psnr", "ssim")


def all_reduce_sum(tensor: torch.Tensor) -> torch.Tensor:
    if is_dist_enabled():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


def compute_ssim_per_frame(recon: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return a lightweight SSIM estimate for NCHW images in [0, 1]."""
    if recon.numel() == 0:
        return torch.empty(0, device=recon.device, dtype=recon.dtype)

    recon = recon.float().clamp(0.0, 1.0)
    target = target.float().clamp(0.0, 1.0)
    height, width = recon.shape[-2:]
    kernel_size = min(11, height, width)
    if kernel_size % 2 == 0:
        kernel_size -= 1
    if kernel_size <= 1:
        recon_flat = recon.flatten(1)
        target_flat = target.flatten(1)
        mu_x = recon_flat.mean(dim=1)
        mu_y = target_flat.mean(dim=1)
        var_x = recon_flat.var(dim=1, unbiased=False)
        var_y = target_flat.var(dim=1, unbiased=False)
        cov_xy = ((recon_flat - mu_x[:, None]) * (target_flat - mu_y[:, None])).mean(dim=1)
        c1 = 0.01**2
        c2 = 0.03**2
        ssim_values = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
        ).clamp_min(1e-8)
        return ssim_values.clamp(-1.0, 1.0)
    else:
        padding = kernel_size // 2
        mu_x_map = F.avg_pool2d(recon, kernel_size=kernel_size, stride=1, padding=padding)
        mu_y_map = F.avg_pool2d(target, kernel_size=kernel_size, stride=1, padding=padding)
        mu_x = mu_x_map
        mu_y = mu_y_map
        var_x = F.avg_pool2d(recon * recon, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_x
        var_y = F.avg_pool2d(target * target, kernel_size=kernel_size, stride=1, padding=padding) - mu_y * mu_y
        cov_xy = F.avg_pool2d(recon * target, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y

    c1 = 0.01**2
    c2 = 0.03**2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov_xy + c2)) / (
        (mu_x.pow(2) + mu_y.pow(2) + c1) * (var_x + var_y + c2)
    ).clamp_min(1e-8)
    return ssim_map.flatten(1).mean(dim=1).clamp(-1.0, 1.0)


def init_stage2_val_stats(num_tasks: int, device: str):
    dtype = torch.float64
    return {
        "loss_sums": {key: torch.zeros((), device=device, dtype=dtype) for key in STAGE2_VAL_LOSS_KEYS},
        "loss_batches": torch.zeros((), device=device, dtype=dtype),
        "frame_sums": {key: torch.zeros((), device=device, dtype=dtype) for key in STAGE2_FRAME_METRIC_KEYS},
        "frame_count": torch.zeros((), device=device, dtype=dtype),
        "task_sums": {
            key: torch.zeros(num_tasks, device=device, dtype=dtype) for key in STAGE2_FRAME_METRIC_KEYS
        },
        "task_frame_count": torch.zeros(num_tasks, device=device, dtype=dtype),
    }


def update_stage2_val_stats(stats: dict, losses: dict, outputs: dict, task_id: torch.Tensor) -> None:
    for key in STAGE2_VAL_LOSS_KEYS:
        stats["loss_sums"][key].add_(losses[key].detach().double())
    stats["loss_batches"].add_(1.0)

    tail_mask = outputs["tail_mask"].detach().bool()
    if not tail_mask.any():
        return

    recon = outputs["recon_tail"].detach().float()
    target = outputs["target_tail"].detach().float()
    batch_size, tail_len = tail_mask.shape
    frame_mask = tail_mask.reshape(-1)
    recon_frames = recon.reshape(batch_size * tail_len, *recon.shape[2:])[frame_mask]
    target_frames = target.reshape(batch_size * tail_len, *target.shape[2:])[frame_mask]

    mse_values = (recon_frames - target_frames).pow(2).flatten(1).mean(dim=1)
    l1_values = (recon_frames - target_frames).abs().flatten(1).mean(dim=1)
    psnr_values = -10.0 * torch.log10(torch.clamp(mse_values, min=1e-10))
    ssim_values = compute_ssim_per_frame(recon_frames, target_frames)
    metric_values = {
        "mse": mse_values,
        "l1": l1_values,
        "psnr": psnr_values,
        "ssim": ssim_values,
    }

    expanded_task_id = task_id.detach().long().view(batch_size, 1).expand(batch_size, tail_len).reshape(-1)
    frame_task_id = expanded_task_id[frame_mask]
    ones = torch.ones_like(mse_values, dtype=torch.float64)

    stats["frame_count"].add_(ones.sum())
    stats["task_frame_count"].scatter_add_(0, frame_task_id, ones)
    for key, values in metric_values.items():
        values = values.double()
        stats["frame_sums"][key].add_(values.sum())
        stats["task_sums"][key].scatter_add_(0, frame_task_id, values)


def finalize_stage2_val_stats(stats: dict, task_names: list[str]) -> tuple[dict, dict]:
    tensors = [stats["loss_batches"], stats["frame_count"], stats["task_frame_count"]]
    tensors.extend(stats["loss_sums"].values())
    tensors.extend(stats["frame_sums"].values())
    tensors.extend(stats["task_sums"].values())
    for tensor in tensors:
        all_reduce_sum(tensor)

    loss_batches = max(float(stats["loss_batches"].item()), 1.0)
    frame_count = max(float(stats["frame_count"].item()), 1.0)
    val_metrics = {
        key: float(stats["loss_sums"][key].item() / loss_batches) for key in STAGE2_VAL_LOSS_KEYS
    }
    for key in STAGE2_FRAME_METRIC_KEYS:
        val_metrics[key] = float(stats["frame_sums"][key].item() / frame_count)
    val_metrics["masked_tail_frames"] = float(stats["frame_count"].item())

    task_metrics = {}
    for task_idx, task_name in enumerate(task_names):
        count = float(stats["task_frame_count"][task_idx].item())
        if count <= 0:
            continue
        for key in STAGE2_FRAME_METRIC_KEYS:
            task_metrics[f"{task_name}/{key}"] = float(stats["task_sums"][key][task_idx].item() / count)
        task_metrics[f"{task_name}/masked_tail_frames"] = count
    return val_metrics, task_metrics


def tensor_to_pil(image: torch.Tensor, size: int) -> Image.Image:
    image = image.detach().cpu().float().clamp(0.0, 1.0)
    if image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    if image.shape[0] > 3:
        image = image[:3]
    array = (image * 255.0).byte().permute(1, 2, 0).numpy()
    pil_image = Image.fromarray(array)
    if size > 0 and pil_image.width != size:
        resample = getattr(Image, "Resampling", Image).BILINEAR
        pil_image = pil_image.resize((size, size), resample=resample)
    return pil_image


def add_image_label(image: Image.Image, label: str) -> Image.Image:
    label_height = 24
    canvas = Image.new("RGB", (image.width, image.height + label_height), color=(255, 255, 255))
    canvas.paste(image, (0, label_height))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 5), label[:36], fill=(0, 0, 0))
    return canvas


def make_error_image(recon: torch.Tensor, target: torch.Tensor, size: int) -> Image.Image:
    error = (recon.detach().float() - target.detach().float()).abs().mean(dim=0, keepdim=True)
    error = error.mul(4.0).clamp(0.0, 1.0).repeat(3, 1, 1)
    return tensor_to_pil(error, size)


def stitch_stage2_panel(rows: list[list[Image.Image]]) -> Image.Image:
    row_images = []
    for cells in rows:
        width = sum(cell.width for cell in cells)
        height = max(cell.height for cell in cells)
        row_image = Image.new("RGB", (width, height), color=(255, 255, 255))
        x_offset = 0
        for cell in cells:
            row_image.paste(cell, (x_offset, 0))
            x_offset += cell.width
        row_images.append(row_image)

    panel_width = max(row.width for row in row_images)
    panel_height = sum(row.height for row in row_images)
    panel = Image.new("RGB", (panel_width, panel_height), color=(255, 255, 255))
    y_offset = 0
    for row in row_images:
        panel.paste(row, (0, y_offset))
        y_offset += row.height
    return panel


def save_stage2_reconstruction_panel(
    output_dir: Path,
    epoch: int,
    global_step: int,
    visual_seq: torch.Tensor,
    tactile_seq: torch.Tensor,
    outputs: dict,
    task_id: torch.Tensor,
    task_names: list[str],
    image_size: int,
    tactile_flow_label: str = "flow",
) -> Path | None:
    tail_mask = outputs["tail_mask"].detach().cpu().bool()
    valid_indices = torch.nonzero(tail_mask.any(dim=1), as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return None

    sample_idx = int(valid_indices[0].item())
    task_idx = int(task_id[sample_idx].detach().cpu().item())
    task_name = task_names[task_idx] if 0 <= task_idx < len(task_names) else f"task_{task_idx}"
    tail_len = outputs["recon_tail"].shape[1]
    tail_start = visual_seq.shape[1] - tail_len
    context_idx = max(0, tail_start - 1)

    rows = []
    for tail_idx in range(tail_len):
        source_idx = tail_start + tail_idx
        mask_value = int(bool(tail_mask[sample_idx, tail_idx].item()))
        row = [
            add_image_label(
                tensor_to_pil(visual_seq[sample_idx, context_idx], image_size),
                f"{task_name} ctx",
            ),
            add_image_label(
                tensor_to_pil(tactile_seq[sample_idx, source_idx], image_size),
                f"tactile t{source_idx}",
            ),
            add_image_label(
                tensor_to_pil(outputs["target_tail"][sample_idx, tail_idx], image_size),
                f"target t{source_idx}",
            ),
            add_image_label(
                tensor_to_pil(outputs["recon_tail"][sample_idx, tail_idx], image_size),
                f"recon t{source_idx}",
            ),
            add_image_label(
                make_error_image(
                    outputs["recon_tail"][sample_idx, tail_idx],
                    outputs["target_tail"][sample_idx, tail_idx],
                    image_size,
                ),
                f"abs err x4 m={mask_value}",
            ),
        ]
        if "target_tactile_flow_tail" in outputs and "recon_tactile_flow_tail" in outputs:
            row.extend(
                [
                    add_image_label(
                        tensor_to_pil(outputs["target_tactile_flow_tail"][sample_idx, tail_idx], image_size),
                        f"target {tactile_flow_label} t{source_idx}",
                    ),
                    add_image_label(
                        tensor_to_pil(outputs["recon_tactile_flow_tail"][sample_idx, tail_idx], image_size),
                        f"recon {tactile_flow_label} t{source_idx}",
                    ),
                ]
            )
        rows.append(row)

    panel = stitch_stage2_panel(rows)
    panel_dir = output_dir / "visualizations" / "stage2_reconstructions"
    panel_dir.mkdir(parents=True, exist_ok=True)
    panel_path = panel_dir / f"epoch_{epoch + 1:04d}_step_{global_step:08d}.png"
    panel.save(panel_path)
    return panel_path


def log_stage2_reconstruction_panel(
    panel_path: Path | None,
    epoch: int,
    global_step: int,
    writer: SummaryWriter | None,
    wandb_run,
) -> None:
    if panel_path is None:
        return
    if writer is not None:
        with Image.open(panel_path) as image:
            writer.add_image(
                "stage2/reconstructions",
                np.asarray(image.convert("RGB")),
                epoch,
                dataformats="HWC",
            )
    if wandb_run is not None and wandb is not None:
        wandb_run.log(
            {
                "stage2/reconstructions": wandb.Image(str(panel_path)),
                "epoch": epoch + 1,
                "global_step": global_step,
            }
        )


def train_stage1(
    model: VTMUSE,
    train_loader,
    val_loader,
    args,
    writer: SummaryWriter | None,
    wandb_run=None,
    curve_plotter: TrainingCurvePlotter | None = None,
):
    if is_main_process():
        print("\n" + "=" * 80)
        print("STAGE 1: Multimodal Alignment (Contrastive Learning)")
        print("=" * 80 + "\n")

    base_model = unwrap_model(model)
    params = unique_trainable_params(
        base_model.visual_encoder,
        base_model.tactile_encoder,
        base_model.encoder,
    )

    optimizer = optim.AdamW(params, lr=args.lr_stage1, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage1)
    scaler = create_grad_scaler(args)
    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs_stage1):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        train_loss_sum = 0.0
        train_loss_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Stage 1 Epoch {epoch + 1}/{args.epochs_stage1}",
            disable=not is_main_process(),
        )
        for batch in pbar:
            visual_seq = batch["visual_seq"].to(args.device, non_blocking=True)
            tactile_seq = batch["tactile_seq"].to(args.device, non_blocking=True)
            delta_steps = batch["delta_steps"].to(args.device, non_blocking=True)
            task_id = batch["task_id"].to(args.device, non_blocking=True)
            action_seq = None

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args):
                losses = compute_stage1_loss(
                    model,
                    visual_seq,
                    tactile_seq,
                    delta_steps,
                    task_id,
                    action_seq,
                    False,
                    args.stage1_cross_weight,
                    args.stage1_temporal_weight,
                    args.stage1_consistency_weight,
                    args.contrastive_temp,
                    args.ddp_contrastive_all_gather,
                )
            loss = losses["contrastive_loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_item = loss.item()
            train_loss_sum += loss_item
            train_loss_count += 1
            pbar.set_postfix({"loss": loss_item})

            if global_step % args.log_interval == 0:
                log_step_metrics(
                    "stage1/train",
                    losses,
                    global_step,
                    writer,
                    wandb_run,
                    curve_plotter,
                )
            global_step += 1

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_loss_count = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not is_main_process()):
                visual_seq = batch["visual_seq"].to(args.device, non_blocking=True)
                tactile_seq = batch["tactile_seq"].to(args.device, non_blocking=True)
                delta_steps = batch["delta_steps"].to(args.device, non_blocking=True)
                task_id = batch["task_id"].to(args.device, non_blocking=True)
                action_seq = None
                with autocast_context(args):
                    losses = compute_stage1_loss(
                        model,
                        visual_seq,
                        tactile_seq,
                        delta_steps,
                        task_id,
                        action_seq,
                        False,
                        args.stage1_cross_weight,
                        args.stage1_temporal_weight,
                        args.stage1_consistency_weight,
                        args.contrastive_temp,
                        args.ddp_contrastive_all_gather,
                    )
                val_loss_sum += losses["contrastive_loss"].item()
                val_loss_count += 1

        avg_train_loss = train_loss_sum / max(train_loss_count, 1)
        avg_val_loss = val_loss_sum / max(val_loss_count, 1)
        avg_train_loss = reduce_scalar(avg_train_loss, args.device)
        avg_val_loss = reduce_scalar(avg_val_loss, args.device)
        if is_main_process():
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        log_epoch_metrics(
            "stage1/epoch",
            {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": scheduler.get_last_lr()[0]},
            epoch,
            global_step,
            writer,
            wandb_run,
        )

        if is_main_process() and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(args.output_dir) / "stage1_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")
        if is_main_process():
            checkpoint_path = Path(args.output_dir) / "stage1_last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved last model to {checkpoint_path}")


def train_stage2(
    model: VTMUSE,
    train_loader,
    val_loader,
    args,
    writer: SummaryWriter | None,
    wandb_run=None,
    curve_plotter: TrainingCurvePlotter | None = None,
):
    if is_main_process():
        print("\n" + "=" * 80)
        print("STAGE 2: Temporal-Memory Latent Modeling (VAE)")
        print("=" * 80 + "\n")

    base_model = unwrap_model(model)
    configure_stage2_frozen_encoders(base_model, args)
    if is_main_process():
        print(f"Stage 2 reuse frozen encoder tokens: {base_model.reuse_frozen_encoder_tokens}")
        print(f"Stage 2 frozen encoders eval mode: {args.stage2_frozen_encoders_eval}")
    modules = [base_model.encoder, base_model.decoder]
    if base_model.tactile_flow_decoder is not None:
        modules.append(base_model.tactile_flow_decoder)
    params = unique_trainable_params(*modules)
    optimizer = optim.AdamW(params, lr=args.lr_stage2, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs_stage2)
    scaler = create_grad_scaler(args)

    vae_loss_fn = MaskedTemporalReconstructionLoss(
        recon_weight=args.recon_weight,
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        tactile_flow_weight=args.tactile_flow_weight,
        tactile_flow_foreground_weight=args.tactile_flow_foreground_weight,
        use_perceptual=args.use_perceptual and args.perceptual_weight > 0.0,
    )

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(args.epochs_stage2):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
        model.train()
        configure_stage2_frozen_encoders(base_model, args)
        train_loss_sum = 0.0
        train_loss_count = 0

        pbar = tqdm(
            train_loader,
            desc=f"Stage 2 Epoch {epoch + 1}/{args.epochs_stage2}",
            disable=not is_main_process(),
        )
        for batch in pbar:
            visual_seq = batch["visual_seq"].to(args.device, non_blocking=True)
            tactile_seq = batch["tactile_seq"].to(args.device, non_blocking=True)
            action_seq = None
            visual_mask = batch["visual_mask"].to(args.device, non_blocking=True)
            delta_steps = batch["delta_steps"].to(args.device, non_blocking=True)
            task_id = batch["task_id"].to(args.device, non_blocking=True)
            tactile_flow_target = (
                batch["tactile_flow_target"].to(args.device, non_blocking=True)
                if "tactile_flow_target" in batch
                else None
            )

            optimizer.zero_grad(set_to_none=True)
            with autocast_context(args):
                losses = compute_stage2_loss(
                    model,
                    visual_seq,
                    tactile_seq,
                    action_seq,
                    visual_mask,
                    delta_steps,
                    task_id,
                    vae_loss_fn,
                    tactile_flow_target,
                )
            loss = losses["total_loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_item = loss.item()
            train_loss_sum += loss_item
            train_loss_count += 1
            pbar.set_postfix({"loss": loss_item})

            if global_step % args.log_interval == 0:
                log_step_metrics(
                    "stage2/train",
                    losses,
                    global_step,
                    writer,
                    wandb_run,
                    curve_plotter,
                )
            global_step += 1

        scheduler.step()

        model.eval()
        val_stats = init_stage2_val_stats(args.num_tasks, args.device)
        stage2_panel_path = None
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation", disable=not is_main_process()):
                visual_seq = batch["visual_seq"].to(args.device, non_blocking=True)
                tactile_seq = batch["tactile_seq"].to(args.device, non_blocking=True)
                action_seq = None
                visual_mask = batch["visual_mask"].to(args.device, non_blocking=True)
                delta_steps = batch["delta_steps"].to(args.device, non_blocking=True)
                task_id = batch["task_id"].to(args.device, non_blocking=True)
                tactile_flow_target = (
                    batch["tactile_flow_target"].to(args.device, non_blocking=True)
                    if "tactile_flow_target" in batch
                    else None
                )

                with autocast_context(args):
                    losses, outputs = compute_stage2_loss(
                        model,
                        visual_seq,
                        tactile_seq,
                        action_seq,
                        visual_mask,
                        delta_steps,
                        task_id,
                        vae_loss_fn,
                        tactile_flow_target,
                        return_outputs=True,
                    )
                update_stage2_val_stats(val_stats, losses, outputs, task_id)
                if (
                    stage2_panel_path is None
                    and is_main_process()
                    and args.stage2_log_val_reconstructions
                ):
                    stage2_panel_path = save_stage2_reconstruction_panel(
                        output_dir=Path(args.output_dir),
                        epoch=epoch,
                        global_step=global_step,
                        visual_seq=visual_seq,
                        tactile_seq=tactile_seq,
                        outputs=outputs,
                        task_id=task_id,
                        task_names=args.task_names,
                        image_size=args.stage2_reconstruction_image_size,
                        tactile_flow_label=args.tactile_flow_target_mode,
                    )

        avg_train_loss = train_loss_sum / max(train_loss_count, 1)
        avg_train_loss = reduce_scalar(avg_train_loss, args.device)
        val_metrics, task_val_metrics = finalize_stage2_val_stats(val_stats, args.task_names)
        avg_val_loss = val_metrics["total_loss"]
        if is_main_process():
            print(
                f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, "
                f"Val Loss = {avg_val_loss:.4f}, "
                f"Val Recon = {val_metrics['recon_loss']:.4f}, "
                f"Val PSNR = {val_metrics['psnr']:.2f}, "
                f"Val SSIM = {val_metrics['ssim']:.4f}"
            )

        log_epoch_metrics(
            "stage2/epoch",
            {"train_loss": avg_train_loss, "val_loss": avg_val_loss, "lr": scheduler.get_last_lr()[0]},
            epoch,
            global_step,
            writer,
            wandb_run,
        )
        log_epoch_metrics(
            "stage2/val",
            val_metrics,
            epoch,
            global_step,
            writer,
            wandb_run,
        )
        if task_val_metrics:
            log_epoch_metrics(
                "stage2/val_task",
                task_val_metrics,
                epoch,
                global_step,
                writer,
                wandb_run,
            )
        if is_main_process():
            log_stage2_reconstruction_panel(stage2_panel_path, epoch, global_step, writer, wandb_run)

        if is_main_process() and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            checkpoint_path = Path(args.output_dir) / "stage2_best.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")
        if is_main_process():
            checkpoint_path = Path(args.output_dir) / "stage2_last.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": unwrap_model(model).state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": avg_val_loss,
                    "config": vars(args),
                },
                checkpoint_path,
            )
            print(f"Saved last model to {checkpoint_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train VT-MUSE")

    parser.add_argument("--train_data", type=str, required=True, help="Path to training data directory")
    parser.add_argument("--val_data", type=str, required=True, help="Path to validation data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints", help="Output directory")

    parser.add_argument("--visual_image_size", type=int, default=224)
    parser.add_argument("--tactile_image_size", type=int, default=224)
    parser.add_argument("--action_dim", type=int, default=7)
    parser.add_argument("--latent_dim", type=int, default=512)
    parser.add_argument("--history_len", type=int, default=5)
    parser.add_argument("--sample_stride", type=int, default=5)
    parser.add_argument("--num_tail_frames", type=int, default=2)
    parser.add_argument("--pretrained_encoders", action="store_true", default=True)
    parser.add_argument("--max_delta_t", type=int, default=32)
    parser.add_argument("--num_memory_layers", type=int, default=4)
    parser.add_argument("--num_latent_layers", type=int, default=4)
    parser.add_argument("--tactile_temporal_mode", type=str, default="raw", choices=["raw", "delta", "flow", "denoised_flow"])
    parser.add_argument(
        "--tactile_flow_target_mode",
        type=str,
        default="none",
        choices=["none", "delta", "flow", "denoised_flow", "marker_flow", "depth_delta"],
        help="Optional Stage 2 tactile-flow reconstruction target. Input tactile mode remains controlled separately.",
    )
    parser.add_argument("--tactile_flow_clip", type=float, default=0.25)
    parser.add_argument(
        "--marker_flow_clip",
        type=float,
        default=1.0,
        help="Clip value in output-image pixels for marker_flow targets.",
    )
    parser.add_argument(
        "--depth_delta_clip",
        type=float,
        default=0.5,
        help="Clip value for dense depth_delta reconstruction targets.",
    )
    parser.add_argument("--tactile_delta_clip", type=float, default=0.25)

    parser.add_argument("--stage", type=int, default=2, choices=[1, 2], help="Training stage")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")

    parser.add_argument("--epochs_stage1", type=int, default=50)
    parser.add_argument("--lr_stage1", type=float, default=1e-4)
    parser.add_argument("--contrastive_temp", type=float, default=0.07)
    parser.add_argument("--stage1_cross_weight", type=float, default=1.0)
    parser.add_argument("--stage1_temporal_weight", type=float, default=1.0)
    parser.add_argument("--stage1_consistency_weight", type=float, default=1.0)
    parser.add_argument(
        "--ddp_contrastive_all_gather",
        dest="ddp_contrastive_all_gather",
        action="store_true",
        default=True,
        help="Use all DDP ranks as contrastive candidates in Stage 1.",
    )
    parser.add_argument(
        "--no_ddp_contrastive_all_gather",
        dest="ddp_contrastive_all_gather",
        action="store_false",
        help="Keep Stage 1 contrastive candidates local to each DDP rank.",
    )
    parser.add_argument(
        "--trainable_vit_layers",
        type=int,
        default=3,
        help="Number of final ViT transformer blocks to train in Stage 1; set -1 to train all.",
    )
    parser.add_argument(
        "--no_action_conditioning",
        action="store_true",
        help="Disable action tokens in Stage 2 encoder and downstream encoder-only features",
    )

    parser.add_argument("--epochs_stage2", type=int, default=100)
    parser.add_argument("--lr_stage2", type=float, default=1e-4)
    parser.add_argument("--recon_weight", type=float, default=1.0)
    parser.add_argument("--tactile_flow_weight", type=float, default=0.0)
    parser.add_argument(
        "--val_random_tail_mask",
        action="store_true",
        default=False,
        help="Use the same random tail-mask pattern distribution for Stage 2 validation as training.",
    )
    parser.add_argument(
        "--tactile_flow_foreground_weight",
        type=float,
        default=0.0,
        help="Extra spatial weight for nonzero marker-flow pixels in tactile-flow reconstruction.",
    )
    parser.add_argument("--kl_weight", type=float, default=0.001)
    parser.add_argument("--perceptual_weight", type=float, default=0.1)
    parser.add_argument("--use_perceptual", action="store_true", default=True)
    parser.add_argument(
        "--stage2_reuse_frozen_encoder_tokens",
        dest="stage2_reuse_frozen_encoder_tokens",
        action="store_true",
        default=False,
        help="Compute frozen visual/tactile ViT tokens once per Stage 2 batch and reuse them for prior/posterior/target paths.",
    )
    parser.add_argument(
        "--no_stage2_reuse_frozen_encoder_tokens",
        dest="stage2_reuse_frozen_encoder_tokens",
        action="store_false",
        help="Keep separate frozen ViT forwards in Stage 2.",
    )
    parser.add_argument(
        "--stage2_frozen_encoders_eval",
        dest="stage2_frozen_encoders_eval",
        action="store_true",
        default=True,
        help="Keep frozen Stage 2 visual/tactile encoders in eval mode.",
    )
    parser.add_argument(
        "--no_stage2_frozen_encoders_eval",
        dest="stage2_frozen_encoders_eval",
        action="store_false",
        help="Leave frozen Stage 2 encoders in train mode.",
    )

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--prefetch_factor", type=int, default=2)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--tf32", dest="tf32", action="store_true", default=False)
    parser.add_argument("--no_tf32", dest="tf32", action="store_false")
    parser.add_argument("--amp_dtype", type=str, default="none", choices=["none", "bf16", "fp16"])
    parser.add_argument("--wandb_mode", type=str, default="disabled", choices=["disabled", "offline", "online"])
    parser.add_argument("--wandb_project", type=str, default="vt-muse")
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument(
        "--plot_training_curves",
        dest="plot_training_curves",
        action="store_true",
        default=True,
        help="Write a continuously updated training-curve PNG and log it to W&B/TensorBoard.",
    )
    parser.add_argument(
        "--no_plot_training_curves",
        dest="plot_training_curves",
        action="store_false",
        help="Disable live training-curve PNG rendering.",
    )
    parser.add_argument("--plot_curve_max_points", type=int, default=2000)
    parser.add_argument("--plot_curve_render_every", type=int, default=30)
    parser.add_argument(
        "--stage2_log_val_reconstructions",
        dest="stage2_log_val_reconstructions",
        action="store_true",
        default=True,
        help="Save and log one Stage 2 validation reconstruction panel after each epoch.",
    )
    parser.add_argument(
        "--no_stage2_log_val_reconstructions",
        dest="stage2_log_val_reconstructions",
        action="store_false",
        help="Disable Stage 2 validation reconstruction panels.",
    )
    parser.add_argument(
        "--stage2_reconstruction_image_size",
        type=int,
        default=128,
        help="Cell size in pixels for Stage 2 validation reconstruction panels.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    setup_distributed(args)
    configure_precision(args)
    args.bilateral_tactile = True
    args.tactile_mode = "bilateral_concat"
    args.use_action_conditioning = False

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if is_main_process():
        with open(output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    writer = SummaryWriter(log_dir=output_dir / "logs") if is_main_process() else None
    wandb_run = maybe_init_wandb(args, output_dir)
    curve_plotter = (
        TrainingCurvePlotter(
            output_dir=output_dir / "visualizations",
            max_points=args.plot_curve_max_points,
            render_every=args.plot_curve_render_every,
        )
        if is_main_process() and args.plot_training_curves
        else None
    )

    train_data_paths = list(Path(args.train_data).glob("*.hdf5"))
    val_data_paths = list(Path(args.val_data).glob("*.hdf5"))

    print(f"Found {len(train_data_paths)} training files")
    print(f"Found {len(val_data_paths)} validation files")

    task_names = order_task_names(
        VTMUSEDataset._infer_task_name(data_path)
        for data_path in [*train_data_paths, *val_data_paths]
    )
    train_dataset = VTMUSEDataset(
        data_paths=train_data_paths,
        history_len=args.history_len,
        image_size=args.visual_image_size,
        sample_stride=args.sample_stride,
        num_tail_frames=args.num_tail_frames,
        random_tail_mask=True,
        augment=False,
        task_names=task_names,
        tactile_temporal_mode=args.tactile_temporal_mode,
        tactile_flow_target_mode=args.tactile_flow_target_mode,
        tactile_flow_clip=args.tactile_flow_clip,
        marker_flow_clip=args.marker_flow_clip,
        depth_delta_clip=args.depth_delta_clip,
        tactile_delta_clip=args.tactile_delta_clip,
    )
    val_dataset = VTMUSEDataset(
        data_paths=val_data_paths,
        history_len=args.history_len,
        image_size=args.visual_image_size,
        sample_stride=args.sample_stride,
        num_tail_frames=args.num_tail_frames,
        random_tail_mask=args.val_random_tail_mask,
        augment=False,
        task_names=task_names,
        tactile_temporal_mode=args.tactile_temporal_mode,
        tactile_flow_target_mode=args.tactile_flow_target_mode,
        tactile_flow_clip=args.tactile_flow_clip,
        marker_flow_clip=args.marker_flow_clip,
        depth_delta_clip=args.depth_delta_clip,
        tactile_delta_clip=args.tactile_delta_clip,
    )
    train_sampler = DistributedSampler(train_dataset, shuffle=True) if args.distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if args.distributed else None
    loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": str(args.device).startswith("cuda"),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = args.prefetch_factor

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        sampler=train_sampler,
        drop_last=True,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        drop_last=False,
        **loader_kwargs,
    )
    num_tasks = max(getattr(train_dataset, "num_tasks", 0), getattr(val_dataset, "num_tasks", 0))
    args.num_tasks = num_tasks
    args.task_names = task_names
    if is_main_process():
        with open(output_dir / "config.json", "w") as f:
            json.dump(vars(args), f, indent=2)

    model = VTMUSE(
        visual_image_size=args.visual_image_size,
        tactile_image_size=args.tactile_image_size,
        action_dim=args.action_dim,
        latent_dim=args.latent_dim,
        history_len=args.history_len,
        pretrained_encoders=args.pretrained_encoders,
        max_delta_t=args.max_delta_t,
        num_memory_layers=args.num_memory_layers,
        num_latent_layers=args.num_latent_layers,
        num_tail_frames=args.num_tail_frames,
        num_tasks=num_tasks,
        reconstruct_tactile_flow=args.tactile_flow_target_mode != "none",
    ).to(args.device)

    if args.stage == 1:
        args.vit_trainability = configure_stage1_vit_trainability(model, args.trainable_vit_layers)
        if is_main_process():
            with open(output_dir / "config.json", "w") as f:
                json.dump(vars(args), f, indent=2)

    if is_main_process():
        print(f"Detected {num_tasks} task(s): {task_names}")
        print(f"Tactile temporal mode: {args.tactile_temporal_mode}")
        print(f"Auxiliary tactile target mode: {args.tactile_flow_target_mode}")
        print(f"Action conditioning: {args.use_action_conditioning}")
        print(f"DDP contrastive all-gather: {args.ddp_contrastive_all_gather}")
        if args.stage == 1:
            print(f"Stage 1 ViT trainability: {args.vit_trainability}")
        print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=args.device)
        load_result = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=args.tactile_flow_target_mode == "none",
        )
        if is_main_process():
            print(f"Resumed from {args.resume}")
            if args.tactile_flow_target_mode != "none":
                print(f"Non-strict load for tactile-flow target head: {load_result}")

    if args.distributed:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    if args.stage == 1:
        train_stage1(model, train_loader, val_loader, args, writer, wandb_run, curve_plotter)
    else:
        train_stage2(model, train_loader, val_loader, args, writer, wandb_run, curve_plotter)

    if writer is not None:
        writer.close()
    if wandb_run is not None:
        wandb_run.finish()
    if is_main_process():
        print("\nTraining complete!")
    cleanup_distributed()


if __name__ == "__main__":
    main()
