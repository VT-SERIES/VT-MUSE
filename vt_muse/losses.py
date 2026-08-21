"""Stage 1 alignment and Stage 2 masked reconstruction objectives for VT-MUSE."""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16
from typing import Dict

try:
    from torch.distributed.nn.functional import all_gather as differentiable_all_gather
except ImportError:
    differentiable_all_gather = None


def _dist_is_active() -> bool:
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1


def _concat_all_gather_with_grad(tensor: torch.Tensor) -> torch.Tensor:
    if not _dist_is_active():
        return tensor

    if differentiable_all_gather is not None:
        return torch.cat(differentiable_all_gather(tensor), dim=0)

    raise RuntimeError(
        "DDP contrastive all-gather requires torch.distributed.nn.functional.all_gather "
        "so gradients flow through gathered negatives."
    )


class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG16 features."""

    def __init__(self, layers: list = None):
        super().__init__()
        if layers is None:
            layers = ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3']

        self.layers = layers

        # Load pretrained VGG16
        vgg = vgg16(pretrained=True)
        self.features = vgg.features

        # Freeze VGG parameters
        for param in self.features.parameters():
            param.requires_grad = False

        # Layer name to index mapping
        self.layer_name_mapping = {
            'relu1_2': 3,
            'relu2_2': 8,
            'relu3_3': 15,
            'relu4_3': 22
        }

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) predicted image
            target: (B, C, H, W) target image

        Returns:
            Perceptual loss
        """
        if next(self.features.parameters()).device != pred.device:
            self.features = self.features.to(pred.device)

        # Normalize to ImageNet stats
        mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=pred.device).view(1, 3, 1, 1)

        pred_norm = (pred - mean) / std
        target_norm = (target - mean) / std

        loss = 0.0
        x_pred = pred_norm
        x_target = target_norm

        for i, layer in enumerate(self.features):
            x_pred = layer(x_pred)
            x_target = layer(x_target)

            if i in self.layer_name_mapping.values():
                loss += F.mse_loss(x_pred, x_target)

        return loss


class ContrastiveLoss(nn.Module):
    """Contrastive loss for multimodal alignment (Stage 1)."""

    def __init__(self, temperature: float = 0.07, gather_distributed: bool = True):
        super().__init__()
        self.temperature = temperature
        self.gather_distributed = gather_distributed

    def forward(
        self,
        visual_features: torch.Tensor,
        tactile_features: torch.Tensor,
        action_features: torch.Tensor = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            visual_features: (B, D) visual features
            tactile_features: (B, D) tactile features
            action_features: (B, D) action features (optional)

        Returns:
            Dict with loss components
        """
        B = visual_features.shape[0]

        # Normalize features
        visual_features = F.normalize(visual_features, dim=-1)
        tactile_features = F.normalize(tactile_features, dim=-1)
        use_gather = self.gather_distributed and _dist_is_active()
        if use_gather:
            gathered_visual_features = _concat_all_gather_with_grad(visual_features)
            gathered_tactile_features = _concat_all_gather_with_grad(tactile_features)
            labels = torch.arange(B, device=visual_features.device) + dist.get_rank() * B
        else:
            gathered_visual_features = visual_features
            gathered_tactile_features = tactile_features
            labels = torch.arange(B, device=visual_features.device)

        # Compute similarity matrix
        logits_vt = torch.matmul(visual_features, gathered_tactile_features.T) / self.temperature
        logits_tv = torch.matmul(tactile_features, gathered_visual_features.T) / self.temperature

        # Cross-entropy loss
        loss_vt = F.cross_entropy(logits_vt, labels)
        loss_tv = F.cross_entropy(logits_tv, labels)

        loss = (loss_vt + loss_tv) / 2

        losses = {
            'contrastive_loss': loss,
            'loss_vt': loss_vt,
            'loss_tv': loss_tv
        }

        # Three-way contrastive if action features provided
        if action_features is not None:
            action_features = F.normalize(action_features, dim=-1)
            gathered_action_features = (
                _concat_all_gather_with_grad(action_features) if use_gather else action_features
            )

            logits_va = torch.matmul(visual_features, gathered_action_features.T) / self.temperature
            logits_ta = torch.matmul(tactile_features, gathered_action_features.T) / self.temperature

            loss_va = F.cross_entropy(logits_va, labels)
            loss_ta = F.cross_entropy(logits_ta, labels)

            losses['loss_va'] = loss_va
            losses['loss_ta'] = loss_ta
            losses['contrastive_loss'] = (loss + loss_va + loss_ta) / 3

        return losses


class MaskedTemporalReconstructionLoss(nn.Module):
    """Conditional VAE loss applied only to masked tail frames."""

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        perceptual_weight: float = 0.1,
        tactile_flow_weight: float = 0.0,
        tactile_flow_foreground_weight: float = 0.0,
        use_perceptual: bool = True,
    ):
        super().__init__()
        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.tactile_flow_weight = tactile_flow_weight
        self.tactile_flow_foreground_weight = tactile_flow_foreground_weight
        self.use_perceptual = use_perceptual
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()

    def _masked_frame_average(
        self,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        if mask.sum() == 0:
            return torch.tensor(0.0, device=values.device)
        mask = mask.float().view(*mask.shape, *([1] * (values.dim() - mask.dim())))
        return (values * mask).sum() / mask.sum().clamp_min(1.0)

    def forward(
        self,
        recon_tail: torch.Tensor,
        target_tail: torch.Tensor,
        tail_mask: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        prior_mu: torch.Tensor,
        prior_logvar: torch.Tensor,
        recon_tactile_flow_tail: torch.Tensor = None,
        target_tactile_flow_tail: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        pixel_error = (recon_tail - target_tail).pow(2).mean(dim=(2, 3, 4))
        recon_loss = self._masked_frame_average(pixel_error, tail_mask)

        if (
            self.tactile_flow_weight > 0.0
            and recon_tactile_flow_tail is not None
            and target_tactile_flow_tail is not None
        ):
            flow_pixel_error = (recon_tactile_flow_tail - target_tactile_flow_tail).pow(2).mean(dim=2)
            if self.tactile_flow_foreground_weight > 0.0:
                foreground = (
                    (target_tactile_flow_tail[:, :, 0].sub(0.5).abs() > 1e-6)
                    | (target_tactile_flow_tail[:, :, 1].abs() > 1e-6)
                    | (target_tactile_flow_tail[:, :, 2].abs() > 1e-6)
                ).float()
                spatial_weight = 1.0 + self.tactile_flow_foreground_weight * foreground
                flow_error = (flow_pixel_error * spatial_weight).sum(dim=(2, 3)) / spatial_weight.sum(
                    dim=(2, 3)
                ).clamp_min(1.0)
            else:
                flow_error = flow_pixel_error.mean(dim=(2, 3))
            tactile_flow_loss = self._masked_frame_average(flow_error, tail_mask)
        else:
            tactile_flow_loss = torch.tensor(0.0, device=recon_tail.device)

        if self.use_perceptual and tail_mask.any():
            flat_recon = recon_tail[tail_mask]
            flat_target = target_tail[tail_mask]
            perceptual_loss = self.perceptual_loss(flat_recon, flat_target)
        else:
            perceptual_loss = torch.tensor(0.0, device=recon_tail.device)

        kl = (
            prior_logvar - logvar
            + (logvar.exp() + (mu - prior_mu).pow(2)) / prior_logvar.exp()
            - 1
        )
        kl_loss = 0.5 * torch.sum(kl) / mu.shape[0]

        total_loss = (
            self.recon_weight * recon_loss +
            self.tactile_flow_weight * tactile_flow_loss +
            self.kl_weight * kl_loss +
            self.perceptual_weight * perceptual_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "tactile_flow_loss": tactile_flow_loss,
            "kl_loss": kl_loss,
            "perceptual_loss": perceptual_loss,
        }


def compute_stage1_loss(
    model,
    visual_seq: torch.Tensor,
    tactile_seq: torch.Tensor,
    delta_steps: torch.Tensor,
    task_id: torch.Tensor = None,
    action_seq: torch.Tensor = None,
    action_cotrain: bool = False,
    cross_weight: float = 1.0,
    temporal_weight: float = 1.0,
    consistency_weight: float = 1.0,
    temperature: float = 0.07,
    ddp_contrastive_all_gather: bool = True,
) -> Dict[str, torch.Tensor]:
    criterion = ContrastiveLoss(
        temperature=temperature,
        gather_distributed=ddp_contrastive_all_gather,
    )
    model_for_alignment = model.module if hasattr(model, "module") else model
    can_reuse_encoder_tokens = (
        hasattr(model_for_alignment, "encode_alignment_base_tokens")
        and hasattr(model_for_alignment, "encode_alignment_tokens_from_base")
    )
    if can_reuse_encoder_tokens:
        visual_base_tokens, tactile_base_tokens, temporal_embed = model_for_alignment.encode_alignment_base_tokens(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            delta_steps=delta_steps,
        )
        visual_tokens, tactile_tokens, action_tokens = model_for_alignment.encode_alignment_tokens_from_base(
            visual_base_tokens=visual_base_tokens,
            tactile_base_tokens=tactile_base_tokens,
            temporal_embed=temporal_embed,
            task_id=task_id,
        )
    else:
        visual_tokens, tactile_tokens, action_tokens = model_for_alignment.encode_alignment_tokens(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            task_id=task_id,
            action_seq=action_seq if action_cotrain else None,
            delta_steps=delta_steps,
        )

    total = None
    logs: Dict[str, torch.Tensor] = {}
    num_steps = visual_tokens.shape[1]
    for step in range(num_steps):
        losses = criterion(
            visual_tokens[:, step, :],
            tactile_tokens[:, step, :],
            action_tokens[:, step, :] if action_cotrain and action_tokens is not None else None,
        )
        step_total = losses["contrastive_loss"]
        total = step_total if total is None else total + step_total
        logs[f"contrastive_t{step}"] = step_total

    assert total is not None
    cross_loss = total / num_steps
    logs["cross_loss"] = cross_loss

    state_tokens = F.normalize(0.5 * (visual_tokens + tactile_tokens), dim=-1)
    if num_steps > 1:
        temporal_losses = []
        for step in range(num_steps - 1):
            temporal_losses.append(
                criterion(
                    state_tokens[:, step, :],
                    state_tokens[:, step + 1, :],
                )["contrastive_loss"]
            )
        temporal_loss = torch.stack(temporal_losses).mean()
    else:
        temporal_loss = torch.tensor(0.0, device=visual_seq.device)
    logs["temporal_loss"] = temporal_loss

    if consistency_weight > 0.0:
        mask_a = torch.rand(visual_seq.shape[:2], device=visual_seq.device) < 0.3
        mask_b = torch.rand(visual_seq.shape[:2], device=visual_seq.device) < 0.3
        if can_reuse_encoder_tokens:
            visual_a, tactile_a, _ = model_for_alignment.encode_alignment_tokens_from_base(
                visual_base_tokens=visual_base_tokens,
                tactile_base_tokens=tactile_base_tokens,
                temporal_embed=temporal_embed,
                task_id=task_id,
                visual_mask=mask_a,
                use_mask=True,
            )
            visual_b, tactile_b, _ = model_for_alignment.encode_alignment_tokens_from_base(
                visual_base_tokens=visual_base_tokens,
                tactile_base_tokens=tactile_base_tokens,
                temporal_embed=temporal_embed,
                task_id=task_id,
                visual_mask=mask_b,
                use_mask=True,
            )
        else:
            visual_a, tactile_a, _ = model_for_alignment.encode_alignment_tokens(
                visual_seq=visual_seq,
                tactile_seq=tactile_seq,
                task_id=task_id,
                action_seq=None,
                delta_steps=delta_steps,
                visual_mask=mask_a,
                use_mask=True,
            )
            visual_b, tactile_b, _ = model_for_alignment.encode_alignment_tokens(
                visual_seq=visual_seq,
                tactile_seq=tactile_seq,
                task_id=task_id,
                action_seq=None,
                delta_steps=delta_steps,
                visual_mask=mask_b,
                use_mask=True,
            )
        state_a = F.normalize(0.5 * (visual_a + tactile_a), dim=-1)
        state_b = F.normalize(0.5 * (visual_b + tactile_b), dim=-1)
        consistency_loss = 1.0 - F.cosine_similarity(state_a, state_b, dim=-1).mean()
    else:
        consistency_loss = torch.tensor(0.0, device=visual_seq.device)
    logs["consistency_loss"] = consistency_loss

    logs["contrastive_loss"] = (
        cross_weight * cross_loss
        + temporal_weight * temporal_loss
        + consistency_weight * consistency_loss
    )
    return logs


def compute_stage2_loss(
    model,
    visual_seq: torch.Tensor,
    tactile_seq: torch.Tensor,
    action_seq: torch.Tensor,
    visual_mask: torch.Tensor,
    delta_steps: torch.Tensor,
    task_id: torch.Tensor,
    vae_loss_fn: MaskedTemporalReconstructionLoss,
    tactile_flow_target: torch.Tensor = None,
    return_outputs: bool = False,
) -> Dict[str, torch.Tensor]:
    outputs = model(
        visual_seq=visual_seq,
        tactile_seq=tactile_seq,
        action_seq=action_seq,
        visual_mask=visual_mask,
        delta_steps=delta_steps,
        task_id=task_id,
        tactile_flow_target=tactile_flow_target,
    )

    losses = vae_loss_fn(
        recon_tail=outputs["recon_tail"],
        target_tail=outputs["target_tail"],
        tail_mask=outputs["tail_mask"],
        mu=outputs["mu"],
        logvar=outputs["logvar"],
        prior_mu=outputs["prior_mu"],
        prior_logvar=outputs["prior_logvar"],
        recon_tactile_flow_tail=outputs.get("recon_tactile_flow_tail"),
        target_tactile_flow_tail=outputs.get("target_tactile_flow_tail"),
    )
    losses.update(
        {
            "masked_tail_frames": outputs["tail_mask"].sum().float(),
        }
    )
    if return_outputs:
        return losses, outputs
    return losses