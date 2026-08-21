"""Temporal tactile representations for VT-MUSE.

The flow mode keeps the ViT input interface unchanged by mapping raw-adjacent
tactile frame pairs to a 3-channel image: horizontal flow, vertical flow, and
magnitude.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _to_sequence(tactile_seq: torch.Tensor) -> tuple[torch.Tensor, bool]:
    if tactile_seq.dim() == 4:
        return tactile_seq.unsqueeze(0), True
    if tactile_seq.dim() == 5:
        return tactile_seq, False
    raise ValueError(f"Expected tactile sequence with rank 4 or 5, got {tuple(tactile_seq.shape)}")


def _rgb_to_gray(images: torch.Tensor) -> torch.Tensor:
    if images.shape[1] >= 3:
        weights = images.new_tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
        return (images[:, :3] * weights).sum(dim=1, keepdim=True)
    return images[:, :1]


def _central_gradients(gray: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    padded_x = F.pad(gray, (1, 1, 0, 0), mode="replicate")
    padded_y = F.pad(gray, (0, 0, 1, 1), mode="replicate")
    grad_x = 0.5 * (padded_x[:, :, :, 2:] - padded_x[:, :, :, :-2])
    grad_y = 0.5 * (padded_y[:, :, 2:, :] - padded_y[:, :, :-2, :])
    return grad_x, grad_y


def _box_blur(images: torch.Tensor, kernel_size: int) -> torch.Tensor:
    if kernel_size <= 1:
        return images
    if kernel_size % 2 == 0:
        raise ValueError(f"Blur kernel size must be odd, got {kernel_size}")
    pad = kernel_size // 2
    return F.avg_pool2d(
        F.pad(images, (pad, pad, pad, pad), mode="replicate"),
        kernel_size=kernel_size,
        stride=1,
    )


def _flow_to_three_channels(flow_u: torch.Tensor, flow_v: torch.Tensor, flow_clip: float) -> torch.Tensor:
    magnitude = torch.sqrt(flow_u.square() + flow_v.square()).clamp(max=flow_clip)
    return torch.cat(
        [
            0.5 + 0.5 * (flow_u / max(flow_clip, 1e-6)),
            0.5 + 0.5 * (flow_v / max(flow_clip, 1e-6)),
            magnitude / max(flow_clip, 1e-6),
        ],
        dim=1,
    )


def tactile_sequence_to_delta(
    tactile_seq: torch.Tensor,
    prev_tactile_seq: torch.Tensor | None = None,
    delta_clip: float = 0.25,
) -> torch.Tensor:
    seq, squeezed = _to_sequence(tactile_seq)
    if prev_tactile_seq is None:
        raise ValueError("delta tactile mode requires prev_tactile_seq with raw previous frames")
    prev, _ = _to_sequence(prev_tactile_seq)
    if prev.shape != seq.shape:
        raise ValueError(
            f"Previous tactile sequence shape {tuple(prev.shape)} does not match current {tuple(seq.shape)}"
        )
    delta = (seq - prev).clamp(min=-delta_clip, max=delta_clip)
    delta = 0.5 + 0.5 * (delta / max(delta_clip, 1e-6))
    return delta.squeeze(0) if squeezed else delta


def tactile_sequence_to_flow(
    tactile_seq: torch.Tensor,
    prev_tactile_seq: torch.Tensor | None = None,
    flow_clip: float = 0.25,
    eps: float = 1e-4,
) -> torch.Tensor:
    seq, squeezed = _to_sequence(tactile_seq)
    batch_size, seq_len, channels, height, width = seq.shape
    if prev_tactile_seq is None:
        raise ValueError("flow tactile mode requires prev_tactile_seq with raw previous frames")
    prev, _ = _to_sequence(prev_tactile_seq)
    if prev.shape != seq.shape:
        raise ValueError(
            f"Previous tactile sequence shape {tuple(prev.shape)} does not match current {tuple(seq.shape)}"
        )

    curr_flat = seq.reshape(batch_size * seq_len, channels, height, width)
    prev_flat = prev.reshape(batch_size * seq_len, channels, height, width)
    curr_gray = _rgb_to_gray(curr_flat)
    prev_gray = _rgb_to_gray(prev_flat)

    grad_x, grad_y = _central_gradients(curr_gray)
    grad_t = curr_gray - prev_gray
    denom = grad_x.square() + grad_y.square() + eps

    flow_u = (-grad_t * grad_x / denom).clamp(min=-flow_clip, max=flow_clip)
    flow_v = (-grad_t * grad_y / denom).clamp(min=-flow_clip, max=flow_clip)

    flow = _flow_to_three_channels(flow_u, flow_v, flow_clip)
    flow = flow.reshape(batch_size, seq_len, 3, height, width)
    return flow.squeeze(0) if squeezed else flow


def tactile_sequence_to_denoised_flow(
    tactile_seq: torch.Tensor,
    prev_tactile_seq: torch.Tensor | None = None,
    flow_clip: float = 0.5,
    eps: float = 1e-4,
    blur_kernel: int = 5,
    grad_threshold: float = 0.03,
) -> torch.Tensor:
    """Estimate a less noisy one-step tactile flow proxy.

    Compared with ``tactile_sequence_to_flow``, this variant smooths grayscale
    frames, down-weights low-gradient pixels, and uses tanh compression instead
    of hard clipping to suppress noise in low-texture regions.
    """

    seq, squeezed = _to_sequence(tactile_seq)
    batch_size, seq_len, channels, height, width = seq.shape
    if prev_tactile_seq is None:
        raise ValueError("denoised_flow tactile mode requires prev_tactile_seq with raw previous frames")
    prev, _ = _to_sequence(prev_tactile_seq)
    if prev.shape != seq.shape:
        raise ValueError(
            f"Previous tactile sequence shape {tuple(prev.shape)} does not match current {tuple(seq.shape)}"
        )

    curr_flat = seq.reshape(batch_size * seq_len, channels, height, width)
    prev_flat = prev.reshape(batch_size * seq_len, channels, height, width)
    curr_gray = _box_blur(_rgb_to_gray(curr_flat), blur_kernel)
    prev_gray = _box_blur(_rgb_to_gray(prev_flat), blur_kernel)

    grad_x, grad_y = _central_gradients(curr_gray)
    grad_t = _box_blur(curr_gray - prev_gray, blur_kernel)
    grad_energy = grad_x.square() + grad_y.square()
    if grad_threshold <= 0:
        confidence = torch.ones_like(grad_energy)
    else:
        confidence = grad_energy / (grad_energy + grad_threshold ** 2)
    denom = grad_energy + eps

    raw_u = -grad_t * grad_x / denom
    raw_v = -grad_t * grad_y / denom
    flow_u = flow_clip * torch.tanh((raw_u * confidence) / max(flow_clip, 1e-6))
    flow_v = flow_clip * torch.tanh((raw_v * confidence) / max(flow_clip, 1e-6))

    flow = _flow_to_three_channels(flow_u, flow_v, flow_clip)
    flow = flow.reshape(batch_size, seq_len, 3, height, width)
    return flow.squeeze(0) if squeezed else flow


def build_tactile_temporal_features(
    tactile_seq: torch.Tensor,
    prev_tactile_seq: torch.Tensor | None = None,
    mode: str = "raw",
    flow_clip: float = 0.25,
    delta_clip: float = 0.25,
    denoised_flow_blur_kernel: int = 5,
    denoised_flow_grad_threshold: float = 0.03,
) -> torch.Tensor:
    if mode == "raw":
        return tactile_seq
    if mode == "delta":
        return tactile_sequence_to_delta(tactile_seq, prev_tactile_seq=prev_tactile_seq, delta_clip=delta_clip)
    if mode == "flow":
        return tactile_sequence_to_flow(tactile_seq, prev_tactile_seq=prev_tactile_seq, flow_clip=flow_clip)
    if mode == "denoised_flow":
        return tactile_sequence_to_denoised_flow(
            tactile_seq,
            prev_tactile_seq=prev_tactile_seq,
            flow_clip=flow_clip,
            blur_kernel=denoised_flow_blur_kernel,
            grad_threshold=denoised_flow_grad_threshold,
        )
    raise ValueError(f"Unsupported tactile temporal mode: {mode}")
