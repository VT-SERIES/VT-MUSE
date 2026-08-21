"""VT-MUSE temporal visual-tactile representation model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
from transformers import ViTModel, ViTConfig
import math
import os
from pathlib import Path

class ImageEncoder(nn.Module):
    """Encodes visual or tactile observations using ViT backbone."""

    @staticmethod
    def _resolve_vit_checkpoint() -> str:
        env_checkpoint = os.environ.get("VT_MUSE_VIT_CHECKPOINT")
        if env_checkpoint:
            env_path = Path(env_checkpoint)
            if env_path.exists():
                return str(env_path)

        repo_local = Path(__file__).resolve().parents[1] / ".hf_cache" / "google_vit_base_patch16_224"
        if repo_local.exists():
            return str(repo_local)

        return "google/vit-base-patch16-224"

    def __init__(
        self,
        modality: str = 'visual',
        image_size: int = 224,
        patch_size: int = 16,
        num_channels: int = 3,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        pretrained: bool = True
    ):
        super().__init__()
        self.modality = modality

        if pretrained:
            # Use pretrained ViT
            vit_checkpoint = self._resolve_vit_checkpoint()
            local_only = Path(vit_checkpoint).exists()
            self.vit = ViTModel.from_pretrained(vit_checkpoint, local_files_only=local_only)
            # Adjust input channels if needed
            if num_channels != 3:
                old_conv = self.vit.embeddings.patch_embeddings.projection
                self.vit.embeddings.patch_embeddings.projection = nn.Conv2d(
                    num_channels, old_conv.out_channels,
                    kernel_size=old_conv.kernel_size,
                    stride=old_conv.stride
                )
        else:
            # Create ViT from scratch
            config = ViTConfig(
                image_size=image_size,
                patch_size=patch_size,
                num_channels=num_channels,
                hidden_size=hidden_size,
                num_hidden_layers=num_hidden_layers,
                num_attention_heads=num_attention_heads
            )
            self.vit = ViTModel(config)

        self.hidden_size = self.vit.config.hidden_size

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: (B, C, H, W) images

        Returns:
            (B, hidden_size) encoded features (CLS token)
        """
        outputs = self.vit(pixel_values=images)
        # Use CLS token
        cls_token = outputs.last_hidden_state[:, 0, :]
        return cls_token

    def get_all_tokens(self, images: torch.Tensor) -> torch.Tensor:
        """Get all patch tokens (not just CLS)."""
        outputs = self.vit(pixel_values=images)
        return outputs.last_hidden_state


class PatchImageDecoder(nn.Module):
    """
    Decoder for VT-MUSE that reconstructs visual observation
    from latent code conditioned on tactile and action.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        embed_dim: int = 768,
        output_size: Tuple[int, int] = (224, 224),
        output_channels: int = 3,
        num_transformer_layers: int = 6,
        patch_size: int = 16
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.embed_dim = embed_dim
        self.output_size = output_size
        self.output_channels = output_channels
        self.patch_size = patch_size

        # Calculate number of patches
        self.num_patches_h = output_size[0] // patch_size
        self.num_patches_w = output_size[1] // patch_size
        self.num_patches = self.num_patches_h * self.num_patches_w

        # Project latent to embedding dimension
        self.latent_proj = nn.Linear(latent_dim, embed_dim)

        # Learnable patch queries
        self.patch_queries = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_transformer_layers)

        # Project patches to pixels
        self.patch_proj = nn.Linear(embed_dim, patch_size * patch_size * output_channels)

    def forward(self, z: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (B, latent_dim) latent code
            condition: (B, embed_dim) conditioning information

        Returns:
            (B, C, H, W) reconstructed visual observation
        """
        B = z.shape[0]

        # Project latent
        z_proj = self.latent_proj(z).unsqueeze(1)  # (B, 1, embed_dim)

        # Combine latent and condition as memory
        memory = torch.cat([z_proj, condition.unsqueeze(1)], dim=1)  # (B, 2, embed_dim)

        # Prepare patch queries
        queries = self.patch_queries.expand(B, -1, -1) + self.pos_encoding  # (B, num_patches, embed_dim)

        # Decode
        patch_features = self.transformer_decoder(queries, memory)  # (B, num_patches, embed_dim)

        # Project to pixels
        patches = self.patch_proj(patch_features)  # (B, num_patches, patch_size^2 * C)

        # Reshape to image
        patches = patches.reshape(B, self.num_patches_h, self.num_patches_w,
                                   self.patch_size, self.patch_size, self.output_channels)
        patches = patches.permute(0, 5, 1, 3, 2, 4)  # (B, C, H_p, patch_size, W_p, patch_size)
        image = patches.reshape(B, self.output_channels, self.output_size[0], self.output_size[1])

        return torch.sigmoid(image)


class TemporalMemoryEncoder(nn.Module):
    """Encode temporally ordered multimodal tokens into contextualized memory."""

    def __init__(self, embed_dim: int, num_layers: int = 2):
        super().__init__()
        self.num_layers = num_layers
        if num_layers <= 0:
            self.encoder = nn.Identity()
            return
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class LatentAggregator(nn.Module):
    """Aggregate a small token set into latent feature statistics."""

    def __init__(self, embed_dim: int, latent_dim: int, num_layers: int = 2):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_mu = nn.Linear(embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(embed_dim, latent_dim)

    def forward(self, tokens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = tokens.shape[0]
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        fused = self.encoder(torch.cat([cls_token, tokens], dim=1))
        cls_output = fused[:, 0, :]
        return self.fc_mu(cls_output), self.fc_logvar(cls_output)


class MultiFrameTailDecoder(nn.Module):
    """Decode a fixed number of tail frames using shared single-frame decoder weights."""

    def __init__(
        self,
        latent_dim: int,
        embed_dim: int,
        num_tail_frames: int,
        output_size: Tuple[int, int] = (224, 224),
        output_channels: int = 3,
    ):
        super().__init__()
        self.num_tail_frames = num_tail_frames
        self.frame_embed = nn.Parameter(torch.zeros(1, num_tail_frames, embed_dim))
        self.decoder = PatchImageDecoder(
            latent_dim=latent_dim,
            embed_dim=embed_dim,
            output_size=output_size,
            output_channels=output_channels,
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        batch_size = z.shape[0]

        conditions = self.frame_embed[:, :self.num_tail_frames, :].expand(batch_size, -1, -1)
        flat_z = z.unsqueeze(1).expand(-1, self.num_tail_frames, -1).reshape(batch_size * self.num_tail_frames, -1)
        flat_conditions = conditions.reshape(batch_size * self.num_tail_frames, -1)
        recon = self.decoder(flat_z, flat_conditions)
        return recon.reshape(batch_size, self.num_tail_frames, *recon.shape[1:])


class VTMUSEEncoder(nn.Module):
    """Build temporal memory from a sampled sequence and read it with tactile queries."""

    def __init__(
        self,
        visual_encoder: ImageEncoder,
        tactile_encoder: ImageEncoder,
        latent_dim: int = 512,
        history_len: int = 5,
        max_delta_t: int = 64,
        num_memory_layers: int = 4,
        num_latent_layers: int = 4,
        num_tail_frames: int = 2,
        num_tasks: int = 0,
    ):
        super().__init__()
        self.visual_encoder = visual_encoder
        self.tactile_encoder = tactile_encoder
        self.history_len = history_len
        self.max_delta_t = max_delta_t
        self.num_tail_frames = num_tail_frames

        embed_dim = visual_encoder.hidden_size
        self.embed_dim = embed_dim
        self.num_tasks = num_tasks

        self.visual_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.tactile_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.masked_visual_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.history_role_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.query_role_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.target_role_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.memory_context_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.visual_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.temporal_slot_embed = nn.Parameter(torch.zeros(1, history_len, embed_dim))
        self.delta_t_embed = nn.Embedding(max_delta_t + 1, embed_dim)
        self.task_embed = nn.Embedding(num_tasks, embed_dim) if num_tasks > 0 else None

        self.memory_encoder = TemporalMemoryEncoder(embed_dim=embed_dim, num_layers=num_memory_layers)
        self.memory_reader = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.task_conditioner = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=8,
            dropout=0.1,
            batch_first=True,
        )
        self.latent_aggregator = LatentAggregator(
            embed_dim=embed_dim,
            latent_dim=latent_dim,
            num_layers=num_latent_layers,
        )

    def _get_task_token(self, task_id: Optional[torch.Tensor], batch_size: int, device: torch.device) -> torch.Tensor:
        if self.task_embed is None or task_id is None:
            return torch.zeros(batch_size, 1, self.embed_dim, device=device)
        return self.task_embed(task_id).unsqueeze(1)

    def _cross_attend_task(
        self,
        task_id: Optional[torch.Tensor],
        memory_tokens: torch.Tensor,
    ) -> torch.Tensor:
        task_token = self._get_task_token(task_id, memory_tokens.shape[0], memory_tokens.device)
        task_context, _ = self.task_conditioner(
            query=task_token,
            key=memory_tokens,
            value=memory_tokens,
            need_weights=False,
        )
        return task_context

    def _build_temporal_encoding(
        self,
        batch_size: int,
        history_len: int,
        delta_steps: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        slot_embed = self.temporal_slot_embed[:, :history_len, :].expand(batch_size, -1, -1)
        if delta_steps is None:
            delta_steps = torch.arange(history_len, 0, -1, device=device)
            delta_steps = delta_steps.unsqueeze(0).expand(batch_size, -1)
        delta_steps = delta_steps.clamp(max=self.max_delta_t)
        dt_embed = self.delta_t_embed(delta_steps)
        return slot_embed + dt_embed

    def _encode_visual_tokens(
        self,
        visual_seq: torch.Tensor,
        temporal_embed: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len = visual_seq.shape[:2]
        visual_flat = visual_seq.reshape(batch_size * seq_len, *visual_seq.shape[2:])
        visual_base_tokens = self.visual_encoder(visual_flat).reshape(batch_size, seq_len, -1)
        return self._apply_visual_context(
            visual_base_tokens=visual_base_tokens,
            temporal_embed=temporal_embed,
            visual_mask=visual_mask,
            use_mask=use_mask,
        )

    def _apply_visual_context(
        self,
        visual_base_tokens: torch.Tensor,
        temporal_embed: torch.Tensor,
        visual_mask: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ) -> torch.Tensor:
        batch_size, seq_len = visual_base_tokens.shape[:2]
        visual_tokens = visual_base_tokens + self.visual_embed + self.history_role_embed + temporal_embed
        if use_mask and visual_mask is not None:
            mask_token = self.visual_mask_token.expand(batch_size, seq_len, -1)
            mask_token = mask_token + self.masked_visual_embed + self.history_role_embed + temporal_embed
            visual_tokens = torch.where(visual_mask.unsqueeze(-1), mask_token, visual_tokens)
        return visual_tokens

    def _encode_tactile_tokens(
        self,
        tactile_seq: torch.Tensor,
        temporal_embed: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len = tactile_seq.shape[:2]
        tactile_flat = tactile_seq.reshape(batch_size * seq_len, *tactile_seq.shape[2:])
        tactile_base_tokens = self.tactile_encoder(tactile_flat).reshape(batch_size, seq_len, -1)
        return self._apply_tactile_context(tactile_base_tokens, temporal_embed)

    def _apply_tactile_context(
        self,
        tactile_base_tokens: torch.Tensor,
        temporal_embed: torch.Tensor,
    ) -> torch.Tensor:
        return tactile_base_tokens + self.tactile_embed + self.history_role_embed + temporal_embed

    def encode_alignment_base_tokens(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        delta_steps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = visual_seq.shape[:2]
        temporal_embed = self._build_temporal_encoding(
            batch_size=batch_size,
            history_len=seq_len,
            delta_steps=delta_steps,
            device=visual_seq.device,
        )
        visual_flat = visual_seq.reshape(batch_size * seq_len, *visual_seq.shape[2:])
        tactile_flat = tactile_seq.reshape(batch_size * seq_len, *tactile_seq.shape[2:])
        visual_base_tokens = self.visual_encoder(visual_flat).reshape(batch_size, seq_len, -1)
        tactile_base_tokens = self.tactile_encoder(tactile_flat).reshape(batch_size, seq_len, -1)
        return visual_base_tokens, tactile_base_tokens, temporal_embed

    def encode_alignment_tokens_from_base(
        self,
        visual_base_tokens: torch.Tensor,
        tactile_base_tokens: torch.Tensor,
        temporal_embed: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        visual_tokens = self._apply_visual_context(
            visual_base_tokens=visual_base_tokens,
            temporal_embed=temporal_embed,
            visual_mask=visual_mask,
            use_mask=use_mask,
        )
        tactile_tokens = self._apply_tactile_context(tactile_base_tokens, temporal_embed)

        action_tokens = None
        task_context = self._cross_attend_task(task_id, torch.cat([visual_tokens, tactile_tokens], dim=1))
        visual_tokens = visual_tokens + task_context
        tactile_tokens = tactile_tokens + task_context
        return visual_tokens, tactile_tokens, action_tokens

    def encode_alignment_tokens(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        action_seq: Optional[torch.Tensor] = None,
        delta_steps: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Build frame-wise temporalized tokens for Stage 1 alignment.
        """
        visual_base_tokens, tactile_base_tokens, temporal_embed = self.encode_alignment_base_tokens(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            delta_steps=delta_steps,
        )
        return self.encode_alignment_tokens_from_base(
            visual_base_tokens=visual_base_tokens,
            tactile_base_tokens=tactile_base_tokens,
            temporal_embed=temporal_embed,
            task_id=task_id,
            visual_mask=visual_mask,
            use_mask=use_mask,
        )

    def _encode_sequence_tokens(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
        visual_mask: Optional[torch.Tensor],
        delta_steps: Optional[torch.Tensor],
        task_id: Optional[torch.Tensor],
        use_mask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        visual_base_tokens, tactile_base_tokens, temporal_embed = self.encode_stage2_base_tokens(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            delta_steps=delta_steps,
        )
        return self._encode_sequence_tokens_from_base(
            visual_base_tokens=visual_base_tokens,
            tactile_base_tokens=tactile_base_tokens,
            temporal_embed=temporal_embed,
            visual_mask=visual_mask,
            task_id=task_id,
            use_mask=use_mask,
        )

    def encode_stage2_base_tokens(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        delta_steps: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = visual_seq.shape[:2]
        temporal_embed = self._build_temporal_encoding(
            batch_size=batch_size,
            history_len=seq_len,
            delta_steps=delta_steps,
            device=visual_seq.device,
        )
        visual_flat = visual_seq.reshape(batch_size * seq_len, *visual_seq.shape[2:])
        tactile_flat = tactile_seq.reshape(batch_size * seq_len, *tactile_seq.shape[2:])
        visual_base_tokens = self.visual_encoder(visual_flat).reshape(batch_size, seq_len, -1)
        tactile_base_tokens = self.tactile_encoder(tactile_flat).reshape(batch_size, seq_len, -1)
        return visual_base_tokens, tactile_base_tokens, temporal_embed

    def _encode_sequence_tokens_from_base(
        self,
        visual_base_tokens: torch.Tensor,
        tactile_base_tokens: torch.Tensor,
        temporal_embed: torch.Tensor,
        visual_mask: Optional[torch.Tensor],
        task_id: Optional[torch.Tensor],
        use_mask: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = visual_base_tokens.shape[1]
        visual_tokens = self._apply_visual_context(
            visual_base_tokens=visual_base_tokens,
            temporal_embed=temporal_embed,
            visual_mask=visual_mask,
            use_mask=use_mask,
        )
        tactile_tokens = self._apply_tactile_context(tactile_base_tokens, temporal_embed)

        memory_tokens = [visual_tokens, tactile_tokens]
        task_context = self._cross_attend_task(task_id, torch.cat(memory_tokens, dim=1))
        visual_tokens = visual_tokens + task_context
        tactile_tokens = tactile_tokens + task_context

        tokens = []
        for t in range(seq_len):
            tokens.append(visual_tokens[:, t : t + 1, :])
            tokens.append(tactile_tokens[:, t : t + 1, :])
        return torch.cat(tokens, dim=1), tactile_tokens

    def _build_query_tokens(self, tactile_tokens: torch.Tensor) -> torch.Tensor:
        query_tokens = tactile_tokens[:, -self.num_tail_frames :, :]
        return query_tokens + self.query_role_embed

    def _build_target_tokens(
        self,
        visual_seq: torch.Tensor,
        delta_steps: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size, seq_len = visual_seq.shape[:2]
        temporal_embed = self._build_temporal_encoding(
            batch_size=batch_size,
            history_len=seq_len,
            delta_steps=delta_steps,
            device=visual_seq.device,
        )
        visual_flat = visual_seq.reshape(batch_size * seq_len, *visual_seq.shape[2:])
        visual_base_tokens = self.visual_encoder(visual_flat).reshape(batch_size, seq_len, -1)
        return self._build_target_tokens_from_base(
            visual_base_tokens=visual_base_tokens,
            temporal_embed=temporal_embed,
        )

    def _build_target_tokens_from_base(
        self,
        visual_base_tokens: torch.Tensor,
        temporal_embed: torch.Tensor,
    ) -> torch.Tensor:
        visual_tokens = self._apply_visual_context(
            visual_base_tokens=visual_base_tokens,
            temporal_embed=temporal_embed,
            visual_mask=None,
            use_mask=False,
        )
        return visual_tokens[:, -self.num_tail_frames :, :] + self.target_role_embed

    def forward(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
        visual_mask: torch.Tensor,
        delta_steps: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        use_posterior: bool = False,
        visual_base_tokens: Optional[torch.Tensor] = None,
        tactile_base_tokens: Optional[torch.Tensor] = None,
        temporal_embed: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if visual_base_tokens is None or tactile_base_tokens is None or temporal_embed is None:
            history_tokens, tactile_tokens = self._encode_sequence_tokens(
                visual_seq=visual_seq,
                tactile_seq=tactile_seq,
                action_seq=action_seq,
                visual_mask=visual_mask,
                delta_steps=delta_steps,
                task_id=task_id,
                use_mask=True,
            )
            target_tokens = None
        else:
            history_tokens, tactile_tokens = self._encode_sequence_tokens_from_base(
                visual_base_tokens=visual_base_tokens,
                tactile_base_tokens=tactile_base_tokens,
                temporal_embed=temporal_embed,
                visual_mask=visual_mask,
                task_id=task_id,
                use_mask=True,
            )
            target_tokens = (
                self._build_target_tokens_from_base(visual_base_tokens, temporal_embed)
                if use_posterior
                else None
            )
        history_memory = self.memory_encoder(history_tokens)
        task_context = self._cross_attend_task(task_id, history_memory)

        query_tokens = self._build_query_tokens(tactile_tokens) + task_context
        memory_context, attention_weights = self.memory_reader(
            query=query_tokens,
            key=history_memory,
            value=history_memory,
            need_weights=True,
        )
        memory_context = memory_context + self.memory_context_embed

        latent_tokens = [
            query_tokens,
            memory_context,
            task_context,
        ]
        if use_posterior:
            if target_tokens is None:
                target_tokens = self._build_target_tokens(visual_seq, delta_steps)
            latent_tokens.append(target_tokens)

        mu, logvar = self.latent_aggregator(torch.cat(latent_tokens, dim=1))
        extras = {
            "history_memory": history_memory,
            "task_context": task_context,
            "memory_context": memory_context,
            "attention_weights": attention_weights,
        }
        return mu, logvar, extras


class VTMUSE(nn.Module):
    """VT-MUSE model used by both representation-learning stages."""

    def __init__(
        self,
        visual_image_size: int = 224,
        tactile_image_size: int = 224,
        action_dim: int = 7,
        latent_dim: int = 512,
        history_len: int = 5,
        pretrained_encoders: bool = True,
        max_delta_t: int = 64,
        num_memory_layers: int = 4,
        num_latent_layers: int = 4,
        num_tail_frames: int = 2,
        num_tasks: int = 0,
        reconstruct_tactile_flow: bool = False,
    ):
        super().__init__()
        self.visual_image_size = visual_image_size
        self.tactile_image_size = tactile_image_size
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        self.history_len = history_len
        self.num_tail_frames = num_tail_frames
        self.num_tasks = num_tasks
        self.reconstruct_tactile_flow = reconstruct_tactile_flow
        self.reuse_frozen_encoder_tokens = False

        self.visual_encoder = ImageEncoder(
            modality="visual",
            image_size=visual_image_size,
            pretrained=pretrained_encoders,
        )
        self.tactile_encoder = ImageEncoder(
            modality="tactile",
            image_size=tactile_image_size,
            pretrained=pretrained_encoders,
        )
        self.encoder = VTMUSEEncoder(
            visual_encoder=self.visual_encoder,
            tactile_encoder=self.tactile_encoder,
            latent_dim=latent_dim,
            history_len=history_len,
            max_delta_t=max_delta_t,
            num_memory_layers=num_memory_layers,
            num_latent_layers=num_latent_layers,
            num_tail_frames=num_tail_frames,
            num_tasks=num_tasks,
        )
        self.decoder = MultiFrameTailDecoder(
            latent_dim=latent_dim,
            embed_dim=self.visual_encoder.hidden_size,
            num_tail_frames=num_tail_frames,
            output_size=(visual_image_size, visual_image_size),
            output_channels=3,
        )
        self.tactile_flow_decoder = (
            MultiFrameTailDecoder(
                latent_dim=latent_dim,
                embed_dim=self.visual_encoder.hidden_size,
                num_tail_frames=num_tail_frames,
                output_size=(tactile_image_size, tactile_image_size),
                output_channels=3,
            )
            if reconstruct_tactile_flow
            else None
        )

    def forward(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
        visual_mask: torch.Tensor,
        delta_steps: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
        tactile_flow_target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        base_token_kwargs = {}
        if self.reuse_frozen_encoder_tokens:
            visual_base_tokens, tactile_base_tokens, temporal_embed = self.encoder.encode_stage2_base_tokens(
                visual_seq=visual_seq,
                tactile_seq=tactile_seq,
                delta_steps=delta_steps,
            )
            base_token_kwargs = {
                "visual_base_tokens": visual_base_tokens,
                "tactile_base_tokens": tactile_base_tokens,
                "temporal_embed": temporal_embed,
            }

        prior_mu, prior_logvar, prior_extras = self.encoder(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            action_seq=action_seq,
            visual_mask=visual_mask,
            delta_steps=delta_steps,
            task_id=task_id,
            use_posterior=False,
            **base_token_kwargs,
        )
        mu, logvar, posterior_extras = self.encoder(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            action_seq=action_seq,
            visual_mask=visual_mask,
            delta_steps=delta_steps,
            task_id=task_id,
            use_posterior=True,
            **base_token_kwargs,
        )
        z = prior_mu
        recon_tail = self.decoder(z)

        outputs = {
            "recon_tail": recon_tail,
            "target_tail": visual_seq[:, -self.num_tail_frames :, :, :, :],
            "tail_mask": visual_mask[:, -self.num_tail_frames :],
            "mu": mu,
            "logvar": logvar,
            "prior_mu": prior_mu,
            "prior_logvar": prior_logvar,
            "z": z,
            "posterior_mu": mu,
            "posterior_logvar": logvar,
            "memory_context": posterior_extras["memory_context"],
            "attention_weights": posterior_extras["attention_weights"],
            "prior_attention_weights": prior_extras["attention_weights"],
        }
        if self.tactile_flow_decoder is not None:
            outputs["recon_tactile_flow_tail"] = self.tactile_flow_decoder(z)
            if tactile_flow_target is not None:
                outputs["target_tactile_flow_tail"] = tactile_flow_target[:, -self.num_tail_frames :, :, :, :]
        return outputs

    def encode_window(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        action_seq: Optional[torch.Tensor],
        visual_mask: torch.Tensor,
        delta_steps: Optional[torch.Tensor] = None,
        task_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mu, _, _ = self.encoder(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            action_seq=action_seq,
            visual_mask=visual_mask,
            delta_steps=delta_steps,
            task_id=task_id,
            use_posterior=False,
        )
        return mu

    def encode_alignment_tokens(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        action_seq: Optional[torch.Tensor] = None,
        delta_steps: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.encoder.encode_alignment_tokens(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            task_id=task_id,
            action_seq=action_seq,
            delta_steps=delta_steps,
            visual_mask=visual_mask,
            use_mask=use_mask,
        )

    def encode_alignment_base_tokens(
        self,
        visual_seq: torch.Tensor,
        tactile_seq: torch.Tensor,
        delta_steps: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.encoder.encode_alignment_base_tokens(
            visual_seq=visual_seq,
            tactile_seq=tactile_seq,
            delta_steps=delta_steps,
        )

    def encode_alignment_tokens_from_base(
        self,
        visual_base_tokens: torch.Tensor,
        tactile_base_tokens: torch.Tensor,
        temporal_embed: torch.Tensor,
        task_id: Optional[torch.Tensor] = None,
        visual_mask: Optional[torch.Tensor] = None,
        use_mask: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return self.encoder.encode_alignment_tokens_from_base(
            visual_base_tokens=visual_base_tokens,
            tactile_base_tokens=tactile_base_tokens,
            temporal_embed=temporal_embed,
            task_id=task_id,
            visual_mask=visual_mask,
            use_mask=use_mask,
        )

    def encode(
        self,
        current_tactile: torch.Tensor,
        visual_history: torch.Tensor,
        tactile_history: torch.Tensor,
        action_history: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = visual_history.shape[0]
        visual_mask = torch.zeros(
            batch_size,
            visual_history.shape[1],
            dtype=torch.bool,
            device=visual_history.device,
        )
        visual_mask[:, -1] = True
        tactile_seq = tactile_history.clone()
        tactile_seq[:, -1] = current_tactile
        return self.encode_window(
            visual_seq=visual_history,
            tactile_seq=tactile_seq,
            action_seq=action_history,
            visual_mask=visual_mask,
            delta_steps=None,
        )
