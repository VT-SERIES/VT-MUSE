"""HDF5 trajectory loading and temporal masking for VT-MUSE."""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from io import BytesIO
from pathlib import Path
from PIL import Image
from typing import Dict, List, Optional, Tuple
# import numpy as np
# import torchvision.transforms as transforms

RGB_JPEG_ENCODING_ATTR = "rgb_jpeg_encoding"
LEGACY_RGB_JPEG_ENCODING = "opencv_rgb_passthrough_legacy"
FIXED_RGB_JPEG_ENCODING = "opencv_bgr_encoded_rgb"


class HDF5EpisodeDataset(Dataset):
    """
    Dataset for VT-MUSE training.

    Loads visual-tactile-action trajectories from HDF5 files.
    """

    def __init__(
        self,
        data_paths: List[Path],
        history_len: int = 5,
        image_size: int = 224,
        stage: int = 1,
        augment: bool = True
    ):
        """
        Args:
            data_paths: List of paths to HDF5 data files
            history_len: Number of historical timesteps
            image_size: Size to resize images to
            stage: Training stage (1: contrastive, 2: VAE, 3: end-to-end)
            augment: Whether to apply data augmentation
        """
        self.data_paths = data_paths
        self.history_len = history_len
        self.image_size = image_size
        self.stage = stage
        self.augment = augment

        # Build index of all valid samples
        self.samples = []
        self._build_index()

        # Keep visual and tactile observations in their recorded form.
        self.transform = None

    def _build_index(self):
        """Build index of all valid samples across all data files."""
        for data_path in self.data_paths:
            with h5py.File(data_path, 'r') as f:
                if self._is_univtac_episode(f):
                    traj_len = self._get_univtac_length(f)
                    for t in range(self.history_len, traj_len):
                        self.samples.append({
                            'file': data_path,
                            'episode': None,
                            'timestep': t,
                            'format': 'univtac'
                        })
                    continue

                # Assume structure: /episode_XXX/observations/...
                for episode_key in f.keys():
                    episode = f[episode_key]

                    # Get trajectory length
                    if 'observation' in episode and 'camera' in episode['observation']:
                        # Get first camera
                        camera_keys = list(episode['observation']['camera'].keys())
                        if len(camera_keys) > 0:
                            first_camera = camera_keys[0]
                            traj_len = len(episode['observation']['camera'][first_camera]['rgb'])

                            # Add valid samples (those with sufficient history)
                            for t in range(self.history_len, traj_len):
                                self.samples.append({
                                    'file': data_path,
                                    'episode': episode_key,
                                    'timestep': t,
                                    'format': 'legacy'
                                })

    def _is_univtac_episode(self, f: h5py.File) -> bool:
        """Detect UniVTAC dataset files where the file itself is one episode."""
        return 'observation' in f and 'tactile' in f and 'embodiment' in f

    def _get_univtac_length(self, f: h5py.File) -> int:
        if 'step' in f:
            return len(f['step'])
        if 'embodiment' in f and 'ee' in f['embodiment']:
            return len(f['embodiment']['ee'])
        return len(f['observation/head/rgb'])

    def _get_first_existing_dataset(self, root: h5py.Group, paths: List[str]):
        for path in paths:
            if path in root:
                return root[path]
        raise KeyError(f"None of these datasets exist: {paths}")

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, h5_dataset, index: int) -> torch.Tensor:
        """Load and transform an image."""
        img = h5_dataset[index]

        if isinstance(img, (bytes, bytearray, np.bytes_)):
            decoded = np.array(Image.open(BytesIO(bytes(img))).convert('RGB'))
            encoding = h5_dataset.file.attrs.get(RGB_JPEG_ENCODING_ATTR, LEGACY_RGB_JPEG_ENCODING)
            if isinstance(encoding, bytes):
                encoding = encoding.decode("utf-8")
            if encoding == LEGACY_RGB_JPEG_ENCODING:
                # Historical UniVTAC HDF5 files were JPEG-encoded with OpenCV
                # from RGB arrays without converting to BGR first. PIL decodes
                # those bytes faithfully, which makes red/blue appear swapped.
                # Flip the channel order for legacy files only.
                img = decoded[:, :, ::-1].copy()
            else:
                img = decoded

        # Convert to uint8 if needed by PIL transforms.
        # if img.dtype == np.float32 or img.dtype == np.float64:
        #     img = (img * 255).astype(np.uint8)
        #
        # if self.transform is not None:
        #     return self.transform(img)

        # Keep images as recorded; only convert layout/range to a tensor.
        if img.ndim == 3:
            img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
        else:
            img_tensor = torch.from_numpy(img).float()

        if img_tensor.max() > 1.0:
            img_tensor = img_tensor / 255.0

        if img_tensor.ndim == 3 and img_tensor.shape[-2:] != (self.image_size, self.image_size):
            img_tensor = F.interpolate(
                img_tensor.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode='bilinear',
                align_corners=False
            ).squeeze(0)

        return img_tensor

    def _load_univtac_action(self, episode: h5py.File, index: int) -> torch.Tensor:
        """Load a 7D action-like signal from UniVTAC trajectories."""
        if 'actor/action' in episode:
            action = episode['actor/action'][index]
        elif 'embodiment/ee' in episode:
            action = episode['embodiment/ee'][min(index + 1, len(episode['embodiment/ee']) - 1)]
        else:
            action = np.zeros(7, dtype=np.float32)
        return torch.tensor(action, dtype=torch.float32)

    def _getitem_univtac(self, episode: h5py.File, t: int) -> Dict[str, torch.Tensor]:
        visual_dataset = self._get_first_existing_dataset(
            episode,
            ['observation/head/rgb', 'observation/wrist/rgb']
        )
        tactile_dataset = self._get_first_existing_dataset(
            episode,
            [
                'tactile/left_gsmini/rgb_marker',
                'tactile/right_gsmini/rgb_marker',
                'tactile/left_gsmini/rgb',
                'tactile/right_gsmini/rgb',
                'tactile/left_tactile/rgb_marker',
                'tactile/right_tactile/rgb_marker',
            ]
        )

        current_visual = self._load_image(visual_dataset, t)
        current_tactile = self._load_image(tactile_dataset, t)

        visual_history = []
        tactile_history = []
        action_history = []
        for hist_t in range(t - self.history_len, t):
            visual_history.append(self._load_image(visual_dataset, hist_t))
            tactile_history.append(self._load_image(tactile_dataset, hist_t))
            action_history.append(self._load_univtac_action(episode, hist_t))

        visual_history = torch.stack(visual_history)
        tactile_history = torch.stack(tactile_history)
        action_history = torch.stack(action_history)

        if self.stage == 1:
            return {
                'visual': current_visual,
                'tactile': current_tactile,
                'action': action_history[-1] if len(action_history) > 0 else torch.zeros(7)
            }

        return {
            'current_visual': current_visual,
            'current_tactile': current_tactile,
            'visual_history': visual_history,
            'tactile_history': tactile_history,
            'action_history': action_history
        }

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample_info = self.samples[idx]

        with h5py.File(sample_info['file'], 'r') as f:
            if sample_info.get('format') == 'univtac':
                return self._getitem_univtac(f, sample_info['timestep'])

            episode = f[sample_info['episode']]
            t = sample_info['timestep']

            # Get camera names
            camera_keys = list(episode['observation']['camera'].keys())
            primary_camera = camera_keys[0]  # Use first camera as primary

            # Load current observations
            current_visual = self._load_image(
                episode['observation']['camera'][primary_camera]['rgb'], t
            )

            # Load current tactile (assume tactile sensor exists)
            if 'tactile' in episode:
                tactile_keys = list(episode['tactile'].keys())
                if len(tactile_keys) > 0:
                    primary_tactile = tactile_keys[0]
                    current_tactile = self._load_image(
                        episode['tactile'][primary_tactile]['rgb'], t
                    )
                else:
                    # Fallback: use visual as tactile (for testing)
                    current_tactile = current_visual.clone()
            else:
                current_tactile = current_visual.clone()

            # Load history
            visual_history = []
            tactile_history = []
            action_history = []

            for hist_t in range(t - self.history_len, t):
                # Visual history
                vis_hist = self._load_image(
                    episode['observation']['camera'][primary_camera]['rgb'], hist_t
                )
                visual_history.append(vis_hist)

                # Tactile history
                if 'tactile' in episode and len(tactile_keys) > 0:
                    tac_hist = self._load_image(
                        episode['tactile'][primary_tactile]['rgb'], hist_t
                    )
                else:
                    tac_hist = vis_hist.clone()
                tactile_history.append(tac_hist)

                # Action history
                if 'actor' in episode:
                    action = torch.tensor(episode['actor']['action'][hist_t], dtype=torch.float32)
                else:
                    # Fallback: zero action
                    action = torch.zeros(7, dtype=torch.float32)
                action_history.append(action)

            visual_history = torch.stack(visual_history)  # (T, C, H, W)
            tactile_history = torch.stack(tactile_history)  # (T, C, H, W)
            action_history = torch.stack(action_history)  # (T, action_dim)

        # Return based on stage
        if self.stage == 1:
            # Stage 1: Contrastive learning
            return {
                'visual': current_visual,
                'tactile': current_tactile,
                'action': action_history[-1] if len(action_history) > 0 else torch.zeros(7)
            }
        else:
            # Stage 2/3: VAE training
            return {
                'current_visual': current_visual,
                'current_tactile': current_tactile,
                'visual_history': visual_history,
                'tactile_history': tactile_history,
                'action_history': action_history
            }


class VTMUSEDataset(HDF5EpisodeDataset):
    """
    Build a fixed-stride temporal window and randomly mask late visual frames.
    """

    def __init__(
        self,
        data_paths: List[Path],
        history_len: int = 5,
        image_size: int = 224,
        sample_stride: int = 5,
        num_tail_frames: int = 2,
        random_tail_mask: bool = True,
        augment: bool = False,
        task_names: List[str] | None = None,
        tactile_temporal_mode: str = "raw",
        tactile_flow_target_mode: str = "none",
        tactile_flow_clip: float = 0.25,
        marker_flow_clip: float = 1.0,
        depth_delta_clip: float = 0.5,
        tactile_delta_clip: float = 0.25,
    ):
        self.sample_stride = sample_stride
        self.num_tail_frames = num_tail_frames
        self.random_tail_mask = random_tail_mask
        self.tactile_temporal_mode = tactile_temporal_mode
        self.tactile_flow_target_mode = tactile_flow_target_mode
        self.tactile_flow_clip = tactile_flow_clip
        self.marker_flow_clip = marker_flow_clip
        self.depth_delta_clip = depth_delta_clip
        self.tactile_delta_clip = tactile_delta_clip
        self.task_names = task_names or sorted({self._infer_task_name(path) for path in data_paths})
        self.task_to_id = {task_name: idx for idx, task_name in enumerate(self.task_names)}
        self.num_tasks = len(self.task_names)
        super().__init__(
            data_paths=data_paths,
            history_len=history_len,
            image_size=image_size,
            stage=2,
            augment=augment,
        )

    @staticmethod
    def _infer_task_name(data_path: Path) -> str:
        stem = data_path.stem
        if "__episode_" in stem:
            return stem.split("__episode_", 1)[0]
        return data_path.parent.name

    def _build_index(self):
        self.samples = []
        min_timestep = (self.history_len - 1) * self.sample_stride
        for data_path in self.data_paths:
            task_name = self._infer_task_name(data_path)
            import h5py

            with h5py.File(data_path, "r") as f:
                if self._is_univtac_episode(f):
                    traj_len = self._get_univtac_length(f)
                    for t in range(min_timestep, traj_len):
                        self.samples.append(
                            {
                                "file": data_path,
                                "episode": None,
                                "timestep": t,
                                "format": "univtac",
                                "task_name": task_name,
                            }
                        )
                    continue

                for episode_key in f.keys():
                    episode = f[episode_key]
                    if "observation" in episode and "camera" in episode["observation"]:
                        camera_keys = list(episode["observation"]["camera"].keys())
                        if camera_keys:
                            first_camera = camera_keys[0]
                            traj_len = len(episode["observation"]["camera"][first_camera]["rgb"])
                            for t in range(min_timestep, traj_len):
                                self.samples.append(
                                    {
                                        "file": data_path,
                                        "episode": episode_key,
                                        "timestep": t,
                                        "format": "legacy",
                                        "task_name": task_name,
                                    }
                                )

    def _sample_indices(self, timestep: int) -> List[int]:
        start = timestep - (self.history_len - 1) * self.sample_stride
        return [start + i * self.sample_stride for i in range(self.history_len)]

    def _compose_bilateral_tactile(self, left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        if left.shape != right.shape:
            raise ValueError(f"Left/right tactile shapes must match, got {left.shape} vs {right.shape}")
        tactile = torch.cat([left, right], dim=2)
        if tactile.shape[-2:] != (self.image_size, self.image_size):
            tactile = F.interpolate(
                tactile.unsqueeze(0),
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(0)
        return tactile

    def _build_tail_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.history_len, dtype=torch.bool)
        tail_start = self.history_len - self.num_tail_frames
        if self.random_tail_mask:
            patterns = [
                [True, False],
                [False, True],
                [True, True],
            ]
            choice = patterns[torch.randint(0, len(patterns), (1,)).item()]
        else:
            choice = [True] * self.num_tail_frames
        mask[tail_start:] = torch.tensor(choice, dtype=torch.bool)
        return mask

    def _build_delta_steps(self) -> torch.Tensor:
        values = [self.sample_stride * (self.history_len - i - 1) for i in range(self.history_len)]
        return torch.tensor(values, dtype=torch.long)

    def _build_tactile_seq(
        self,
        frames: List[torch.Tensor],
        prev_frames: List[torch.Tensor],
    ) -> torch.Tensor:
        return build_tactile_temporal_features(
            torch.stack(frames),
            prev_tactile_seq=torch.stack(prev_frames),
            mode=self.tactile_temporal_mode,
            flow_clip=self.tactile_flow_clip,
            delta_clip=self.tactile_delta_clip,
        )

    def _build_tactile_flow_target(
        self,
        frames: List[torch.Tensor],
        prev_frames: List[torch.Tensor],
    ) -> torch.Tensor | None:
        if self.tactile_flow_target_mode == "none":
            return None
        return build_tactile_temporal_features(
            torch.stack(frames),
            prev_tactile_seq=torch.stack(prev_frames),
            mode=self.tactile_flow_target_mode,
            flow_clip=self.tactile_flow_clip,
            delta_clip=self.tactile_delta_clip,
        )

    def _rasterize_marker_flow_pair(
        self,
        left_marker,
        prev_left_marker,
        right_marker,
        prev_right_marker,
    ) -> torch.Tensor:
        """Rasterize sparse marker displacement into the bilateral tactile image space."""
        height = self.image_size
        width = self.image_size
        marker_source_width = 320.0
        marker_source_height = 240.0
        side_output_width = width / 2.0
        max_source_displacement = 20.0

        value_sum = torch.zeros(3, height, width, dtype=torch.float32)
        count = torch.zeros(1, height, width, dtype=torch.float32)

        for side_idx, (marker, prev_marker) in enumerate(
            ((left_marker, prev_left_marker), (right_marker, prev_right_marker))
        ):
            marker = torch.as_tensor(marker, dtype=torch.float32).reshape(-1, 2)
            prev_marker = torch.as_tensor(prev_marker, dtype=torch.float32).reshape(-1, 2)
            delta = marker - prev_marker
            raw_magnitude = torch.linalg.norm(delta, dim=-1)
            valid = (
                torch.isfinite(marker).all(dim=-1)
                & torch.isfinite(prev_marker).all(dim=-1)
                & (marker[:, 0] >= 0)
                & (marker[:, 0] <= marker_source_width)
                & (marker[:, 1] >= 0)
                & (marker[:, 1] <= marker_source_height)
                & (raw_magnitude <= max_source_displacement)
            )
            if not valid.any():
                continue

            marker = marker[valid]
            delta = delta[valid]
            x = marker[:, 0] / marker_source_width * side_output_width + side_idx * side_output_width
            y = marker[:, 1] / marker_source_height * height
            u = delta[:, 0] / marker_source_width * side_output_width
            v = delta[:, 1] / marker_source_height * height
            magnitude = torch.linalg.norm(torch.stack([u, v], dim=-1), dim=-1)

            x_idx = torch.round(x).long().clamp(0, width - 1)
            y_idx = torch.round(y).long().clamp(0, height - 1)
            values = torch.stack([u, v, magnitude], dim=0)

            # A small 3x3 splat makes the sparse marker field readable by the
            # image decoder while preserving the physical displacement values.
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    xx = (x_idx + dx).clamp(0, width - 1)
                    yy = (y_idx + dy).clamp(0, height - 1)
                    flat_idx = yy * width + xx
                    value_sum.view(3, -1).scatter_add_(1, flat_idx.unsqueeze(0).expand(3, -1), values)
                    count.view(1, -1).scatter_add_(
                        1,
                        flat_idx.unsqueeze(0),
                        torch.ones(1, flat_idx.numel(), dtype=torch.float32),
                    )

        averaged = value_sum / count.clamp_min(1.0)
        flow_clip = max(self.marker_flow_clip, 1e-6)
        u_norm = 0.5 + 0.5 * averaged[0:1].clamp(-flow_clip, flow_clip) / flow_clip
        v_norm = 0.5 + 0.5 * averaged[1:2].clamp(-flow_clip, flow_clip) / flow_clip
        mag_norm = averaged[2:3].clamp(0.0, flow_clip) / flow_clip
        return torch.cat([u_norm, v_norm, mag_norm], dim=0)

    def _build_marker_flow_target(
        self,
        left_marker_dataset,
        right_marker_dataset,
        indices: List[int],
    ) -> torch.Tensor:
        maps = []
        for slot_idx, idx in enumerate(indices):
            # Align the marker-flow target with the sampled temporal window.
            # For sample_stride=5 and indices [t-20, t-15, ..., t], this uses
            # [0, marker_{t-15}-marker_{t-20}, ..., marker_t-marker_{t-5}]
            # rather than 30Hz one-frame differences that are often too small
            # and noise-dominated.
            prev_idx = idx if slot_idx == 0 else indices[slot_idx - 1]
            maps.append(
                self._rasterize_marker_flow_pair(
                    left_marker_dataset[idx],
                    left_marker_dataset[prev_idx],
                    right_marker_dataset[idx],
                    right_marker_dataset[prev_idx],
                )
            )
        return torch.stack(maps)

    def _build_depth_delta_target(
        self,
        left_depth_dataset,
        right_depth_dataset,
        indices: List[int],
    ) -> torch.Tensor:
        maps = []
        clip = max(self.depth_delta_clip, 1e-6)
        for slot_idx, idx in enumerate(indices):
            prev_idx = idx if slot_idx == 0 else indices[slot_idx - 1]
            left_delta = torch.as_tensor(left_depth_dataset[idx] - left_depth_dataset[prev_idx], dtype=torch.float32)
            right_delta = torch.as_tensor(right_depth_dataset[idx] - right_depth_dataset[prev_idx], dtype=torch.float32)
            delta = torch.cat([left_delta, right_delta], dim=1)
            delta = torch.nan_to_num(delta, nan=0.0, posinf=clip, neginf=-clip)
            if delta.shape[-2:] != (self.image_size, self.image_size):
                delta = F.interpolate(
                    delta.unsqueeze(0).unsqueeze(0),
                    size=(self.image_size, self.image_size),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(0).squeeze(0)
            signed = 0.5 + 0.5 * delta.clamp(-clip, clip) / clip
            positive = delta.clamp(0.0, clip) / clip
            negative = (-delta).clamp(0.0, clip) / clip
            maps.append(torch.stack([signed, positive, negative], dim=0))
        return torch.stack(maps)

    def _getitem_univtac(self, episode, t: int, task_name: str) -> Dict[str, torch.Tensor]:
        visual_dataset = self._get_first_existing_dataset(
            episode,
            ["observation/head/rgb", "observation/wrist/rgb"],
        )
        left_tactile_dataset = self._get_first_existing_dataset(
            episode,
            [
                "tactile/left_tactile/rgb_marker",
                "tactile/left_gsmini/rgb_marker",
                "tactile/left_tactile/rgb",
                "tactile/left_gsmini/rgb",
            ],
        )
        right_tactile_dataset = self._get_first_existing_dataset(
            episode,
            [
                "tactile/right_tactile/rgb_marker",
                "tactile/right_gsmini/rgb_marker",
                "tactile/right_tactile/rgb",
                "tactile/right_gsmini/rgb",
            ],
        )
        left_marker_dataset = right_marker_dataset = None
        left_depth_dataset = right_depth_dataset = None
        if self.tactile_flow_target_mode == "marker_flow":
            left_marker_dataset = self._get_first_existing_dataset(
                episode,
                ["tactile/left_tactile/marker", "tactile/left_gsmini/marker"],
            )
            right_marker_dataset = self._get_first_existing_dataset(
                episode,
                ["tactile/right_tactile/marker", "tactile/right_gsmini/marker"],
            )
        elif self.tactile_flow_target_mode == "depth_delta":
            left_depth_dataset = self._get_first_existing_dataset(
                episode,
                ["tactile/left_tactile/depth", "tactile/left_gsmini/depth"],
            )
            right_depth_dataset = self._get_first_existing_dataset(
                episode,
                ["tactile/right_tactile/depth", "tactile/right_gsmini/depth"],
            )

        indices = self._sample_indices(t)
        visual_seq = torch.stack([self._load_image(visual_dataset, idx) for idx in indices])
        tactile_frames = []
        prev_tactile_frames = []
        for idx in indices:
            tactile_frames.append(
                self._compose_bilateral_tactile(
                    self._load_image(left_tactile_dataset, idx),
                    self._load_image(right_tactile_dataset, idx),
                )
            )
            prev_idx = max(idx - 1, 0)
            prev_tactile_frames.append(
                self._compose_bilateral_tactile(
                    self._load_image(left_tactile_dataset, prev_idx),
                    self._load_image(right_tactile_dataset, prev_idx),
                )
            )
        tactile_seq = self._build_tactile_seq(tactile_frames, prev_tactile_frames)
        if self.tactile_flow_target_mode == "marker_flow":
            tactile_flow_target = self._build_marker_flow_target(left_marker_dataset, right_marker_dataset, indices)
        elif self.tactile_flow_target_mode == "depth_delta":
            tactile_flow_target = self._build_depth_delta_target(left_depth_dataset, right_depth_dataset, indices)
        else:
            tactile_flow_target = self._build_tactile_flow_target(tactile_frames, prev_tactile_frames)
        action_seq = torch.stack([self._load_univtac_action(episode, idx) for idx in indices])
        task_id = torch.tensor(self.task_to_id[task_name], dtype=torch.long)

        sample = {
            "visual_seq": visual_seq,
            "tactile_seq": tactile_seq,
            "action_seq": action_seq,
            "visual_mask": self._build_tail_mask(),
            "delta_steps": self._build_delta_steps(),
            "task_id": task_id,
        }
        if tactile_flow_target is not None:
            sample["tactile_flow_target"] = tactile_flow_target
        return sample

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        import h5py

        sample_info = self.samples[idx]
        with h5py.File(sample_info["file"], "r") as f:
            if sample_info.get("format") == "univtac":
                return self._getitem_univtac(f, sample_info["timestep"], sample_info["task_name"])

            episode = f[sample_info["episode"]]
            timestep = sample_info["timestep"]
            indices = self._sample_indices(timestep)

            camera_keys = list(episode["observation"]["camera"].keys())
            primary_camera = camera_keys[0]

            left_tactile_dataset = None
            right_tactile_dataset = None
            if "tactile" in episode:
                tactile_group = episode["tactile"]
                left_candidates = [
                    "left_tactile/rgb_marker",
                    "left_gsmini/rgb_marker",
                    "left_tactile/rgb",
                    "left_gsmini/rgb",
                ]
                right_candidates = [
                    "right_tactile/rgb_marker",
                    "right_gsmini/rgb_marker",
                    "right_tactile/rgb",
                    "right_gsmini/rgb",
                ]
                left_candidates = [f"tactile/{path}" for path in left_candidates]
                right_candidates = [f"tactile/{path}" for path in right_candidates]
                try:
                    left_tactile_dataset = self._get_first_existing_dataset(episode, left_candidates)
                except KeyError:
                    left_tactile_dataset = None
                try:
                    right_tactile_dataset = self._get_first_existing_dataset(episode, right_candidates)
                except KeyError:
                    right_tactile_dataset = None
                if left_tactile_dataset is None or right_tactile_dataset is None:
                    raise KeyError(
                        "Legacy episode is missing left or right tactile observations. "
                        "VT-MUSE requires both tactile sides."
                    )
                left_marker_dataset = right_marker_dataset = None
                left_depth_dataset = right_depth_dataset = None
                if self.tactile_flow_target_mode == "marker_flow":
                    left_marker_dataset = self._get_first_existing_dataset(
                        episode,
                        ["tactile/left_tactile/marker", "tactile/left_gsmini/marker"],
                    )
                    right_marker_dataset = self._get_first_existing_dataset(
                        episode,
                        ["tactile/right_tactile/marker", "tactile/right_gsmini/marker"],
                    )
                elif self.tactile_flow_target_mode == "depth_delta":
                    left_depth_dataset = self._get_first_existing_dataset(
                        episode,
                        ["tactile/left_tactile/depth", "tactile/left_gsmini/depth"],
                    )
                    right_depth_dataset = self._get_first_existing_dataset(
                        episode,
                        ["tactile/right_tactile/depth", "tactile/right_gsmini/depth"],
                    )

            visual_seq = []
            tactile_seq = []
            prev_tactile_seq = []
            action_seq = []
            for frame_idx in indices:
                visual = self._load_image(
                    episode["observation"]["camera"][primary_camera]["rgb"],
                    frame_idx,
                )
                visual_seq.append(visual)

                if left_tactile_dataset is not None and right_tactile_dataset is not None:
                    left_tactile = self._load_image(left_tactile_dataset, frame_idx)
                    right_tactile = self._load_image(right_tactile_dataset, frame_idx)
                    tactile = self._compose_bilateral_tactile(left_tactile, right_tactile)
                    prev_frame_idx = max(frame_idx - 1, 0)
                    prev_tactile = self._compose_bilateral_tactile(
                        self._load_image(left_tactile_dataset, prev_frame_idx),
                        self._load_image(right_tactile_dataset, prev_frame_idx),
                    )
                else:
                    raise KeyError(
                        "Legacy episode is missing tactile observations. "
                        "VT-MUSE requires tactile/left and tactile/right "
                        "or a compatible single-side tactile stream."
                    )
                tactile_seq.append(tactile)
                prev_tactile_seq.append(prev_tactile)

                if "actor" in episode:
                    action = torch.tensor(episode["actor"]["action"][frame_idx], dtype=torch.float32)
                else:
                    action = torch.zeros(7, dtype=torch.float32)
                action_seq.append(action)

        tactile_input = self._build_tactile_seq(tactile_seq, prev_tactile_seq)
        if self.tactile_flow_target_mode == "marker_flow":
            tactile_flow_target = self._build_marker_flow_target(left_marker_dataset, right_marker_dataset, indices)
        elif self.tactile_flow_target_mode == "depth_delta":
            tactile_flow_target = self._build_depth_delta_target(left_depth_dataset, right_depth_dataset, indices)
        else:
            tactile_flow_target = self._build_tactile_flow_target(tactile_seq, prev_tactile_seq)
        sample = {
            "visual_seq": torch.stack(visual_seq),
            "tactile_seq": tactile_input,
            "action_seq": torch.stack(action_seq),
            "visual_mask": self._build_tail_mask(),
            "delta_steps": self._build_delta_steps(),
            "task_id": torch.tensor(self.task_to_id[sample_info["task_name"]], dtype=torch.long),
        }
        if tactile_flow_target is not None:
            sample["tactile_flow_target"] = tactile_flow_target
        return sample


def create_dataloaders(
    train_data_paths: List[Path],
    val_data_paths: List[Path],
    history_len: int = 5,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
    sample_stride: int = 5,
    num_tail_frames: int = 2,
    tactile_temporal_mode: str = "raw",
    tactile_flow_target_mode: str = "none",
    tactile_flow_clip: float = 0.25,
    marker_flow_clip: float = 1.0,
    depth_delta_clip: float = 0.5,
    tactile_delta_clip: float = 0.25,
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    task_names = sorted(
        {
            VTMUSEDataset._infer_task_name(data_path)
            for data_path in [*train_data_paths, *val_data_paths]
        }
    )
    train_dataset = VTMUSEDataset(
        data_paths=train_data_paths,
        history_len=history_len,
        image_size=image_size,
        sample_stride=sample_stride,
        num_tail_frames=num_tail_frames,
        random_tail_mask=True,
        augment=False,
        task_names=task_names,
        tactile_temporal_mode=tactile_temporal_mode,
        tactile_flow_target_mode=tactile_flow_target_mode,
        tactile_flow_clip=tactile_flow_clip,
        marker_flow_clip=marker_flow_clip,
        depth_delta_clip=depth_delta_clip,
        tactile_delta_clip=tactile_delta_clip,
    )
    val_dataset = VTMUSEDataset(
        data_paths=val_data_paths,
        history_len=history_len,
        image_size=image_size,
        sample_stride=sample_stride,
        num_tail_frames=num_tail_frames,
        random_tail_mask=False,
        augment=False,
        task_names=task_names,
        tactile_temporal_mode=tactile_temporal_mode,
        tactile_flow_target_mode=tactile_flow_target_mode,
        tactile_flow_clip=tactile_flow_clip,
        marker_flow_clip=marker_flow_clip,
        depth_delta_clip=depth_delta_clip,
        tactile_delta_clip=tactile_delta_clip,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
