"""Training-curve visualization helpers.

The plotter keeps a lightweight in-memory metric history and rewrites one PNG
as training advances. Training scripts can then log that same image to W&B or
TensorBoard without coupling plotting logic to the optimization loop.
"""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, Iterable, Tuple

import numpy as np
from PIL import Image

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class TrainingCurvePlotter:
    """Maintain and render live training curves to a single PNG file."""

    def __init__(
        self,
        output_dir: str | Path,
        filename: str = "training_curves.png",
        max_points: int = 2000,
        render_every: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.output_dir / filename
        self.max_points = max_points
        self.render_every = max(1, render_every)
        self._history: Dict[str, Deque[Tuple[int, float]]] = defaultdict(
            lambda: deque(maxlen=max_points)
        )
        self._updates = 0

    @staticmethod
    def _should_plot(metric_name: str) -> bool:
        name = metric_name.lower()
        return (
            "loss" in name
            or "contrastive_t" in name
            or name.endswith("/lr")
            or name.endswith("_lr")
        )

    @staticmethod
    def _clean_label(metric_name: str) -> str:
        for prefix in ("stage1/train/", "stage2/train/", "stage1/epoch/", "stage2/epoch/"):
            if metric_name.startswith(prefix):
                return metric_name[len(prefix) :]
        return metric_name

    def update(self, metrics: Dict[str, float], step: int) -> Path | None:
        """Add scalar metrics and render a refreshed plot when due."""
        for name, value in metrics.items():
            if not self._should_plot(name):
                continue
            try:
                scalar = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(scalar):
                self._history[name].append((int(step), scalar))

        if not self._history:
            return None

        self._updates += 1
        if self._updates != 1 and self._updates % self.render_every != 0:
            return None

        self.render()
        return self.path

    def render(self) -> Path:
        groups = self._group_metric_names()
        ncols = 1
        nrows = max(1, len(groups))
        fig_height = 3.8 * nrows
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(10, fig_height), squeeze=False)

        for ax, (title, names) in zip(axes[:, 0], groups):
            for name in names:
                points = list(self._history[name])
                if not points:
                    continue
                steps, values = zip(*points)
                ax.plot(steps, values, label=self._clean_label(name), linewidth=1.7)
            ax.set_title(title)
            ax.set_xlabel("global_step")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best", fontsize=8)

        fig.tight_layout()
        fig.savefig(self.path, dpi=140)
        plt.close(fig)
        return self.path

    def read_image(self) -> np.ndarray:
        """Return the latest plot as an HWC uint8 image for TensorBoard."""
        with Image.open(self.path) as image:
            return np.asarray(image.convert("RGB"))

    def _group_metric_names(self) -> Iterable[Tuple[str, list[str]]]:
        names = sorted(self._history)
        stage1 = [name for name in names if name.startswith("stage1")]
        stage2 = [name for name in names if name.startswith("stage2")]
        other = [name for name in names if name not in set(stage1 + stage2)]

        groups = []
        if stage1:
            groups.append(("Stage 1 training curves", stage1))
        if stage2:
            groups.append(("Stage 2 training curves", stage2))
        if other:
            groups.append(("Other training curves", other))
        return groups
