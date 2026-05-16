from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Union

import torch
from torch_geometric.data import Data

CorrelationMode = Literal["positive", "negative", "zero"]
FeatureType = Literal["binary", "categorical", "continuous"]


@dataclass(frozen=True)
class FeatureManipulationSpec:
    feature: Union[int, str]
    feature_type: FeatureType
    p: float = 0.8
    alpha: float = 1.0
    sigma: float = 1.0
    seed: int = 0


class FeatureManipulator:
    def __init__(self, spec: FeatureManipulationSpec):
        self.spec = spec
        self._cached_feature_idx: Optional[int] = None
        self._cached_column: Optional[torch.Tensor] = None
        self._cached_data_id: Optional[int] = None

    def resolve_feature_index(self, data: Data) -> int:
        if isinstance(self.spec.feature, int):
            return self.spec.feature
        feature_names = getattr(data, "feature_names", None)
        if feature_names is None:
            raise ValueError("Feature names not available on data; cannot resolve string feature.")
        try:
            return feature_names.index(self.spec.feature)
        except ValueError as exc:
            raise ValueError(
                f"Feature '{self.spec.feature}' not found. Available: {feature_names[:10]} ..."
            ) from exc

    def cache_original(self, data: Data) -> None:
        feature_idx = self.resolve_feature_index(data)
        cache_key = id(data)
        if self._cached_data_id == cache_key and self._cached_column is not None:
            return
        self._cached_feature_idx = feature_idx
        self._cached_column = data.x[:, feature_idx].detach().cpu().clone()
        self._cached_data_id = cache_key

    def restore(self, data: Data) -> None:
        if self._cached_column is None or self._cached_feature_idx is None:
            raise RuntimeError("No cached feature column to restore.")
        data.x[:, self._cached_feature_idx] = self._cached_column.to(data.x.device)

    def apply_split(
        self,
        data: Data,
        split: Literal["train", "val", "test"],
        corr: CorrelationMode,
        seed_offset: Optional[int] = None,
    ) -> None:
        mask = getattr(data, f"{split}_mask", None)
        if mask is None:
            raise ValueError(f"Data has no {split}_mask.")
        offset = seed_offset if seed_offset is not None else {"train": 0, "val": 1, "test": 2}[split]
        self.apply_mask(data, mask, corr=corr, seed=self.spec.seed + offset)

    def apply_mask(self, data: Data, mask: torch.Tensor, corr: CorrelationMode, seed: int) -> None:
        if data.y is None:
            raise ValueError("Data has no labels (y).")
        unique_labels = torch.unique(data.y).tolist()
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(f"Feature manipulation requires binary labels, got {unique_labels}.")

        self.cache_original(data)
        if self._cached_feature_idx is None:
            raise RuntimeError("Failed to resolve target feature index.")

        idx = mask.nonzero(as_tuple=False).view(-1)
        if idx.numel() == 0:
            return

        x = data.x
        y = data.y[idx].to(torch.float32)
        device = x.device
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))

        if self.spec.feature_type in ("binary", "categorical"):
            if corr in ("positive", "negative") and not (0.5 < self.spec.p < 1.0):
                raise ValueError("Parameter p must be in (0.5, 1.0) for positive/negative correlation.")
            if corr == "zero":
                probs = torch.full_like(y, 0.5, device=device)
            elif corr == "positive":
                probs = torch.where(y > 0.5, self.spec.p, 1.0 - self.spec.p).to(device)
            else:
                probs = torch.where(y > 0.5, 1.0 - self.spec.p, self.spec.p).to(device)
            sampled = torch.bernoulli(probs, generator=generator).to(x.dtype)
            x[idx, self._cached_feature_idx] = sampled
            return

        if self.spec.feature_type == "continuous":
            eps = torch.randn(idx.size(0), generator=generator, device=device) * self.spec.sigma
            if corr == "positive":
                base = self.spec.alpha * y.to(device)
            elif corr == "negative":
                base = self.spec.alpha * (1.0 - y.to(device))
            else:
                base = torch.zeros_like(y, device=device)
            x[idx, self._cached_feature_idx] = (base + eps).to(x.dtype)
            return

        raise ValueError(f"Unsupported feature_type: {self.spec.feature_type}")
