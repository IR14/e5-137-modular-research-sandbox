"""Redshift binning utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class RedshiftBin:
    """A half-open redshift bin, except the final configured bin may include its high edge."""

    z_min: float
    z_max: float

    @property
    def label(self) -> str:
        return f"z{self.z_min:.3f}_{self.z_max:.3f}".replace(".", "p")

    def contains(self, z: ArrayLike, include_right: bool = False) -> NDArray[np.bool_]:
        values = np.asarray(z, dtype=float)
        if include_right:
            return (values >= self.z_min) & (values <= self.z_max)
        return (values >= self.z_min) & (values < self.z_max)


def make_redshift_bins(edges: ArrayLike) -> list[RedshiftBin]:
    """Create ordered bins from monotonically increasing redshift edges."""
    values = np.asarray(edges, dtype=float)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("At least two redshift bin edges are required.")
    if np.any(np.diff(values) <= 0):
        raise ValueError("Redshift bin edges must be strictly increasing.")
    return [RedshiftBin(float(lo), float(hi)) for lo, hi in zip(values[:-1], values[1:])]


def assign_redshift_bins(z: ArrayLike, edges: ArrayLike) -> NDArray[np.int64]:
    """Assign each redshift to a configured bin index, or -1 outside the range."""
    values = np.asarray(z, dtype=float)
    edges_array = np.asarray(edges, dtype=float)
    bins = np.searchsorted(edges_array, values, side="right") - 1
    outside = (values < edges_array[0]) | (values > edges_array[-1])
    bins[outside] = -1
    bins[values == edges_array[-1]] = len(edges_array) - 2
    return bins.astype(np.int64)


def subset_by_redshift_bin(frame: pd.DataFrame, zbin: RedshiftBin, final_bin: bool = False) -> pd.DataFrame:
    """Return rows inside one redshift bin."""
    mask = zbin.contains(frame["z"].to_numpy(), include_right=final_bin)
    return frame.loc[mask].copy()
