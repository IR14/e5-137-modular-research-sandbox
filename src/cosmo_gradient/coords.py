"""Coordinate utilities for sky-vector operations."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def radec_to_unit(ra_deg: ArrayLike, dec_deg: ArrayLike) -> NDArray[np.float64]:
    """Convert right ascension and declination in degrees to Cartesian unit vectors."""
    ra = np.deg2rad(np.asarray(ra_deg, dtype=float))
    dec = np.deg2rad(np.asarray(dec_deg, dtype=float))
    cos_dec = np.cos(dec)
    vectors = np.column_stack(
        [
            cos_dec * np.cos(ra),
            cos_dec * np.sin(ra),
            np.sin(dec),
        ]
    )
    return normalize_vectors(vectors)


def unit_to_radec(vectors: ArrayLike) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Convert Cartesian unit vectors to RA/DEC in degrees."""
    vec = normalize_vectors(np.asarray(vectors, dtype=float))
    ra = np.rad2deg(np.arctan2(vec[..., 1], vec[..., 0])) % 360.0
    dec = np.rad2deg(np.arcsin(np.clip(vec[..., 2], -1.0, 1.0)))
    return ra, dec


def normalize_vectors(vectors: ArrayLike) -> NDArray[np.float64]:
    """Normalize vectors along the final axis."""
    arr = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Cannot normalize zero-length vector.")
    return arr / norms


def angular_separation_deg(a: ArrayLike, b: ArrayLike) -> NDArray[np.float64]:
    """Return angular separation between vectors in degrees."""
    va = normalize_vectors(a)
    vb = normalize_vectors(b)
    dot = np.sum(va * vb, axis=-1)
    return np.rad2deg(np.arccos(np.clip(dot, -1.0, 1.0)))
