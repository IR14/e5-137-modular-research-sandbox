"""Sky map construction from data and random catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cosmo_gradient.coords import radec_to_unit


@dataclass(frozen=True)
class SkyMap:
    """Weighted count and overdensity maps for one tracer/redshift bin."""

    nside: int
    backend: str
    data_counts: NDArray[np.float64]
    random_counts: NDArray[np.float64]
    alpha: float
    delta: NDArray[np.float64]
    valid: NDArray[np.bool_]
    pixel_vectors: NDArray[np.float64]


def build_overdensity_map(
    data: pd.DataFrame,
    randoms: pd.DataFrame,
    nside: int,
    min_random_per_pixel: float = 1.0,
) -> SkyMap:
    """Build weighted count maps and a random-corrected overdensity map."""
    data_counts, npix, backend = weighted_counts(data, nside)
    random_counts, random_npix, _ = weighted_counts(randoms, nside)
    if npix != random_npix:
        raise ValueError("Data and random maps have incompatible pixel counts.")
    return counts_to_overdensity_map(
        data_counts=data_counts,
        random_counts=random_counts,
        nside=nside,
        backend=backend,
        min_random_per_pixel=min_random_per_pixel,
    )


def counts_to_overdensity_map(
    data_counts: NDArray[np.float64],
    random_counts: NDArray[np.float64],
    nside: int,
    backend: str,
    min_random_per_pixel: float = 1.0,
) -> SkyMap:
    """Build a random-corrected overdensity map from pre-accumulated count maps."""
    if data_counts.shape != random_counts.shape:
        raise ValueError("Data and random count maps must have the same shape.")
    npix = len(data_counts)
    data_total = float(np.sum(data_counts))
    random_total = float(np.sum(random_counts))
    if data_total <= 0.0:
        raise ValueError("No weighted data objects remain in this bin.")
    if random_total <= 0.0:
        raise ValueError("No weighted random objects remain in this bin.")
    alpha = data_total / random_total
    expected = alpha * random_counts
    valid = expected >= min_random_per_pixel
    delta = np.full(npix, np.nan, dtype=float)
    delta[valid] = data_counts[valid] / expected[valid] - 1.0
    return SkyMap(
        nside=nside,
        backend=backend,
        data_counts=data_counts,
        random_counts=random_counts,
        alpha=alpha,
        delta=delta,
        valid=valid,
        pixel_vectors=pixel_vectors(nside, npix=npix, backend=backend),
    )


def weighted_counts(frame: pd.DataFrame, nside: int) -> tuple[NDArray[np.float64], int, str]:
    """Return weighted pixel counts for a standardized catalog."""
    pix, npix, backend = pixelize_radec(frame["ra"].to_numpy(), frame["dec"].to_numpy(), nside)
    weights = frame["weight"].to_numpy(dtype=float) if "weight" in frame else None
    counts = np.bincount(pix, weights=weights, minlength=npix).astype(float)
    return counts, npix, backend


def pixelize_radec(
    ra_deg: NDArray[np.float64],
    dec_deg: NDArray[np.float64],
    nside: int,
) -> tuple[NDArray[np.int64], int, str]:
    """Assign sky coordinates to pixels.

    Uses HEALPix via ``healpy`` when installed. The fallback is a deterministic
    equal-angle grid for tests and dry runs; scientific DESI runs should install
    the ``healpix`` optional dependency.
    """
    if nside <= 0:
        raise ValueError("nside must be positive.")
    try:
        import healpy as hp

        theta = np.deg2rad(90.0 - dec_deg)
        phi = np.deg2rad(ra_deg)
        return hp.ang2pix(nside, theta, phi, nest=False).astype(np.int64), hp.nside2npix(nside), "healpy"
    except ImportError:
        return _pixelize_equal_angle(ra_deg, dec_deg, nside)


def pixel_vectors(nside: int, npix: Optional[int] = None, backend: Optional[str] = None) -> NDArray[np.float64]:
    """Return Cartesian unit vectors for pixel centers."""
    try:
        if backend in (None, "healpy"):
            import healpy as hp

            n_pixels = hp.nside2npix(nside) if npix is None else npix
            theta, phi = hp.pix2ang(nside, np.arange(n_pixels), nest=False)
            ra = np.rad2deg(phi)
            dec = 90.0 - np.rad2deg(theta)
            return radec_to_unit(ra, dec)
    except ImportError:
        pass
    n_ra = 4 * nside
    n_dec = 2 * nside
    if npix is not None and npix != n_ra * n_dec:
        raise ValueError("Fallback pixel count does not match nside.")
    dec_index = np.repeat(np.arange(n_dec), n_ra)
    ra_index = np.tile(np.arange(n_ra), n_dec)
    ra = (ra_index + 0.5) * 360.0 / n_ra
    sin_edges = np.linspace(-1.0, 1.0, n_dec + 1)
    sin_dec = 0.5 * (sin_edges[dec_index] + sin_edges[dec_index + 1])
    dec = np.rad2deg(np.arcsin(np.clip(sin_dec, -1.0, 1.0)))
    return radec_to_unit(ra, dec)


def _pixelize_equal_angle(
    ra_deg: NDArray[np.float64],
    dec_deg: NDArray[np.float64],
    nside: int,
) -> tuple[NDArray[np.int64], int, str]:
    n_ra = 4 * nside
    n_dec = 2 * nside
    ra_bin = np.floor((ra_deg % 360.0) / 360.0 * n_ra).astype(int)
    sin_dec = np.sin(np.deg2rad(dec_deg))
    dec_bin = np.floor((sin_dec + 1.0) / 2.0 * n_dec).astype(int)
    ra_bin = np.clip(ra_bin, 0, n_ra - 1)
    dec_bin = np.clip(dec_bin, 0, n_dec - 1)
    pix = dec_bin * n_ra + ra_bin
    return pix.astype(np.int64), n_ra * n_dec, "equal_angle"


def save_sky_map(path: str, sky_map: SkyMap) -> None:
    """Save a sky map as a compressed NumPy archive."""
    np.savez_compressed(
        path,
        nside=sky_map.nside,
        backend=sky_map.backend,
        data_counts=sky_map.data_counts,
        random_counts=sky_map.random_counts,
        alpha=sky_map.alpha,
        delta=sky_map.delta,
        valid=sky_map.valid,
        pixel_vectors=sky_map.pixel_vectors,
    )


def load_sky_map(path: str) -> SkyMap:
    """Load a sky map saved with :func:`save_sky_map`."""
    loaded = np.load(path, allow_pickle=False)
    return SkyMap(
        nside=int(loaded["nside"]),
        backend=str(loaded["backend"]),
        data_counts=loaded["data_counts"],
        random_counts=loaded["random_counts"],
        alpha=float(loaded["alpha"]),
        delta=loaded["delta"],
        valid=loaded["valid"],
        pixel_vectors=loaded["pixel_vectors"],
    )
