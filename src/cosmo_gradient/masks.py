"""Survey mask helpers.

The real DESI analysis should use the DESI LSS random catalogs and known
systematics maps. The synthetic mask here only gives the local smoke test a
non-trivial footprint so that the random-catalog correction is exercised.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def synthetic_survey_mask(ra_deg: ArrayLike, dec_deg: ArrayLike) -> NDArray[np.bool_]:
    """A deterministic, survey-like sky footprint used only in synthetic mode."""
    ra = np.asarray(ra_deg, dtype=float) % 360.0
    dec = np.asarray(dec_deg, dtype=float)

    broad_footprint = (dec > -35.0) & (dec < 78.0)
    dust_like_band = np.abs(dec + 10.0 * np.sin(np.deg2rad(ra))) < 7.0
    ra_gap = (ra > 118.0) & (ra < 146.0) & (dec < 35.0)
    far_south_gap = (dec < -15.0) & (ra > 260.0) & (ra < 330.0)
    return broad_footprint & ~dust_like_band & ~ra_gap & ~far_south_gap


def mask_fraction(ra_deg: ArrayLike, dec_deg: ArrayLike) -> float:
    """Return the fraction of points accepted by the synthetic mask."""
    mask = synthetic_survey_mask(ra_deg, dec_deg)
    return float(np.mean(mask))
