"""Dipole fitting, bootstrap uncertainty, and null tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from cosmo_gradient.coords import angular_separation_deg, unit_to_radec
from cosmo_gradient.maps import SkyMap


@dataclass(frozen=True)
class DipoleFit:
    """A fitted dipole model ``delta(n) = monopole + vector dot n``."""

    amplitude: float
    vector: NDArray[np.float64]
    ra_deg: float
    dec_deg: float
    monopole: float
    n_pixels: int
    amplitude_std: Optional[float] = None
    axis_ra_std_deg: Optional[float] = None
    axis_dec_std_deg: Optional[float] = None
    null_p_value: Optional[float] = None
    null_permutations: Optional[int] = None
    null_amplitudes: Optional[NDArray[np.float64]] = None
    poisson_p_value: Optional[float] = None
    poisson_mocks: Optional[int] = None
    poisson_amplitudes: Optional[NDArray[np.float64]] = None
    block_null_p_value: Optional[float] = None
    block_null_mocks: Optional[int] = None
    block_null_regions: Optional[int] = None
    block_null_amplitudes: Optional[NDArray[np.float64]] = None
    jackknife_regions: Optional[int] = None
    jackknife_amplitude_std: Optional[float] = None
    jackknife_amplitude_min: Optional[float] = None
    jackknife_amplitude_max: Optional[float] = None
    jackknife_axis_median_shift_deg: Optional[float] = None
    jackknife_axis_max_shift_deg: Optional[float] = None

    def to_record(self, tracer: str, z_min: float, z_max: float, n_data: int, n_random: int) -> dict[str, object]:
        """Return a flat record suitable for CSV output."""
        record = asdict(self)
        record.pop("vector")
        record.pop("null_amplitudes")
        record.pop("poisson_amplitudes")
        record.pop("block_null_amplitudes")
        record.update(
            {
                "tracer": tracer,
                "z_min": z_min,
                "z_max": z_max,
                "n_data": n_data,
                "n_random": n_random,
                "vector_x": float(self.vector[0]),
                "vector_y": float(self.vector[1]),
                "vector_z": float(self.vector[2]),
            }
        )
        return record


def fit_dipole_map(
    sky_map: SkyMap,
    weights: Optional[NDArray[np.float64]] = None,
    pixel_indices: Optional[NDArray[np.int64]] = None,
) -> DipoleFit:
    """Fit a weighted dipole to an overdensity map."""
    valid = sky_map.valid & np.isfinite(sky_map.delta)
    if pixel_indices is None:
        selected = np.where(valid)[0]
    else:
        selected = np.asarray(pixel_indices, dtype=np.int64)
        selected = selected[valid[selected]]
    if len(selected) < 4:
        raise ValueError("At least four valid pixels are required to fit a dipole.")

    y = sky_map.delta[selected]
    vectors = sky_map.pixel_vectors[selected]
    design = np.column_stack([np.ones(len(selected)), vectors])
    if weights is None:
        fit_weights = sky_map.random_counts[selected].astype(float)
    else:
        fit_weights = np.asarray(weights, dtype=float)[selected]
    fit_weights = np.clip(fit_weights, 1e-12, None)
    sqrt_w = np.sqrt(fit_weights)
    params, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    monopole = float(params[0])
    vector = params[1:4].astype(float)
    amplitude = float(np.linalg.norm(vector))
    if amplitude > 0:
        ra, dec = unit_to_radec(vector)
        ra_value = float(np.asarray(ra))
        dec_value = float(np.asarray(dec))
    else:
        ra_value = float("nan")
        dec_value = float("nan")
    return DipoleFit(
        amplitude=amplitude,
        vector=vector,
        ra_deg=ra_value,
        dec_deg=dec_value,
        monopole=monopole,
        n_pixels=int(len(selected)),
    )


def fit_with_resampling(
    sky_map: SkyMap,
    rng: Generator,
    bootstrap_samples: int = 80,
    null_permutations: int = 120,
    poisson_mocks: int = 0,
    block_null_mocks: int = 0,
    block_null_regions: int = 12,
    jackknife_regions: int = 12,
) -> DipoleFit:
    """Fit the observed dipole and attach bootstrap, permutation, and jackknife diagnostics."""
    observed = fit_dipole_map(sky_map)
    valid_indices = np.where(sky_map.valid & np.isfinite(sky_map.delta))[0]
    bootstrap = _bootstrap_amplitudes(sky_map, valid_indices, rng, bootstrap_samples)
    null_amplitudes = permutation_null_amplitudes(sky_map, rng, null_permutations)
    p_value = (1.0 + np.sum(null_amplitudes >= observed.amplitude)) / (len(null_amplitudes) + 1.0)
    poisson_amplitudes = poisson_null_amplitudes(sky_map, rng, poisson_mocks)
    poisson_p_value = (
        (1.0 + np.sum(poisson_amplitudes >= observed.amplitude)) / (len(poisson_amplitudes) + 1.0)
        if len(poisson_amplitudes)
        else float("nan")
    )
    block_null_amplitudes = block_signflip_null_amplitudes(
        sky_map,
        rng,
        n_mocks=block_null_mocks,
        n_regions=block_null_regions,
    )
    block_null_p_value = (
        (1.0 + np.sum(block_null_amplitudes >= observed.amplitude))
        / (len(block_null_amplitudes) + 1.0)
        if len(block_null_amplitudes)
        else float("nan")
    )
    jackknife = jackknife_dipoles(sky_map, observed, jackknife_regions)

    axis_samples = _bootstrap_axes(sky_map, valid_indices, rng, bootstrap_samples)
    ra_std = _circular_std_deg(axis_samples[:, 0]) if len(axis_samples) else float("nan")
    dec_std = float(np.nanstd(axis_samples[:, 1], ddof=1)) if len(axis_samples) > 1 else float("nan")
    return DipoleFit(
        amplitude=observed.amplitude,
        vector=observed.vector,
        ra_deg=observed.ra_deg,
        dec_deg=observed.dec_deg,
        monopole=observed.monopole,
        n_pixels=observed.n_pixels,
        amplitude_std=float(np.nanstd(bootstrap, ddof=1)) if len(bootstrap) > 1 else float("nan"),
        axis_ra_std_deg=ra_std,
        axis_dec_std_deg=dec_std,
        null_p_value=float(p_value),
        null_permutations=int(len(null_amplitudes)),
        null_amplitudes=null_amplitudes,
        poisson_p_value=float(poisson_p_value),
        poisson_mocks=int(len(poisson_amplitudes)),
        poisson_amplitudes=poisson_amplitudes,
        block_null_p_value=float(block_null_p_value),
        block_null_mocks=int(len(block_null_amplitudes)),
        block_null_regions=int(block_null_regions if len(block_null_amplitudes) else 0),
        block_null_amplitudes=block_null_amplitudes,
        jackknife_regions=int(jackknife["n_regions"]),
        jackknife_amplitude_std=jackknife["amplitude_std"],
        jackknife_amplitude_min=jackknife["amplitude_min"],
        jackknife_amplitude_max=jackknife["amplitude_max"],
        jackknife_axis_median_shift_deg=jackknife["axis_median_shift_deg"],
        jackknife_axis_max_shift_deg=jackknife["axis_max_shift_deg"],
    )


def permutation_null_amplitudes(
    sky_map: SkyMap,
    rng: Generator,
    n_permutations: int,
) -> NDArray[np.float64]:
    """Shuffle valid-pixel overdensities to estimate a mask-preserving null distribution."""
    valid_indices = np.where(sky_map.valid & np.isfinite(sky_map.delta))[0]
    if n_permutations <= 0:
        return np.array([], dtype=float)
    amplitudes = np.empty(n_permutations, dtype=float)
    original = sky_map.delta.copy()
    for index in range(n_permutations):
        shuffled = original.copy()
        shuffled[valid_indices] = rng.permutation(original[valid_indices])
        permuted = SkyMap(
            nside=sky_map.nside,
            backend=sky_map.backend,
            data_counts=sky_map.data_counts,
            random_counts=sky_map.random_counts,
            alpha=sky_map.alpha,
            delta=shuffled,
            valid=sky_map.valid,
            pixel_vectors=sky_map.pixel_vectors,
        )
        amplitudes[index] = fit_dipole_map(permuted).amplitude
    return amplitudes


def poisson_null_amplitudes(
    sky_map: SkyMap,
    rng: Generator,
    n_mocks: int,
) -> NDArray[np.float64]:
    """Draw Poisson count mocks from the random-catalog selection map."""
    if n_mocks <= 0:
        return np.array([], dtype=float)
    valid_indices = np.where(sky_map.valid & np.isfinite(sky_map.delta))[0]
    if len(valid_indices) < 4:
        return np.array([], dtype=float)
    expected = sky_map.alpha * sky_map.random_counts
    amplitudes = np.empty(n_mocks, dtype=float)
    for index in range(n_mocks):
        mock_counts = np.zeros_like(expected, dtype=float)
        mock_counts[valid_indices] = rng.poisson(expected[valid_indices]).astype(float)
        delta = np.full_like(expected, np.nan, dtype=float)
        delta[valid_indices] = mock_counts[valid_indices] / expected[valid_indices] - 1.0
        mock_map = SkyMap(
            nside=sky_map.nside,
            backend=sky_map.backend,
            data_counts=mock_counts,
            random_counts=sky_map.random_counts,
            alpha=sky_map.alpha,
            delta=delta,
            valid=sky_map.valid,
            pixel_vectors=sky_map.pixel_vectors,
        )
        amplitudes[index] = fit_dipole_map(mock_map).amplitude
    return amplitudes


def block_signflip_null_amplitudes(
    sky_map: SkyMap,
    rng: Generator,
    n_mocks: int,
    n_regions: int = 12,
) -> NDArray[np.float64]:
    """Estimate a coarse spatial null by flipping overdensity signs in RA blocks.

    This diagnostic preserves the angular footprint and large contiguous residual
    structures more than a pixel permutation does. It is not a replacement for
    survey mocks, but low values here are a useful warning that a dipole-like
    residual survives coarse sky-block randomization.
    """
    if n_mocks <= 0 or n_regions <= 1:
        return np.array([], dtype=float)
    valid_indices = np.where(sky_map.valid & np.isfinite(sky_map.delta))[0]
    if len(valid_indices) < 8:
        return np.array([], dtype=float)

    ra, _ = unit_to_radec(sky_map.pixel_vectors[valid_indices])
    ordered = valid_indices[np.argsort(ra)]
    blocks = [
        block.astype(np.int64)
        for block in np.array_split(ordered, min(n_regions, len(ordered)))
        if len(block) > 0
    ]
    if len(blocks) < 2:
        return np.array([], dtype=float)

    original = sky_map.delta.copy()
    amplitudes = np.empty(n_mocks, dtype=float)
    for index in range(n_mocks):
        flipped = original.copy()
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(blocks), replace=True)
        for sign, block in zip(signs, blocks):
            flipped[block] = sign * original[block]
        mock_map = SkyMap(
            nside=sky_map.nside,
            backend=sky_map.backend,
            data_counts=sky_map.data_counts,
            random_counts=sky_map.random_counts,
            alpha=sky_map.alpha,
            delta=flipped,
            valid=sky_map.valid,
            pixel_vectors=sky_map.pixel_vectors,
        )
        amplitudes[index] = fit_dipole_map(mock_map).amplitude
    return amplitudes


def jackknife_dipoles(
    sky_map: SkyMap,
    observed: DipoleFit,
    n_regions: int,
) -> dict[str, float]:
    """Leave-one-region-out dipole diagnostics over coarse RA sky regions."""
    valid_indices = np.where(sky_map.valid & np.isfinite(sky_map.delta))[0]
    if n_regions <= 1 or len(valid_indices) < 8:
        return {
            "n_regions": 0.0,
            "amplitude_std": float("nan"),
            "amplitude_min": float("nan"),
            "amplitude_max": float("nan"),
            "axis_median_shift_deg": float("nan"),
            "axis_max_shift_deg": float("nan"),
        }

    ra, _ = unit_to_radec(sky_map.pixel_vectors[valid_indices])
    order = np.argsort(ra)
    region_members = [
        valid_indices[chunk]
        for chunk in np.array_split(order, min(n_regions, len(valid_indices)))
        if len(chunk) > 0
    ]
    amplitudes: list[float] = []
    axis_shifts: list[float] = []
    valid_mask = np.zeros(len(sky_map.delta), dtype=bool)
    valid_mask[valid_indices] = True
    for members in region_members:
        keep_mask = valid_mask.copy()
        keep_mask[members] = False
        keep_indices = np.where(keep_mask)[0]
        if len(keep_indices) < 4:
            continue
        fit = fit_dipole_map(sky_map, pixel_indices=keep_indices)
        amplitudes.append(fit.amplitude)
        shift = float(angular_separation_deg(observed.vector, fit.vector))
        axis_shifts.append(min(shift, 180.0 - shift))

    if not amplitudes:
        return {
            "n_regions": 0.0,
            "amplitude_std": float("nan"),
            "amplitude_min": float("nan"),
            "amplitude_max": float("nan"),
            "axis_median_shift_deg": float("nan"),
            "axis_max_shift_deg": float("nan"),
        }
    amplitude_values = np.asarray(amplitudes, dtype=float)
    shift_values = np.asarray(axis_shifts, dtype=float)
    return {
        "n_regions": float(len(amplitude_values)),
        "amplitude_std": float(np.nanstd(amplitude_values, ddof=1)) if len(amplitude_values) > 1 else float("nan"),
        "amplitude_min": float(np.nanmin(amplitude_values)),
        "amplitude_max": float(np.nanmax(amplitude_values)),
        "axis_median_shift_deg": float(np.nanmedian(shift_values)),
        "axis_max_shift_deg": float(np.nanmax(shift_values)),
    }


def jackknife_region_table(sky_map: SkyMap, n_regions: int = 12) -> pd.DataFrame:
    """Return leave-one-RA-region-out dipole fits with region locations."""
    observed = fit_dipole_map(sky_map)
    valid_indices = np.where(sky_map.valid & np.isfinite(sky_map.delta))[0]
    if n_regions <= 1 or len(valid_indices) < 8:
        return pd.DataFrame()

    ra, dec = unit_to_radec(sky_map.pixel_vectors[valid_indices])
    order = np.argsort(ra)
    chunks = [
        valid_indices[chunk]
        for chunk in np.array_split(order, min(n_regions, len(valid_indices)))
        if len(chunk) > 0
    ]
    rows: list[dict[str, float]] = []
    valid_mask = np.zeros(len(sky_map.delta), dtype=bool)
    valid_mask[valid_indices] = True
    for index, members in enumerate(chunks):
        keep_mask = valid_mask.copy()
        keep_mask[members] = False
        keep_indices = np.where(keep_mask)[0]
        if len(keep_indices) < 4:
            continue
        fit = fit_dipole_map(sky_map, pixel_indices=keep_indices)
        shift = float(angular_separation_deg(observed.vector, fit.vector))
        member_ra, member_dec = unit_to_radec(sky_map.pixel_vectors[members])
        member_random = sky_map.random_counts[members]
        rows.append(
            {
                "region_index": float(index),
                "removed_pixels": float(len(members)),
                "removed_random_weight": float(np.sum(member_random)),
                "removed_ra_min_deg": float(np.nanmin(member_ra)),
                "removed_ra_max_deg": float(np.nanmax(member_ra)),
                "removed_dec_min_deg": float(np.nanmin(member_dec)),
                "removed_dec_max_deg": float(np.nanmax(member_dec)),
                "leave_one_out_amplitude": fit.amplitude,
                "leave_one_out_ra_deg": fit.ra_deg,
                "leave_one_out_dec_deg": fit.dec_deg,
                "amplitude_change": fit.amplitude - observed.amplitude,
                "axis_shift_deg": min(shift, 180.0 - shift),
                "observed_amplitude": observed.amplitude,
                "observed_ra_deg": observed.ra_deg,
                "observed_dec_deg": observed.dec_deg,
            }
        )
    return pd.DataFrame(rows).sort_values("axis_shift_deg", ascending=False).reset_index(drop=True)


def _bootstrap_amplitudes(
    sky_map: SkyMap,
    valid_indices: NDArray[np.int64],
    rng: Generator,
    n_samples: int,
) -> NDArray[np.float64]:
    amplitudes = np.empty(max(n_samples, 0), dtype=float)
    for index in range(len(amplitudes)):
        sample = rng.choice(valid_indices, size=len(valid_indices), replace=True)
        amplitudes[index] = fit_dipole_map(sky_map, pixel_indices=sample).amplitude
    return amplitudes


def _bootstrap_axes(
    sky_map: SkyMap,
    valid_indices: NDArray[np.int64],
    rng: Generator,
    n_samples: int,
) -> NDArray[np.float64]:
    axes = []
    for _ in range(max(n_samples, 0)):
        sample = rng.choice(valid_indices, size=len(valid_indices), replace=True)
        fit = fit_dipole_map(sky_map, pixel_indices=sample)
        axes.append([fit.ra_deg, fit.dec_deg])
    return np.asarray(axes, dtype=float)


def _circular_std_deg(values: NDArray[np.float64]) -> float:
    radians = np.deg2rad(values)
    resultant = np.abs(np.mean(np.exp(1j * radians)))
    if resultant <= 0:
        return 180.0
    return float(np.rad2deg(np.sqrt(max(-2.0 * np.log(resultant), 0.0))))
