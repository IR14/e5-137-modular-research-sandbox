"""Low-order angular mode diagnostics for sky overdensity maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cosmo_gradient.coords import angular_separation_deg, unit_to_radec
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, load_sky_map


@dataclass(frozen=True)
class LinearModeFit:
    """Weighted linear fit of low-order angular templates."""

    model_name: str
    amplitude: float
    ra_deg: float
    dec_deg: float
    vector: NDArray[np.float64]
    monopole: float
    coefficients: NDArray[np.float64]
    coefficient_names: list[str]
    n_pixels: int
    weighted_r2: float
    residual_std: float
    weighted_design_condition: float


def fit_dipole_quadrupole_map(sky_map: SkyMap) -> LinearModeFit:
    """Fit monopole, dipole, and five quadrupole-like templates to a sky map."""
    valid = sky_map.valid & np.isfinite(sky_map.delta)
    selected = np.where(valid)[0]
    if len(selected) < 9:
        raise ValueError("At least nine valid pixels are required for dipole+quadrupole fitting.")

    y = sky_map.delta[selected]
    vectors = sky_map.pixel_vectors[selected]
    x = vectors[:, 0]
    yv = vectors[:, 1]
    z = vectors[:, 2]
    qxx = x * x - 1.0 / 3.0
    qyy = yv * yv - 1.0 / 3.0
    qxy = x * yv
    qxz = x * z
    qyz = yv * z
    design = np.column_stack([np.ones(len(selected)), x, yv, z, qxx, qyy, qxy, qxz, qyz])
    names = ["monopole", "dipole_x", "dipole_y", "dipole_z", "q_xx", "q_yy", "q_xy", "q_xz", "q_yz"]
    weights = sky_map.random_counts[selected]
    params = _weighted_lstsq(design, y, weights)
    vector = params[1:4].astype(float)
    amplitude = float(np.linalg.norm(vector))
    if amplitude > 0:
        ra, dec = unit_to_radec(vector)
        ra_deg = float(np.asarray(ra))
        dec_deg = float(np.asarray(dec))
    else:
        ra_deg = float("nan")
        dec_deg = float("nan")
    model = design @ params
    weighted_r2 = _weighted_r2(y, model, weights)
    condition = _weighted_design_condition(design, weights)
    residual_std = float(np.nanstd(y - model, ddof=1)) if len(y) > 1 else float("nan")
    return LinearModeFit(
        model_name="dipole_plus_quadrupole",
        amplitude=amplitude,
        ra_deg=ra_deg,
        dec_deg=dec_deg,
        vector=vector,
        monopole=float(params[0]),
        coefficients=params,
        coefficient_names=names,
        n_pixels=int(len(selected)),
        weighted_r2=weighted_r2,
        residual_std=residual_std,
        weighted_design_condition=condition,
    )


def map_multipole_diagnostics(sky_map: SkyMap) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return summary and coefficient tables for dipole-only vs quadrupole-marginalized fits."""
    dipole = fit_dipole_map(sky_map)
    quadrupole = fit_dipole_quadrupole_map(sky_map)
    sep = float(angular_separation_deg(dipole.vector, quadrupole.vector))
    sep = min(sep, 180.0 - sep)
    quadrupole_norm = float(np.linalg.norm(quadrupole.coefficients[4:]))
    summary = pd.DataFrame(
        [
            {
                "model": "dipole_only",
                "dipole_amplitude": dipole.amplitude,
                "ra_deg": dipole.ra_deg,
                "dec_deg": dipole.dec_deg,
                "monopole": dipole.monopole,
                "n_pixels": dipole.n_pixels,
                "weighted_r2": np.nan,
                "residual_std": np.nan,
                "weighted_design_condition": np.nan,
                "quadrupole_norm": 0.0,
                "axis_shift_from_dipole_only_deg": 0.0,
                "amplitude_ratio_to_dipole_only": 1.0,
            },
            {
                "model": "dipole_plus_quadrupole",
                "dipole_amplitude": quadrupole.amplitude,
                "ra_deg": quadrupole.ra_deg,
                "dec_deg": quadrupole.dec_deg,
                "monopole": quadrupole.monopole,
                "n_pixels": quadrupole.n_pixels,
                "weighted_r2": quadrupole.weighted_r2,
                "residual_std": quadrupole.residual_std,
                "weighted_design_condition": quadrupole.weighted_design_condition,
                "quadrupole_norm": quadrupole_norm,
                "axis_shift_from_dipole_only_deg": sep,
                "amplitude_ratio_to_dipole_only": (
                    quadrupole.amplitude / dipole.amplitude if dipole.amplitude > 0 else np.nan
                ),
            },
        ]
    )
    coefficients = pd.DataFrame(
        {
            "model": quadrupole.model_name,
            "term": quadrupole.coefficient_names,
            "coefficient": quadrupole.coefficients.astype(float),
        }
    )
    return summary, coefficients


def write_map_multipole_diagnostics(
    map_path: Path,
    output_prefix: str,
    tables_dir: Path,
    reports_dir: Path,
) -> tuple[Path, Path, Path]:
    """Load a saved map, write multipole diagnostics, and return artifact paths."""
    sky_map = load_sky_map(str(map_path))
    summary, coefficients = map_multipole_diagnostics(sky_map)
    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    summary_path = tables_dir / f"{output_prefix}.csv"
    coefficient_path = tables_dir / f"{output_prefix}_coefficients.csv"
    report_path = reports_dir / f"{output_prefix}.md"
    summary.to_csv(summary_path, index=False)
    coefficients.to_csv(coefficient_path, index=False)
    _write_report(report_path, map_path, summary, coefficients)
    return summary_path, coefficient_path, report_path


def _weighted_lstsq(
    design: NDArray[np.float64],
    y: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    fit_weights = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    sqrt_w = np.sqrt(fit_weights)
    params, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    return params.astype(float)


def _weighted_r2(
    y: NDArray[np.float64],
    model: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    fit_weights = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    y_bar = float(np.average(y, weights=fit_weights))
    total = float(np.sum(fit_weights * (y - y_bar) ** 2))
    if total <= 0:
        return float("nan")
    residual = float(np.sum(fit_weights * (y - model) ** 2))
    return float(1.0 - residual / total)


def _weighted_design_condition(
    design: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> float:
    fit_weights = np.clip(np.asarray(weights, dtype=float), 1e-12, None)
    sqrt_w = np.sqrt(fit_weights)
    return float(np.linalg.cond(design * sqrt_w[:, None]))


def _write_report(
    report_path: Path,
    map_path: Path,
    summary: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> None:
    lines = [
        "# Multipole diagnostics",
        "",
        f"- Map: `{map_path}`",
        "",
        "This diagnostic compares the standard dipole-only fit with a fit that also",
        "marginalizes over five quadrupole-like angular templates. If the dipole amplitude",
        "or axis changes strongly, the candidate is likely entangled with broader footprint",
        "or low-order angular structure rather than behaving as an isolated dipole.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Coefficients",
        "",
        coefficients.to_markdown(index=False),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
