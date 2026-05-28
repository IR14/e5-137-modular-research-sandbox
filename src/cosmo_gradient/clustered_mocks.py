"""Clustered lognormal mock calibration for sky-map dipole tests."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, load_sky_map
from cosmo_gradient.systematics import load_external_template_maps, regress_templates

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LognormalMockOutputs:
    """Paths written by a clustered lognormal mock calibration run."""

    null_csv: Path
    summary_csv: Path
    report: Path


def run_lognormal_mock_calibration(
    map_path: Path,
    external_templates: Sequence[str],
    output_prefix: Path,
    mocks: int = 1000,
    sigmas: Sequence[str | float] = ("auto",),
    smoothing_deg: float = 8.0,
    cl_slope: float = 1.4,
    lmax: int | None = None,
    seed: int = 20260525,
) -> LognormalMockOutputs:
    """Calibrate dipole amplitudes with mask-matched clustered lognormal mocks.

    The mocks preserve the survey window through the random-count map and add a
    correlated angular density field before Poisson sampling. This is still a
    parametric approximation; it is stronger than a shot-noise-only Poisson null
    but not a substitute for official DESI mocks.
    """
    sky_map = load_sky_map(str(map_path))
    resolved_lmax = int(lmax if lmax is not None else max(3 * sky_map.nside - 1, 1))
    sigma_values = resolve_lognormal_sigmas(sky_map, sigmas)
    template_matrix, template_names = load_external_template_maps(
        external_templates,
        valid=sky_map.valid & np.isfinite(sky_map.delta),
        expected_npix=len(sky_map.delta),
    )
    observed_raw = fit_dipole_map(sky_map)
    observed_regression = regress_templates(sky_map, template_matrix, template_names)
    observed_corrected = fit_dipole_map(observed_regression.corrected_map)

    rng = default_rng(seed)
    frames = []
    for sigma_label, sigma in sigma_values:
        LOGGER.info(
            "Running lognormal mocks for %s: sigma=%s mocks=%d",
            map_path.name,
            sigma_label,
            mocks,
        )
        frames.append(
            _run_lognormal_null_mocks(
                sky_map=sky_map,
                template_matrix=template_matrix,
                template_names=template_names,
                rng=rng,
                n_mocks=mocks,
                sigma=sigma,
                sigma_label=sigma_label,
                smoothing_deg=smoothing_deg,
                cl_slope=cl_slope,
                lmax=resolved_lmax,
            )
        )
    null = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    null = _attach_observed_p_values(
        null,
        observed_raw_amplitude=observed_raw.amplitude,
        observed_corrected_amplitude=observed_corrected.amplitude,
    )
    summary = _summarize_lognormal_null(
        null,
        observed_raw=observed_raw,
        observed_corrected=observed_corrected,
        n_templates=len(template_names),
        map_path=map_path,
    )

    null_csv = output_prefix.with_name(f"{output_prefix.name}_null_mocks.csv")
    summary_csv = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    report_parent = output_prefix.parent
    if report_parent.name == "tables":
        report_parent = report_parent.parent / "reports"
    report = report_parent / f"{output_prefix.name}.md"
    for path in [null_csv, summary_csv, report]:
        path.parent.mkdir(parents=True, exist_ok=True)
    null.to_csv(null_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _write_lognormal_report(
        report_path=report,
        map_path=map_path,
        summary=summary,
        mocks=mocks,
        sigmas=sigma_values,
        smoothing_deg=smoothing_deg,
        cl_slope=cl_slope,
        lmax=resolved_lmax,
        n_templates=len(template_names),
    )
    return LognormalMockOutputs(null_csv=null_csv, summary_csv=summary_csv, report=report)


def lognormal_mock_map(
    sky_map: SkyMap,
    rng: Generator,
    sigma: float,
    smoothing_deg: float = 8.0,
    cl_slope: float = 1.4,
    lmax: int | None = None,
) -> SkyMap:
    """Draw one clustered lognormal count mock from a map's selection function."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    expected = sky_map.alpha * sky_map.random_counts
    valid = sky_map.valid & np.isfinite(sky_map.delta) & (expected > 0.0)
    density = np.ones_like(expected, dtype=float)
    if sigma > 0.0:
        field = correlated_gaussian_field(
            sky_map=sky_map,
            rng=rng,
            smoothing_deg=smoothing_deg,
            cl_slope=cl_slope,
            lmax=lmax,
        )
        density[valid] = np.exp(sigma * field[valid])
        mean_density = float(np.average(density[valid], weights=expected[valid]))
        if mean_density <= 0.0 or not np.isfinite(mean_density):
            raise ValueError("Invalid lognormal density normalization.")
        density[valid] /= mean_density

    rate = expected * density
    mock_counts = np.zeros_like(expected, dtype=float)
    mock_counts[valid] = rng.poisson(np.clip(rate[valid], 0.0, None)).astype(float)
    delta = np.full_like(expected, np.nan, dtype=float)
    delta[valid] = mock_counts[valid] / expected[valid] - 1.0
    return SkyMap(
        nside=sky_map.nside,
        backend=sky_map.backend,
        data_counts=mock_counts,
        random_counts=sky_map.random_counts,
        alpha=sky_map.alpha,
        delta=delta,
        valid=sky_map.valid,
        pixel_vectors=sky_map.pixel_vectors,
    )


def correlated_gaussian_field(
    sky_map: SkyMap,
    rng: Generator,
    smoothing_deg: float = 8.0,
    cl_slope: float = 1.4,
    lmax: int | None = None,
) -> NDArray[np.float64]:
    """Generate a standardized correlated Gaussian field on the map pixels."""
    expected = sky_map.alpha * sky_map.random_counts
    valid = sky_map.valid & np.isfinite(sky_map.delta) & (expected > 0.0)
    if np.sum(valid) < 4:
        raise ValueError("At least four valid pixels are required for clustered mocks.")
    if sky_map.backend == "healpy":
        try:
            raw = _healpy_gaussian_field(
                nside=sky_map.nside,
                rng=rng,
                smoothing_deg=smoothing_deg,
                cl_slope=cl_slope,
                lmax=lmax,
            )
            return _standardize_field(raw, valid=valid, weights=expected)
        except ImportError:
            pass
    raw = _kernel_gaussian_field(sky_map.pixel_vectors, rng=rng, smoothing_deg=smoothing_deg)
    return _standardize_field(raw, valid=valid, weights=expected)


def estimate_excess_lognormal_sigma(sky_map: SkyMap) -> float:
    """Estimate a crude clustered-field RMS after subtracting Poisson variance."""
    expected = sky_map.alpha * sky_map.random_counts
    valid = sky_map.valid & np.isfinite(sky_map.delta) & (expected > 0.0)
    if np.sum(valid) < 4:
        return 0.0
    weights = sky_map.random_counts[valid].astype(float)
    delta = sky_map.delta[valid].astype(float)
    mean = float(np.average(delta, weights=weights))
    observed_var = float(np.average((delta - mean) ** 2, weights=weights))
    shot_var = float(np.average(1.0 / expected[valid], weights=weights))
    return float(np.sqrt(max(observed_var - shot_var, 0.0)))


def resolve_lognormal_sigmas(
    sky_map: SkyMap,
    specs: Sequence[str | float],
) -> list[tuple[str, float]]:
    """Resolve CLI sigma specs, supporting ``auto`` from the observed map."""
    if not specs:
        specs = ("auto",)
    resolved: list[tuple[str, float]] = []
    for spec in specs:
        if isinstance(spec, str) and spec.strip().lower() == "auto":
            sigma = estimate_excess_lognormal_sigma(sky_map)
            resolved.append(("auto", sigma))
            continue
        sigma = float(spec)
        if sigma < 0.0:
            raise ValueError("Lognormal sigma values must be non-negative.")
        resolved.append((f"{sigma:g}", sigma))
    return resolved


def _run_lognormal_null_mocks(
    sky_map: SkyMap,
    template_matrix: NDArray[np.float64],
    template_names: Sequence[str],
    rng: Generator,
    n_mocks: int,
    sigma: float,
    sigma_label: str,
    smoothing_deg: float,
    cl_slope: float,
    lmax: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for mock_index in range(n_mocks):
        mock = lognormal_mock_map(
            sky_map=sky_map,
            rng=rng,
            sigma=sigma,
            smoothing_deg=smoothing_deg,
            cl_slope=cl_slope,
            lmax=lmax,
        )
        raw = fit_dipole_map(mock)
        corrected_map = regress_templates(mock, template_matrix, template_names).corrected_map
        corrected = fit_dipole_map(corrected_map)
        records.extend(
            [
                _mock_record(
                    mock_index=mock_index,
                    sigma_label=sigma_label,
                    sigma=sigma,
                    smoothing_deg=smoothing_deg,
                    cl_slope=cl_slope,
                    lmax=lmax,
                    stage="raw_no_template_regression",
                    fit=raw,
                ),
                _mock_record(
                    mock_index=mock_index,
                    sigma_label=sigma_label,
                    sigma=sigma,
                    smoothing_deg=smoothing_deg,
                    cl_slope=cl_slope,
                    lmax=lmax,
                    stage="after_external_template_regression",
                    fit=corrected,
                ),
            ]
        )
    return pd.DataFrame.from_records(records)


def _mock_record(
    mock_index: int,
    sigma_label: str,
    sigma: float,
    smoothing_deg: float,
    cl_slope: float,
    lmax: int,
    stage: str,
    fit,
) -> dict[str, object]:
    return {
        "mock_index": mock_index,
        "mock_family": "lognormal_angular",
        "sigma_label": sigma_label,
        "sigma_lognormal": sigma,
        "smoothing_deg": smoothing_deg,
        "cl_slope": cl_slope,
        "lmax": lmax,
        "stage": stage,
        "amplitude": fit.amplitude,
        "ra_deg": fit.ra_deg,
        "dec_deg": fit.dec_deg,
        "vector_x": float(fit.vector[0]),
        "vector_y": float(fit.vector[1]),
        "vector_z": float(fit.vector[2]),
    }


def _attach_observed_p_values(
    null: pd.DataFrame,
    observed_raw_amplitude: float,
    observed_corrected_amplitude: float,
) -> pd.DataFrame:
    frame = null.copy()
    observed = {
        "raw_no_template_regression": observed_raw_amplitude,
        "after_external_template_regression": observed_corrected_amplitude,
    }
    p_values: dict[tuple[str, str], float] = {}
    for (sigma_label, stage), group in frame.groupby(["sigma_label", "stage"]):
        values = group["amplitude"].to_numpy(dtype=float)
        amplitude = observed[str(stage)]
        p_values[(str(sigma_label), str(stage))] = (
            (1.0 + np.sum(values >= amplitude)) / (len(values) + 1.0)
        )
    frame["observed_empirical_p_value"] = [
        p_values[(str(row.sigma_label), str(row.stage))] for row in frame.itertuples()
    ]
    return frame


def _summarize_lognormal_null(
    null: pd.DataFrame,
    observed_raw,
    observed_corrected,
    n_templates: int,
    map_path: Path,
) -> pd.DataFrame:
    observed = {
        "raw_no_template_regression": observed_raw,
        "after_external_template_regression": observed_corrected,
    }
    rows: list[dict[str, object]] = []
    for (sigma_label, stage), group in null.groupby(["sigma_label", "stage"]):
        values = group["amplitude"].to_numpy(dtype=float)
        fit = observed[str(stage)]
        rows.append(
            {
                "mock_family": "lognormal_angular",
                "map_path": str(map_path),
                "sigma_label": sigma_label,
                "sigma_lognormal": float(group["sigma_lognormal"].iloc[0]),
                "smoothing_deg": float(group["smoothing_deg"].iloc[0]),
                "cl_slope": float(group["cl_slope"].iloc[0]),
                "lmax": int(group["lmax"].iloc[0]),
                "stage": stage,
                "n_mocks": len(group),
                "n_templates": n_templates,
                "null_amplitude_median": float(np.median(values)),
                "null_amplitude_p68": float(np.quantile(values, 0.68)),
                "null_amplitude_p95": float(np.quantile(values, 0.95)),
                "null_amplitude_p99": float(np.quantile(values, 0.99)),
                "observed_amplitude": fit.amplitude,
                "observed_ra_deg": fit.ra_deg,
                "observed_dec_deg": fit.dec_deg,
                "observed_empirical_p_value": (
                    (1.0 + np.sum(values >= fit.amplitude)) / (len(values) + 1.0)
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_lognormal_report(
    report_path: Path,
    map_path: Path,
    summary: pd.DataFrame,
    mocks: int,
    sigmas: Sequence[tuple[str, float]],
    smoothing_deg: float,
    cl_slope: float,
    lmax: int,
    n_templates: int,
) -> None:
    brief = summary.copy()
    for column in brief.select_dtypes(include="number").columns:
        if column not in {"n_mocks", "n_templates", "lmax"}:
            brief[column] = brief[column].map(lambda value: f"{value:.6g}")
    sigma_text = ", ".join(f"{label}={value:.6g}" for label, value in sigmas)
    lines = [
        "# Clustered lognormal mock calibration",
        "",
        f"- Map: `{map_path}`",
        "- Mock family: angular lognormal field plus Poisson sampling",
        f"- Mocks per sigma: {mocks}",
        f"- Sigma values: {sigma_text}",
        f"- Smoothing scale: {smoothing_deg:g} deg",
        f"- Power-spectrum slope: {cl_slope:g}",
        f"- lmax: {lmax}",
        f"- External templates regressed: {n_templates}",
        "",
        "These mocks preserve the angular selection function encoded by the DESI",
        "random-count map and add correlated angular density structure before",
        "Poisson sampling. They are meant to reduce the overconfidence of",
        "shot-noise-only p-values.",
        "",
        "They are not official DESI mocks and do not encode the full DESI",
        "selection, radial evolution, reconstruction, fiber assignment, or survey",
        "covariance model.",
        "",
        "## Summary",
        "",
        brief.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Compare these p-values with the earlier Poisson-only calibration; a large",
        "increase means the old null was too narrow.",
        "- If a signal remains unusual over a plausible sigma grid, the next step is",
        "official DESI-like mocks or a tuned 3D lognormal/mock-catalog generator.",
        "- Template-regressed rows should be read together with injection-recovery",
        "tests, because template regression can suppress real dipole power.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _healpy_gaussian_field(
    nside: int,
    rng: Generator,
    smoothing_deg: float,
    cl_slope: float,
    lmax: int | None,
) -> NDArray[np.float64]:
    import healpy as hp

    resolved_lmax = int(lmax if lmax is not None else max(3 * nside - 1, 1))
    cl = _power_law_cl(resolved_lmax, smoothing_deg=smoothing_deg, cl_slope=cl_slope)
    state = np.random.get_state()
    healpy_logger = logging.getLogger("healpy")
    previous_level = healpy_logger.level
    healpy_logger.setLevel(logging.WARNING)
    np.random.seed(int(rng.integers(0, np.iinfo(np.uint32).max)))
    try:
        return np.asarray(
            hp.synfast(cl, nside=nside, lmax=resolved_lmax, new=True),
            dtype=float,
        )
    finally:
        np.random.set_state(state)
        healpy_logger.setLevel(previous_level)


def _power_law_cl(
    lmax: int,
    smoothing_deg: float,
    cl_slope: float,
) -> NDArray[np.float64]:
    ells = np.arange(lmax + 1, dtype=float)
    cl = np.zeros(lmax + 1, dtype=float)
    selected = ells >= 1
    cl[selected] = np.power(ells[selected], -float(cl_slope))
    if smoothing_deg > 0.0:
        sigma_rad = np.deg2rad(smoothing_deg) / np.sqrt(8.0 * np.log(2.0))
        cl[selected] *= np.exp(-ells[selected] * (ells[selected] + 1.0) * sigma_rad**2)
    return cl


def _kernel_gaussian_field(
    vectors: NDArray[np.float64],
    rng: Generator,
    smoothing_deg: float,
) -> NDArray[np.float64]:
    if len(vectors) > 2048:
        raise RuntimeError("healpy is required for lognormal mocks with more than 2048 pixels.")
    cos_theta = np.clip(vectors @ vectors.T, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    scale = max(np.deg2rad(smoothing_deg), 1e-3)
    covariance = np.exp(-0.5 * (theta / scale) ** 2)
    covariance.flat[:: len(vectors) + 1] += 1e-8
    return rng.multivariate_normal(np.zeros(len(vectors)), covariance).astype(float)


def _standardize_field(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    field = np.asarray(values, dtype=float).copy()
    selected = valid & np.isfinite(field)
    if np.sum(selected) < 4:
        raise ValueError("Generated field has too few valid pixels.")
    selected_weights = np.clip(weights[selected], 1e-12, None)
    mean = float(np.average(field[selected], weights=selected_weights))
    field -= mean
    variance = float(np.average(field[selected] ** 2, weights=selected_weights))
    if variance <= 0.0 or not np.isfinite(variance):
        raise ValueError("Generated field has zero variance on the survey footprint.")
    field /= np.sqrt(variance)
    return field
