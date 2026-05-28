"""Selection-preserving mock calibration for dipole estimators."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from cosmo_gradient.coords import angular_separation_deg, radec_to_unit
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, load_sky_map
from cosmo_gradient.systematics import load_external_template_maps, regress_templates


@dataclass(frozen=True)
class CalibrationOutputs:
    """Paths written by a mock calibration run."""

    null_csv: Path
    injections_csv: Path
    summary_csv: Path
    report: Path


def run_mock_calibration(
    map_path: Path,
    external_templates: Sequence[str],
    output_prefix: Path,
    amplitudes: Sequence[float],
    axes: Sequence[tuple[str, float, float]],
    null_mocks: int = 1000,
    injection_mocks: int = 300,
    seed: int = 20260525,
    detection_alpha: float = 0.05,
) -> CalibrationOutputs:
    """Calibrate raw and template-regressed dipole amplitudes with Poisson mocks.

    These mocks preserve the survey window encoded by the random-count map. They
    do not include realistic large-scale structure covariance, so they are an
    intermediate calibration step rather than a replacement for official mocks.
    """
    sky_map = load_sky_map(str(map_path))
    template_matrix, template_names = load_external_template_maps(
        external_templates,
        valid=sky_map.valid & np.isfinite(sky_map.delta),
        expected_npix=len(sky_map.delta),
    )
    rng = default_rng(seed)
    observed_raw = fit_dipole_map(sky_map)
    observed_regression = regress_templates(sky_map, template_matrix, template_names)
    observed_corrected = fit_dipole_map(observed_regression.corrected_map)

    null = _run_null_mocks(
        sky_map=sky_map,
        template_matrix=template_matrix,
        template_names=template_names,
        rng=rng,
        n_mocks=null_mocks,
    )
    thresholds = _stage_thresholds(null, detection_alpha=detection_alpha)
    null = _attach_empirical_p_values(
        null,
        observed_raw_amplitude=observed_raw.amplitude,
        observed_corrected_amplitude=observed_corrected.amplitude,
    )

    injections = _run_injection_mocks(
        sky_map=sky_map,
        template_matrix=template_matrix,
        template_names=template_names,
        rng=rng,
        amplitudes=amplitudes,
        axes=axes,
        n_mocks=injection_mocks,
        thresholds=thresholds,
    )
    summary = _summarize_calibration(
        injections,
        null,
        observed_raw=observed_raw,
        observed_corrected=observed_corrected,
        thresholds=thresholds,
        detection_alpha=detection_alpha,
    )

    null_csv = output_prefix.with_name(f"{output_prefix.name}_null_mocks.csv")
    injections_csv = output_prefix.with_name(f"{output_prefix.name}_injection_mocks.csv")
    summary_csv = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    report_parent = output_prefix.parent
    if report_parent.name == "tables":
        report_parent = report_parent.parent / "reports"
    report = report_parent / f"{output_prefix.name}.md"
    for path in [null_csv, injections_csv, summary_csv, report]:
        path.parent.mkdir(parents=True, exist_ok=True)
    null.to_csv(null_csv, index=False)
    injections.to_csv(injections_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _write_calibration_report(
        report,
        map_path=map_path,
        null=null,
        injections=injections,
        summary=summary,
        n_templates=len(template_names),
        detection_alpha=detection_alpha,
    )
    return CalibrationOutputs(
        null_csv=null_csv,
        injections_csv=injections_csv,
        summary_csv=summary_csv,
        report=report,
    )


def poisson_mock_map(
    sky_map: SkyMap,
    rng: Generator,
    amplitude: float = 0.0,
    axis_vector: NDArray[np.float64] | None = None,
) -> SkyMap:
    """Draw a Poisson mock from the random-count selection map."""
    expected = sky_map.alpha * sky_map.random_counts
    rate = expected.copy()
    valid = sky_map.valid & np.isfinite(sky_map.delta) & (expected > 0.0)
    if amplitude != 0.0:
        if axis_vector is None:
            raise ValueError("axis_vector is required for non-zero injected amplitude.")
        axis = np.asarray(axis_vector, dtype=float)
        axis = axis / np.linalg.norm(axis)
        modulation = 1.0 + amplitude * (sky_map.pixel_vectors @ axis)
        rate = expected * np.clip(modulation, 0.0, None)
    mock_counts = np.zeros_like(expected, dtype=float)
    mock_counts[valid] = rng.poisson(rate[valid]).astype(float)
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


def _run_null_mocks(
    sky_map: SkyMap,
    template_matrix: NDArray[np.float64],
    template_names: Sequence[str],
    rng: Generator,
    n_mocks: int,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for mock_index in range(n_mocks):
        mock = poisson_mock_map(sky_map, rng)
        raw = fit_dipole_map(mock)
        corrected_map = regress_templates(mock, template_matrix, template_names).corrected_map
        corrected = fit_dipole_map(corrected_map)
        records.extend(
            [
                _fit_record(mock_index, "raw_no_template_regression", raw),
                _fit_record(mock_index, "after_external_template_regression", corrected),
            ]
        )
    return pd.DataFrame.from_records(records)


def _run_injection_mocks(
    sky_map: SkyMap,
    template_matrix: NDArray[np.float64],
    template_names: Sequence[str],
    rng: Generator,
    amplitudes: Sequence[float],
    axes: Sequence[tuple[str, float, float]],
    n_mocks: int,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for axis_name, axis_ra, axis_dec in axes:
        axis_vector = radec_to_unit([axis_ra], [axis_dec])[0]
        for amplitude in amplitudes:
            for mock_index in range(n_mocks):
                mock = poisson_mock_map(
                    sky_map,
                    rng,
                    amplitude=float(amplitude),
                    axis_vector=axis_vector,
                )
                raw = fit_dipole_map(mock)
                regression = regress_templates(mock, template_matrix, template_names)
                corrected_map = regression.corrected_map
                corrected = fit_dipole_map(corrected_map)
                records.extend(
                    [
                        _injection_fit_record(
                            mock_index=mock_index,
                            stage="raw_no_template_regression",
                            fit=raw,
                            injected_amplitude=float(amplitude),
                            axis_name=axis_name,
                            axis_ra=axis_ra,
                            axis_dec=axis_dec,
                            axis_vector=axis_vector,
                            threshold=thresholds["raw_no_template_regression"],
                        ),
                        _injection_fit_record(
                            mock_index=mock_index,
                            stage="after_external_template_regression",
                            fit=corrected,
                            injected_amplitude=float(amplitude),
                            axis_name=axis_name,
                            axis_ra=axis_ra,
                            axis_dec=axis_dec,
                            axis_vector=axis_vector,
                            threshold=thresholds["after_external_template_regression"],
                        ),
                    ]
                )
    return pd.DataFrame.from_records(records)


def _fit_record(mock_index: int, stage: str, fit) -> dict[str, object]:
    return {
        "mock_index": mock_index,
        "stage": stage,
        "amplitude": fit.amplitude,
        "ra_deg": fit.ra_deg,
        "dec_deg": fit.dec_deg,
        "vector_x": float(fit.vector[0]),
        "vector_y": float(fit.vector[1]),
        "vector_z": float(fit.vector[2]),
    }


def _injection_fit_record(
    mock_index: int,
    stage: str,
    fit,
    injected_amplitude: float,
    axis_name: str,
    axis_ra: float,
    axis_dec: float,
    axis_vector: NDArray[np.float64],
    threshold: float,
) -> dict[str, object]:
    projection = float(np.dot(fit.vector, axis_vector))
    axis_error = (
        float(angular_separation_deg(fit.vector, axis_vector))
        if fit.amplitude > 0
        else np.nan
    )
    return {
        "mock_index": mock_index,
        "axis_name": axis_name,
        "axis_ra_deg": axis_ra,
        "axis_dec_deg": axis_dec,
        "stage": stage,
        "injected_amplitude": injected_amplitude,
        "fit_amplitude": fit.amplitude,
        "fit_ra_deg": fit.ra_deg,
        "fit_dec_deg": fit.dec_deg,
        "axis_error_deg": axis_error,
        "axis_projection": projection,
        "projection_recovery_fraction": (
            projection / injected_amplitude if injected_amplitude > 0 else np.nan
        ),
        "detected_at_threshold": bool(fit.amplitude >= threshold),
        "detection_threshold": threshold,
    }


def _stage_thresholds(null: pd.DataFrame, detection_alpha: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for stage, group in null.groupby("stage"):
        thresholds[str(stage)] = float(group["amplitude"].quantile(1.0 - detection_alpha))
    return thresholds


def _attach_empirical_p_values(
    null: pd.DataFrame,
    observed_raw_amplitude: float,
    observed_corrected_amplitude: float,
) -> pd.DataFrame:
    frame = null.copy()
    observed = {
        "raw_no_template_regression": observed_raw_amplitude,
        "after_external_template_regression": observed_corrected_amplitude,
    }
    p_values = {}
    for stage, amplitude in observed.items():
        values = frame.loc[frame["stage"].eq(stage), "amplitude"].to_numpy(dtype=float)
        p_values[stage] = (1.0 + np.sum(values >= amplitude)) / (len(values) + 1.0)
    frame["observed_empirical_p_value"] = frame["stage"].map(p_values)
    return frame


def _summarize_calibration(
    injections: pd.DataFrame,
    null: pd.DataFrame,
    observed_raw,
    observed_corrected,
    thresholds: dict[str, float],
    detection_alpha: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed = {
        "raw_no_template_regression": observed_raw,
        "after_external_template_regression": observed_corrected,
    }
    for stage, group in null.groupby("stage"):
        fit = observed[str(stage)]
        values = group["amplitude"].to_numpy(dtype=float)
        rows.append(
            {
                "row_type": "null_mocks",
                "axis_name": "",
                "stage": stage,
                "injected_amplitude": 0.0,
                "n_mocks": len(group),
                "threshold_alpha": detection_alpha,
                "detection_threshold": thresholds[str(stage)],
                "null_amplitude_median": float(np.median(values)),
                "null_amplitude_p95": float(np.quantile(values, 0.95)),
                "observed_amplitude": fit.amplitude,
                "observed_ra_deg": fit.ra_deg,
                "observed_dec_deg": fit.dec_deg,
                "observed_empirical_p_value": (
                    (1.0 + np.sum(values >= fit.amplitude)) / (len(values) + 1.0)
                ),
                "detection_efficiency": np.nan,
                "median_projection_recovery_fraction": np.nan,
                "median_axis_error_deg": np.nan,
            }
        )
    group_columns = ["axis_name", "stage", "injected_amplitude"]
    for key, group in injections.groupby(group_columns):
        axis_name, stage, amplitude = key
        rows.append(
            {
                "row_type": "injection",
                "axis_name": axis_name,
                "stage": stage,
                "injected_amplitude": amplitude,
                "n_mocks": len(group),
                "threshold_alpha": detection_alpha,
                "detection_threshold": group["detection_threshold"].iloc[0],
                "null_amplitude_median": np.nan,
                "null_amplitude_p95": np.nan,
                "observed_amplitude": np.nan,
                "observed_ra_deg": np.nan,
                "observed_dec_deg": np.nan,
                "observed_empirical_p_value": np.nan,
                "detection_efficiency": float(group["detected_at_threshold"].mean()),
                "median_projection_recovery_fraction": float(
                    group["projection_recovery_fraction"].median()
                ),
                "median_axis_error_deg": float(group["axis_error_deg"].median()),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_calibration_report(
    report_path: Path,
    map_path: Path,
    null: pd.DataFrame,
    injections: pd.DataFrame,
    summary: pd.DataFrame,
    n_templates: int,
    detection_alpha: float,
) -> None:
    brief = summary.copy()
    for column in brief.select_dtypes(include="number").columns:
        if column not in {"n_mocks"}:
            brief[column] = brief[column].map(lambda value: f"{value:.6g}")
    lines = [
        "# Mock calibration",
        "",
        f"- Map: `{map_path}`",
        f"- External templates: {n_templates}",
        f"- Null mock rows: {len(null)}",
        f"- Injection mock rows: {len(injections)}",
        f"- Detection alpha: {detection_alpha:g}",
        "",
        "These are selection-preserving Poisson mocks based on the DESI random-count",
        "map. They calibrate the estimator against the survey window and shot noise,",
        "but they do not include realistic clustered large-scale structure covariance.",
        "",
        "## Summary",
        "",
        brief.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Low observed empirical p-values here would be a stronger warning than a",
        "single-map permutation p-value.",
        "- Injection detection efficiency shows whether the current template-regressed",
        "estimator would notice a real dipole of a given amplitude.",
        "- Official DESI mocks are still required before quoting cosmological upper limits.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
