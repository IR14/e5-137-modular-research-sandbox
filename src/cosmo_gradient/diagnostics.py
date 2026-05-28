"""Diagnostics for template overfitting and injected dipole recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cosmo_gradient.coords import angular_separation_deg, radec_to_unit, unit_to_radec
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, load_sky_map
from cosmo_gradient.systematics import load_external_template_maps, regress_templates


@dataclass(frozen=True)
class TemplateSet:
    """A named subset of template columns."""

    name: str
    include_patterns: tuple[str, ...]
    exclude_patterns: tuple[str, ...] = ()


DEFAULT_TEMPLATE_SETS: tuple[TemplateSet, ...] = (
    TemplateSet("all", ("",)),
    TemplateSet("no_dust", ("",), ("ebv",)),
    TemplateSet("no_stellar_density", ("",), ("stardens",)),
    TemplateSet("no_depth", ("",), ("galdepth", "psfdepth")),
    TemplateSet("no_seeing", ("",), ("psfsize",)),
    TemplateSet("no_sky_brightness", ("",), ("cosky",)),
    TemplateSet("dust_only", ("ebv",)),
    TemplateSet("stellar_density_only", ("stardens",)),
    TemplateSet("depth_only", ("galdepth", "psfdepth")),
    TemplateSet("seeing_only", ("psfsize",)),
    TemplateSet("sky_brightness_only", ("cosky",)),
)


def write_template_group_audit(
    map_path: Path,
    external_templates: Sequence[str],
    output_csv: Path,
    output_report: Path,
    template_sets: Sequence[TemplateSet] = DEFAULT_TEMPLATE_SETS,
) -> pd.DataFrame:
    """Regress selected template groups and write a leave-one/group-only audit."""
    sky_map = load_sky_map(str(map_path))
    template_matrix, template_names = load_external_template_maps(
        external_templates,
        valid=sky_map.valid & np.isfinite(sky_map.delta),
        expected_npix=len(sky_map.delta),
    )
    raw_fit = fit_dipole_map(sky_map)
    records: list[dict[str, object]] = []
    for template_set in template_sets:
        selected_matrix, selected_names = _select_templates(
            template_matrix,
            template_names,
            template_set,
        )
        regression = regress_templates(sky_map, selected_matrix, selected_names)
        corrected_fit = fit_dipole_map(regression.corrected_map)
        axis_shift = _folded_axis_separation(raw_fit.vector, corrected_fit.vector)
        records.append(
            {
                "template_set": template_set.name,
                "n_templates": len(regression.template_names),
                "template_names": ",".join(regression.template_names),
                "template_weighted_r2": regression.weighted_r2,
                "raw_amplitude": raw_fit.amplitude,
                "raw_ra_deg": raw_fit.ra_deg,
                "raw_dec_deg": raw_fit.dec_deg,
                "corrected_amplitude": corrected_fit.amplitude,
                "corrected_ra_deg": corrected_fit.ra_deg,
                "corrected_dec_deg": corrected_fit.dec_deg,
                "amplitude_ratio_corrected_to_raw": (
                    corrected_fit.amplitude / raw_fit.amplitude if raw_fit.amplitude > 0 else np.nan
                ),
                "axis_shift_raw_to_corrected_deg": axis_shift,
            }
        )
    frame = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_group_audit_report(output_report, map_path, frame)
    return frame


def write_injection_recovery(
    map_path: Path,
    external_templates: Sequence[str],
    output_csv: Path,
    output_report: Path,
    amplitudes: Sequence[float],
    axes: Sequence[tuple[str, float, float]],
) -> pd.DataFrame:
    """Inject known dipoles into one sky map and measure recovery after regression."""
    sky_map = load_sky_map(str(map_path))
    template_matrix, template_names = load_external_template_maps(
        external_templates,
        valid=sky_map.valid & np.isfinite(sky_map.delta),
        expected_npix=len(sky_map.delta),
    )
    baseline_raw = fit_dipole_map(sky_map)
    baseline_regression = regress_templates(sky_map, template_matrix, template_names)
    baseline_corrected = fit_dipole_map(baseline_regression.corrected_map)

    records: list[dict[str, object]] = []
    for axis_name, axis_ra, axis_dec in axes:
        axis_vector = radec_to_unit([axis_ra], [axis_dec])[0]
        for amplitude in amplitudes:
            injected = inject_dipole(sky_map, amplitude=amplitude, axis_vector=axis_vector)
            injected_raw = fit_dipole_map(injected)
            injected_regression = regress_templates(injected, template_matrix, template_names)
            injected_corrected = fit_dipole_map(injected_regression.corrected_map)

            raw_recovered = injected_raw.vector - baseline_raw.vector
            corrected_recovered = injected_corrected.vector - baseline_corrected.vector
            records.extend(
                [
                    _injection_record(
                        axis_name=axis_name,
                        axis_ra=axis_ra,
                        axis_dec=axis_dec,
                        injected_amplitude=amplitude,
                        stage="raw_no_template_regression",
                        baseline_amplitude=baseline_raw.amplitude,
                        absolute_amplitude=injected_raw.amplitude,
                        recovered_vector=raw_recovered,
                        true_axis=axis_vector,
                    ),
                    _injection_record(
                        axis_name=axis_name,
                        axis_ra=axis_ra,
                        axis_dec=axis_dec,
                        injected_amplitude=amplitude,
                        stage="after_external_template_regression",
                        baseline_amplitude=baseline_corrected.amplitude,
                        absolute_amplitude=injected_corrected.amplitude,
                        recovered_vector=corrected_recovered,
                        true_axis=axis_vector,
                    ),
                ]
            )
    frame = pd.DataFrame.from_records(records)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_csv, index=False)
    _write_injection_report(output_report, map_path, frame, len(template_names))
    return frame


def inject_dipole(
    sky_map: SkyMap,
    amplitude: float,
    axis_vector: NDArray[np.float64],
) -> SkyMap:
    """Return a map with ``amplitude * dot(axis, n)`` added to valid pixels."""
    axis = np.asarray(axis_vector, dtype=float)
    axis = axis / np.linalg.norm(axis)
    injected_delta = sky_map.delta.copy()
    valid = sky_map.valid & np.isfinite(sky_map.delta)
    injected_delta[valid] = (
        injected_delta[valid] + amplitude * (sky_map.pixel_vectors[valid] @ axis)
    )
    return _replace_delta(sky_map, injected_delta)


def parse_axis_specs(axis_specs: Sequence[str] | None) -> list[tuple[str, float, float]]:
    """Parse CLI axis specs like ``name=ra,dec``."""
    if not axis_specs:
        return [("fiducial", 215.0, 25.0)]
    axes: list[tuple[str, float, float]] = []
    for spec in axis_specs:
        if "=" in spec:
            name, values = spec.split("=", 1)
        else:
            name, values = f"axis_{len(axes) + 1}", spec
        parts = [part.strip() for part in values.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Axis spec must be name=ra,dec or ra,dec: {spec}")
        axes.append((name.strip(), float(parts[0]), float(parts[1])))
    return axes


def _select_templates(
    template_matrix: NDArray[np.float64],
    template_names: Sequence[str],
    template_set: TemplateSet,
) -> tuple[NDArray[np.float64], list[str]]:
    selected_indices: list[int] = []
    for index, name in enumerate(template_names):
        lower = name.lower()
        include = any(pattern in lower for pattern in template_set.include_patterns)
        exclude = any(pattern in lower for pattern in template_set.exclude_patterns)
        if include and not exclude:
            selected_indices.append(index)
    if not selected_indices:
        return np.empty((template_matrix.shape[0], 0), dtype=float), []
    selected_names = [template_names[index] for index in selected_indices]
    return template_matrix[:, selected_indices], selected_names


def _injection_record(
    axis_name: str,
    axis_ra: float,
    axis_dec: float,
    injected_amplitude: float,
    stage: str,
    baseline_amplitude: float,
    absolute_amplitude: float,
    recovered_vector: NDArray[np.float64],
    true_axis: NDArray[np.float64],
) -> dict[str, object]:
    recovered_amplitude = float(np.linalg.norm(recovered_vector))
    if recovered_amplitude > 0:
        recovered_ra, recovered_dec = unit_to_radec(recovered_vector)
        axis_error = float(angular_separation_deg(recovered_vector, true_axis))
    else:
        recovered_ra = np.array(float("nan"))
        recovered_dec = np.array(float("nan"))
        axis_error = float("nan")
    return {
        "axis_name": axis_name,
        "axis_ra_deg": axis_ra,
        "axis_dec_deg": axis_dec,
        "stage": stage,
        "injected_amplitude": injected_amplitude,
        "baseline_amplitude": baseline_amplitude,
        "absolute_fit_amplitude": absolute_amplitude,
        "recovered_amplitude": recovered_amplitude,
        "recovery_fraction": (
            recovered_amplitude / injected_amplitude if injected_amplitude > 0 else np.nan
        ),
        "recovered_ra_deg": float(np.asarray(recovered_ra)),
        "recovered_dec_deg": float(np.asarray(recovered_dec)),
        "axis_error_deg": axis_error,
    }


def _folded_axis_separation(
    left_vector: NDArray[np.float64],
    right_vector: NDArray[np.float64],
) -> float:
    sep = float(angular_separation_deg(left_vector, right_vector))
    return min(sep, 180.0 - sep)


def _replace_delta(sky_map: SkyMap, delta: NDArray[np.float64]) -> SkyMap:
    return SkyMap(
        nside=sky_map.nside,
        backend=sky_map.backend,
        data_counts=sky_map.data_counts,
        random_counts=sky_map.random_counts,
        alpha=sky_map.alpha,
        delta=np.asarray(delta, dtype=float),
        valid=sky_map.valid,
        pixel_vectors=sky_map.pixel_vectors,
    )


def _write_group_audit_report(report_path: Path, map_path: Path, frame: pd.DataFrame) -> None:
    lines = [
        "# Template group audit",
        "",
        f"- Map: `{map_path}`",
        "",
        "This diagnostic regresses subsets of external templates from the same fixed sky map.",
        "Large differences between rows identify which systematics groups carry the",
        "dipole-like residual.",
        "",
    ]
    if len(frame):
        lines.append(frame.drop(columns=["template_names"]).to_markdown(index=False))
    else:
        lines.append("No audit rows were produced.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_injection_report(
    report_path: Path,
    map_path: Path,
    frame: pd.DataFrame,
    n_templates: int,
) -> None:
    lines = [
        "# Injection recovery",
        "",
        f"- Map: `{map_path}`",
        f"- External templates: {n_templates}",
        "",
        "This diagnostic injects a known dipole into a fixed observed map. Recovery is measured",
        "by subtracting the baseline fitted vector before and after injection, so the table",
        "reports",
        "the recovered injected component rather than the map's pre-existing residual.",
        "",
    ]
    if len(frame):
        lines.append(frame.to_markdown(index=False))
    else:
        lines.append("No injection rows were produced.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
