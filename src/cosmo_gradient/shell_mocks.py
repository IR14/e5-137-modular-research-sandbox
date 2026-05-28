"""Tuned redshift-shell lognormal mocks for DESI-like dipole calibration."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.random import Generator, default_rng
from numpy.typing import NDArray

from cosmo_gradient.clustered_mocks import correlated_gaussian_field
from cosmo_gradient.config import ProjectConfig
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.io.desi import CatalogPair, desi_file_tracer, load_catalog_pair
from cosmo_gradient.maps import SkyMap, counts_to_overdensity_map, weighted_counts
from cosmo_gradient.randoms import expected_random_paths
from cosmo_gradient.systematics import load_external_template_maps, regress_templates

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShellMockOutputs:
    """Paths written by a redshift-shell mock calibration run."""

    null_csv: Path
    summary_csv: Path
    report: Path


@dataclass(frozen=True)
class ShellCube:
    """Observed and random count maps split by redshift shell."""

    tracer: str
    regions: tuple[str, ...]
    z_min: float
    z_max: float
    z_edges: NDArray[np.float64]
    nside: int
    backend: str
    data_counts: NDArray[np.float64]
    random_counts: NDArray[np.float64]
    shell_alpha: NDArray[np.float64]
    data_rows: NDArray[np.int64]
    random_rows: NDArray[np.int64]


def run_shell_lognormal_calibration(
    config: ProjectConfig,
    tracer: str,
    regions: Sequence[str],
    z_min: float,
    z_max: float,
    output_prefix: Path,
    external_templates: Sequence[str] = (),
    random_indices: Sequence[int] | None = None,
    nside: int | None = None,
    weight_mode: str = "desi",
    mocks: int = 1000,
    shells: int = 6,
    sigma: str | float = "auto",
    radial_corr: float = 0.05,
    smoothing_deg: float = 8.0,
    cl_slope: float = 1.4,
    lmax: int | None = None,
    seed: int = 20260525,
) -> ShellMockOutputs:
    """Run tuned 3D-ish shell lognormal mocks under the real DESI selection."""
    if nside is not None:
        config = _replace_nside(config, int(nside))
    if random_indices is not None:
        config = _replace_random_indices(config, list(random_indices))
    cube = build_shell_cube(
        config=config,
        tracer=tracer,
        regions=regions,
        z_min=z_min,
        z_max=z_max,
        shells=shells,
        weight_mode=weight_mode,
    )
    observed_map = cube_to_sky_map(cube, data_counts=cube.data_counts)
    template_matrix, template_names = load_external_template_maps(
        external_templates,
        valid=observed_map.valid & np.isfinite(observed_map.delta),
        expected_npix=len(observed_map.delta),
    )
    observed_raw = fit_dipole_map(observed_map)
    observed_corrected_map = regress_templates(
        observed_map,
        template_matrix,
        template_names,
    ).corrected_map
    observed_corrected = fit_dipole_map(observed_corrected_map)
    sigma_value = resolve_shell_sigma(cube, sigma)

    rng = default_rng(seed)
    null = _run_shell_mocks(
        cube=cube,
        template_matrix=template_matrix,
        template_names=template_names,
        rng=rng,
        mocks=mocks,
        sigma=sigma_value,
        sigma_label=str(sigma),
        radial_corr=radial_corr,
        smoothing_deg=smoothing_deg,
        cl_slope=cl_slope,
        lmax=lmax,
    )
    null = _attach_p_values(
        null,
        observed_raw_amplitude=observed_raw.amplitude,
        observed_corrected_amplitude=observed_corrected.amplitude,
    )
    summary = _summarize_shell_null(
        null=null,
        cube=cube,
        observed_raw=observed_raw,
        observed_corrected=observed_corrected,
        sigma_value=sigma_value,
        sigma_label=str(sigma),
        n_templates=len(template_names),
        external_templates=external_templates,
        weight_mode=weight_mode,
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
    _write_shell_report(report, summary, cube, mocks=mocks)
    return ShellMockOutputs(null_csv=null_csv, summary_csv=summary_csv, report=report)


def build_shell_cube(
    config: ProjectConfig,
    tracer: str,
    regions: Sequence[str],
    z_min: float,
    z_max: float,
    shells: int,
    weight_mode: str = "desi",
) -> ShellCube:
    """Load real catalogs and accumulate data/random count maps by z shell."""
    if shells <= 0:
        raise ValueError("shells must be positive.")
    z_edges = np.linspace(float(z_min), float(z_max), int(shells) + 1)
    data_counts: NDArray[np.float64] | None = None
    random_counts: NDArray[np.float64] | None = None
    backend: str | None = None
    data_rows = np.zeros(shells, dtype=np.int64)
    random_rows = np.zeros(shells, dtype=np.int64)

    for region in regions:
        pair = _load_real_pair(config, tracer=tracer, region=region, weight_mode=weight_mode)
        for shell_index in range(shells):
            lo = z_edges[shell_index]
            hi = z_edges[shell_index + 1]
            final_shell = shell_index == shells - 1
            data_mask = _z_mask(pair.data["z"].to_numpy(dtype=float), lo, hi, final_shell)
            random_mask = _z_mask(pair.randoms["z"].to_numpy(dtype=float), lo, hi, final_shell)
            data_bin = pair.data.loc[data_mask]
            random_bin = pair.randoms.loc[random_mask]
            shell_data, npix, shell_backend = weighted_counts(data_bin, config.analysis.nside)
            shell_random, random_npix, random_backend = weighted_counts(
                random_bin,
                config.analysis.nside,
            )
            if npix != random_npix or shell_backend != random_backend:
                raise ValueError("Data/random shell maps have incompatible pixel geometry.")
            if data_counts is None:
                data_counts = np.zeros((shells, npix), dtype=float)
                random_counts = np.zeros((shells, npix), dtype=float)
                backend = shell_backend
            if data_counts.shape[1] != npix or backend != shell_backend:
                raise ValueError("Region shell maps have incompatible pixel geometry.")
            data_counts[shell_index] += shell_data
            random_counts[shell_index] += shell_random
            data_rows[shell_index] += len(data_bin)
            random_rows[shell_index] += len(random_bin)
        del pair
        gc.collect()

    if data_counts is None or random_counts is None or backend is None:
        raise ValueError("No shell maps were built.")
    shell_alpha = np.divide(
        data_counts.sum(axis=1),
        random_counts.sum(axis=1),
        out=np.zeros(shells, dtype=float),
        where=random_counts.sum(axis=1) > 0.0,
    )
    return ShellCube(
        tracer=tracer,
        regions=tuple(regions),
        z_min=float(z_min),
        z_max=float(z_max),
        z_edges=z_edges,
        nside=config.analysis.nside,
        backend=backend,
        data_counts=data_counts,
        random_counts=random_counts,
        shell_alpha=shell_alpha,
        data_rows=data_rows,
        random_rows=random_rows,
    )


def cube_to_sky_map(
    cube: ShellCube,
    data_counts: NDArray[np.float64],
) -> SkyMap:
    """Collapse a shell cube to the 2D overdensity map used by the dipole fitter."""
    return counts_to_overdensity_map(
        data_counts=np.asarray(data_counts, dtype=float).sum(axis=0),
        random_counts=cube.random_counts.sum(axis=0),
        nside=cube.nside,
        backend=cube.backend,
        min_random_per_pixel=1.0,
    )


def shell_lognormal_mock_counts(
    cube: ShellCube,
    rng: Generator,
    sigma: float,
    radial_corr: float = 0.05,
    smoothing_deg: float = 8.0,
    cl_slope: float = 1.4,
    lmax: int | None = None,
) -> NDArray[np.float64]:
    """Draw shell-by-shell counts with correlated angular and radial structure."""
    if sigma < 0.0:
        raise ValueError("sigma must be non-negative.")
    expected = cube.shell_alpha[:, None] * cube.random_counts
    mock_counts = np.zeros_like(expected, dtype=float)
    shell_fields = _correlated_shell_fields(
        cube=cube,
        rng=rng,
        radial_corr=radial_corr,
        smoothing_deg=smoothing_deg,
        cl_slope=cl_slope,
        lmax=lmax,
    )
    for shell_index in range(len(cube.z_edges) - 1):
        valid = expected[shell_index] > 0.0
        rate = expected[shell_index].copy()
        if sigma > 0.0 and np.sum(valid) >= 4:
            density = np.exp(sigma * shell_fields[shell_index])
            mean_density = float(np.average(density[valid], weights=expected[shell_index, valid]))
            density /= mean_density
            rate *= density
        mock_counts[shell_index, valid] = rng.poisson(np.clip(rate[valid], 0.0, None))
    return mock_counts


def estimate_shell_sigma(cube: ShellCube) -> float:
    """Estimate excess shell overdensity RMS after crude shot-noise subtraction."""
    expected = cube.shell_alpha[:, None] * cube.random_counts
    selected = expected > 0.0
    if np.sum(selected) < 4:
        return 0.0
    delta = np.full_like(expected, np.nan, dtype=float)
    delta[selected] = cube.data_counts[selected] / expected[selected] - 1.0
    weights = cube.random_counts[selected].astype(float)
    values = delta[selected].astype(float)
    mean = float(np.average(values, weights=weights))
    observed_var = float(np.average((values - mean) ** 2, weights=weights))
    shot_var = float(np.average(1.0 / expected[selected], weights=weights))
    return float(np.sqrt(max(observed_var - shot_var, 0.0)))


def resolve_shell_sigma(cube: ShellCube, sigma: str | float) -> float:
    """Resolve ``auto`` or numeric shell-lognormal sigma."""
    if isinstance(sigma, str) and sigma.strip().lower() == "auto":
        return estimate_shell_sigma(cube)
    value = float(sigma)
    if value < 0.0:
        raise ValueError("sigma must be non-negative.")
    return value


def _run_shell_mocks(
    cube: ShellCube,
    template_matrix: NDArray[np.float64],
    template_names: Sequence[str],
    rng: Generator,
    mocks: int,
    sigma: float,
    sigma_label: str,
    radial_corr: float,
    smoothing_deg: float,
    cl_slope: float,
    lmax: int | None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for mock_index in range(mocks):
        mock_counts = shell_lognormal_mock_counts(
            cube=cube,
            rng=rng,
            sigma=sigma,
            radial_corr=radial_corr,
            smoothing_deg=smoothing_deg,
            cl_slope=cl_slope,
            lmax=lmax,
        )
        mock_map = cube_to_sky_map(cube, mock_counts)
        raw = fit_dipole_map(mock_map)
        corrected_map = regress_templates(mock_map, template_matrix, template_names).corrected_map
        corrected = fit_dipole_map(corrected_map)
        records.extend(
            [
                _fit_record(
                    mock_index=mock_index,
                    stage="raw_no_template_regression",
                    fit=raw,
                    sigma=sigma,
                    sigma_label=sigma_label,
                    radial_corr=radial_corr,
                    smoothing_deg=smoothing_deg,
                    cl_slope=cl_slope,
                    shells=len(cube.z_edges) - 1,
                ),
                _fit_record(
                    mock_index=mock_index,
                    stage="after_external_template_regression",
                    fit=corrected,
                    sigma=sigma,
                    sigma_label=sigma_label,
                    radial_corr=radial_corr,
                    smoothing_deg=smoothing_deg,
                    cl_slope=cl_slope,
                    shells=len(cube.z_edges) - 1,
                ),
            ]
        )
    return pd.DataFrame.from_records(records)


def _fit_record(
    mock_index: int,
    stage: str,
    fit,
    sigma: float,
    sigma_label: str,
    radial_corr: float,
    smoothing_deg: float,
    cl_slope: float,
    shells: int,
) -> dict[str, object]:
    return {
        "mock_index": mock_index,
        "mock_family": "redshift_shell_lognormal",
        "stage": stage,
        "shells": shells,
        "sigma_label": sigma_label,
        "sigma_lognormal": sigma,
        "radial_corr": radial_corr,
        "smoothing_deg": smoothing_deg,
        "cl_slope": cl_slope,
        "amplitude": fit.amplitude,
        "ra_deg": fit.ra_deg,
        "dec_deg": fit.dec_deg,
        "vector_x": float(fit.vector[0]),
        "vector_y": float(fit.vector[1]),
        "vector_z": float(fit.vector[2]),
    }


def _attach_p_values(
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
    for stage, group in frame.groupby("stage"):
        values = group["amplitude"].to_numpy(dtype=float)
        p_values[str(stage)] = (1.0 + np.sum(values >= observed[str(stage)])) / (
            len(values) + 1.0
        )
    frame["observed_empirical_p_value"] = frame["stage"].map(p_values)
    return frame


def _summarize_shell_null(
    null: pd.DataFrame,
    cube: ShellCube,
    observed_raw,
    observed_corrected,
    sigma_value: float,
    sigma_label: str,
    n_templates: int,
    external_templates: Sequence[str],
    weight_mode: str,
) -> pd.DataFrame:
    observed = {
        "raw_no_template_regression": observed_raw,
        "after_external_template_regression": observed_corrected,
    }
    rows: list[dict[str, object]] = []
    for stage, group in null.groupby("stage"):
        values = group["amplitude"].to_numpy(dtype=float)
        fit = observed[str(stage)]
        rows.append(
            {
                "mock_family": "redshift_shell_lognormal",
                "tracer": cube.tracer,
                "regions": "+".join(cube.regions),
                "z_min": cube.z_min,
                "z_max": cube.z_max,
                "nside": cube.nside,
                "shells": len(cube.z_edges) - 1,
                "z_edges": ",".join(f"{value:.6g}" for value in cube.z_edges),
                "sigma_label": sigma_label,
                "sigma_lognormal": sigma_value,
                "radial_corr": float(group["radial_corr"].iloc[0]),
                "smoothing_deg": float(group["smoothing_deg"].iloc[0]),
                "cl_slope": float(group["cl_slope"].iloc[0]),
                "stage": stage,
                "n_mocks": len(group),
                "n_templates": n_templates,
                "external_templates": ",".join(external_templates),
                "weight_mode": weight_mode,
                "n_data": int(cube.data_rows.sum()),
                "n_random": int(cube.random_rows.sum()),
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


def _write_shell_report(
    report_path: Path,
    summary: pd.DataFrame,
    cube: ShellCube,
    mocks: int,
) -> None:
    brief = summary.copy()
    for column in brief.select_dtypes(include="number").columns:
        if column not in {"nside", "shells", "n_mocks", "n_templates", "n_data", "n_random"}:
            brief[column] = brief[column].map(lambda value: f"{value:.6g}")
    shell_table = pd.DataFrame(
        {
            "shell": np.arange(len(cube.z_edges) - 1),
            "z_min": cube.z_edges[:-1],
            "z_max": cube.z_edges[1:],
            "data_rows": cube.data_rows,
            "random_rows": cube.random_rows,
            "alpha": cube.shell_alpha,
        }
    )
    lines = [
        "# Redshift-shell lognormal mock calibration",
        "",
        f"- Tracer: `{cube.tracer}`",
        f"- Regions: `{'+'.join(cube.regions)}`",
        f"- Redshift range: `{cube.z_min:g}-{cube.z_max:g}`",
        f"- Nside: {cube.nside}",
        f"- Mocks: {mocks}",
        "",
        "This tuned 3D-ish mock backend preserves the DESI random-catalog angular",
        "and redshift selection by generating a lognormal density field in redshift",
        "shells, Poisson sampling each shell, summing the shell maps, and running",
        "the same dipole/template-regression estimator as for the data.",
        "",
        "It is stronger than a purely angular or Poisson-only null, but it is still",
        "not an official DESI covariance product.",
        "",
        "## Shell selection",
        "",
        shell_table.to_markdown(index=False),
        "",
        "## Summary",
        "",
        brief.to_markdown(index=False),
        "",
        "## Interpretation",
        "",
        "- Use this as a local tuned 3D sanity check while official DESI mocks are",
        "being downloaded or staged.",
        "- A final cosmology-grade p-value should be based on many official EZmock",
        "or AbacusSummit realizations processed through the same pipeline.",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _correlated_shell_fields(
    cube: ShellCube,
    rng: Generator,
    radial_corr: float,
    smoothing_deg: float,
    cl_slope: float,
    lmax: int | None,
) -> NDArray[np.float64]:
    shells = len(cube.z_edges) - 1
    template_map = cube_to_sky_map(cube, cube.data_counts)
    fields = np.empty((shells, cube.data_counts.shape[1]), dtype=float)
    previous: NDArray[np.float64] | None = None
    mids = 0.5 * (cube.z_edges[:-1] + cube.z_edges[1:])
    for shell_index in range(shells):
        fresh = correlated_gaussian_field(
            sky_map=template_map,
            rng=rng,
            smoothing_deg=smoothing_deg,
            cl_slope=cl_slope,
            lmax=lmax,
        )
        if previous is None or radial_corr <= 0.0:
            field = fresh
        else:
            dz = abs(float(mids[shell_index] - mids[shell_index - 1]))
            rho = float(np.exp(-dz / radial_corr))
            field = rho * previous + np.sqrt(max(1.0 - rho**2, 0.0)) * fresh
        fields[shell_index] = _standardize_shell_field(
            field,
            weights=cube.random_counts[shell_index],
        )
        previous = fields[shell_index]
    return fields


def _standardize_shell_field(
    field: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    values = np.asarray(field, dtype=float).copy()
    valid = np.isfinite(values) & (weights > 0.0)
    if np.sum(valid) < 4:
        return values
    selected_weights = np.clip(weights[valid], 1e-12, None)
    mean = float(np.average(values[valid], weights=selected_weights))
    values -= mean
    variance = float(np.average(values[valid] ** 2, weights=selected_weights))
    if variance > 0.0 and np.isfinite(variance):
        values /= np.sqrt(variance)
    return values


def _load_real_pair(
    config: ProjectConfig,
    tracer: str,
    region: str,
    weight_mode: str,
) -> CatalogPair:
    data_template = config.desi.file_templates["data"]
    file_tracer = desi_file_tracer(config.desi, tracer)
    first_data_path: Path | None = None
    first_random_paths: list[Path] = []
    first_missing: list[Path] = []
    for root in config.paths.raw_search_roots():
        data_path = root / data_template.format(tracer=file_tracer, region=region)
        random_paths = expected_random_paths(
            root,
            tracer=file_tracer,
            region=region,
            indices=config.desi.random_indices,
            template=config.desi.file_templates["random"],
        )
        missing = [path for path in [data_path, *random_paths] if not path.exists()]
        if not missing:
            break
        if first_data_path is None:
            first_data_path = data_path
            first_random_paths = random_paths
            first_missing = missing
    else:
        data_path = first_data_path
        random_paths = first_random_paths
        missing = first_missing
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        roots_text = "\n".join(str(path) for path in config.paths.raw_search_roots())
        raise FileNotFoundError(
            f"Missing real DESI files for shell mocks.\nSearched raw roots:\n{roots_text}\n"
            f"Expected missing paths:\n{missing_text}"
        )
    if data_path is None:
        raise ValueError("No raw data roots configured.")
    return load_catalog_pair(
        data_path=data_path,
        random_paths=random_paths,
        tracer=tracer,
        region=region,
        config=config.desi,
        min_redshift=config.analysis.min_redshift,
        max_redshift=config.analysis.max_redshift,
        weight_mode=weight_mode,
    )


def _z_mask(
    z: NDArray[np.float64],
    z_min: float,
    z_max: float,
    final_shell: bool,
) -> NDArray[np.bool_]:
    if final_shell:
        return (z >= z_min) & (z <= z_max)
    return (z >= z_min) & (z < z_max)


def _replace_nside(config: ProjectConfig, nside: int) -> ProjectConfig:
    from dataclasses import replace

    return replace(config, analysis=replace(config.analysis, nside=nside))


def _replace_random_indices(config: ProjectConfig, indices: Sequence[int]) -> ProjectConfig:
    from dataclasses import replace

    return replace(config, desi=replace(config.desi, random_indices=list(indices)))
