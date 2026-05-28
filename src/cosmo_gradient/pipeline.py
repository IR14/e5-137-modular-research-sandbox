"""End-to-end first-pass pipeline."""

from __future__ import annotations

import itertools
import gc
import logging
from dataclasses import replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.random import default_rng

from cosmo_gradient.binning import make_redshift_bins, subset_by_redshift_bin
from cosmo_gradient.config import ProjectConfig
from cosmo_gradient.dipole import fit_with_resampling
from cosmo_gradient.io.desi import (
    CatalogPair,
    build_lss_urls,
    desi_file_tracer,
    generate_synthetic_pair,
    load_catalog_pair,
    write_download_manifest,
)
from cosmo_gradient.maps import (
    SkyMap,
    build_overdensity_map,
    counts_to_overdensity_map,
    save_sky_map,
    weighted_counts,
)
from cosmo_gradient.plots import (
    plot_amplitude_by_redshift,
    plot_axis_by_redshift,
    plot_null_distribution,
    plot_overdensity_map,
)
from cosmo_gradient.randoms import expected_random_paths
from cosmo_gradient.statistics import (
    add_multiple_testing_corrections,
    pairwise_axis_separations,
    result_quality_flags,
)
from cosmo_gradient.systematics import (
    build_template_maps,
    combine_template_matrices,
    load_external_template_maps,
    regress_templates,
)
from cosmo_gradient.coords import angular_separation_deg

LOGGER = logging.getLogger(__name__)


def prepare_downloads(
    config: ProjectConfig,
    tracers: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    random_indices: Optional[Sequence[int]] = None,
) -> Path:
    """Create a dry-run DESI download manifest."""
    selected_tracers = list(tracers or config.analysis.tracers)
    selected_regions = list(regions or config.analysis.regions)
    selected_indices = list(random_indices or config.desi.random_indices)
    config.paths.ensure_dirs()
    manifest = config.paths.data_raw / "desi_dr1_lss_download_manifest.md"
    return write_download_manifest(
        manifest,
        config=config.desi,
        tracers=selected_tracers,
        regions=selected_regions,
        random_indices=selected_indices,
    )


def run_first_pass(
    config: ProjectConfig,
    mode: Optional[str] = None,
    tracers: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    random_indices: Optional[Sequence[int]] = None,
    nside: Optional[int] = None,
    null_permutations: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    poisson_mocks: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
    weight_mode: str = "desi",
    results_filename: str = "first_pass_dipoles.csv",
    artifact_label: Optional[str] = None,
    write_summary_artifacts: bool = True,
) -> pd.DataFrame:
    """Run the first-pass map and dipole analysis."""
    if random_indices is not None:
        config = replace(config, desi=replace(config.desi, random_indices=list(random_indices)))
    if nside is not None:
        config = replace(config, analysis=replace(config.analysis, nside=int(nside)))
    if any(
        value is not None
        for value in (
            null_permutations,
            bootstrap_samples,
            jackknife_regions,
            poisson_mocks,
            block_null_mocks,
            block_null_regions,
        )
    ):
        config = replace(
            config,
            dipole=replace(
                config.dipole,
                null_permutations=(
                    int(null_permutations)
                    if null_permutations is not None
                    else config.dipole.null_permutations
                ),
                bootstrap_samples=(
                    int(bootstrap_samples)
                    if bootstrap_samples is not None
                    else config.dipole.bootstrap_samples
                ),
                jackknife_regions=(
                    int(jackknife_regions)
                    if jackknife_regions is not None
                    else config.dipole.jackknife_regions
                ),
                poisson_mocks=(
                    int(poisson_mocks)
                    if poisson_mocks is not None
                    else config.dipole.poisson_mocks
                ),
                block_null_mocks=(
                    int(block_null_mocks)
                    if block_null_mocks is not None
                    else config.dipole.block_null_mocks
                ),
                block_null_regions=(
                    int(block_null_regions)
                    if block_null_regions is not None
                    else config.dipole.block_null_regions
                ),
            ),
        )
    config.paths.ensure_dirs()
    run_mode = mode or config.analysis.mode
    selected_tracers = list(tracers or config.analysis.tracers)
    selected_regions = list(regions or config.analysis.regions)
    rng = default_rng(config.analysis.random_seed)
    LOGGER.info(
        "Running first pass in %s mode for tracers=%s regions=%s",
        run_mode,
        ",".join(selected_tracers),
        ",".join(selected_regions),
    )

    records: list[dict[str, object]] = []
    null_store: dict[str, np.ndarray] = {}
    for tracer in selected_tracers:
        for region in selected_regions:
            pair = _load_pair_for_mode(config, run_mode, tracer, region, rng, weight_mode=weight_mode)
            records.extend(_analyze_pair(config, pair, rng, null_store, artifact_label=artifact_label))

    results = pd.DataFrame.from_records(records)
    if len(results):
        results["weight_mode"] = weight_mode
    results_path = config.paths.tables / results_filename
    results.to_csv(results_path, index=False)
    LOGGER.info("Wrote %s", results_path)
    if write_summary_artifacts:
        make_plots(config, results, null_store)
        write_report(config, results)
    return results


def run_combined_regions(
    config: ProjectConfig,
    mode: Optional[str] = None,
    tracers: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    random_indices: Optional[Sequence[int]] = None,
    nside: Optional[int] = None,
    null_permutations: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    poisson_mocks: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
    weight_mode: str = "desi",
    results_filename: str = "combined_region_dipoles.csv",
) -> pd.DataFrame:
    """Run a joint map/dipole analysis after summing count maps across sky regions."""
    config = _runtime_config(
        config,
        random_indices=random_indices,
        nside=nside,
        null_permutations=null_permutations,
        bootstrap_samples=bootstrap_samples,
        jackknife_regions=jackknife_regions,
        poisson_mocks=poisson_mocks,
        block_null_mocks=block_null_mocks,
        block_null_regions=block_null_regions,
    )
    config.paths.ensure_dirs()
    run_mode = mode or config.analysis.mode
    selected_tracers = list(tracers or config.analysis.tracers)
    selected_regions = list(regions or config.analysis.regions)
    rng = default_rng(config.analysis.random_seed)
    LOGGER.info(
        "Running combined-region first pass in %s mode for tracers=%s regions=%s",
        run_mode,
        ",".join(selected_tracers),
        ",".join(selected_regions),
    )

    records: list[dict[str, object]] = []
    null_store: dict[str, np.ndarray] = {}
    for tracer in selected_tracers:
        records.extend(
            _analyze_combined_regions_for_tracer(
                config=config,
                mode=run_mode,
                tracer=tracer,
                regions=selected_regions,
                rng=rng,
                null_store=null_store,
                weight_mode=weight_mode,
            )
        )
    results = pd.DataFrame.from_records(records)
    if len(results):
        results["weight_mode"] = weight_mode
    results_path = config.paths.tables / results_filename
    results.to_csv(results_path, index=False)
    LOGGER.info("Wrote %s", results_path)
    make_plots(config, results, null_store)
    write_report(config, results)
    return results


def _runtime_config(
    config: ProjectConfig,
    random_indices: Optional[Sequence[int]] = None,
    nside: Optional[int] = None,
    null_permutations: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    poisson_mocks: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
) -> ProjectConfig:
    """Return a config copy with common runtime CLI overrides applied."""
    if random_indices is not None:
        config = replace(config, desi=replace(config.desi, random_indices=list(random_indices)))
    if nside is not None:
        config = replace(config, analysis=replace(config.analysis, nside=int(nside)))
    if any(
        value is not None
        for value in (
            null_permutations,
            bootstrap_samples,
            jackknife_regions,
            poisson_mocks,
            block_null_mocks,
            block_null_regions,
        )
    ):
        config = replace(
            config,
            dipole=replace(
                config.dipole,
                null_permutations=(
                    int(null_permutations)
                    if null_permutations is not None
                    else config.dipole.null_permutations
                ),
                bootstrap_samples=(
                    int(bootstrap_samples)
                    if bootstrap_samples is not None
                    else config.dipole.bootstrap_samples
                ),
                jackknife_regions=(
                    int(jackknife_regions)
                    if jackknife_regions is not None
                    else config.dipole.jackknife_regions
                ),
                poisson_mocks=(
                    int(poisson_mocks)
                    if poisson_mocks is not None
                    else config.dipole.poisson_mocks
                ),
                block_null_mocks=(
                    int(block_null_mocks)
                    if block_null_mocks is not None
                    else config.dipole.block_null_mocks
                ),
                block_null_regions=(
                    int(block_null_regions)
                    if block_null_regions is not None
                    else config.dipole.block_null_regions
                ),
            ),
        )
    return config


def run_nside_robustness_grid(
    config: ProjectConfig,
    nsides: Sequence[int],
    mode: Optional[str] = None,
    tracers: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    random_indices: Optional[Sequence[int]] = None,
    null_permutations: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    poisson_mocks: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
    weight_mode: str = "desi",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the first-pass analysis over several HEALPix nsides."""
    config.paths.ensure_dirs()
    selected_nsides = [int(nside) for nside in nsides]
    if any(nside <= 0 for nside in selected_nsides):
        raise ValueError("All nsides must be positive.")
    frames = []
    for nside in selected_nsides:
        LOGGER.info("Running nside robustness pass for nside=%d", nside)
        nside_config = replace(config, analysis=replace(config.analysis, nside=nside))
        frame = run_first_pass(
            nside_config,
            mode=mode,
            tracers=tracers,
            regions=regions,
            random_indices=random_indices,
            null_permutations=null_permutations,
            bootstrap_samples=bootstrap_samples,
            jackknife_regions=jackknife_regions,
            poisson_mocks=poisson_mocks,
            block_null_mocks=block_null_mocks,
            block_null_regions=block_null_regions,
            weight_mode=weight_mode,
            results_filename=f"first_pass_dipoles_nside{nside}.csv",
            artifact_label=f"nside{nside}",
            write_summary_artifacts=False,
        )
        frame["nside"] = nside
        frames.append(frame)
    grid = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    grid_path = config.paths.tables / "nside_robustness_grid.csv"
    grid.to_csv(grid_path, index=False)
    LOGGER.info("Wrote %s", grid_path)

    stability = summarize_nside_stability(grid)
    stability_path = config.paths.tables / "nside_axis_stability.csv"
    stability.to_csv(stability_path, index=False)
    LOGGER.info("Wrote %s", stability_path)
    write_nside_robustness_report(config, grid, stability)
    return grid, stability


def summarize_nside_stability(grid: pd.DataFrame) -> pd.DataFrame:
    """Summarize how fitted axes change across nside values."""
    if len(grid) == 0:
        return pd.DataFrame()
    rows = []
    group_columns = ["tracer", "region", "z_min", "z_max"]
    for key, group in grid.groupby(group_columns, dropna=False):
        separations = []
        for (_, left), (_, right) in itertools.combinations(group.iterrows(), 2):
            left_vector = np.array([left["vector_x"], left["vector_y"], left["vector_z"]], dtype=float)
            right_vector = np.array([right["vector_x"], right["vector_y"], right["vector_z"]], dtype=float)
            separations.append(float(angular_separation_deg(left_vector, right_vector)))
        row = dict(zip(group_columns, key))
        row.update(
            {
                "nside_count": int(group["nside"].nunique()),
                "max_axis_shift_deg": float(max(separations)) if separations else 0.0,
                "median_axis_shift_deg": float(np.median(separations)) if separations else 0.0,
                "amplitude_min": float(group["amplitude"].min()),
                "amplitude_max": float(group["amplitude"].max()),
                "amplitude_range": float(group["amplitude"].max() - group["amplitude"].min()),
                "p_value_min": float(group["null_p_value"].min()),
                "p_value_max": float(group["null_p_value"].max()),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def write_nside_robustness_report(
    config: ProjectConfig,
    grid: pd.DataFrame,
    stability: pd.DataFrame,
) -> Path:
    """Write a markdown report for the nside robustness pass."""
    report_path = config.paths.reports / "nside_robustness_report.md"
    lines = [
        "# Nside robustness report",
        "",
        "This report checks whether first-pass dipole axes are stable under changes",
        "to the HEALPix map resolution. Stable axes are not by themselves evidence",
        "for a cosmological signal; unstable axes are a warning sign for pixelization,",
        "noise, footprint, or selection effects.",
        "",
    ]
    if len(grid):
        lines.extend(
            [
                "## Grid",
                "",
                grid[
                    [
                        "nside",
                        "tracer",
                        "region",
                        "z_min",
                        "z_max",
                        "amplitude",
                        "ra_deg",
                        "dec_deg",
                        "null_p_value",
                    ]
                ].to_markdown(index=False),
                "",
            ]
        )
    if len(stability):
        lines.extend(
            [
                "## Axis stability",
                "",
                stability.to_markdown(index=False),
                "",
                "## Interpretation rule",
                "",
                "- Large axis shifts across nside weaken any directional interpretation.",
                "- High p-values remain compatible with the isotropic null model.",
                "- A bin that is stable in nside still needs random-catalog, NGC/SGC, and systematics checks.",
                "",
            ]
        )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    return report_path


def run_systematics_audit(
    config: ProjectConfig,
    tracer: str,
    region: str,
    mode: str = "real",
    random_indices: Optional[Sequence[int]] = None,
    nside: Optional[int] = None,
    null_permutations: Optional[int] = None,
    poisson_mocks: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    weight_mode: str = "desi",
    split_columns: Optional[Sequence[str]] = None,
    ra_sectors: int = 4,
) -> pd.DataFrame:
    """Run split tests over available survey/systematics columns."""
    if random_indices is not None:
        config = replace(config, desi=replace(config.desi, random_indices=list(random_indices)))
    if nside is not None:
        config = replace(config, analysis=replace(config.analysis, nside=int(nside)))
    if any(
        value is not None
        for value in (
            null_permutations,
            bootstrap_samples,
            jackknife_regions,
            poisson_mocks,
            block_null_mocks,
            block_null_regions,
        )
    ):
        config = replace(
            config,
            dipole=replace(
                config.dipole,
                null_permutations=int(null_permutations or config.dipole.null_permutations),
                bootstrap_samples=int(bootstrap_samples or config.dipole.bootstrap_samples),
                jackknife_regions=int(jackknife_regions or config.dipole.jackknife_regions),
                poisson_mocks=int(poisson_mocks or config.dipole.poisson_mocks),
                block_null_mocks=int(block_null_mocks or config.dipole.block_null_mocks),
                block_null_regions=int(block_null_regions or config.dipole.block_null_regions),
            ),
        )
    rng = default_rng(config.analysis.random_seed)
    pair = _load_pair_for_mode(config, mode, tracer, region, rng, weight_mode=weight_mode)
    splits = _build_split_definitions(pair, split_columns=split_columns, ra_sectors=ra_sectors)
    all_records: list[dict[str, object]] = []
    null_store: dict[str, np.ndarray] = {}
    for split_variable, split_name, data_mask, random_mask in splits:
        data_split = pair.data.loc[data_mask].copy()
        random_split = pair.randoms.loc[random_mask].copy()
        if len(data_split) < 100 or len(random_split) < 1000:
            LOGGER.warning(
                "Skipping split %s=%s: too few rows (data=%d random=%d)",
                split_variable,
                split_name,
                len(data_split),
                len(random_split),
            )
            continue
        split_pair = CatalogPair(
            tracer=pair.tracer,
            region=pair.region,
            data=data_split,
            randoms=random_split,
            data_source=pair.data_source,
            random_source=pair.random_source,
            synthetic=pair.synthetic,
        )
        safe_name = _safe_label(f"{split_variable}_{split_name}")
        records = _analyze_pair(
            config,
            split_pair,
            rng,
            null_store,
            artifact_label=f"audit_{safe_name}",
            extra_fields={
                "split_variable": split_variable,
                "split_name": split_name,
                "split_data_rows": len(data_split),
                "split_random_rows": len(random_split),
                "weight_mode": weight_mode,
            },
        )
        all_records.extend(records)
    results = pd.DataFrame.from_records(all_records)
    if len(results) and "null_p_value" in results.columns:
        results = add_multiple_testing_corrections(results)
    output_path = config.paths.tables / f"systematics_audit_{tracer}_{region}.csv"
    results.to_csv(output_path, index=False)
    LOGGER.info("Wrote %s", output_path)
    report_path = config.paths.reports / f"systematics_audit_{tracer}_{region}.md"
    _write_systematics_audit_report(report_path, results)
    return results


def run_template_systematics(
    config: ProjectConfig,
    tracer: str,
    region: str,
    mode: str = "real",
    random_indices: Optional[Sequence[int]] = None,
    nside: Optional[int] = None,
    null_permutations: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
    weight_mode: str = "desi",
    template_columns: Optional[Sequence[str]] = None,
    external_templates: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Regress survey-template maps from overdensity maps and refit dipoles."""
    config = _runtime_config(
        config,
        random_indices=random_indices,
        nside=nside,
        null_permutations=null_permutations,
        bootstrap_samples=bootstrap_samples,
        jackknife_regions=jackknife_regions,
        block_null_mocks=block_null_mocks,
        block_null_regions=block_null_regions,
    )
    config.paths.ensure_dirs()
    rng = default_rng(config.analysis.random_seed)
    pair = _load_pair_for_mode(config, mode, tracer, region, rng, weight_mode=weight_mode)
    columns = [column.lower() for column in (template_columns or ["photsys", "ntile", "frac_tlobs_tiles"])]
    external_template_specs = list(external_templates or [])
    records: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    z_bins = make_redshift_bins(config.analysis.z_bins[tracer])
    for index, zbin in enumerate(z_bins):
        final_bin = index == len(z_bins) - 1
        data_bin = subset_by_redshift_bin(pair.data, zbin, final_bin=final_bin)
        random_bin = subset_by_redshift_bin(pair.randoms, zbin, final_bin=final_bin)
        if len(data_bin) < 20 or len(random_bin) < 100:
            LOGGER.warning(
                "Skipping template regression %s %s: too few rows (data=%d random=%d)",
                tracer,
                zbin.label,
                len(data_bin),
                len(random_bin),
            )
            continue
        sky_map = build_overdensity_map(
            data_bin,
            random_bin,
            nside=config.analysis.nside,
            min_random_per_pixel=config.analysis.min_random_per_pixel,
        )
        catalog_templates, catalog_template_names = build_template_maps(
            random_bin,
            nside=config.analysis.nside,
            columns=columns,
            valid=sky_map.valid & np.isfinite(sky_map.delta),
            backend=sky_map.backend,
        )
        external_template_matrix, external_template_names = load_external_template_maps(
            external_template_specs,
            valid=sky_map.valid & np.isfinite(sky_map.delta),
            expected_npix=len(sky_map.delta),
        )
        templates, template_names = combine_template_matrices(
            (catalog_templates, catalog_template_names),
            (external_template_matrix, external_template_names),
        )
        regression = regress_templates(sky_map, templates, template_names)
        raw_fit = fit_with_resampling(
            sky_map,
            rng=rng,
            bootstrap_samples=config.dipole.bootstrap_samples,
            null_permutations=config.dipole.null_permutations,
            poisson_mocks=0,
            block_null_mocks=config.dipole.block_null_mocks,
            block_null_regions=config.dipole.block_null_regions,
            jackknife_regions=config.dipole.jackknife_regions,
        )
        corrected_fit = fit_with_resampling(
            regression.corrected_map,
            rng=rng,
            bootstrap_samples=config.dipole.bootstrap_samples,
            null_permutations=config.dipole.null_permutations,
            poisson_mocks=0,
            block_null_mocks=config.dipole.block_null_mocks,
            block_null_regions=config.dipole.block_null_regions,
            jackknife_regions=config.dipole.jackknife_regions,
        )
        suffix = f"{tracer}_{region}_{zbin.label}_{_nside_label(config)}_template_regressed"
        raw_map_path = config.paths.data_processed / f"{suffix}_raw_map.npz"
        corrected_map_path = config.paths.data_processed / f"{suffix}_corrected_map.npz"
        save_sky_map(str(raw_map_path), sky_map)
        save_sky_map(str(corrected_map_path), regression.corrected_map)
        if len(regression.coefficients):
            coeffs = regression.coefficients.copy()
            coeffs.insert(0, "z_max", zbin.z_max)
            coeffs.insert(0, "z_min", zbin.z_min)
            coeffs.insert(0, "region", region)
            coeffs.insert(0, "tracer", tracer)
            coefficient_frames.append(coeffs)

        axis_sep = float(angular_separation_deg(raw_fit.vector, corrected_fit.vector))
        axis_sep = min(axis_sep, 180.0 - axis_sep)
        records.append(
            {
                "tracer": tracer,
                "region": region,
                "z_min": zbin.z_min,
                "z_max": zbin.z_max,
                "n_data": len(data_bin),
                "n_random": len(random_bin),
                "n_templates": len(regression.template_names),
                "template_names": ",".join(regression.template_names),
                "catalog_template_columns_requested": ",".join(columns),
                "external_templates_requested": ",".join(external_template_specs),
                "template_weighted_r2": regression.weighted_r2,
                "raw_amplitude": raw_fit.amplitude,
                "raw_ra_deg": raw_fit.ra_deg,
                "raw_dec_deg": raw_fit.dec_deg,
                "raw_null_p_value": raw_fit.null_p_value,
                "raw_block_null_p_value": raw_fit.block_null_p_value,
                "corrected_amplitude": corrected_fit.amplitude,
                "corrected_ra_deg": corrected_fit.ra_deg,
                "corrected_dec_deg": corrected_fit.dec_deg,
                "corrected_null_p_value": corrected_fit.null_p_value,
                "corrected_block_null_p_value": corrected_fit.block_null_p_value,
                "amplitude_ratio_corrected_to_raw": (
                    corrected_fit.amplitude / raw_fit.amplitude if raw_fit.amplitude > 0 else np.nan
                ),
                "axis_shift_raw_to_corrected_deg": axis_sep,
                "raw_jackknife_axis_max_shift_deg": raw_fit.jackknife_axis_max_shift_deg,
                "corrected_jackknife_axis_max_shift_deg": corrected_fit.jackknife_axis_max_shift_deg,
                "raw_map_path": str(raw_map_path),
                "corrected_map_path": str(corrected_map_path),
                "weight_mode": weight_mode,
            }
        )
        LOGGER.info(
            "%s %s %s template regression: raw amp=%.4g p=%.3g -> corrected amp=%.4g p=%.3g",
            tracer,
            region,
            zbin.label,
            raw_fit.amplitude,
            raw_fit.null_p_value,
            corrected_fit.amplitude,
            corrected_fit.null_p_value,
        )

    results = pd.DataFrame.from_records(records)
    results_path = config.paths.tables / f"template_systematics_{tracer}_{region}.csv"
    results.to_csv(results_path, index=False)
    coefficients = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    coefficients_path = config.paths.tables / f"template_systematics_{tracer}_{region}_coefficients.csv"
    coefficients.to_csv(coefficients_path, index=False)
    _write_template_systematics_report(
        config.paths.reports / f"template_systematics_{tracer}_{region}.md",
        results,
        coefficients,
    )
    LOGGER.info("Wrote %s", results_path)
    LOGGER.info("Wrote %s", coefficients_path)
    return results


def run_combined_template_systematics(
    config: ProjectConfig,
    mode: str = "real",
    tracers: Optional[Sequence[str]] = None,
    regions: Optional[Sequence[str]] = None,
    random_indices: Optional[Sequence[int]] = None,
    nside: Optional[int] = None,
    null_permutations: Optional[int] = None,
    bootstrap_samples: Optional[int] = None,
    jackknife_regions: Optional[int] = None,
    block_null_mocks: Optional[int] = None,
    block_null_regions: Optional[int] = None,
    weight_mode: str = "desi",
    external_templates: Optional[Sequence[str]] = None,
    output_prefix: str = "combined_template_systematics",
) -> pd.DataFrame:
    """Regress external templates from maps summed over multiple survey regions."""
    config = _runtime_config(
        config,
        random_indices=random_indices,
        nside=nside,
        null_permutations=null_permutations,
        bootstrap_samples=bootstrap_samples,
        jackknife_regions=jackknife_regions,
        block_null_mocks=block_null_mocks,
        block_null_regions=block_null_regions,
    )
    config.paths.ensure_dirs()
    rng = default_rng(config.analysis.random_seed)
    selected_tracers = list(tracers or config.analysis.tracers)
    selected_regions = list(regions or config.analysis.regions)
    combined_label = "+".join(selected_regions)
    safe_region_label = _safe_label(combined_label)
    records: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    for tracer in selected_tracers:
        tracer_records, tracer_coefficients = _analyze_combined_template_for_tracer(
            config=config,
            mode=mode,
            tracer=tracer,
            regions=selected_regions,
            rng=rng,
            weight_mode=weight_mode,
            external_templates=list(external_templates or []),
        )
        records.extend(tracer_records)
        if len(tracer_coefficients):
            coefficient_frames.append(tracer_coefficients)

    results = pd.DataFrame.from_records(records)
    output_stem = f"{output_prefix}_{safe_region_label}"
    results_path = config.paths.tables / f"{output_stem}.csv"
    results.to_csv(results_path, index=False)
    coefficients = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    coefficients_path = config.paths.tables / f"{output_stem}_coefficients.csv"
    coefficients.to_csv(coefficients_path, index=False)
    _write_template_systematics_report(
        config.paths.reports / f"{output_stem}.md",
        results,
        coefficients,
    )
    LOGGER.info("Wrote %s", results_path)
    LOGGER.info("Wrote %s", coefficients_path)
    return results


def _write_template_systematics_report(
    report_path: Path,
    results: pd.DataFrame,
    coefficients: pd.DataFrame,
) -> Path:
    lines = [
        "# Template systematics regression",
        "",
        "This diagnostic regresses per-pixel survey templates from the overdensity map and",
        "then refits the dipole on the residual map. A strong reduction or axis shift means",
        "the candidate dipole is entangled with known survey metadata.",
        "",
    ]
    if len(results):
        lines.extend(
            [
                "## Raw vs template-corrected dipoles",
                "",
                results[
                    [
                        "tracer",
                        "region",
                        "z_min",
                        "z_max",
                        "n_templates",
                        "template_weighted_r2",
                        "raw_amplitude",
                        "raw_null_p_value",
                        "raw_block_null_p_value",
                        "corrected_amplitude",
                        "corrected_null_p_value",
                        "corrected_block_null_p_value",
                        "amplitude_ratio_corrected_to_raw",
                        "axis_shift_raw_to_corrected_deg",
                    ]
                ].to_markdown(index=False),
                "",
            ]
        )
    if len(coefficients):
        lines.extend(["## Template coefficients", "", coefficients.to_markdown(index=False), ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "Template regression is a diagnostic, not a final correction model. It can reveal",
            "whether a candidate axis is aligned with known survey metadata, but final claims",
            "need validated survey mocks and external imaging-systematics maps.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    return report_path


def _load_pair_for_mode(
    config: ProjectConfig,
    mode: str,
    tracer: str,
    region: str,
    rng: np.random.Generator,
    weight_mode: str = "desi",
) -> CatalogPair:
    if mode == "synthetic":
        return generate_synthetic_pair(tracer, region, config.synthetic, rng)
    if mode != "real":
        raise ValueError("analysis mode must be either 'synthetic' or 'real'.")

    data_path, random_paths, missing = _resolve_real_catalog_paths(config, tracer, region)
    if missing:
        data_url, random_urls = build_lss_urls(config.desi, tracer, region, config.desi.random_indices)
        missing_text = "\n".join(str(path) for path in missing)
        roots_text = "\n".join(str(path) for path in config.paths.raw_search_roots())
        raise FileNotFoundError(
            "Missing DESI LSS files. Run `cosmo-gradient prepare` and download the reviewed files.\n"
            f"Searched raw roots:\n{roots_text}\n"
            f"Expected missing paths:\n{missing_text}\n"
            f"Data URL: {data_url}\nRandom URLs: {random_urls}"
        )
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


def _resolve_real_catalog_paths(
    config: ProjectConfig,
    tracer: str,
    region: str,
) -> tuple[Path, list[Path], list[Path]]:
    """Find a real DESI catalog pair in any configured raw-data root."""
    data_template = config.desi.file_templates["data"]
    file_tracer = desi_file_tracer(config.desi, tracer)
    first_missing: list[Path] = []
    first_data_path: Path | None = None
    first_random_paths: list[Path] = []
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
            return data_path, random_paths, []
        if first_data_path is None:
            first_data_path = data_path
            first_random_paths = random_paths
            first_missing = missing
    if first_data_path is None:
        raise ValueError("No raw data roots configured.")
    return first_data_path, first_random_paths, first_missing


def _analyze_combined_regions_for_tracer(
    config: ProjectConfig,
    mode: str,
    tracer: str,
    regions: Sequence[str],
    rng: np.random.Generator,
    null_store: dict[str, np.ndarray],
    weight_mode: str = "desi",
) -> list[dict[str, object]]:
    """Analyze one tracer after summing data/random count maps over several regions."""
    if tracer not in config.analysis.z_bins:
        raise KeyError(f"No redshift bins configured for tracer {tracer}")
    z_bins = make_redshift_bins(config.analysis.z_bins[tracer])
    combined_label = "+".join(regions)
    safe_region_label = _safe_label(combined_label)
    accumulators: dict[str, dict[str, object]] = {
        zbin.label: {
            "zbin": zbin,
            "data_counts": None,
            "random_counts": None,
            "n_data": 0,
            "n_random": 0,
            "backend": None,
            "npix": None,
            "data_sources": [],
            "random_sources": [],
        }
        for zbin in z_bins
    }

    for region in regions:
        pair = _load_pair_for_mode(config, mode, tracer, region, rng, weight_mode=weight_mode)
        for index, zbin in enumerate(z_bins):
            final_bin = index == len(z_bins) - 1
            data_bin = subset_by_redshift_bin(pair.data, zbin, final_bin=final_bin)
            random_bin = subset_by_redshift_bin(pair.randoms, zbin, final_bin=final_bin)
            data_counts, npix, backend = weighted_counts(data_bin, config.analysis.nside)
            random_counts, random_npix, random_backend = weighted_counts(random_bin, config.analysis.nside)
            if npix != random_npix or backend != random_backend:
                raise ValueError("Data and random maps have incompatible pixel geometry.")

            acc = accumulators[zbin.label]
            if acc["data_counts"] is None:
                acc["data_counts"] = np.zeros(npix, dtype=float)
                acc["random_counts"] = np.zeros(npix, dtype=float)
                acc["backend"] = backend
                acc["npix"] = npix
            if acc["npix"] != npix or acc["backend"] != backend:
                raise ValueError("Region maps have incompatible pixel geometry.")
            acc["data_counts"] = np.asarray(acc["data_counts"], dtype=float) + data_counts
            acc["random_counts"] = np.asarray(acc["random_counts"], dtype=float) + random_counts
            acc["n_data"] = int(acc["n_data"]) + len(data_bin)
            acc["n_random"] = int(acc["n_random"]) + len(random_bin)
            acc["data_sources"].append(pair.data_source)
            acc["random_sources"].append(pair.random_source)
        del pair
        gc.collect()

    records: list[dict[str, object]] = []
    for zbin in z_bins:
        acc = accumulators[zbin.label]
        n_data = int(acc["n_data"])
        n_random = int(acc["n_random"])
        if n_data < 20 or n_random < 100:
            LOGGER.warning(
                "Skipping combined %s %s: too few rows after z filtering (data=%d, random=%d)",
                tracer,
                zbin.label,
                n_data,
                n_random,
            )
            continue
        sky_map = counts_to_overdensity_map(
            data_counts=np.asarray(acc["data_counts"], dtype=float),
            random_counts=np.asarray(acc["random_counts"], dtype=float),
            nside=config.analysis.nside,
            backend=str(acc["backend"]),
            min_random_per_pixel=config.analysis.min_random_per_pixel,
        )
        fit = fit_with_resampling(
            sky_map,
            rng=rng,
            bootstrap_samples=config.dipole.bootstrap_samples,
            null_permutations=config.dipole.null_permutations,
            poisson_mocks=config.dipole.poisson_mocks,
            block_null_mocks=config.dipole.block_null_mocks,
            block_null_regions=config.dipole.block_null_regions,
            jackknife_regions=config.dipole.jackknife_regions,
        )
        map_path = (
            config.paths.data_processed
            / f"{tracer}_{safe_region_label}_{zbin.label}_{_nside_label(config)}_combined_map.npz"
        )
        save_sky_map(str(map_path), sky_map)
        null_key = f"{tracer}_{safe_region_label}_{zbin.label}_combined"
        if fit.null_amplitudes is not None:
            null_store[null_key] = fit.null_amplitudes
            np.save(config.paths.tables / f"{null_key}_null_amplitudes.npy", fit.null_amplitudes)
        if fit.poisson_amplitudes is not None and len(fit.poisson_amplitudes):
            np.save(config.paths.tables / f"{null_key}_poisson_amplitudes.npy", fit.poisson_amplitudes)
        if fit.block_null_amplitudes is not None and len(fit.block_null_amplitudes):
            np.save(config.paths.tables / f"{null_key}_block_null_amplitudes.npy", fit.block_null_amplitudes)

        record = fit.to_record(tracer=tracer, z_min=zbin.z_min, z_max=zbin.z_max, n_data=n_data, n_random=n_random)
        record.update(
            {
                "region": combined_label,
                "combined_regions": combined_label,
                "synthetic": mode == "synthetic",
                "data_source": ", ".join(sorted(set(acc["data_sources"]))),
                "random_source": ", ".join(sorted(set(acc["random_sources"]))),
                "map_path": str(map_path),
                "nside": config.analysis.nside,
                "artifact_label": "combined_regions",
                "bootstrap_samples": config.dipole.bootstrap_samples,
            }
        )
        LOGGER.info(
            "%s %s combined %s: amp=%.4g axis=(RA %.1f, DEC %.1f) p=%.3g",
            tracer,
            combined_label,
            zbin.label,
            fit.amplitude,
            fit.ra_deg,
            fit.dec_deg,
            fit.null_p_value if fit.null_p_value is not None else float("nan"),
        )
        records.append(record)
    return records


def _analyze_combined_template_for_tracer(
    config: ProjectConfig,
    mode: str,
    tracer: str,
    regions: Sequence[str],
    rng: np.random.Generator,
    weight_mode: str,
    external_templates: Sequence[str],
) -> tuple[list[dict[str, object]], pd.DataFrame]:
    """Regress external templates after summing one tracer over several regions."""
    if tracer not in config.analysis.z_bins:
        raise KeyError(f"No redshift bins configured for tracer {tracer}")
    z_bins = make_redshift_bins(config.analysis.z_bins[tracer])
    combined_label = "+".join(regions)
    safe_region_label = _safe_label(combined_label)
    accumulators: dict[str, dict[str, object]] = {
        zbin.label: {
            "zbin": zbin,
            "data_counts": None,
            "random_counts": None,
            "n_data": 0,
            "n_random": 0,
            "backend": None,
            "npix": None,
            "data_sources": [],
            "random_sources": [],
        }
        for zbin in z_bins
    }

    for region in regions:
        pair = _load_pair_for_mode(config, mode, tracer, region, rng, weight_mode=weight_mode)
        for index, zbin in enumerate(z_bins):
            final_bin = index == len(z_bins) - 1
            data_bin = subset_by_redshift_bin(pair.data, zbin, final_bin=final_bin)
            random_bin = subset_by_redshift_bin(pair.randoms, zbin, final_bin=final_bin)
            data_counts, npix, backend = weighted_counts(data_bin, config.analysis.nside)
            random_counts, random_npix, random_backend = weighted_counts(random_bin, config.analysis.nside)
            if npix != random_npix or backend != random_backend:
                raise ValueError("Data and random maps have incompatible pixel geometry.")
            acc = accumulators[zbin.label]
            if acc["data_counts"] is None:
                acc["data_counts"] = np.zeros(npix, dtype=float)
                acc["random_counts"] = np.zeros(npix, dtype=float)
                acc["backend"] = backend
                acc["npix"] = npix
            if acc["npix"] != npix or acc["backend"] != backend:
                raise ValueError("Region maps have incompatible pixel geometry.")
            acc["data_counts"] = np.asarray(acc["data_counts"], dtype=float) + data_counts
            acc["random_counts"] = np.asarray(acc["random_counts"], dtype=float) + random_counts
            acc["n_data"] = int(acc["n_data"]) + len(data_bin)
            acc["n_random"] = int(acc["n_random"]) + len(random_bin)
            acc["data_sources"].append(pair.data_source)
            acc["random_sources"].append(pair.random_source)
        del pair
        gc.collect()

    records: list[dict[str, object]] = []
    coefficient_frames: list[pd.DataFrame] = []
    for zbin in z_bins:
        acc = accumulators[zbin.label]
        n_data = int(acc["n_data"])
        n_random = int(acc["n_random"])
        if n_data < 20 or n_random < 100:
            LOGGER.warning(
                "Skipping combined template %s %s: too few rows (data=%d, random=%d)",
                tracer,
                zbin.label,
                n_data,
                n_random,
            )
            continue
        sky_map = counts_to_overdensity_map(
            data_counts=np.asarray(acc["data_counts"], dtype=float),
            random_counts=np.asarray(acc["random_counts"], dtype=float),
            nside=config.analysis.nside,
            backend=str(acc["backend"]),
            min_random_per_pixel=config.analysis.min_random_per_pixel,
        )
        template_matrix, template_names = load_external_template_maps(
            external_templates,
            valid=sky_map.valid & np.isfinite(sky_map.delta),
            expected_npix=len(sky_map.delta),
        )
        regression = regress_templates(sky_map, template_matrix, template_names)
        raw_fit = fit_with_resampling(
            sky_map,
            rng=rng,
            bootstrap_samples=config.dipole.bootstrap_samples,
            null_permutations=config.dipole.null_permutations,
            poisson_mocks=0,
            block_null_mocks=config.dipole.block_null_mocks,
            block_null_regions=config.dipole.block_null_regions,
            jackknife_regions=config.dipole.jackknife_regions,
        )
        corrected_fit = fit_with_resampling(
            regression.corrected_map,
            rng=rng,
            bootstrap_samples=config.dipole.bootstrap_samples,
            null_permutations=config.dipole.null_permutations,
            poisson_mocks=0,
            block_null_mocks=config.dipole.block_null_mocks,
            block_null_regions=config.dipole.block_null_regions,
            jackknife_regions=config.dipole.jackknife_regions,
        )
        suffix = (
            f"{tracer}_{safe_region_label}_{zbin.label}_{_nside_label(config)}"
            "_combined_template_regressed"
        )
        raw_map_path = config.paths.data_processed / f"{suffix}_raw_map.npz"
        corrected_map_path = config.paths.data_processed / f"{suffix}_corrected_map.npz"
        save_sky_map(str(raw_map_path), sky_map)
        save_sky_map(str(corrected_map_path), regression.corrected_map)
        if len(regression.coefficients):
            coeffs = regression.coefficients.copy()
            coeffs.insert(0, "z_max", zbin.z_max)
            coeffs.insert(0, "z_min", zbin.z_min)
            coeffs.insert(0, "region", combined_label)
            coeffs.insert(0, "tracer", tracer)
            coefficient_frames.append(coeffs)

        axis_sep = float(angular_separation_deg(raw_fit.vector, corrected_fit.vector))
        axis_sep = min(axis_sep, 180.0 - axis_sep)
        records.append(
            {
                "tracer": tracer,
                "region": combined_label,
                "combined_regions": combined_label,
                "z_min": zbin.z_min,
                "z_max": zbin.z_max,
                "n_data": n_data,
                "n_random": n_random,
                "n_templates": len(regression.template_names),
                "template_names": ",".join(regression.template_names),
                "external_templates_requested": ",".join(external_templates),
                "template_weighted_r2": regression.weighted_r2,
                "raw_amplitude": raw_fit.amplitude,
                "raw_ra_deg": raw_fit.ra_deg,
                "raw_dec_deg": raw_fit.dec_deg,
                "raw_null_p_value": raw_fit.null_p_value,
                "raw_block_null_p_value": raw_fit.block_null_p_value,
                "corrected_amplitude": corrected_fit.amplitude,
                "corrected_ra_deg": corrected_fit.ra_deg,
                "corrected_dec_deg": corrected_fit.dec_deg,
                "corrected_null_p_value": corrected_fit.null_p_value,
                "corrected_block_null_p_value": corrected_fit.block_null_p_value,
                "amplitude_ratio_corrected_to_raw": (
                    corrected_fit.amplitude / raw_fit.amplitude if raw_fit.amplitude > 0 else np.nan
                ),
                "axis_shift_raw_to_corrected_deg": axis_sep,
                "raw_jackknife_axis_max_shift_deg": raw_fit.jackknife_axis_max_shift_deg,
                "corrected_jackknife_axis_max_shift_deg": (
                    corrected_fit.jackknife_axis_max_shift_deg
                ),
                "raw_map_path": str(raw_map_path),
                "corrected_map_path": str(corrected_map_path),
                "weight_mode": weight_mode,
                "nside": config.analysis.nside,
                "data_source": ", ".join(sorted(set(acc["data_sources"]))),
                "random_source": ", ".join(sorted(set(acc["random_sources"]))),
            }
        )
        LOGGER.info(
            "%s %s combined template %s: raw amp=%.4g p=%.3g -> corrected amp=%.4g p=%.3g",
            tracer,
            combined_label,
            zbin.label,
            raw_fit.amplitude,
            raw_fit.null_p_value,
            corrected_fit.amplitude,
            corrected_fit.null_p_value,
        )
    coefficients = pd.concat(coefficient_frames, ignore_index=True) if coefficient_frames else pd.DataFrame()
    return records, coefficients


def _analyze_pair(
    config: ProjectConfig,
    pair: CatalogPair,
    rng: np.random.Generator,
    null_store: dict[str, np.ndarray],
    artifact_label: Optional[str] = None,
    extra_fields: Optional[dict[str, object]] = None,
) -> list[dict[str, object]]:
    if pair.tracer not in config.analysis.z_bins:
        raise KeyError(f"No redshift bins configured for tracer {pair.tracer}")
    z_bins = make_redshift_bins(config.analysis.z_bins[pair.tracer])
    records = []
    for index, zbin in enumerate(z_bins):
        data_bin = subset_by_redshift_bin(pair.data, zbin, final_bin=index == len(z_bins) - 1)
        random_bin = subset_by_redshift_bin(pair.randoms, zbin, final_bin=index == len(z_bins) - 1)
        if len(data_bin) < 20 or len(random_bin) < 100:
            LOGGER.warning(
                "Skipping %s %s: too few rows after z filtering (data=%d, random=%d)",
                pair.tracer,
                zbin.label,
                len(data_bin),
                len(random_bin),
            )
            continue
        sky_map = build_overdensity_map(
            data_bin,
            random_bin,
            nside=config.analysis.nside,
            min_random_per_pixel=config.analysis.min_random_per_pixel,
        )
        fit = fit_with_resampling(
            sky_map,
            rng=rng,
            bootstrap_samples=config.dipole.bootstrap_samples,
            null_permutations=config.dipole.null_permutations,
            poisson_mocks=config.dipole.poisson_mocks,
            block_null_mocks=config.dipole.block_null_mocks,
            block_null_regions=config.dipole.block_null_regions,
            jackknife_regions=config.dipole.jackknife_regions,
        )
        nside_label = _nside_label(config)
        suffix = f"_{artifact_label}" if artifact_label and artifact_label != nside_label else ""
        map_path = (
            config.paths.data_processed
            / f"{pair.tracer}_{pair.region}_{zbin.label}_{nside_label}{suffix}_map.npz"
        )
        save_sky_map(str(map_path), sky_map)
        if fit.null_amplitudes is not None:
            null_key = f"{pair.tracer}_{pair.region}_{zbin.label}{suffix}"
            null_store[null_key] = fit.null_amplitudes
            np.save(config.paths.tables / f"{null_key}_null_amplitudes.npy", fit.null_amplitudes)
        if fit.poisson_amplitudes is not None and len(fit.poisson_amplitudes):
            np.save(config.paths.tables / f"{pair.tracer}_{pair.region}_{zbin.label}{suffix}_poisson_amplitudes.npy", fit.poisson_amplitudes)
        if fit.block_null_amplitudes is not None and len(fit.block_null_amplitudes):
            np.save(
                config.paths.tables / f"{pair.tracer}_{pair.region}_{zbin.label}{suffix}_block_null_amplitudes.npy",
                fit.block_null_amplitudes,
            )
        record = fit.to_record(
            tracer=pair.tracer,
            z_min=zbin.z_min,
            z_max=zbin.z_max,
            n_data=len(data_bin),
            n_random=len(random_bin),
        )
        record.update(
            {
                "region": pair.region,
                "synthetic": pair.synthetic,
                "data_source": pair.data_source,
                "random_source": pair.random_source,
                "map_path": str(map_path),
                "nside": config.analysis.nside,
                "artifact_label": artifact_label or "",
                "bootstrap_samples": config.dipole.bootstrap_samples,
            }
        )
        if extra_fields:
            record.update(extra_fields)
        LOGGER.info(
            "%s %s: amp=%.4g axis=(RA %.1f, DEC %.1f) p=%.3g",
            pair.tracer,
            zbin.label,
            fit.amplitude,
            fit.ra_deg,
            fit.dec_deg,
            fit.null_p_value if fit.null_p_value is not None else float("nan"),
        )
        records.append(record)
    return records


def _build_split_definitions(
    pair: CatalogPair,
    split_columns: Optional[Sequence[str]],
    ra_sectors: int,
) -> list[tuple[str, str, pd.Series, pd.Series]]:
    splits: list[tuple[str, str, pd.Series, pd.Series]] = []
    requested = [column.lower() for column in (split_columns or ["photsys", "ntile", "frac_tlobs_tiles"])]
    for column in requested:
        if column not in pair.data.columns or column not in pair.randoms.columns:
            LOGGER.warning("Split column %s not available for %s %s", column, pair.tracer, pair.region)
            continue
        if pd.api.types.is_numeric_dtype(pair.data[column]):
            splits.extend(_numeric_splits(pair, column))
        else:
            splits.extend(_categorical_splits(pair, column))
    if ra_sectors > 1:
        edges = np.linspace(0.0, 360.0, ra_sectors + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            name = f"{lo:.0f}_{hi:.0f}"
            data_mask = (pair.data["ra"] >= lo) & (pair.data["ra"] < hi)
            random_mask = (pair.randoms["ra"] >= lo) & (pair.randoms["ra"] < hi)
            splits.append(("ra_sector", name, data_mask, random_mask))
    return splits


def _categorical_splits(pair: CatalogPair, column: str) -> list[tuple[str, str, pd.Series, pd.Series]]:
    values = sorted(set(pair.data[column].dropna().astype(str)) & set(pair.randoms[column].dropna().astype(str)))
    splits = []
    for value in values:
        splits.append((column, value, pair.data[column].astype(str) == value, pair.randoms[column].astype(str) == value))
    return splits


def _numeric_splits(pair: CatalogPair, column: str) -> list[tuple[str, str, pd.Series, pd.Series]]:
    data_values = pd.to_numeric(pair.data[column], errors="coerce")
    random_values = pd.to_numeric(pair.randoms[column], errors="coerce")
    unique_values = sorted(set(data_values.dropna().unique()) & set(random_values.dropna().unique()))
    if 1 < len(unique_values) <= 12 and all(float(value).is_integer() for value in unique_values):
        return [
            (
                column,
                str(int(value)),
                data_values == value,
                random_values == value,
            )
            for value in unique_values
        ]
    quantiles = np.unique(np.nanquantile(data_values.to_numpy(dtype=float), [0.0, 1 / 3, 2 / 3, 1.0]))
    splits = []
    for index, (lo, hi) in enumerate(zip(quantiles[:-1], quantiles[1:])):
        if index == len(quantiles) - 2:
            data_mask = (data_values >= lo) & (data_values <= hi)
            random_mask = (random_values >= lo) & (random_values <= hi)
        else:
            data_mask = (data_values >= lo) & (data_values < hi)
            random_mask = (random_values >= lo) & (random_values < hi)
        splits.append((column, f"q{index + 1}_{lo:.3g}_{hi:.3g}", data_mask, random_mask))
    return splits


def _safe_label(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _nside_label(config: ProjectConfig) -> str:
    return f"nside{config.analysis.nside}"


def _write_systematics_audit_report(report_path: Path, results: pd.DataFrame) -> Path:
    if len(results) == 0:
        report_path.write_text("# Systematics audit\n\nNo split results were produced.\n", encoding="utf-8")
        return report_path
    columns = [
        "split_variable",
        "split_name",
        "tracer",
        "region",
        "z_min",
        "z_max",
        "amplitude",
        "null_p_value",
        "bonferroni_p",
        "bh_fdr_p",
        "poisson_p_value",
        "block_null_p_value",
        "jackknife_axis_max_shift_deg",
        "n_data",
        "n_random",
    ]
    columns = [column for column in columns if column in results.columns]
    lines = [
        "# Systematics split audit",
        "",
        "These split tests check whether a candidate dipole is stable across available survey columns.",
        "A split-specific low p-value is a systematics warning, not a cosmological claim.",
        "",
        results[columns].sort_values(["split_variable", "split_name", "z_min"]).to_markdown(index=False),
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    return report_path


def make_plots(
    config: ProjectConfig,
    results: Optional[pd.DataFrame] = None,
    null_store: Optional[dict[str, np.ndarray]] = None,
) -> list[Path]:
    """Generate summary plots from current results."""
    config.paths.ensure_dirs()
    if results is None:
        results_path = config.paths.tables / "first_pass_dipoles.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"No results table found: {results_path}")
        results = pd.read_csv(results_path)
    outputs: list[Path] = []
    if len(results) == 0:
        return outputs
    try:
        outputs.append(plot_amplitude_by_redshift(results, config.paths.figures / "dipole_amplitude_by_z.png"))
        outputs.append(plot_axis_by_redshift(results, config.paths.figures / "dipole_axis_by_z.png"))

        first = results.iloc[0]
        if Path(first["map_path"]).exists():
            loaded = np.load(first["map_path"], allow_pickle=False)
            from cosmo_gradient.maps import SkyMap

            sky_map = SkyMap(
                nside=int(loaded["nside"]),
                backend=str(loaded["backend"]),
                data_counts=loaded["data_counts"],
                random_counts=loaded["random_counts"],
                alpha=float(loaded["alpha"]),
                delta=loaded["delta"],
                valid=loaded["valid"],
                pixel_vectors=loaded["pixel_vectors"],
            )
            outputs.append(
                plot_overdensity_map(
                    sky_map,
                    config.paths.figures / "example_overdensity_map.png",
                    title=f"{first['tracer']} {first['z_min']:.2f}-{first['z_max']:.2f}",
                )
            )

        if null_store:
            first_key = next(iter(null_store))
            tracer, _, _ = first_key.partition("_")
            matching = results[results["tracer"] == tracer]
            observed = float(matching.iloc[0]["amplitude"]) if len(matching) else float(results.iloc[0]["amplitude"])
            outputs.append(
                plot_null_distribution(
                    null_store[first_key],
                    observed,
                    config.paths.figures / f"{first_key}_null_distribution.png",
                )
            )
    except ImportError:
        LOGGER.warning("matplotlib is not installed; skipping plots. Install with `uv sync --extra plot`.")
    return outputs


def write_report(config: ProjectConfig, results: Optional[pd.DataFrame] = None) -> Path:
    """Write the markdown first-pass report."""
    config.paths.ensure_dirs()
    if results is None:
        results_path = config.paths.tables / "first_pass_dipoles.csv"
        if not results_path.exists():
            raise FileNotFoundError(f"No results table found: {results_path}")
        results = pd.read_csv(results_path)

    report_path = config.paths.reports / config.report.filename
    lines = [
        "# First-pass directional-gradient report",
        "",
        "## Scope",
        "",
        "This report tests whether tracer density maps are compatible with an isotropic null model",
        "after correcting the angular footprint with random catalogs. It does not claim evidence",
        "for a genesis-gradient model or any alternative to Lambda-CDM.",
        "",
        "## Data mode",
        "",
    ]
    if len(results) and bool(results["synthetic"].any()):
        lines.extend(
            [
                "- Mode: synthetic validation.",
                "- Interpretation: pipeline smoke test only; no DESI measurement is reported.",
                "",
            ]
        )
    else:
        lines.extend(["- Mode: real DESI-style catalog inputs.", ""])

    if len(results):
        display_cols = [
            "tracer",
            "region",
            "z_min",
            "z_max",
            "n_data",
            "n_random",
            "amplitude",
            "amplitude_std",
            "ra_deg",
            "dec_deg",
            "null_p_value",
        ]
        table = results[display_cols].copy()
        lines.extend(["## Fitted dipoles", "", table.to_markdown(index=False), ""])
        separations = pairwise_axis_separations(results)
        if len(separations):
            lines.extend(
                [
                    "## Cross-bin and cross-tracer axis separations",
                    "",
                    separations.head(30).to_markdown(index=False),
                    "",
                ]
            )
    else:
        lines.extend(["No fitted bins were produced.", ""])

    lines.extend(["## Warnings and null-model discipline", ""])
    for flag in result_quality_flags(results):
        lines.append(f"- {flag}")
    lines.extend(
        [
            "- A robust DESI analysis must audit survey geometry, extinction, depth, target selection, redshift failures, stellar contamination, and local-motion effects.",
            "- A stable axis must be tested across independent tracer classes and redshift bins before physical interpretation.",
            "",
            "## First-pass conclusion",
            "",
        ]
    )
    if len(results) and bool(results["synthetic"].any()):
        lines.append(
            "Synthetic mode completed. The outputs validate plumbing and estimator behavior, not the hypothesis."
        )
    elif len(results):
        lines.append(
            "Real-data first pass completed. Treat any low p-values as prompts for systematics checks, not discoveries."
        )
    else:
        lines.append("No conclusion: no valid fitted bins were available.")
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOGGER.info("Wrote %s", report_path)
    return report_path
