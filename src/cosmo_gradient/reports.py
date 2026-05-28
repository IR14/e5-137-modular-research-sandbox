"""Science-report helpers for first-pass DESI directional checks."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from cosmo_gradient.config import ProjectConfig
from cosmo_gradient.statistics import (
    add_multiple_testing_corrections,
    compare_to_reference_axes,
    region_axis_consistency,
    tracer_axis_consistency,
)


def write_axis_diagnostics(
    results_path: Path,
    output_prefix: str,
    tables_dir: Path,
    reports_dir: Path,
) -> tuple[Path, Path, Path, Path]:
    """Write CMB/reference, cross-region, and cross-tracer axis diagnostics."""
    results = pd.read_csv(results_path)
    reference = compare_to_reference_axes(results)
    region = region_axis_consistency(results)
    tracer = tracer_axis_consistency(results)

    tables_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    reference_path = tables_dir / f"{output_prefix}_reference_axes.csv"
    region_path = tables_dir / f"{output_prefix}_region_consistency.csv"
    tracer_path = tables_dir / f"{output_prefix}_tracer_consistency.csv"
    report_path = reports_dir / f"{output_prefix}_axis_diagnostics.md"

    reference.to_csv(reference_path, index=False)
    region.to_csv(region_path, index=False)
    tracer.to_csv(tracer_path, index=False)
    _write_axis_diagnostics_markdown(report_path, reference, region, tracer)
    return reference_path, region_path, tracer_path, report_path


def write_master_science_report(
    config: ProjectConfig,
    all_results_path: Path,
    output_path: Optional[Path] = None,
    combined_results_path: Optional[Path] = None,
    nside_stability_path: Optional[Path] = None,
    weight_summary_path: Optional[Path] = None,
    template_summary_path: Optional[Path] = None,
    multipole_summary_path: Optional[Path] = None,
    systematics_path: Optional[Path] = None,
    focused_null_path: Optional[Path] = None,
    jackknife_raw_path: Optional[Path] = None,
    jackknife_corrected_path: Optional[Path] = None,
    axis_reference_path: Optional[Path] = None,
    region_consistency_path: Optional[Path] = None,
    tracer_consistency_path: Optional[Path] = None,
) -> Path:
    """Write a consolidated markdown report for the current first-pass campaign."""
    output = output_path or (config.paths.reports / "master_first_pass_science_report.md")
    output.parent.mkdir(parents=True, exist_ok=True)

    all_results = pd.read_csv(all_results_path)
    corrected = (
        all_results
        if {"bonferroni_p", "bh_fdr_p"} <= set(all_results.columns)
        else add_multiple_testing_corrections(all_results)
    )

    lines = [
        "# Master first-pass science report",
        "",
        "## Scope",
        "",
        "This report summarizes first-pass DESI DR1 LSS directional-density checks for the",
        "genesis-gradient hypothesis. The analysis is framed as a null test against an",
        "isotropic Lambda-CDM-compatible sky after survey-footprint correction with random catalogs.",
        "",
        "It is not a claim of evidence for a new cosmology. Low local p-values are treated as",
        "systematics prompts until they survive masking, weighting, tracer, region, redshift,",
        "and look-elsewhere checks.",
        "",
        "## Primary all-tracer result",
        "",
        f"- Input table: `{all_results_path}`",
        f"- Tested bins: {len(corrected)}",
        f"- Minimum raw permutation p-value: {corrected['null_p_value'].min():.6g}",
        f"- Minimum Bonferroni-adjusted p-value: {corrected['bonferroni_p'].min():.6g}",
        f"- Minimum BH-FDR adjusted p-value: {corrected['bh_fdr_p'].min():.6g}",
        "",
        corrected[
            [
                "tracer",
                "region",
                "z_min",
                "z_max",
                "amplitude",
                "ra_deg",
                "dec_deg",
                "null_p_value",
                "bonferroni_p",
                "bh_fdr_p",
                "jackknife_axis_max_shift_deg",
            ]
        ].sort_values("null_p_value").head(10).to_markdown(index=False),
        "",
    ]

    _append_optional_table(
        lines,
        "## Combined NGC+SGC maps",
        combined_results_path,
        [
            "tracer",
            "region",
            "z_min",
            "z_max",
            "amplitude",
            "ra_deg",
            "dec_deg",
            "null_p_value",
            "block_null_p_value",
            "n_data",
            "n_random",
        ],
    )
    _append_optional_table(
        lines,
        "## Nside robustness for ELG SGC",
        nside_stability_path,
        [
            "tracer",
            "region",
            "z_min",
            "z_max",
            "nside_count",
            "max_axis_shift_deg",
            "amplitude_range",
            "p_value_min",
            "p_value_max",
        ],
    )
    _append_optional_table(
        lines,
        "## ELG SGC weight-mode robustness",
        weight_summary_path,
        [
            "z_min",
            "z_max",
            "weight_mode",
            "amplitude",
            "ra_deg",
            "dec_deg",
            "null_p_value",
            "axis_sep_from_desi_deg",
        ],
    )
    _append_optional_table(
        lines,
        "## ELG SGC template-regression weight-mode summary",
        template_summary_path,
        [
            "template_weight_mode",
            "z_min",
            "z_max",
            "template_weighted_r2",
            "raw_amplitude",
            "raw_null_p_value",
            "corrected_amplitude",
            "corrected_null_p_value",
            "amplitude_ratio_corrected_to_raw",
            "axis_shift_raw_to_corrected_deg",
            "corrected_block_null_p_value",
        ],
        max_rows=12,
    )
    _append_optional_table(
        lines,
        "## ELG low-z multipole diagnostics",
        multipole_summary_path,
        [
            "map_case",
            "model",
            "dipole_amplitude",
            "ra_deg",
            "dec_deg",
            "weighted_r2",
            "weighted_design_condition",
            "quadrupole_norm",
            "axis_shift_from_dipole_only_deg",
            "amplitude_ratio_to_dipole_only",
        ],
        max_rows=12,
    )
    _append_optional_table(
        lines,
        "## ELG SGC systematics splits",
        systematics_path,
        [
            "split_variable",
            "split_name",
            "tracer",
            "region",
            "z_min",
            "z_max",
            "null_p_value",
            "bonferroni_p",
            "bh_fdr_p",
            "amplitude",
        ],
        sort_by="null_p_value",
        max_rows=12,
    )
    _append_optional_table(
        lines,
        "## Focused block-null diagnostic",
        focused_null_path,
        [
            "tracer",
            "region",
            "z_min",
            "z_max",
            "amplitude",
            "null_p_value",
            "poisson_p_value",
            "block_null_p_value",
            "jackknife_axis_max_shift_deg",
        ],
    )
    _append_optional_table(
        lines,
        "## ELG SGC raw-map RA jackknife",
        jackknife_raw_path,
        [
            "region_index",
            "removed_ra_min_deg",
            "removed_ra_max_deg",
            "removed_dec_min_deg",
            "removed_dec_max_deg",
            "leave_one_out_amplitude",
            "amplitude_change",
            "axis_shift_deg",
        ],
        sort_by="axis_shift_deg",
        sort_desc=True,
        max_rows=6,
    )
    _append_optional_table(
        lines,
        "## ELG SGC template-corrected RA jackknife",
        jackknife_corrected_path,
        [
            "region_index",
            "removed_ra_min_deg",
            "removed_ra_max_deg",
            "removed_dec_min_deg",
            "removed_dec_max_deg",
            "leave_one_out_amplitude",
            "amplitude_change",
            "axis_shift_deg",
        ],
        sort_by="axis_shift_deg",
        sort_desc=True,
        max_rows=6,
    )
    _append_optional_table(
        lines,
        "## Reference-axis comparison",
        axis_reference_path,
        [
            "reference",
            "tracer",
            "region",
            "z_min",
            "z_max",
            "axis_separation_deg",
            "null_p_value",
            "amplitude",
        ],
        sort_by="axis_separation_deg",
        max_rows=12,
    )
    _append_optional_table(
        lines,
        "## Region consistency",
        region_consistency_path,
        [
            "tracer",
            "z_min",
            "z_max",
            "left_region",
            "right_region",
            "axis_separation_deg",
            "left_p",
            "right_p",
        ],
        sort_by="axis_separation_deg",
        max_rows=12,
    )
    _append_optional_table(
        lines,
        "## Cross-tracer consistency",
        tracer_consistency_path,
        [
            "region",
            "left_tracer",
            "right_tracer",
            "left_z_min",
            "left_z_max",
            "right_z_min",
            "right_z_max",
            "z_overlap",
            "axis_separation_deg",
            "left_p",
            "right_p",
        ],
        sort_by="axis_separation_deg",
        max_rows=12,
    )

    lines.extend(
        [
            "## Interpretation",
            "",
            "- The current all-tracer grid does not reject the isotropic null after look-elsewhere correction.",
            "- The lowest local p-value is the DESI-weighted ELG SGC low-redshift bin, but it is not significant after correction and has large jackknife sensitivity.",
            "- The same feature is not mirrored by LRG or QSO, and the NGC counterpart is weaker.",
            "- ELG SGC is strongly weight-sensitive: removing or simplifying DESI weights produces much larger amplitudes and different axes, which is a survey-selection warning.",
            "- Any future positive claim would require end-to-end survey mocks and deeper systematics modeling beyond this first-pass estimator.",
            "",
            "## Recommended next checks",
            "",
            "- Replace empirical nulls with official or mock-based DESI clustering mocks when available for this exact catalog selection.",
            "- Add Galactic extinction, stellar density, imaging depth, and redshift-failure maps as explicit regressors.",
            "- Run the same estimator on independent catalog versions or alternative target selections.",
        ]
    )
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _append_optional_table(
    lines: list[str],
    heading: str,
    path: Optional[Path],
    columns: list[str],
    sort_by: Optional[str] = None,
    sort_desc: bool = False,
    max_rows: int = 10,
) -> None:
    if path is None or not path.exists():
        return
    frame = pd.read_csv(path)
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=not sort_desc)
    available = [column for column in columns if column in frame.columns]
    if not available:
        return
    lines.extend([heading, "", f"- Source: `{path}`", "", frame[available].head(max_rows).to_markdown(index=False), ""])


def _write_axis_diagnostics_markdown(
    report_path: Path,
    reference: pd.DataFrame,
    region: pd.DataFrame,
    tracer: pd.DataFrame,
) -> None:
    lines = [
        "# Axis diagnostics",
        "",
        "These diagnostics compare fitted axes with reference directions and with each other.",
        "They are consistency checks only; a small angular separation is not significant unless",
        "the underlying dipole measurement is itself significant under the null model.",
        "",
    ]
    if not reference.empty:
        lines.extend(
            [
                "## Reference axes",
                "",
                reference[
                    [
                        "reference",
                        "tracer",
                        "region",
                        "z_min",
                        "z_max",
                        "axis_separation_deg",
                        "null_p_value",
                        "amplitude",
                    ]
                ].head(12).to_markdown(index=False),
                "",
            ]
        )
    if not region.empty:
        lines.extend(["## Region consistency", "", region.head(12).to_markdown(index=False), ""])
    if not tracer.empty:
        lines.extend(["## Cross-tracer consistency", "", tracer.head(12).to_markdown(index=False), ""])
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
