"""Summary statistics for fitted dipole axes."""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from cosmo_gradient.coords import angular_separation_deg, radec_to_unit


DEFAULT_REFERENCE_AXES: dict[str, tuple[float, float, str]] = {
    "cmb_dipole_apex": (
        167.942,
        -6.944,
        "Approximate CMB dipole apex in equatorial J2000 coordinates.",
    )
}


def pairwise_axis_separations(results: pd.DataFrame) -> pd.DataFrame:
    """Return pairwise angular separations for fitted dipole axes."""
    rows = []
    for (_, a), (_, b) in itertools.combinations(results.iterrows(), 2):
        vector_a = np.array([a["vector_x"], a["vector_y"], a["vector_z"]], dtype=float)
        vector_b = np.array([b["vector_x"], b["vector_y"], b["vector_z"]], dtype=float)
        rows.append(
            {
                "left": f"{a['tracer']} {a['z_min']:.3f}-{a['z_max']:.3f}",
                "right": f"{b['tracer']} {b['z_min']:.3f}-{b['z_max']:.3f}",
                "separation_deg": float(angular_separation_deg(vector_a, vector_b)),
            }
        )
    return pd.DataFrame(rows)


def result_quality_flags(results: pd.DataFrame) -> list[str]:
    """Return human-readable warnings about first-pass result limitations."""
    flags = [
        "This is a first-pass density-gradient analysis, not evidence for or against a new cosmology.",
        "Survey mask, selection function, extinction, depth, and redshift failures must be audited before interpretation.",
    ]
    if "synthetic" in results.columns and bool(results["synthetic"].any()):
        flags.append("Synthetic mode was used; fitted axes are validation outputs, not DESI measurements.")
    if "null_p_value" in results.columns and (results["null_p_value"] < 0.05).any():
        flags.append("At least one bin has a low permutation p-value; inspect systematics before any physical claim.")
    return flags


def add_multiple_testing_corrections(
    results: pd.DataFrame,
    p_value_column: str = "null_p_value",
) -> pd.DataFrame:
    """Return results sorted by p-value with Bonferroni and BH-FDR corrections."""
    if p_value_column not in results.columns:
        raise KeyError(f"Missing p-value column: {p_value_column}")
    corrected = results.copy()
    corrected[p_value_column] = pd.to_numeric(corrected[p_value_column], errors="coerce")
    corrected = corrected.dropna(subset=[p_value_column]).sort_values(p_value_column)
    corrected = corrected.reset_index(drop=True)
    if corrected.empty:
        corrected["test_rank"] = []
        corrected["bonferroni_p"] = []
        corrected["bh_fdr_p"] = []
        return corrected

    n_tests = len(corrected)
    ranks = np.arange(1, n_tests + 1, dtype=float)
    corrected["test_rank"] = ranks.astype(int)
    corrected["bonferroni_p"] = np.minimum(corrected[p_value_column].to_numpy(dtype=float) * n_tests, 1.0)

    bh_raw = corrected[p_value_column].to_numpy(dtype=float) * n_tests / ranks
    bh_adjusted = np.minimum.accumulate(bh_raw[::-1])[::-1]
    corrected["bh_fdr_p"] = np.minimum(bh_adjusted, 1.0)
    return corrected


def overlapping_redshift_axis_separations(results: pd.DataFrame) -> pd.DataFrame:
    """Return axis separations for rows in the same region with overlapping redshift bins."""
    rows = []
    required = {"region", "tracer", "z_min", "z_max", "vector_x", "vector_y", "vector_z"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing columns for axis-separation summary: {sorted(missing)}")
    working = results.copy()
    if "z_bin" not in working.columns:
        working["z_bin"] = working["z_min"].map(lambda value: f"{value:g}") + "-" + working["z_max"].map(lambda value: f"{value:g}")
    for (_, left), (_, right) in itertools.combinations(working.iterrows(), 2):
        if left["region"] != right["region"]:
            continue
        overlap = max(0.0, min(float(left["z_max"]), float(right["z_max"])) - max(float(left["z_min"]), float(right["z_min"])))
        if overlap <= 0.0:
            continue
        left_vector = np.array([left["vector_x"], left["vector_y"], left["vector_z"]], dtype=float)
        right_vector = np.array([right["vector_x"], right["vector_y"], right["vector_z"]], dtype=float)
        rows.append(
            {
                "region": left["region"],
                "left": f"{left['tracer']} {left['z_bin']}",
                "right": f"{right['tracer']} {right['z_bin']}",
                "z_overlap": float(overlap),
                "axis_separation_deg": float(angular_separation_deg(left_vector, right_vector)),
                "left_p": float(left["null_p_value"]) if "null_p_value" in left else np.nan,
                "right_p": float(right["null_p_value"]) if "null_p_value" in right else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("axis_separation_deg").reset_index(drop=True)


def compare_to_reference_axes(
    results: pd.DataFrame,
    references: dict[str, tuple[float, float, str]] | None = None,
) -> pd.DataFrame:
    """Return angular distances from fitted axes to named reference directions."""
    required = {"tracer", "region", "z_min", "z_max", "vector_x", "vector_y", "vector_z"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing columns for reference-axis comparison: {sorted(missing)}")
    refs = references or DEFAULT_REFERENCE_AXES
    rows = []
    for ref_name, (ref_ra, ref_dec, description) in refs.items():
        ref_vector = radec_to_unit([ref_ra], [ref_dec])[0]
        for _, row in results.iterrows():
            axis = np.array([row["vector_x"], row["vector_y"], row["vector_z"]], dtype=float)
            sep = float(angular_separation_deg(axis, ref_vector))
            rows.append(
                {
                    "reference": ref_name,
                    "reference_ra_deg": float(ref_ra),
                    "reference_dec_deg": float(ref_dec),
                    "reference_description": description,
                    "tracer": row["tracer"],
                    "region": row["region"],
                    "z_min": float(row["z_min"]),
                    "z_max": float(row["z_max"]),
                    "axis_ra_deg": float(row["ra_deg"]) if "ra_deg" in row else np.nan,
                    "axis_dec_deg": float(row["dec_deg"]) if "dec_deg" in row else np.nan,
                    "axis_separation_deg": min(sep, 180.0 - sep),
                    "null_p_value": float(row["null_p_value"]) if "null_p_value" in row else np.nan,
                    "amplitude": float(row["amplitude"]) if "amplitude" in row else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values(["reference", "axis_separation_deg"]).reset_index(drop=True)


def region_axis_consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Compare axes for the same tracer/redshift bin across different regions."""
    required = {"tracer", "region", "z_min", "z_max", "vector_x", "vector_y", "vector_z"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing columns for region-axis consistency: {sorted(missing)}")
    rows = []
    for key, group in results.groupby(["tracer", "z_min", "z_max"], dropna=False):
        if group["region"].nunique() < 2:
            continue
        for (_, left), (_, right) in itertools.combinations(group.iterrows(), 2):
            left_vector = np.array([left["vector_x"], left["vector_y"], left["vector_z"]], dtype=float)
            right_vector = np.array([right["vector_x"], right["vector_y"], right["vector_z"]], dtype=float)
            sep = float(angular_separation_deg(left_vector, right_vector))
            rows.append(
                {
                    "tracer": key[0],
                    "z_min": float(key[1]),
                    "z_max": float(key[2]),
                    "left_region": left["region"],
                    "right_region": right["region"],
                    "axis_separation_deg": min(sep, 180.0 - sep),
                    "left_p": float(left["null_p_value"]) if "null_p_value" in left else np.nan,
                    "right_p": float(right["null_p_value"]) if "null_p_value" in right else np.nan,
                    "left_amplitude": float(left["amplitude"]) if "amplitude" in left else np.nan,
                    "right_amplitude": float(right["amplitude"]) if "amplitude" in right else np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values("axis_separation_deg").reset_index(drop=True) if rows else pd.DataFrame()


def tracer_axis_consistency(results: pd.DataFrame) -> pd.DataFrame:
    """Compare axes between different tracers in the same region and overlapping redshift bins."""
    required = {"region", "tracer", "z_min", "z_max", "vector_x", "vector_y", "vector_z"}
    missing = required - set(results.columns)
    if missing:
        raise KeyError(f"Missing columns for tracer-axis consistency: {sorted(missing)}")
    rows = []
    for (_, left), (_, right) in itertools.combinations(results.iterrows(), 2):
        if left["region"] != right["region"] or left["tracer"] == right["tracer"]:
            continue
        overlap = max(0.0, min(float(left["z_max"]), float(right["z_max"])) - max(float(left["z_min"]), float(right["z_min"])))
        if overlap <= 0.0:
            continue
        left_vector = np.array([left["vector_x"], left["vector_y"], left["vector_z"]], dtype=float)
        right_vector = np.array([right["vector_x"], right["vector_y"], right["vector_z"]], dtype=float)
        sep = float(angular_separation_deg(left_vector, right_vector))
        rows.append(
            {
                "region": left["region"],
                "left_tracer": left["tracer"],
                "right_tracer": right["tracer"],
                "left_z_min": float(left["z_min"]),
                "left_z_max": float(left["z_max"]),
                "right_z_min": float(right["z_min"]),
                "right_z_max": float(right["z_max"]),
                "z_overlap": float(overlap),
                "axis_separation_deg": min(sep, 180.0 - sep),
                "left_p": float(left["null_p_value"]) if "null_p_value" in left else np.nan,
                "right_p": float(right["null_p_value"]) if "null_p_value" in right else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("axis_separation_deg").reset_index(drop=True) if rows else pd.DataFrame()


def write_look_elsewhere_summary(
    results_path: Path,
    output_csv: Path,
    output_report: Path,
    axis_output_csv: Path | None = None,
    description: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write multiple-testing and overlapping-axis summaries for a results CSV."""
    results = pd.read_csv(results_path)
    corrected = add_multiple_testing_corrections(results)
    if "z_bin" not in corrected.columns and {"z_min", "z_max"} <= set(corrected.columns):
        corrected["z_bin"] = corrected["z_min"].map(lambda value: f"{value:g}") + "-" + corrected["z_max"].map(lambda value: f"{value:g}")
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_csv, index=False)

    separations = overlapping_redshift_axis_separations(corrected) if not corrected.empty else pd.DataFrame()
    if axis_output_csv is not None:
        axis_output_csv.parent.mkdir(parents=True, exist_ok=True)
        separations.to_csv(axis_output_csv, index=False)

    output_report.parent.mkdir(parents=True, exist_ok=True)
    _write_look_elsewhere_markdown(output_report, corrected, separations, description)
    return corrected, separations


def _write_look_elsewhere_markdown(
    output_report: Path,
    corrected: pd.DataFrame,
    separations: pd.DataFrame,
    description: str,
) -> None:
    lines = ["# Look-elsewhere summary", ""]
    if description:
        lines.extend([description, ""])
    if corrected.empty:
        lines.append("No valid p-values were available.")
        output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return

    lines.extend(
        [
            f"- Number of tested tracer/region/redshift bins: {len(corrected)}",
            f"- Minimum permutation p-value: {corrected['null_p_value'].min():.6g}",
            f"- Minimum Bonferroni-adjusted p-value: {corrected['bonferroni_p'].min():.6g}",
            f"- Minimum BH-FDR adjusted p-value: {corrected['bh_fdr_p'].min():.6g}",
            "",
            "## Lowest p-value bins",
            "",
        ]
    )
    preferred_columns = [
        "test_rank",
        "tracer",
        "region",
        "z_bin",
        "n_data",
        "n_random",
        "amplitude",
        "ra_deg",
        "dec_deg",
        "null_p_value",
        "bonferroni_p",
        "bh_fdr_p",
        "poisson_p_value",
        "jackknife_axis_max_shift_deg",
    ]
    table_columns = [column for column in preferred_columns if column in corrected.columns]
    lines.append(corrected[table_columns].head(10).to_markdown(index=False))
    lines.append("")
    if not separations.empty:
        lines.extend(
            [
                "## Closest axes among overlapping redshift intervals",
                "",
                separations.head(10).to_markdown(index=False),
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "No directional claim should be made from local p-values without accounting for the",
            "number of tracer, region, redshift, resolution, weight, and split tests that were tried.",
            "Poisson p-values are shot-noise diagnostics only; the permutation p-value is the",
            "first-pass null statistic used for this correction table.",
        ]
    )
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")
