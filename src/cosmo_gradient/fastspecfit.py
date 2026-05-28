"""DESI FastSpecFit/LSS population-gradient validation utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cosmo_gradient.coords import unit_to_radec
from cosmo_gradient.dipole import DipoleFit, fit_dipole_map
from cosmo_gradient.maps import SkyMap, pixel_vectors, pixelize_radec

LOGGER = logging.getLogger(__name__)

SUPPORTED_SUFFIXES = (".fits", ".fits.gz", ".parquet")
LSS_REQUIRED = {"TARGETID", "RA", "DEC", "Z"}
FASTSPEC_QUALITY_COLUMNS = (
    "ZWARN",
    "RCHI2_CONT",
    "RCHI2_LINE",
    "DELTACHI2",
    "DELTA_LINECHI2",
    "DN4000_IVAR",
)
MGFE_INDEX_COLUMNS = ("MGB", "FE5270", "FE5335")
LUMINOSITY_PROXY_COLUMNS = (
    "FLUX_SYNTH_G",
    "FLUX_SYNTH_R",
    "FLUX_SYNTH_Z",
    "FLUX_G",
    "FLUX_R",
    "FLUX_Z",
    "FLUX_W1",
    "FLUX_W2",
)
POPULATION_ALIASES = {
    "DN4000": ("DN4000", "D4000", "D4000_N", "D4000N", "DN4000_MODEL"),
    "DN4000_MODEL": ("DN4000_MODEL",),
    "STELLAR_MASS": (
        "STEL_MASS",
        "STELLAR_MASS",
        "LOGMSTAR",
        "LOGMASS",
        "LOGM_STELLAR",
        "MSTAR",
    ),
    "METALLICITY": (
        "MGFE",
        "MGFE_PRIME",
        *MGFE_INDEX_COLUMNS,
        "METALLICITY",
        "ZZSUN",
        "LOGZ",
        "LOGZ_SOLAR",
        "Z_STELLAR",
        "ZMETAL",
        "OH12",
        "OIIIHB_NIIHA_METALLICITY",
    ),
    "AGE": ("AGE", "AGE_LW", "LIGHT_WEIGHTED_AGE"),
}


@dataclass(frozen=True)
class FastSpecQualityConfig:
    """Quality cuts for FastSpecFit population-gradient analyses."""

    require_zwarn_zero: bool = True
    max_rchi2_cont: float = 2.0
    min_deltachi2: float = 25.0
    min_dn4000_ivar: float = 0.0
    winsorize_column: str = "DN4000_MODEL"
    winsor_lower_percentile: float = 1.0
    winsor_upper_percentile: float = 99.0


@dataclass(frozen=True)
class FileInventoryRecord:
    """One local table discovered during metadata inventory."""

    path: Path
    format: str
    size_gib: float
    n_rows: int | None
    n_columns: int
    has_targetid: bool
    has_lss_columns: bool
    population_columns: tuple[str, ...]
    columns_preview: tuple[str, ...]
    error: str = ""


@dataclass(frozen=True)
class GradientRunResult:
    """Paths and row counts emitted by one FastSpecFit gradient validation run."""

    table_path: Path
    report_path: Path
    figure_paths: tuple[Path, ...]
    n_joined: int
    n_used: int
    dipole: DipoleFit | None


def inventory_local_tables(
    roots: list[Path],
    output_prefix: Path,
    max_files: int | None = None,
) -> pd.DataFrame:
    """Scan local FITS/Parquet metadata and write FastSpecFit/LSS inventory files."""
    records: list[FileInventoryRecord] = []
    paths = _iter_table_paths(roots)
    if max_files is not None:
        paths = paths[:max_files]
    LOGGER.info("Inspecting %s candidate local tables", len(paths))
    for index, path in enumerate(paths, start=1):
        if index % 250 == 0:
            LOGGER.info("Inventoried %s/%s files", index, len(paths))
        records.append(inspect_table(path))

    frame = pd.DataFrame(
        {
            "path": [str(record.path) for record in records],
            "format": [record.format for record in records],
            "size_gib": [record.size_gib for record in records],
            "n_rows": [record.n_rows for record in records],
            "n_columns": [record.n_columns for record in records],
            "has_targetid": [record.has_targetid for record in records],
            "has_lss_columns": [record.has_lss_columns for record in records],
            "population_columns": [";".join(record.population_columns) for record in records],
            "columns_preview": [";".join(record.columns_preview) for record in records],
            "error": [record.error for record in records],
        }
    )
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_prefix.with_suffix(".csv")
    md_path = _report_path_for_output_prefix(output_prefix)
    frame.to_csv(csv_path, index=False)
    write_inventory_report(frame, md_path)
    LOGGER.info("Wrote %s", csv_path)
    LOGGER.info("Wrote %s", md_path)
    return frame


def inspect_table(path: Path) -> FileInventoryRecord:
    """Read only metadata/schema for one local table."""
    try:
        columns, n_rows, fmt = _table_metadata(path)
        upper = {column.upper(): column for column in columns}
        population = [
            original
            for aliases in POPULATION_ALIASES.values()
            for alias in aliases
            if (original := upper.get(alias.upper())) is not None
        ]
        return FileInventoryRecord(
            path=path,
            format=fmt,
            size_gib=path.stat().st_size / 1024**3,
            n_rows=n_rows,
            n_columns=len(columns),
            has_targetid="TARGETID" in upper,
            has_lss_columns=LSS_REQUIRED.issubset(set(upper)),
            population_columns=tuple(dict.fromkeys(population)),
            columns_preview=tuple(columns[:80]),
        )
    except Exception as exc:  # pragma: no cover - defensive for broken local files
        return FileInventoryRecord(
            path=path,
            format=_guess_format(path),
            size_gib=path.stat().st_size / 1024**3 if path.exists() else float("nan"),
            n_rows=None,
            n_columns=0,
            has_targetid=False,
            has_lss_columns=False,
            population_columns=(),
            columns_preview=(),
            error=str(exc),
        )


def write_inventory_report(frame: pd.DataFrame, path: Path) -> None:
    """Write a compact markdown inventory summary."""
    path.parent.mkdir(parents=True, exist_ok=True)
    population = frame[frame["population_columns"].fillna("").astype(str) != ""].copy()
    lss = frame[frame["has_lss_columns"]].copy()
    lines = [
        "# DESI local table inventory",
        "",
        "This report is metadata-only: no large science tables were loaded into memory.",
        "",
        f"- candidate tables inspected: {len(frame)}",
        f"- LSS-like tables with TARGETID/RA/DEC/Z: {len(lss)}",
        f"- population/VAC-like tables with known columns: {len(population)}",
        f"- files with metadata errors: {int((frame['error'].fillna('') != '').sum())}",
        "",
        "## LSS candidates",
        "",
        _markdown_table(
            lss[["path", "size_gib", "n_rows", "columns_preview"]].head(20)
            if len(lss)
            else pd.DataFrame(columns=["path", "size_gib", "n_rows", "columns_preview"])
        ),
        "",
        "## Population/VAC candidates",
        "",
        _markdown_table(
            population[["path", "size_gib", "n_rows", "population_columns", "columns_preview"]].head(30)
            if len(population)
            else pd.DataFrame(
                columns=["path", "size_gib", "n_rows", "population_columns", "columns_preview"]
            )
        ),
        "",
        "## Next action",
        "",
    ]
    if len(population) == 0:
        lines.extend(
            [
                "No FastSpecFit/FastPhot-style population table was found in the scanned roots.",
                "Place or symlink the DESI VAC files into `data/raw` or the external data root, then rerun:",
                "",
                "```bash",
                "uv run cosmo-gradient inventory-fastspecfit --root data/raw --root '/Volumes/T7 Shield/cosmo_genesis_gradient/raw'",
                "```",
            ]
        )
    else:
        lines.extend(
            [
                "Use one LSS candidate and one population candidate with:",
                "",
                "```bash",
                "uv run cosmo-gradient fastspecfit-gradient --lss <LSS.fits> --vac <VAC.fits-or.parquet> --observable DN4000",
                "```",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_fastspecfit_gradient_validation(
    lss_path: Path,
    vac_path: Path,
    output_prefix: Path,
    random_paths: list[Path] | None = None,
    observable: str = "DN4000",
    z_min: float | None = None,
    z_max: float | None = None,
    nside: int = 16,
    max_rows: int | None = None,
    external_templates: list[str] | None = None,
    min_objects_per_pixel: float = 5.0,
    quality_config: FastSpecQualityConfig | None = FastSpecQualityConfig(),
    block_null_mocks: int = 500,
    block_nside: int = 2,
    seed: int = 20260527,
) -> GradientRunResult:
    """Join LSS and FastSpecFit-like tables, residualize population proxies, and fit a dipole."""
    external_templates = external_templates or []
    random_paths = random_paths or []
    lss_columns = ["TARGETID", "RA", "DEC", "Z", "WEIGHT", "WEIGHT_COMP", "WEIGHT_FKP", "PHOTSYS"]
    lss = _read_table_columns(lss_path, lss_columns)
    lss = _standardize_columns(lss)
    if z_min is not None:
        lss = lss[lss["z"] >= z_min]
    if z_max is not None:
        lss = lss[lss["z"] < z_max]
    if max_rows is not None:
        lss = lss.head(max_rows)

    vac_columns = _select_vac_columns(vac_path, observable)
    vac = _read_table_columns(vac_path, vac_columns)
    vac = _standardize_columns(vac)
    joined = lss.merge(vac, on="targetid", how="inner", suffixes=("", "_vac"))
    if joined.empty:
        raise ValueError("LSS/VAC join on TARGETID produced zero rows.")

    joined = _add_mgfe_proxy(joined)
    observable_column = _resolve_column(joined, observable, POPULATION_ALIASES.get(observable.upper(), (observable,)))
    joined, quality_stats = apply_quality_mask(joined, observable_column, quality_config)
    if joined.empty:
        raise ValueError("All LSS/VAC joined rows were removed by FastSpecFit quality cuts.")
    observable_column = _resolve_column(joined, observable, POPULATION_ALIASES.get(observable.upper(), (observable,)))
    mass_column = _resolve_optional_column(joined, POPULATION_ALIASES["STELLAR_MASS"])
    controls = ["z", "z2"]
    joined["z2"] = joined["z"] ** 2
    if mass_column is not None:
        controls.append(mass_column)
    line_rchi2_column = (
        _resolve_optional_column(joined, ("RCHI2_LINE",))
        if _is_line_observable(observable_column)
        else None
    )
    if line_rchi2_column is not None:
        controls.append(line_rchi2_column)
    line_deltachi2_column = (
        _resolve_optional_column(joined, ("DELTA_LINECHI2",))
        if _is_line_observable(observable_column)
        else None
    )
    if line_deltachi2_column is not None:
        controls.append(line_deltachi2_column)
    for proxy_column in _available_luminosity_proxy_columns(joined):
        controls.append(proxy_column)

    pixels, npix, backend = pixelize_radec(
        joined["ra"].to_numpy(dtype=float),
        joined["dec"].to_numpy(dtype=float),
        nside,
    )
    joined["pixel"] = pixels
    template_matrix, template_names = _object_external_templates(
        external_templates,
        pixels,
        npix,
    )
    for idx, name in enumerate(template_names):
        column = f"template_{idx}_{_safe_name(name)}"
        joined[column] = template_matrix[:, idx]
        controls.append(column)

    survey_weight = _analysis_weight(joined)
    ivar_weight = _ivar_weight(joined, observable_column)
    regression_weight = survey_weight * ivar_weight
    residual, coefficients = residualize_observable(
        joined,
        observable_column=observable_column,
        control_columns=controls,
        weights=regression_weight,
    )
    joined["population_residual"] = residual

    residual_sum = np.bincount(
        pixels,
        weights=ivar_weight * np.nan_to_num(residual, nan=0.0),
        minlength=npix,
    ).astype(float)
    ivar_counts = np.bincount(pixels, weights=ivar_weight, minlength=npix).astype(float)
    finite_residual_objects = np.isfinite(residual) & np.isfinite(ivar_weight) & (ivar_weight > 0.0)
    object_counts = np.bincount(
        pixels,
        weights=finite_residual_objects.astype(float),
        minlength=npix,
    ).astype(float)
    residual_mean = np.full(npix, np.nan, dtype=float)
    good_data = (object_counts >= min_objects_per_pixel) & (ivar_counts > 0.0)
    residual_mean[good_data] = residual_sum[good_data] / ivar_counts[good_data]

    random_counts = _random_pixel_counts(random_paths, nside, npix)
    valid = good_data.copy()
    if random_counts is not None:
        valid &= random_counts > 0.0
        map_weights = np.where(random_counts > 0.0, random_counts, ivar_counts)
    else:
        random_counts = ivar_counts.copy()
        map_weights = ivar_counts.copy()

    sky_map = SkyMap(
        nside=nside,
        backend=backend,
        data_counts=object_counts,
        random_counts=map_weights,
        alpha=1.0,
        delta=residual_mean,
        valid=valid,
        pixel_vectors=pixel_vectors(nside, npix=npix, backend=backend),
    )
    dipole = fit_dipole_map(sky_map, weights=ivar_counts)
    if block_null_mocks > 0:
        block_null_amplitudes = spatial_block_permutation_null_amplitudes(
            sky_map=sky_map,
            rng=np.random.default_rng(seed),
            n_permutations=block_null_mocks,
            block_nside=block_nside,
            weights=ivar_counts,
        )
        block_p_value = (
            (1.0 + np.sum(block_null_amplitudes >= dipole.amplitude))
            / (len(block_null_amplitudes) + 1.0)
            if len(block_null_amplitudes)
            else float("nan")
        )
        dipole = DipoleFit(
            amplitude=dipole.amplitude,
            vector=dipole.vector,
            ra_deg=dipole.ra_deg,
            dec_deg=dipole.dec_deg,
            monopole=dipole.monopole,
            n_pixels=dipole.n_pixels,
            block_null_p_value=float(block_p_value),
            block_null_mocks=int(len(block_null_amplitudes)),
            block_null_regions=int(12 * block_nside * block_nside),
            block_null_amplitudes=block_null_amplitudes,
        )

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    table_path = output_prefix.with_suffix(".csv")
    report_path = _report_path_for_output_prefix(output_prefix)
    fig_dir = Path("outputs/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)
    scatter_path = fig_dir / f"{output_prefix.name}_residual_vs_z.png"
    hist_path = fig_dir / f"{output_prefix.name}_residual_hist.png"
    _write_gradient_tables(
        table_path,
        joined,
        observable_column=observable_column,
        controls=controls,
        coefficients=coefficients,
        dipole=dipole,
    )
    _write_gradient_report(
        report_path,
        lss_path=lss_path,
        vac_path=vac_path,
        random_paths=random_paths,
        n_joined=len(joined),
        n_used=int(np.sum(np.isfinite(residual))),
        observable_column=observable_column,
        controls=controls,
        coefficients=coefficients,
        dipole=dipole,
        quality_stats=quality_stats,
        figures=[scatter_path, hist_path],
        z_min=z_min,
        z_max=z_max,
    )
    _write_gradient_figures(
        joined,
        observable_column=observable_column,
        scatter_path=scatter_path,
        hist_path=hist_path,
    )
    return GradientRunResult(
        table_path=table_path,
        report_path=report_path,
        figure_paths=(scatter_path, hist_path),
        n_joined=len(joined),
        n_used=int(np.sum(np.isfinite(residual))),
        dipole=dipole,
    )


def write_fastspecfit_preflight_report(
    path: Path,
    *,
    inventory_frame: pd.DataFrame,
    roots: list[Path],
) -> None:
    """Write the validation report when the local population VAC is not present yet."""
    population = inventory_frame[
        inventory_frame["population_columns"].fillna("").astype(str) != ""
    ].copy()
    lss = inventory_frame[inventory_frame["has_lss_columns"]].copy()
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DESI FastSpecFit gradient validation",
        "",
        "Status: preflight completed; population-gradient fit was not run.",
        "",
        "This report is intentionally conservative. It records what is locally available and blocks",
        "the Dn4000/metallicity residual analysis until a FastSpecFit/FastPhot VAC with population",
        "columns is present.",
        "",
        "## Scanned Roots",
        "",
        *[f"- `{root}`" for root in roots],
        "",
        "## Inventory Summary",
        "",
        f"- candidate FITS/Parquet tables inspected: {len(inventory_frame)}",
        f"- LSS-like tables with TARGETID/RA/DEC/Z: {len(lss)}",
        f"- population/VAC-like tables with known columns: {len(population)}",
        f"- metadata errors: {int((inventory_frame['error'].fillna('') != '').sum())}",
        "",
        "## LSS Candidates",
        "",
        _markdown_table(
            lss[["path", "size_gib", "n_rows", "columns_preview"]].head(20)
            if len(lss)
            else pd.DataFrame(columns=["path", "size_gib", "n_rows", "columns_preview"])
        ),
        "",
        "## Population/VAC Candidates",
        "",
        _markdown_table(
            population[["path", "size_gib", "n_rows", "population_columns", "columns_preview"]].head(30)
            if len(population)
            else pd.DataFrame(
                columns=["path", "size_gib", "n_rows", "population_columns", "columns_preview"]
            )
        ),
        "",
        "## Scientific Hold",
        "",
        "No residual Dn4000/metallicity dipole is reported from this preflight. A real result requires",
        "a joined LSS + FastSpecFit/FastPhot table on TARGETID, then the same estimator must be run",
        "through the EZmock ensemble for calibrated p-values.",
        "",
        "## Next Command",
        "",
        "After placing the VAC file locally, rerun inventory and then execute:",
        "",
        "```bash",
        "uv run cosmo-gradient fastspecfit-gradient \\",
        "  --lss data/raw/<TRACER>_<REGION>_clustering.dat.fits \\",
        "  --vac /path/to/fastspecfit-or-fastphot-vac.fits \\",
        "  --random data/raw/<TRACER>_<REGION>_0_clustering.ran.fits \\",
        "  --observable DN4000",
        "```",
        "",
        "## Official VAC Targets",
        "",
        "DESI DR1 FastSpecFit Iron v2.1 catalogs are published under:",
        "",
        "`https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/`",
        "",
        "For a pipeline smoke test with the current SSD free space, use a small catalog first:",
        "",
        "```bash",
        "mkdir -p '/Volumes/T7 Shield/cosmo_genesis_gradient/raw/vac/fastspecfit/iron/v2.1/catalogs'",
        "cd '/Volumes/T7 Shield/cosmo_genesis_gradient/raw/vac/fastspecfit/iron/v2.1/catalogs'",
        "curl -L -C - -O https://data.desi.lbl.gov/public/dr1/vac/dr1/fastspecfit/iron/v2.1/catalogs/fastspec-iron-sv3-dark.fits",
        "```",
        "",
        "For main-survey science, the relevant large files are `fastspec-iron-main-dark.fits`",
        "for LRG/ELG/QSO-like dark-time targets and `fastspec-iron-main-bright.fits` for",
        "BGS-like bright-time targets. The current SSD free space must be checked before",
        "downloading them.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def residualize_observable(
    frame: pd.DataFrame,
    observable_column: str,
    control_columns: list[str],
    weights: NDArray[np.float64],
) -> tuple[NDArray[np.float64], pd.DataFrame]:
    """Regress a population observable on redshift/mass/systematics controls."""
    y = pd.to_numeric(frame[observable_column], errors="coerce").to_numpy(dtype=float)
    design_columns: list[NDArray[np.float64]] = [np.ones(len(frame), dtype=float)]
    terms = ["intercept"]
    for column in control_columns:
        if column not in frame:
            continue
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        if np.all(~np.isfinite(values)):
            continue
        median = float(np.nanmedian(values))
        values = np.where(np.isfinite(values), values, median)
        scale = float(np.nanstd(values))
        if scale <= 0.0 or not np.isfinite(scale):
            continue
        design_columns.append((values - median) / scale)
        terms.append(column)
    design = np.column_stack(design_columns)
    finite = np.isfinite(y) & np.isfinite(weights) & (weights > 0.0)
    finite &= np.all(np.isfinite(design), axis=1)
    residual = np.full(len(frame), np.nan, dtype=float)
    if int(np.sum(finite)) <= len(terms):
        raise ValueError("Not enough finite rows to fit population residual model.")
    sqrt_w = np.sqrt(np.clip(weights[finite], 1e-12, None))
    params, *_ = np.linalg.lstsq(design[finite] * sqrt_w[:, None], y[finite] * sqrt_w, rcond=None)
    residual[finite] = y[finite] - design[finite] @ params
    coefficients = pd.DataFrame({"term": terms, "coefficient": params.astype(float)})
    return residual, coefficients


def apply_quality_mask(
    frame: pd.DataFrame,
    observable_column: str,
    quality_config: FastSpecQualityConfig | None = FastSpecQualityConfig(),
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Apply FastSpecFit quality cuts and winsorize configured observables."""
    if quality_config is None:
        return frame.copy(), {"quality_cuts": "disabled", "n_input": int(len(frame)), "n_after": int(len(frame))}
    filtered = frame.copy()
    initial = len(filtered)
    mask = np.ones(initial, dtype=bool)
    stats: dict[str, float | int | str] = {"quality_cuts": "enabled", "n_input": int(initial)}

    zwarn_column = _resolve_optional_column(filtered, ("ZWARN",))
    if quality_config.require_zwarn_zero and zwarn_column is not None:
        values = pd.to_numeric(filtered[zwarn_column], errors="coerce").to_numpy(dtype=float)
        current = np.isfinite(values) & (values == 0.0)
        stats["n_fail_zwarn"] = int(np.sum(mask & ~current))
        mask &= current

    rchi2_column = _quality_rchi2_column(filtered, observable_column)
    if rchi2_column is not None:
        values = pd.to_numeric(filtered[rchi2_column], errors="coerce").to_numpy(dtype=float)
        current = np.isfinite(values) & (values < quality_config.max_rchi2_cont)
        stats[f"n_fail_{rchi2_column.lower()}"] = int(np.sum(mask & ~current))
        mask &= current

    deltachi2_column = _quality_deltachi2_column(filtered, observable_column)
    if deltachi2_column is not None:
        values = pd.to_numeric(filtered[deltachi2_column], errors="coerce").to_numpy(dtype=float)
        current = np.isfinite(values) & (values > quality_config.min_deltachi2)
        stats[f"n_fail_{deltachi2_column.lower()}"] = int(np.sum(mask & ~current))
        mask &= current

    ivar_column = _observable_ivar_column(filtered, observable_column)
    if ivar_column is not None:
        values = pd.to_numeric(filtered[ivar_column], errors="coerce").to_numpy(dtype=float)
        current = np.isfinite(values) & (values > quality_config.min_dn4000_ivar)
        stats[f"n_fail_{ivar_column.lower()}"] = int(np.sum(mask & ~current))
        mask &= current

    observable = pd.to_numeric(filtered[observable_column], errors="coerce").to_numpy(dtype=float)
    current = np.isfinite(observable)
    stats["n_fail_observable_finite"] = int(np.sum(mask & ~current))
    mask &= current

    filtered = filtered.loc[mask].copy()
    stats["n_after"] = int(len(filtered))
    stats["n_removed"] = int(initial - len(filtered))
    winsor_column = _resolve_optional_column(filtered, (quality_config.winsorize_column, observable_column))
    if winsor_column is not None and len(filtered):
        lower, upper = winsorize_column(
            filtered,
            winsor_column,
            quality_config.winsor_lower_percentile,
            quality_config.winsor_upper_percentile,
        )
        stats["winsorized_column"] = winsor_column
        stats["winsor_lower"] = float(lower)
        stats["winsor_upper"] = float(upper)
    return filtered, stats


def winsorize_column(
    frame: pd.DataFrame,
    column: str,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.0,
) -> tuple[float, float]:
    """Clip a numeric column in-place to percentile bounds and return the bounds."""
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        raise ValueError(f"Cannot winsorize {column}; no finite values.")
    lower, upper = np.nanpercentile(values[finite], [lower_percentile, upper_percentile])
    frame[column] = np.clip(values, lower, upper)
    return float(lower), float(upper)


def spatial_block_permutation_null_amplitudes(
    sky_map: SkyMap,
    rng: np.random.Generator,
    n_permutations: int = 500,
    block_nside: int = 2,
    weights: NDArray[np.float64] | None = None,
) -> NDArray[np.float64]:
    """Shuffle residual maps as coarse sky blocks while preserving within-block patterns."""
    if n_permutations <= 0:
        return np.array([], dtype=float)
    if block_nside <= 0 or block_nside >= sky_map.nside:
        raise ValueError("block_nside must be positive and smaller than map nside.")

    ra, dec = unit_to_radec(sky_map.pixel_vectors)
    block_ids, _, _ = pixelize_radec(ra, dec, block_nside)
    valid_pixels = sky_map.valid & np.isfinite(sky_map.delta)
    unique_blocks = np.unique(block_ids[valid_pixels])
    block_pixels = [np.where((block_ids == block_id) & valid_pixels)[0] for block_id in unique_blocks]
    block_pixels = [pixels for pixels in block_pixels if len(pixels)]
    if len(block_pixels) < 2:
        return np.array([], dtype=float)
    amplitudes = np.empty(n_permutations, dtype=float)
    original = sky_map.delta.copy()
    for index in range(n_permutations):
        shuffled = original.copy()
        source_order = rng.permutation(len(block_pixels))
        for target_index, source_index in enumerate(source_order):
            target_pixels = block_pixels[target_index]
            source_pixels = block_pixels[source_index]
            if len(target_pixels) == len(source_pixels):
                shuffled[target_pixels] = original[source_pixels]
            else:
                take = rng.choice(source_pixels, size=len(target_pixels), replace=len(source_pixels) < len(target_pixels))
                shuffled[target_pixels] = original[take]
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
        amplitudes[index] = fit_dipole_map(permuted, weights=weights).amplitude
    return amplitudes


def _iter_table_paths(roots: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            LOGGER.warning("Inventory root does not exist: %s", root)
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if path.name.startswith("._"):
                continue
            lower = path.name.lower()
            if lower.endswith(SUPPORTED_SUFFIXES):
                paths.append(path)
    return sorted(paths)


def _table_metadata(path: Path) -> tuple[list[str], int | None, str]:
    if path.name.lower().endswith((".fits", ".fits.gz")):
        schemas = _fits_table_schemas(path)
        columns = list(dict.fromkeys(column for _, _, schema_columns, _ in schemas for column in schema_columns))
        n_rows = max((rows for _, _, _, rows in schemas if rows is not None), default=None)
        return columns, n_rows, "fits"
    if path.name.lower().endswith(".parquet"):
        import pyarrow.parquet as pq

        metadata = pq.read_metadata(path)
        return list(metadata.schema.names), int(metadata.num_rows), "parquet"
    raise ValueError(f"Unsupported table format: {path}")


def _read_table_columns(path: Path, columns: list[str]) -> pd.DataFrame:
    available, _, fmt = _table_metadata(path)
    by_upper = {column.upper(): column for column in available}
    selected = [by_upper[column.upper()] for column in columns if column.upper() in by_upper]
    if not selected:
        raise ValueError(f"None of requested columns found in {path}")
    if fmt == "fits":
        return _read_fits_columns(path, selected)
    import pyarrow.parquet as pq

    return pq.read_table(path, columns=selected).to_pandas()


def _read_fits_columns(path: Path, selected: list[str]) -> pd.DataFrame:
    """Read FITS columns, merging multiple table HDUs on TARGETID when needed."""
    import fitsio

    schemas = _fits_table_schemas(path)
    requested_order = [column.upper() for column in selected]
    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for hdu_index, _, hdu_columns, _ in schemas:
        hdu_by_upper = {column.upper(): column for column in hdu_columns}
        hdu_selected: list[str] = []
        for requested in requested_order:
            if requested == "TARGETID":
                continue
            if requested in seen:
                continue
            if requested in hdu_by_upper:
                hdu_selected.append(hdu_by_upper[requested])
                seen.add(requested)
        if not hdu_selected:
            continue
        if "TARGETID" in requested_order and "TARGETID" in hdu_by_upper:
            hdu_selected.insert(0, hdu_by_upper["TARGETID"])
        data = fitsio.read(str(path), ext=hdu_index, columns=list(dict.fromkeys(hdu_selected)))
        frames.append(_native_endian_frame(pd.DataFrame(data)))

    missing = [column for column in requested_order if column != "TARGETID" and column not in seen]
    if missing:
        LOGGER.debug("Missing requested FITS columns in %s: %s", path, ", ".join(missing))
    if not frames:
        raise ValueError(f"None of requested columns found in a FITS table HDU for {path}")
    if len(frames) == 1:
        return frames[0]

    result = frames[0]
    for frame in frames[1:]:
        if "TARGETID" not in result or "TARGETID" not in frame:
            raise ValueError(f"Cannot merge multiple FITS HDUs without TARGETID in {path}")
        duplicate_columns = [column for column in frame.columns if column in result.columns and column != "TARGETID"]
        frame = frame.drop(columns=duplicate_columns)
        result = result.merge(frame, on="TARGETID", how="left")
    return result


def _native_endian_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert FITS big-endian numeric columns to native endian for pandas joins."""
    converted = frame.copy(deep=False)
    for column in converted.columns:
        dtype = converted[column].dtype
        byteorder = getattr(dtype, "byteorder", "=")
        if byteorder not in (">", "<"):
            continue
        if byteorder == "=" or byteorder == np.dtype(dtype).newbyteorder("=").byteorder:
            continue
        values = converted[column].to_numpy(copy=False)
        native_dtype = values.dtype.newbyteorder("=")
        converted[column] = values.byteswap().view(native_dtype)
    return converted


def _fits_table_schemas(path: Path) -> list[tuple[int, str, list[str], int | None]]:
    import fitsio

    schemas: list[tuple[int, str, list[str], int | None]] = []
    with fitsio.FITS(str(path)) as fits:
        for hdu_index in range(1, len(fits)):
            hdu = fits[hdu_index]
            try:
                columns = list(hdu.get_colnames())
            except Exception:
                continue
            try:
                n_rows = int(hdu.get_nrows())
            except Exception:
                n_rows = None
            try:
                extname = str(hdu.get_extname())
            except Exception:
                extname = str(hdu_index)
            schemas.append((hdu_index, extname, columns, n_rows))
    if not schemas:
        raise ValueError(f"No readable FITS table HDUs found in {path}")
    return schemas


def _select_vac_columns(path: Path, observable: str) -> list[str]:
    columns, _, _ = _table_metadata(path)
    by_upper = {column.upper(): column for column in columns}
    selected = ["TARGETID"]
    if observable.upper() in {"METALLICITY", "MGFE", "MGFE_PRIME"}:
        for column in MGFE_INDEX_COLUMNS:
            if column.upper() in by_upper:
                selected.append(by_upper[column.upper()])
    alias_groups = [
        POPULATION_ALIASES.get(observable.upper(), (observable,)),
        _observable_ivar_aliases(observable),
        POPULATION_ALIASES["STELLAR_MASS"],
        POPULATION_ALIASES["METALLICITY"],
        LUMINOSITY_PROXY_COLUMNS,
        ("Z", "REDSHIFT"),
        FASTSPEC_QUALITY_COLUMNS,
    ]
    for aliases in alias_groups:
        keep_all_matches = aliases in {FASTSPEC_QUALITY_COLUMNS, LUMINOSITY_PROXY_COLUMNS}
        for alias in aliases:
            if alias.upper() in by_upper:
                selected.append(by_upper[alias.upper()])
                if not keep_all_matches:
                    break
    return list(dict.fromkeys(selected))


def _standardize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename: dict[str, str] = {}
    for column in frame.columns:
        upper = column.upper()
        if upper == "TARGETID":
            rename[column] = "targetid"
        elif upper == "RA":
            rename[column] = "ra"
        elif upper == "DEC":
            rename[column] = "dec"
        elif upper == "Z":
            rename[column] = "z"
        elif upper == "WEIGHT":
            rename[column] = "weight"
    return frame.rename(columns=rename)


def _resolve_column(frame: pd.DataFrame, requested: str, aliases: tuple[str, ...]) -> str:
    by_upper = {column.upper(): column for column in frame.columns}
    for alias in (requested, *aliases):
        if alias.upper() in by_upper:
            return by_upper[alias.upper()]
    raise ValueError(f"Could not resolve column {requested}; available columns: {list(frame.columns)}")


def _resolve_optional_column(frame: pd.DataFrame, aliases: tuple[str, ...]) -> str | None:
    by_upper = {column.upper(): column for column in frame.columns}
    for alias in aliases:
        if alias.upper() in by_upper:
            return by_upper[alias.upper()]
    return None


def _analysis_weight(frame: pd.DataFrame) -> NDArray[np.float64]:
    if "weight" in frame:
        weight = pd.to_numeric(frame["weight"], errors="coerce").to_numpy(dtype=float)
    else:
        weight = np.ones(len(frame), dtype=float)
    return np.where(np.isfinite(weight) & (weight > 0.0), weight, 1.0)


def _dn4000_ivar_weight(frame: pd.DataFrame) -> NDArray[np.float64]:
    """Return strictly positive DN4000_IVAR weights for residual map construction."""
    return _ivar_weight(frame, "DN4000")


def _ivar_weight(frame: pd.DataFrame, observable_column: str) -> NDArray[np.float64]:
    """Return strictly positive IVAR weights for a FastSpecFit observable."""
    ivar_column = _observable_ivar_column(frame, observable_column)
    if ivar_column is None:
        raise ValueError(
            f"{observable_column}_IVAR or a known observable IVAR column is required "
            "for IVAR-weighted FastSpecFit residual maps."
        )
    values = pd.to_numeric(frame[ivar_column], errors="coerce").to_numpy(dtype=float)
    if not np.any(np.isfinite(values) & (values > 0.0)):
        raise ValueError(f"{ivar_column} contains no positive finite weights.")
    return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)


def _observable_ivar_aliases(observable: str) -> tuple[str, ...]:
    upper = observable.upper()
    aliases = [f"{upper}_IVAR"]
    if upper == "DN4000_MODEL":
        aliases.append("DN4000_IVAR")
    if upper == "DN4000":
        aliases.extend(["DN4000_IVAR", "DN4000_MODEL_IVAR"])
    return tuple(dict.fromkeys(aliases))


def _observable_ivar_column(frame: pd.DataFrame, observable_column: str) -> str | None:
    return _resolve_optional_column(frame, _observable_ivar_aliases(observable_column))


def _is_line_observable(observable_column: str) -> bool:
    upper = observable_column.upper()
    if upper.startswith("FLUX_SYNTH"):
        return False
    return upper.endswith("_EW") or upper.endswith("_FLUX") or "_FLUX" in upper


def _quality_rchi2_column(frame: pd.DataFrame, observable_column: str) -> str | None:
    if _is_line_observable(observable_column):
        return None
    return _resolve_optional_column(frame, ("RCHI2_CONT",))


def _quality_deltachi2_column(frame: pd.DataFrame, observable_column: str) -> str | None:
    if _is_line_observable(observable_column):
        return None
    return _resolve_optional_column(frame, ("DELTACHI2",))


def _available_luminosity_proxy_columns(frame: pd.DataFrame) -> list[str]:
    by_upper = {column.upper(): column for column in frame.columns}
    return [by_upper[column] for column in LUMINOSITY_PROXY_COLUMNS if column in by_upper]


def _add_mgfe_proxy(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a composite [MgFe] metallicity proxy when Mgb, Fe5270 and Fe5335 exist."""
    if "MGFE" in {column.upper() for column in frame.columns}:
        return frame
    mgb_col = _resolve_optional_column(frame, ("MGB",))
    fe5270_col = _resolve_optional_column(frame, ("FE5270",))
    fe5335_col = _resolve_optional_column(frame, ("FE5335",))
    if mgb_col is None or fe5270_col is None or fe5335_col is None:
        return frame
    enriched = frame.copy()
    mgb = pd.to_numeric(enriched[mgb_col], errors="coerce").to_numpy(dtype=float)
    fe5270 = pd.to_numeric(enriched[fe5270_col], errors="coerce").to_numpy(dtype=float)
    fe5335 = pd.to_numeric(enriched[fe5335_col], errors="coerce").to_numpy(dtype=float)
    fe_blend = 0.72 * fe5270 + 0.28 * fe5335
    product = mgb * fe_blend
    enriched["MGFE"] = np.where(np.isfinite(product) & (product > 0.0), np.sqrt(product), np.nan)
    return enriched


def _object_external_templates(
    specs: list[str],
    pixels: NDArray[np.int64],
    npix: int,
) -> tuple[NDArray[np.float64], list[str]]:
    if not specs:
        return np.empty((len(pixels), 0), dtype=float), []
    values: list[NDArray[np.float64]] = []
    names: list[str] = []
    for spec in specs:
        name, raw_path = _parse_template_spec(spec)
        template = _load_npz_template(raw_path)
        if len(template) != npix:
            raise ValueError(f"Template {raw_path} has {len(template)} pixels, expected {npix}")
        values.append(template[pixels].astype(float))
        names.append(name)
    return np.column_stack(values), names


def _parse_template_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", 1)
    elif ":" in spec:
        name, path = spec.split(":", 1)
    else:
        path = spec
        name = Path(path).stem
    return name, Path(path)


def _load_npz_template(path: Path) -> NDArray[np.float64]:
    loaded = np.load(path, allow_pickle=False)
    if isinstance(loaded, np.lib.npyio.NpzFile):
        if "template" in loaded:
            return np.asarray(loaded["template"], dtype=float)
        first_key = loaded.files[0]
        return np.asarray(loaded[first_key], dtype=float)
    return np.asarray(loaded, dtype=float)


def _random_pixel_counts(
    random_paths: list[Path],
    nside: int,
    npix: int,
) -> NDArray[np.float64] | None:
    if not random_paths:
        return None
    total = np.zeros(npix, dtype=float)
    for path in random_paths:
        randoms = _read_table_columns(path, ["RA", "DEC", "WEIGHT"])
        randoms = _standardize_columns(randoms)
        pix, random_npix, _ = pixelize_radec(
            randoms["ra"].to_numpy(dtype=float),
            randoms["dec"].to_numpy(dtype=float),
            nside,
        )
        if random_npix != npix:
            raise ValueError("Random map pixel count mismatch.")
        weights = _analysis_weight(randoms)
        total += np.bincount(pix, weights=weights, minlength=npix).astype(float)
    return total


def _write_gradient_tables(
    path: Path,
    joined: pd.DataFrame,
    observable_column: str,
    controls: list[str],
    coefficients: pd.DataFrame,
    dipole: DipoleFit,
) -> None:
    summary = pd.DataFrame(
        [
            {
                "kind": "dipole",
                "observable": observable_column,
                "n_joined": len(joined),
                "n_finite_residual": int(np.sum(np.isfinite(joined["population_residual"]))),
                "amplitude": dipole.amplitude,
                "ra_deg": dipole.ra_deg,
                "dec_deg": dipole.dec_deg,
                "monopole": dipole.monopole,
                "block_null_p_value": dipole.block_null_p_value,
                "block_null_mocks": dipole.block_null_mocks,
                "block_null_regions": dipole.block_null_regions,
                "controls": ";".join(controls),
            }
        ]
    )
    coefficient_rows = coefficients.copy()
    coefficient_rows.insert(0, "kind", "coefficient")
    coefficient_rows["observable"] = observable_column
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat([summary, coefficient_rows], ignore_index=True, sort=False).to_csv(path, index=False)


def _write_gradient_report(
    path: Path,
    *,
    lss_path: Path,
    vac_path: Path,
    random_paths: list[Path],
    n_joined: int,
    n_used: int,
    observable_column: str,
    controls: list[str],
    coefficients: pd.DataFrame,
    dipole: DipoleFit,
    quality_stats: dict[str, float | int | str],
    figures: list[Path],
    z_min: float | None,
    z_max: float | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# DESI FastSpecFit gradient validation",
        "",
        "This is an observational validation report. It does not introduce new analytic constants.",
        "",
        "## Inputs",
        "",
        f"- LSS catalog: `{lss_path}`",
        f"- VAC/population catalog: `{vac_path}`",
        f"- random catalogs: {len(random_paths)}",
        f"- redshift range: `{z_min}` to `{z_max}`",
        "",
        "## Sample",
        "",
        f"- joined rows on TARGETID: {n_joined}",
        f"- finite residual rows: {n_used}",
        f"- observable: `{observable_column}`",
        f"- controls: `{'; '.join(controls)}`",
        "",
        "## Dipole Fit",
        "",
        f"- amplitude: {dipole.amplitude:.6g}",
        f"- axis RA: {dipole.ra_deg:.3f} deg",
        f"- axis DEC: {dipole.dec_deg:.3f} deg",
        f"- fitted pixels: {dipole.n_pixels}",
        f"- block-null p-value: {dipole.block_null_p_value}",
        f"- block-null mocks: {dipole.block_null_mocks}",
        "",
        "## Quality Cuts",
        "",
        _markdown_table(pd.DataFrame([quality_stats])),
        "",
        "## Residual Model Coefficients",
        "",
        _markdown_table(coefficients),
        "",
        "## Figures",
        "",
        *[f"- `{figure}`" for figure in figures],
        "",
        "## Caveats",
        "",
        "- This first local run fits a population-residual dipole only.",
        "- Mock-calibrated p-values require rerunning the same estimator on the EZmock ensemble.",
        "- A scientifically interpretable result requires stability across tracers, redshift bins, and systematics templates.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_gradient_figures(
    frame: pd.DataFrame,
    *,
    observable_column: str,
    scatter_path: Path,
    hist_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    finite = np.isfinite(frame["population_residual"]) & np.isfinite(frame["z"])
    sample = frame.loc[finite]
    if len(sample) > 200_000:
        sample = sample.sample(200_000, random_state=20260527)
    plt.figure(figsize=(7, 4.5))
    plt.scatter(sample["z"], sample["population_residual"], s=1, alpha=0.08)
    plt.axhline(0.0, color="black", linewidth=1)
    plt.xlabel("redshift z")
    plt.ylabel(f"{observable_column} residual")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=160)
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.hist(sample["population_residual"], bins=100, histtype="stepfilled", alpha=0.75)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.xlabel(f"{observable_column} residual")
    plt.ylabel("objects")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=160)
    plt.close()


def _guess_format(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".fits", ".fits.gz")):
        return "fits"
    if name.endswith(".parquet"):
        return "parquet"
    return "unknown"


def _report_path_for_output_prefix(output_prefix: Path) -> Path:
    if output_prefix.parent.name == "tables":
        return output_prefix.parent.parent / "reports" / f"{output_prefix.name}.md"
    return output_prefix.with_suffix(".md")


def _safe_name(name: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in name).strip("_").lower()


def _markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    return frame.to_markdown(index=False)
