"""Dry-run manifests for official DESI DR1 LSS mock catalogs."""

from __future__ import annotations

import gc
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from cosmo_gradient.config import ProjectConfig
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.io.desi import CatalogPair, load_catalog_pair
from cosmo_gradient.maps import counts_to_overdensity_map, load_sky_map, weighted_counts
from cosmo_gradient.systematics import load_external_template_maps, regress_templates

DESI_MOCK_BASE_URL = "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/mocks"
DESI_MOCK_MIRROR_BASE_URL = "https://webdav-hdfs.pic.es/data/public/DESI/DR1/survey/catalogs/dr1/mocks"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfficialMockFile:
    """One official mock file entry."""

    family: str
    program: str
    version: str
    realization: int
    flavor: str
    tracer: str
    region: str
    kind: str
    random_index: int | None
    filename: str
    url: str
    mirror_url: str


@dataclass(frozen=True)
class OfficialMockEnsembleOutputs:
    """Paths written by an official mock ensemble run."""

    mocks_csv: Path
    summary_csv: Path
    report: Path
    null_distribution_png: Path | None = None


def build_official_mock_entries(
    family: str,
    tracers: Sequence[str],
    regions: Sequence[str],
    realizations: Sequence[int],
    random_indices: Sequence[int],
    program: str = "dark",
    flavor: str | None = None,
) -> list[OfficialMockFile]:
    """Build expected official DESI DR1 mock catalog URLs without downloading."""
    normalized_family = family.strip()
    if normalized_family == "EZmock":
        version = "v1"
        resolved_flavor = flavor or "ffa"
        mock_prefix = "mock"
    elif normalized_family == "AbacusSummit":
        version = "v4.2"
        resolved_flavor = flavor or "complete"
        mock_prefix = "mock"
    else:
        raise ValueError("family must be 'EZmock' or 'AbacusSummit'.")

    entries: list[OfficialMockFile] = []
    for realization in realizations:
        realization_label = f"{mock_prefix}{int(realization)}"
        base = "/".join(
            [
                DESI_MOCK_BASE_URL,
                normalized_family,
                program,
                version,
                realization_label,
            ]
        )
        mirror_base = "/".join(
            [
                DESI_MOCK_MIRROR_BASE_URL,
                normalized_family,
                program,
                version,
                realization_label,
            ]
        )
        for tracer in tracers:
            mock_tracer = _mock_file_tracer(tracer)
            for region in regions:
                data_name = f"{mock_tracer}_{resolved_flavor}_{region}_clustering.dat.fits"
                entries.append(
                    OfficialMockFile(
                        family=normalized_family,
                        program=program,
                        version=version,
                        realization=int(realization),
                        flavor=resolved_flavor,
                        tracer=tracer,
                        region=region,
                        kind="data",
                        random_index=None,
                        filename=data_name,
                        url=f"{base}/{data_name}",
                        mirror_url=f"{mirror_base}/{data_name}",
                    )
                )
                for random_index in random_indices:
                    random_name = (
                        f"{mock_tracer}_{resolved_flavor}_{region}_{int(random_index)}"
                        "_clustering.ran.fits"
                    )
                    entries.append(
                        OfficialMockFile(
                            family=normalized_family,
                            program=program,
                            version=version,
                            realization=int(realization),
                            flavor=resolved_flavor,
                            tracer=tracer,
                            region=region,
                            kind="random",
                            random_index=int(random_index),
                            filename=random_name,
                            url=f"{base}/{random_name}",
                            mirror_url=f"{mirror_base}/{random_name}",
                        )
                    )
    return entries


def write_official_mock_manifest(
    output_path: Path,
    family: str,
    tracers: Sequence[str],
    regions: Sequence[str],
    realizations: Sequence[int],
    random_indices: Sequence[int],
    program: str = "dark",
    flavor: str | None = None,
) -> Path:
    """Write a dry-run manifest for selected official DESI mock files."""
    entries = build_official_mock_entries(
        family=family,
        tracers=tracers,
        regions=regions,
        realizations=realizations,
        random_indices=random_indices,
        program=program,
        flavor=flavor,
    )
    lines = [
        "# DESI DR1 official mock download manifest",
        "",
        "This is a dry-run manifest. Files are large; review sizes in the DESI",
        "directory listing and prefer Globus for bulk transfers.",
        "",
        "Official DR1 docs describe two public LSS mock datasets:",
        "",
        "- EZmock: 1000 low-resolution covariance realizations.",
        "- AbacusSummit: 25 high-resolution N-body validation realizations.",
        "",
        f"- Family: `{family}`",
        f"- Program: `{program}`",
        f"- Tracers: `{', '.join(tracers)}`",
        f"- Regions: `{', '.join(regions)}`",
        f"- Realizations: `{', '.join(str(value) for value in realizations)}`",
        f"- Random indices: `{', '.join(str(value) for value in random_indices)}`",
        "",
        "## Files",
        "",
    ]
    for entry in entries:
        random_text = "" if entry.random_index is None else f" random={entry.random_index}"
        lines.extend(
            [
                f"### {entry.family} {entry.flavor} mock{entry.realization} "
                f"{entry.tracer} {entry.region} {entry.kind}{random_text}",
                "",
                f"- primary: {entry.url}",
                f"- mirror: {entry.mirror_url}",
                "",
                "```bash",
                f"curl -L -C - -O {entry.url}",
                "# or mirror:",
                f"curl -L -C - -O {entry.mirror_url}",
                "```",
                "",
            ]
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def official_mock_local_path(root: Path, entry: OfficialMockFile) -> Path:
    """Return the local path used for a mock file entry."""
    return (
        root
        / entry.family
        / entry.program
        / entry.version
        / f"mock{entry.realization}"
        / entry.filename
    )


def write_official_mock_download_queue(
    output_path: Path,
    root: Path,
    family: str,
    tracers: Sequence[str],
    regions: Sequence[str],
    realizations: Sequence[int],
    random_indices: Sequence[int],
    program: str = "dark",
    flavor: str | None = None,
    use_mirror: bool = False,
) -> Path:
    """Write a TSV queue of URL and local destination paths for resumable downloads."""
    entries = build_official_mock_entries(
        family=family,
        tracers=tracers,
        regions=regions,
        realizations=realizations,
        random_indices=random_indices,
        program=program,
        flavor=flavor,
    )
    lines = []
    for entry in entries:
        url = entry.mirror_url if use_mirror else entry.url
        lines.append(f"{url}\t{official_mock_local_path(root, entry)}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def run_official_mock_ensemble(
    config: ProjectConfig,
    family: str,
    tracer: str,
    regions: Sequence[str],
    realizations: Sequence[int],
    random_indices: Sequence[int],
    z_min: float,
    z_max: float,
    observed_map_path: Path,
    output_prefix: Path,
    external_templates: Sequence[str] = (),
    program: str = "dark",
    flavor: str | None = None,
    nside: int | None = None,
    weight_mode: str = "desi",
    mocks_root: Path | None = None,
    mocks_roots: Sequence[Path] | None = None,
    allow_partial: bool = True,
) -> OfficialMockEnsembleOutputs:
    """Run the dipole/template estimator over downloaded official mock catalogs."""
    if nside is not None:
        from dataclasses import replace

        config = replace(config, analysis=replace(config.analysis, nside=int(nside)))
    roots = _mock_search_roots(config=config, mocks_root=mocks_root, mocks_roots=mocks_roots)
    observed_map = load_sky_map(str(observed_map_path))
    template_matrix, template_names = load_external_template_maps(
        external_templates,
        valid=observed_map.valid & np.isfinite(observed_map.delta),
        expected_npix=len(observed_map.delta),
    )
    observed_raw = fit_dipole_map(observed_map)
    observed_corrected = fit_dipole_map(
        regress_templates(observed_map, template_matrix, template_names).corrected_map,
    )

    records: list[dict[str, object]] = []
    skipped: list[str] = []
    for realization in realizations:
        missing = _missing_mock_paths(
            roots=roots,
            family=family,
            tracer=tracer,
            regions=regions,
            realization=int(realization),
            random_indices=random_indices,
            program=program,
            flavor=flavor,
        )
        if missing:
            message = f"mock{realization}: missing {len(missing)} files"
            if allow_partial:
                LOGGER.warning("Skipping %s", message)
                skipped.append(message)
                continue
            missing_text = "\n".join(str(path) for path in missing)
            raise FileNotFoundError(f"{message}\n{missing_text}")
        try:
            record = _analyze_one_official_mock(
                config=config,
                roots=roots,
                family=family,
                tracer=tracer,
                regions=regions,
                realization=int(realization),
                random_indices=random_indices,
                z_min=z_min,
                z_max=z_max,
                template_matrix=template_matrix,
                template_names=template_names,
                program=program,
                flavor=flavor,
                weight_mode=weight_mode,
            )
        except OSError as error:
            message = f"mock{realization}: unreadable or incomplete FITS files ({error})"
            if allow_partial:
                LOGGER.warning("Skipping %s", message)
                skipped.append(message)
                continue
            raise
        records.append(record)
        LOGGER.info(
            "%s mock%d %s %s: raw=%.4g corrected=%.4g",
            family,
            realization,
            tracer,
            "+".join(regions),
            record["raw_amplitude"],
            record["corrected_amplitude"],
        )
        gc.collect()

    mocks = pd.DataFrame.from_records(records)
    summary = _summarize_official_ensemble(
        mocks=mocks,
        observed_raw=observed_raw,
        observed_corrected=observed_corrected,
        skipped=skipped,
        family=family,
        tracer=tracer,
        regions=regions,
        realizations=realizations,
        z_min=z_min,
        z_max=z_max,
        nside=config.analysis.nside,
        n_templates=len(template_names),
    )
    mocks_csv = output_prefix.with_name(f"{output_prefix.name}_mocks.csv")
    summary_csv = output_prefix.with_name(f"{output_prefix.name}_summary.csv")
    report_parent = output_prefix.parent
    if report_parent.name == "tables":
        report_parent = report_parent.parent / "reports"
    report = report_parent / f"{output_prefix.name}.md"
    figure_parent = output_prefix.parent
    if figure_parent.name == "tables":
        figure_parent = figure_parent.parent / "figures"
    null_plot = figure_parent / f"{output_prefix.name}_null_distribution.png"
    paths_to_create = [mocks_csv, summary_csv, report]
    if len(mocks):
        paths_to_create.append(null_plot)
    for path in paths_to_create:
        path.parent.mkdir(parents=True, exist_ok=True)
    mocks.to_csv(mocks_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    _write_official_ensemble_report(report, mocks, summary, skipped)
    if len(mocks):
        _write_official_ensemble_null_plot(null_plot, mocks, summary)
    return OfficialMockEnsembleOutputs(
        mocks_csv=mocks_csv,
        summary_csv=summary_csv,
        report=report,
        null_distribution_png=null_plot if len(mocks) else None,
    )


def _mock_file_tracer(tracer: str) -> str:
    normalized = tracer.strip()
    if normalized == "ELG":
        return "ELG_LOP"
    return normalized


def _mock_search_roots(
    config: ProjectConfig,
    mocks_root: Path | None,
    mocks_roots: Sequence[Path] | None,
) -> list[Path]:
    if mocks_roots:
        return [Path(root).expanduser().resolve() for root in mocks_roots]
    return [(mocks_root or (config.paths.data_raw / "mocks")).expanduser().resolve()]


def _resolve_official_mock_local_path(roots: Sequence[Path], entry: OfficialMockFile) -> Path:
    """Return the first existing local path for an official mock entry."""
    candidates = [official_mock_local_path(root, entry) for root in roots]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _missing_mock_paths(
    roots: Sequence[Path],
    family: str,
    tracer: str,
    regions: Sequence[str],
    realization: int,
    random_indices: Sequence[int],
    program: str,
    flavor: str | None,
) -> list[Path]:
    entries = build_official_mock_entries(
        family=family,
        tracers=[tracer],
        regions=regions,
        realizations=[realization],
        random_indices=random_indices,
        program=program,
        flavor=flavor,
    )
    paths = (_resolve_official_mock_local_path(roots, entry) for entry in entries)
    return [path for path in paths if not path.exists()]


def _analyze_one_official_mock(
    config: ProjectConfig,
    roots: Sequence[Path],
    family: str,
    tracer: str,
    regions: Sequence[str],
    realization: int,
    random_indices: Sequence[int],
    z_min: float,
    z_max: float,
    template_matrix: np.ndarray,
    template_names: Sequence[str],
    program: str,
    flavor: str | None,
    weight_mode: str,
) -> dict[str, object]:
    data_counts: np.ndarray | None = None
    random_counts: np.ndarray | None = None
    backend: str | None = None
    n_data = 0
    n_random = 0
    data_sources: list[str] = []
    random_sources: list[str] = []
    for region in regions:
        pair = _load_official_mock_pair(
            config=config,
            roots=roots,
            family=family,
            tracer=tracer,
            region=region,
            realization=realization,
            random_indices=random_indices,
            program=program,
            flavor=flavor,
            weight_mode=weight_mode,
        )
        data_bin = pair.data.loc[_z_mask(pair.data["z"].to_numpy(dtype=float), z_min, z_max)]
        random_bin = pair.randoms.loc[
            _z_mask(pair.randoms["z"].to_numpy(dtype=float), z_min, z_max)
        ]
        shell_data, npix, shell_backend = weighted_counts(data_bin, config.analysis.nside)
        shell_random, random_npix, random_backend = weighted_counts(
            random_bin,
            config.analysis.nside,
        )
        if npix != random_npix or shell_backend != random_backend:
            raise ValueError("Official mock data/random maps have incompatible pixel geometry.")
        if data_counts is None:
            data_counts = np.zeros(npix, dtype=float)
            random_counts = np.zeros(npix, dtype=float)
            backend = shell_backend
        if len(data_counts) != npix or backend != shell_backend:
            raise ValueError("Official mock region maps have incompatible pixel geometry.")
        data_counts += shell_data
        random_counts += shell_random
        n_data += len(data_bin)
        n_random += len(random_bin)
        data_sources.append(pair.data_source)
        random_sources.append(pair.random_source)
        del pair
    if data_counts is None or random_counts is None or backend is None:
        raise ValueError(f"No count maps built for {family} mock{realization}.")
    sky_map = counts_to_overdensity_map(
        data_counts=data_counts,
        random_counts=random_counts,
        nside=config.analysis.nside,
        backend=backend,
        min_random_per_pixel=config.analysis.min_random_per_pixel,
    )
    raw = fit_dipole_map(sky_map)
    corrected_map = regress_templates(sky_map, template_matrix, template_names).corrected_map
    corrected = fit_dipole_map(corrected_map)
    return {
        "family": family,
        "program": program,
        "flavor": flavor or ("ffa" if family == "EZmock" else "complete"),
        "realization": realization,
        "tracer": tracer,
        "regions": "+".join(regions),
        "z_min": z_min,
        "z_max": z_max,
        "nside": config.analysis.nside,
        "random_indices": ",".join(str(value) for value in random_indices),
        "n_data": n_data,
        "n_random": n_random,
        "raw_amplitude": raw.amplitude,
        "raw_ra_deg": raw.ra_deg,
        "raw_dec_deg": raw.dec_deg,
        "raw_vector_x": float(raw.vector[0]),
        "raw_vector_y": float(raw.vector[1]),
        "raw_vector_z": float(raw.vector[2]),
        "corrected_amplitude": corrected.amplitude,
        "corrected_ra_deg": corrected.ra_deg,
        "corrected_dec_deg": corrected.dec_deg,
        "corrected_vector_x": float(corrected.vector[0]),
        "corrected_vector_y": float(corrected.vector[1]),
        "corrected_vector_z": float(corrected.vector[2]),
        "data_source": ", ".join(data_sources),
        "random_source": ", ".join(random_sources),
    }


def _load_official_mock_pair(
    config: ProjectConfig,
    roots: Sequence[Path],
    family: str,
    tracer: str,
    region: str,
    realization: int,
    random_indices: Sequence[int],
    program: str,
    flavor: str | None,
    weight_mode: str,
) -> CatalogPair:
    entries = build_official_mock_entries(
        family=family,
        tracers=[tracer],
        regions=[region],
        realizations=[realization],
        random_indices=random_indices,
        program=program,
        flavor=flavor,
    )
    data_entries = [entry for entry in entries if entry.kind == "data"]
    random_entries = [entry for entry in entries if entry.kind == "random"]
    if len(data_entries) != 1:
        raise ValueError("Expected exactly one official mock data entry.")
    return load_catalog_pair(
        data_path=_resolve_official_mock_local_path(roots, data_entries[0]),
        random_paths=[_resolve_official_mock_local_path(roots, entry) for entry in random_entries],
        tracer=tracer,
        region=region,
        config=config.desi,
        min_redshift=config.analysis.min_redshift,
        max_redshift=config.analysis.max_redshift,
        weight_mode=weight_mode,
    )


def _z_mask(z: np.ndarray, z_min: float, z_max: float) -> np.ndarray:
    return (z >= z_min) & (z <= z_max)


def _summarize_official_ensemble(
    mocks: pd.DataFrame,
    observed_raw,
    observed_corrected,
    skipped: Sequence[str],
    family: str,
    tracer: str,
    regions: Sequence[str],
    realizations: Sequence[int],
    z_min: float,
    z_max: float,
    nside: int,
    n_templates: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    observed = {
        "raw_no_template_regression": ("raw_amplitude", observed_raw),
        "after_external_template_regression": ("corrected_amplitude", observed_corrected),
    }
    for stage, (column, fit) in observed.items():
        values = mocks[column].to_numpy(dtype=float) if len(mocks) else np.array([], dtype=float)
        rows.append(
            {
                "family": family,
                "tracer": tracer,
                "regions": "+".join(regions),
                "z_min": z_min,
                "z_max": z_max,
                "nside": nside,
                "stage": stage,
                "requested_realizations": ",".join(str(value) for value in realizations),
                "n_requested": len(realizations),
                "n_processed": len(mocks),
                "n_skipped": len(skipped),
                "n_templates": n_templates,
                "mock_amplitude_median": float(np.median(values)) if len(values) else np.nan,
                "mock_amplitude_p68": float(np.quantile(values, 0.68)) if len(values) else np.nan,
                "mock_amplitude_p95": float(np.quantile(values, 0.95)) if len(values) else np.nan,
                "mock_amplitude_p99": float(np.quantile(values, 0.99)) if len(values) else np.nan,
                "observed_amplitude": fit.amplitude,
                "observed_ra_deg": fit.ra_deg,
                "observed_dec_deg": fit.dec_deg,
                "observed_empirical_p_value": (
                    (1.0 + np.sum(values >= fit.amplitude)) / (len(values) + 1.0)
                    if len(values)
                    else np.nan
                ),
            }
        )
    return pd.DataFrame.from_records(rows)


def _write_official_ensemble_report(
    report_path: Path,
    mocks: pd.DataFrame,
    summary: pd.DataFrame,
    skipped: Sequence[str],
) -> None:
    brief = summary.copy()
    for column in brief.select_dtypes(include="number").columns:
        if column not in {"nside", "n_requested", "n_processed", "n_skipped", "n_templates"}:
            brief[column] = brief[column].map(lambda value: f"{value:.6g}")
    lines = [
        "# Official DESI mock ensemble dipole calibration",
        "",
        "This report runs the same DESI sky-map dipole estimator over downloaded",
        "official mock clustering catalogs.",
        "",
        "## Summary",
        "",
        brief.to_markdown(index=False),
        "",
    ]
    if len(mocks):
        display = mocks[
            [
                "family",
                "realization",
                "tracer",
                "regions",
                "n_data",
                "n_random",
                "raw_amplitude",
                "raw_ra_deg",
                "raw_dec_deg",
                "corrected_amplitude",
                "corrected_ra_deg",
                "corrected_dec_deg",
            ]
        ].copy()
        lines.extend(["## Per-realization fits", "", display.to_markdown(index=False), ""])
    if skipped:
        lines.extend(["## Skipped realizations", "", *[f"- {item}" for item in skipped], ""])
    lines.extend(
        [
            "## Interpretation",
            "",
            "- This is the preferred direction for cosmology-grade calibration because",
            "the covariance comes from DESI-distributed mock catalogs.",
            "- A 20-realization subset is useful as a smoke ensemble. Stable p-values",
            "need a larger EZmock batch.",
        ]
    )
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_official_ensemble_null_plot(
    output_path: Path,
    mocks: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    import matplotlib.pyplot as plt

    stages = [
        ("raw_no_template_regression", "raw_amplitude", "Raw map"),
        ("after_external_template_regression", "corrected_amplitude", "Template-corrected map"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
    for ax, (stage, column, title) in zip(axes, stages, strict=True):
        row = summary.loc[summary["stage"] == stage].iloc[0]
        observed = float(row["observed_amplitude"])
        empirical_p = float(row["observed_empirical_p_value"])
        values = mocks[column].to_numpy(dtype=float)
        ax.hist(values, bins=30, color="#8394ad", edgecolor="white")
        ax.axvline(observed, color="#b33a3a", linewidth=2, label="observed")
        ax.set_title(f"{title}\np={empirical_p:.4g}")
        ax.set_xlabel("dipole amplitude")
        ax.legend()
    axes[0].set_ylabel("official mocks")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
