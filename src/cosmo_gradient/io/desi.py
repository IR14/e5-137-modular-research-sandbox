"""DESI DR1 LSS catalog loading and synthetic catalog generation."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from numpy.random import Generator
from numpy.typing import NDArray

from cosmo_gradient.config import DESIConfig, SyntheticConfig
from cosmo_gradient.coords import radec_to_unit
from cosmo_gradient.masks import synthetic_survey_mask

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CatalogPair:
    """A data catalog and its matching random catalog in standardized columns."""

    tracer: str
    region: str
    data: pd.DataFrame
    randoms: pd.DataFrame
    data_source: str
    random_source: str
    synthetic: bool = False


def build_lss_urls(
    config: DESIConfig,
    tracer: str,
    region: Optional[str] = None,
    random_indices: Optional[Sequence[int]] = None,
) -> tuple[str, list[str]]:
    """Build expected DESI DR1 LSS clustering catalog and random catalog URLs."""
    use_region = region or config.default_region
    file_tracer = desi_file_tracer(config, tracer)
    indices = list(random_indices or config.random_indices)
    data_name = config.file_templates["data"].format(tracer=file_tracer, region=use_region)
    random_names = [
        config.file_templates["random"].format(tracer=file_tracer, region=use_region, index=index)
        for index in indices
    ]
    base = config.base_url.rstrip("/")
    return f"{base}/{data_name}", [f"{base}/{name}" for name in random_names]


def desi_file_tracer(config: DESIConfig, tracer: str) -> str:
    """Return the DESI filename tracer for a user-facing tracer label."""
    return config.tracer_aliases.get(tracer, tracer)


def write_download_manifest(
    output_path: Path,
    config: DESIConfig,
    tracers: Sequence[str],
    regions: Sequence[str],
    random_indices: Sequence[int],
) -> Path:
    """Write a dry-run manifest with URLs and explicit download commands."""
    lines = [
        "# DESI DR1 LSS download manifest",
        "",
        "This file is generated in dry-run mode. Review file sizes and DESI",
        "documentation before downloading large catalogs.",
        "",
    ]
    for tracer in tracers:
        for region in regions:
            data_url, random_urls = build_lss_urls(config, tracer, region, random_indices)
            lines.extend([f"## {tracer} {region}", "", f"- data: {data_url}"])
            for url in random_urls:
                lines.append(f"- random: {url}")
            lines.extend(
                [
                    "",
                    "```bash",
                    f"curl -L -O {data_url}",
                    *[f"curl -L -O {url}" for url in random_urls],
                    "```",
                    "",
                ]
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def download_manifest_files(manifest_urls: Sequence[str], destination: Path) -> list[Path]:
    """Download explicitly requested files into *destination*."""
    destination.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for url in manifest_urls:
        target = destination / url.rstrip("/").split("/")[-1]
        urlretrieve(url, target)
        downloaded.append(target)
    return downloaded


def load_catalog_pair(
    data_path: Path,
    random_paths: Sequence[Path],
    tracer: str,
    region: str,
    config: DESIConfig,
    min_redshift: float,
    max_redshift: float,
    weight_mode: str = "desi",
) -> CatalogPair:
    """Load and standardize one DESI clustering catalog with one or more random catalogs."""
    data = standardize_catalog(
        _read_table(data_path, config=config),
        config=config,
        tracer=tracer,
        min_redshift=min_redshift,
        max_redshift=max_redshift,
        weight_mode=weight_mode,
    )
    random_frames = [
        standardize_catalog(
            _read_table(path, config=config),
            config=config,
            tracer=tracer,
            min_redshift=min_redshift,
            max_redshift=max_redshift,
            is_random=True,
            weight_mode=weight_mode,
        )
        for path in random_paths
    ]
    randoms = pd.concat(random_frames, ignore_index=True)
    return CatalogPair(
        tracer=tracer,
        region=region,
        data=data,
        randoms=randoms,
        data_source=str(data_path),
        random_source=", ".join(str(path) for path in random_paths),
        synthetic=False,
    )


def _read_table(path: Path, config: Optional[DESIConfig] = None) -> pd.DataFrame:
    _validate_not_error_document(path)
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".parquet"):
        columns = _candidate_columns(config)
        try:
            return pd.read_parquet(path, columns=columns)
        except (KeyError, ValueError):
            return pd.read_parquet(path)
    if suffixes.endswith(".csv"):
        return pd.read_csv(path)
    if suffixes.endswith(".fits") or suffixes.endswith(".fits.gz"):
        return _read_fits(path, config=config)
    raise ValueError(f"Unsupported catalog format: {path}")


def _validate_not_error_document(path: Path) -> None:
    """Fail early when a failed HTTP response was saved with a FITS filename."""
    with path.open("rb") as stream:
        prefix = stream.read(256).lstrip()
    if prefix.startswith((b"<?xml", b"<Error>", b"<!DOCTYPE", b"<html", b"<HTML")):
        raise OSError(
            f"{path} is not a FITS table; it looks like an HTTP/XML/HTML error response. "
            "Delete it and download the correct DESI catalog filename."
        )


def _read_fits(path: Path, config: Optional[DESIConfig] = None) -> pd.DataFrame:
    try:
        import fitsio

        columns = _existing_fits_columns(path, config)
        try:
            return _records_to_native_dataframe(fitsio.read(path, ext=1, columns=columns))
        except UnicodeDecodeError:
            core_columns = _existing_fits_columns(path, config, include_metadata=False)
            if core_columns == columns:
                raise
            LOGGER.warning(
                "Retrying FITS read without metadata string columns after Unicode decode "
                "failure: %s",
                path,
            )
            return _records_to_native_dataframe(fitsio.read(path, ext=1, columns=core_columns))
    except ImportError:
        from astropy.table import Table

        columns = _candidate_columns(config)
        if columns:
            try:
                return Table.read(path, hdu=1, include_names=columns).to_pandas()
            except TypeError:
                pass
        return Table.read(path, hdu=1).to_pandas()


def _records_to_native_dataframe(records: np.ndarray) -> pd.DataFrame:
    """Build a DataFrame while converting FITS big-endian arrays to native endian."""
    if records.dtype.fields is None:
        return pd.DataFrame(_native_endian_array(records))
    columns = {
        name: _native_endian_array(records[name])
        for name in records.dtype.names or []
    }
    return pd.DataFrame(columns)


def _native_endian_array(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.dtype.byteorder in ("=", "|"):
        return arr
    native_order = "<" if np.little_endian else ">"
    if arr.dtype.byteorder == native_order:
        return arr.view(arr.dtype.newbyteorder("="))
    return arr.byteswap().view(arr.dtype.newbyteorder("="))


def _candidate_columns(
    config: Optional[DESIConfig],
    include_metadata: bool = True,
) -> Optional[list[str]]:
    if config is None:
        return None
    candidates: list[str] = []
    for group_name, names in config.columns.items():
        if group_name == "metadata" and not include_metadata:
            continue
        for name in names:
            if name not in candidates:
                candidates.append(name)
    return candidates


def _existing_fits_columns(
    path: Path,
    config: Optional[DESIConfig],
    include_metadata: bool = True,
) -> Optional[list[str]]:
    candidates = _candidate_columns(config, include_metadata=include_metadata)
    if not candidates:
        return None
    import fitsio

    with fitsio.FITS(path) as fits:
        existing = set(fits[1].get_colnames())
    selected = [name for name in candidates if name in existing]
    return selected or None


def standardize_catalog(
    frame: pd.DataFrame,
    config: DESIConfig,
    tracer: str,
    min_redshift: float,
    max_redshift: float,
    is_random: bool = False,
    weight_mode: str = "desi",
) -> pd.DataFrame:
    """Standardize DESI-like tables to ``ra``, ``dec``, ``z``, ``weight``, ``tracer``."""
    ra_col = _first_existing(frame, config.columns["ra"])
    dec_col = _first_existing(frame, config.columns["dec"])
    z_col = _first_existing(frame, config.columns["redshift"])
    weight = _extract_weight(frame, config, weight_mode=weight_mode)
    standardized = pd.DataFrame(
        {
            "ra": pd.to_numeric(frame[ra_col], errors="coerce"),
            "dec": pd.to_numeric(frame[dec_col], errors="coerce"),
            "z": pd.to_numeric(frame[z_col], errors="coerce"),
            "weight": weight,
            "tracer": tracer,
            "is_random": is_random,
        }
    )
    for column in config.columns.get("metadata", []):
        if column in frame.columns:
            standardized[column.lower()] = frame[column].to_numpy()
    quality = (
        np.isfinite(standardized["ra"])
        & np.isfinite(standardized["dec"])
        & np.isfinite(standardized["z"])
        & np.isfinite(standardized["weight"])
        & (standardized["weight"] > 0.0)
        & (standardized["ra"] >= 0.0)
        & (standardized["ra"] < 360.0)
        & (standardized["dec"] >= -90.0)
        & (standardized["dec"] <= 90.0)
        & (standardized["z"] >= min_redshift)
        & (standardized["z"] <= max_redshift)
    )
    return standardized.loc[quality].reset_index(drop=True)


def _first_existing(frame: pd.DataFrame, candidates: Sequence[str]) -> str:
    for name in candidates:
        if name in frame.columns:
            return name
    raise KeyError(f"None of the expected columns exist: {candidates}")


def _extract_weight(frame: pd.DataFrame, config: DESIConfig, weight_mode: str = "desi") -> pd.Series:
    if weight_mode == "uniform":
        return pd.Series(np.ones(len(frame), dtype=float), index=frame.index)
    if weight_mode not in {"desi", "no_fkp"}:
        raise ValueError("weight_mode must be one of: desi, uniform, no_fkp")
    if weight_mode == "no_fkp":
        component_cols = [
            col
            for col in config.columns["weight_components"]
            if col in frame.columns and col != "WEIGHT_FKP"
        ]
        if component_cols:
            weight = np.ones(len(frame), dtype=float)
            for col in component_cols:
                weight *= pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
            return pd.Series(weight, index=frame.index)
    object_weight_cols = [col for col in config.columns["object_weight"] if col in frame.columns]
    if object_weight_cols:
        return pd.to_numeric(frame[object_weight_cols[0]], errors="coerce").astype(float)
    component_cols = [col for col in config.columns["weight_components"] if col in frame.columns]
    if not component_cols:
        return pd.Series(np.ones(len(frame), dtype=float), index=frame.index)
    weight = np.ones(len(frame), dtype=float)
    for col in component_cols:
        weight *= pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
    return pd.Series(weight, index=frame.index)


def generate_synthetic_pair(
    tracer: str,
    region: str,
    config: SyntheticConfig,
    rng: Generator,
) -> CatalogPair:
    """Generate synthetic DESI-like data and random catalogs for local smoke tests."""
    axis = radec_to_unit([config.dipole_axis_ra_deg], [config.dipole_axis_dec_deg])[0]
    amplitude = float(config.dipole_amplitude.get(tracer, 0.03))
    randoms = _sample_masked_sky(
        config.n_random_per_tracer,
        rng,
        apply_mask=config.apply_survey_like_mask,
    )
    randoms["z"] = rng.uniform(config.z_min, config.z_max, len(randoms))
    randoms["weight"] = 1.0
    randoms["tracer"] = tracer
    randoms["is_random"] = True

    data = _sample_dipole_modulated_sky(
        config.n_data_per_tracer,
        rng,
        axis=axis,
        amplitude=amplitude,
        apply_mask=config.apply_survey_like_mask,
    )
    data["z"] = rng.uniform(config.z_min, config.z_max, len(data))
    data["weight"] = rng.lognormal(mean=0.0, sigma=0.05, size=len(data))
    data["tracer"] = tracer
    data["is_random"] = False
    return CatalogPair(
        tracer=tracer,
        region=region,
        data=data,
        randoms=randoms,
        data_source="synthetic",
        random_source="synthetic",
        synthetic=True,
    )


def _sample_masked_sky(n_rows: int, rng: Generator, apply_mask: bool) -> pd.DataFrame:
    ra_values: list[NDArray[np.float64]] = []
    dec_values: list[NDArray[np.float64]] = []
    accepted = 0
    while accepted < n_rows:
        batch = max(2048, int((n_rows - accepted) * 1.4))
        ra = rng.uniform(0.0, 360.0, batch)
        sin_dec = rng.uniform(-1.0, 1.0, batch)
        dec = np.rad2deg(np.arcsin(sin_dec))
        mask = synthetic_survey_mask(ra, dec) if apply_mask else np.ones(batch, dtype=bool)
        keep_ra = ra[mask]
        keep_dec = dec[mask]
        need = n_rows - accepted
        ra_values.append(keep_ra[:need])
        dec_values.append(keep_dec[:need])
        accepted += min(need, len(keep_ra))
    return pd.DataFrame({"ra": np.concatenate(ra_values), "dec": np.concatenate(dec_values)})


def _sample_dipole_modulated_sky(
    n_rows: int,
    rng: Generator,
    axis: NDArray[np.float64],
    amplitude: float,
    apply_mask: bool,
) -> pd.DataFrame:
    if not 0.0 <= amplitude < 1.0:
        raise ValueError("Synthetic dipole amplitude must be in [0, 1).")
    ra_values: list[NDArray[np.float64]] = []
    dec_values: list[NDArray[np.float64]] = []
    accepted = 0
    while accepted < n_rows:
        proposal = _sample_masked_sky(
            max(4096, int((n_rows - accepted) * 1.8)),
            rng,
            apply_mask=apply_mask,
        )
        vectors = radec_to_unit(proposal["ra"].to_numpy(), proposal["dec"].to_numpy())
        modulation = 1.0 + amplitude * (vectors @ axis)
        probability = modulation / (1.0 + amplitude)
        keep = rng.uniform(0.0, 1.0, len(proposal)) < probability
        keep_ra = proposal.loc[keep, "ra"].to_numpy()
        keep_dec = proposal.loc[keep, "dec"].to_numpy()
        need = n_rows - accepted
        ra_values.append(keep_ra[:need])
        dec_values.append(keep_dec[:need])
        accepted += min(need, len(keep_ra))
    return pd.DataFrame({"ra": np.concatenate(ra_values), "dec": np.concatenate(dec_values)})
