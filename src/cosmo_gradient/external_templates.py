"""Build external HEALPix systematics templates for regression diagnostics."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray


@dataclass(frozen=True)
class ExternalTemplateRecord:
    """Metadata for one generated external template."""

    name: str
    source: str
    column: str
    nside: int
    path: Path
    n_finite_pixels: int


def build_pixweight_templates(
    pixweight_path: Path,
    output_dir: Path,
    nside_out: int,
    columns: Sequence[str],
    prefix: str = "pixweight_dark",
) -> pd.DataFrame:
    """Build nside_out RING templates from a DESI pixweight HEALPix FITS table."""
    import fitsio

    output_dir.mkdir(parents=True, exist_ok=True)
    with fitsio.FITS(pixweight_path) as fits:
        header = fits[1].read_header()
        nside_in = int(header["HPXNSIDE"])
        nested = bool(header.get("HPXNEST", False))
        available = set(fits[1].get_colnames())
        selected = ["HPXPIXEL", "FRACAREA", *[column for column in columns if column in available]]
        if len(selected) <= 2:
            raise ValueError("None of the requested pixweight columns were found.")
        table = fits[1].read(columns=selected)

    pixel_out = _pixweight_output_pixels(
        table["HPXPIXEL"].astype(np.int64),
        nside_in=nside_in,
        nside_out=nside_out,
        nested=nested,
    )
    weights = np.asarray(table["FRACAREA"], dtype=float)

    records: list[ExternalTemplateRecord] = []
    for column in selected[2:]:
        values = np.asarray(table[column], dtype=float)
        template = _weighted_pixel_mean(pixel_out, values, weights, nside_out)
        path = output_dir / f"{prefix}_{column.lower()}_nside{nside_out}_ring.npz"
        _save_template(path, template)
        records.append(
            ExternalTemplateRecord(
                name=f"{prefix}_{column.lower()}",
                source=str(pixweight_path),
                column=column,
                nside=nside_out,
                path=path,
                n_finite_pixels=int(np.sum(np.isfinite(template))),
            )
        )
    return _records_to_frame(records)


def build_brick_templates(
    brick_paths: Sequence[Path],
    output_dir: Path,
    nside_out: int,
    columns: Sequence[str],
    prefix: str = "legacy_dr9_bricks",
) -> pd.DataFrame:
    """Build nside_out RING templates from Legacy Survey brick summary tables."""
    import fitsio

    output_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    for brick_path in brick_paths:
        with fitsio.FITS(brick_path) as fits:
            available = set(fits[1].get_colnames())
            selected_columns = [
                column
                for column in ["ra", "dec", "area", "survey_primary", "in_desi", *columns]
                if column in available
            ]
            table = fits[1].read(columns=selected_columns)
        frames.append(pd.DataFrame({name: table[name] for name in table.dtype.names or []}))
    if not frames:
        raise ValueError("No brick summary files were provided.")
    bricks = pd.concat(frames, ignore_index=True)

    pixel_out = _radec_output_pixels(
        bricks["ra"].to_numpy(dtype=float),
        bricks["dec"].to_numpy(dtype=float),
        nside_out,
    )
    weights = bricks["area"].to_numpy(dtype=float) if "area" in bricks else np.ones(len(bricks))
    if "survey_primary" in bricks:
        weights = weights * bricks["survey_primary"].astype(bool).to_numpy(dtype=float)
    if "in_desi" in bricks:
        weights = weights * bricks["in_desi"].astype(bool).to_numpy(dtype=float)

    records: list[ExternalTemplateRecord] = []
    for column in columns:
        if column not in bricks.columns:
            continue
        values = bricks[column].to_numpy(dtype=float)
        template = _weighted_pixel_mean(pixel_out, values, weights, nside_out)
        path = output_dir / f"{prefix}_{column.lower()}_nside{nside_out}_ring.npz"
        _save_template(path, template)
        records.append(
            ExternalTemplateRecord(
                name=f"{prefix}_{column.lower()}",
                source=",".join(str(path) for path in brick_paths),
                column=column,
                nside=nside_out,
                path=path,
                n_finite_pixels=int(np.sum(np.isfinite(template))),
            )
        )
    if not records:
        raise ValueError("None of the requested brick columns were found.")
    return _records_to_frame(records)


def _pixweight_output_pixels(
    pixels: NDArray[np.int64],
    nside_in: int,
    nside_out: int,
    nested: bool,
) -> NDArray[np.int64]:
    import healpy as hp

    theta, phi = hp.pix2ang(nside_in, pixels, nest=nested)
    return hp.ang2pix(nside_out, theta, phi, nest=False).astype(np.int64)


def _radec_output_pixels(
    ra_deg: NDArray[np.float64],
    dec_deg: NDArray[np.float64],
    nside: int,
) -> NDArray[np.int64]:
    import healpy as hp

    theta = np.deg2rad(90.0 - dec_deg)
    phi = np.deg2rad(ra_deg)
    return hp.ang2pix(nside, theta, phi, nest=False).astype(np.int64)


def _weighted_pixel_mean(
    pixels: NDArray[np.int64],
    values: NDArray[np.float64],
    weights: NDArray[np.float64],
    nside: int,
) -> NDArray[np.float64]:
    import healpy as hp

    npix = hp.nside2npix(nside)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    weighted_sum = np.bincount(
        pixels[finite],
        weights=weights[finite] * values[finite],
        minlength=npix,
    ).astype(float)
    weight_sum = np.bincount(pixels[finite], weights=weights[finite], minlength=npix).astype(float)
    template = np.full(npix, np.nan, dtype=float)
    good = weight_sum > 0.0
    template[good] = weighted_sum[good] / weight_sum[good]
    return template


def _save_template(path: Path, template: NDArray[np.float64]) -> None:
    np.savez_compressed(path, template=np.asarray(template, dtype=float))


def _records_to_frame(records: Sequence[ExternalTemplateRecord]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": [record.name for record in records],
            "source": [record.source for record in records],
            "column": [record.column for record in records],
            "nside": [record.nside for record in records],
            "path": [str(record.path) for record in records],
            "n_finite_pixels": [record.n_finite_pixels for record in records],
        }
    )
