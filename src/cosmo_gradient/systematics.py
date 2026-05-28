"""Template-regression diagnostics for survey systematics maps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from cosmo_gradient.maps import SkyMap, pixelize_radec


@dataclass(frozen=True)
class TemplateRegressionResult:
    """Result of a weighted template regression on an overdensity map."""

    corrected_map: SkyMap
    coefficients: pd.DataFrame
    template_names: list[str]
    weighted_r2: float


def build_template_maps(
    frame: pd.DataFrame,
    nside: int,
    columns: Sequence[str],
    valid: NDArray[np.bool_],
    backend: str,
) -> tuple[NDArray[np.float64], list[str]]:
    """Build standardized per-pixel systematics templates from a catalog frame."""
    raw_templates: list[NDArray[np.float64]] = []
    names: list[str] = []
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column]
        if pd.api.types.is_numeric_dtype(values):
            template = _numeric_pixel_mean(frame, column, nside)
            raw_templates.append(template)
            names.append(column)
        else:
            cat_templates, cat_names = _categorical_pixel_fractions(frame, column, nside)
            raw_templates.extend(cat_templates)
            names.extend(cat_names)
    if not raw_templates:
        return np.empty((len(valid), 0), dtype=float), []

    standardized: list[NDArray[np.float64]] = []
    standardized_names: list[str] = []
    weights = np.ones(len(valid), dtype=float)
    for name, template in zip(names, raw_templates, strict=True):
        if len(template) != len(valid):
            raise ValueError("Template map size does not match sky map size.")
        normalized = _standardize_template(template, valid=valid, weights=weights)
        if normalized is None:
            continue
        standardized.append(normalized)
        standardized_names.append(name)
    if not standardized:
        return np.empty((len(valid), 0), dtype=float), []
    template_matrix = np.column_stack(standardized)
    if backend != "healpy":
        # Backend is included to make accidental mixed pixelizations easier to spot in debugging.
        template_matrix = np.asarray(template_matrix, dtype=float)
    return template_matrix, standardized_names


def load_external_template_maps(
    specs: Sequence[str],
    valid: NDArray[np.bool_],
    expected_npix: int,
) -> tuple[NDArray[np.float64], list[str]]:
    """Load standardized external per-pixel templates from disk.

    Specs use ``name=path`` or ``name:path``. Supported files are one-dimensional
    ``.npy`` arrays, ``.npz`` archives, and HEALPix-readable FITS maps. External
    templates must already be in the same pixel ordering and resolution as the
    analysis map.
    """
    templates: list[NDArray[np.float64]] = []
    names: list[str] = []
    weights = np.ones(len(valid), dtype=float)
    for spec in specs:
        name, path = _parse_external_template_spec(spec)
        values = _load_template_array(path)
        if values.ndim != 1:
            raise ValueError(f"External template {path} must be one-dimensional.")
        if len(values) != expected_npix:
            raise ValueError(
                f"External template {path} has {len(values)} pixels, expected {expected_npix}. "
                "Reproject or downgrade it to the analysis map nside/order first."
            )
        normalized = _standardize_template(values.astype(float), valid=valid, weights=weights)
        if normalized is None:
            continue
        templates.append(normalized)
        names.append(f"external:{name}")
    if not templates:
        return np.empty((len(valid), 0), dtype=float), []
    return np.column_stack(templates), names


def combine_template_matrices(
    *template_sets: tuple[NDArray[np.float64], Sequence[str]],
) -> tuple[NDArray[np.float64], list[str]]:
    """Combine zero or more template matrices with matching pixel rows."""
    matrices: list[NDArray[np.float64]] = []
    names: list[str] = []
    row_count: int | None = None
    for matrix, matrix_names in template_sets:
        arr = np.asarray(matrix, dtype=float)
        if arr.ndim == 1:
            arr = arr[:, None]
        if row_count is None:
            row_count = arr.shape[0]
        elif arr.shape[0] != row_count:
            raise ValueError("Template matrices must have matching row counts.")
        if arr.shape[1] == 0:
            continue
        matrices.append(arr)
        names.extend(str(name) for name in matrix_names)
    if row_count is None:
        return np.empty((0, 0), dtype=float), []
    if not matrices:
        return np.empty((row_count, 0), dtype=float), []
    return np.column_stack(matrices), names


def regress_templates(
    sky_map: SkyMap,
    template_maps: NDArray[np.float64],
    template_names: Sequence[str],
) -> TemplateRegressionResult:
    """Regress template maps out of a sky overdensity map with random-count weights."""
    valid = sky_map.valid & np.isfinite(sky_map.delta)
    templates = np.asarray(template_maps, dtype=float)
    if templates.ndim == 1:
        templates = templates[:, None]
    if templates.shape[0] != len(sky_map.delta):
        raise ValueError("Template maps must have one row per sky pixel.")
    if templates.shape[1] != len(template_names):
        raise ValueError("Template name count does not match template matrix width.")

    corrected_delta = sky_map.delta.copy()
    if templates.shape[1] == 0:
        coefficients = pd.DataFrame(columns=["term", "coefficient"])
        return TemplateRegressionResult(
            corrected_map=_replace_delta(sky_map, corrected_delta),
            coefficients=coefficients,
            template_names=[],
            weighted_r2=float("nan"),
        )

    selected_templates = templates[valid]
    finite_columns = np.all(np.isfinite(selected_templates), axis=0)
    selected_templates = selected_templates[:, finite_columns]
    selected_names = [
        name for name, keep in zip(template_names, finite_columns, strict=True) if keep
    ]
    if selected_templates.shape[1] == 0:
        coefficients = pd.DataFrame(columns=["term", "coefficient"])
        return TemplateRegressionResult(
            corrected_map=_replace_delta(sky_map, corrected_delta),
            coefficients=coefficients,
            template_names=[],
            weighted_r2=float("nan"),
        )

    y = sky_map.delta[valid]
    design = np.column_stack([np.ones(len(y)), selected_templates])
    weights = np.clip(sky_map.random_counts[valid].astype(float), 1e-12, None)
    sqrt_w = np.sqrt(weights)
    params, *_ = np.linalg.lstsq(design * sqrt_w[:, None], y * sqrt_w, rcond=None)
    model = design @ params
    residual = y - model
    corrected_delta[valid] = residual

    y_bar = float(np.average(y, weights=weights))
    total_ss = float(np.sum(weights * (y - y_bar) ** 2))
    residual_ss = float(np.sum(weights * residual**2))
    weighted_r2 = 1.0 - residual_ss / total_ss if total_ss > 0.0 else float("nan")
    coefficients = pd.DataFrame(
        {
            "term": ["intercept", *selected_names],
            "coefficient": params.astype(float),
        }
    )
    return TemplateRegressionResult(
        corrected_map=_replace_delta(sky_map, corrected_delta),
        coefficients=coefficients,
        template_names=selected_names,
        weighted_r2=float(weighted_r2),
    )


def _numeric_pixel_mean(frame: pd.DataFrame, column: str, nside: int) -> NDArray[np.float64]:
    pix, npix, _ = pixelize_radec(frame["ra"].to_numpy(), frame["dec"].to_numpy(), nside)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    weights = _catalog_weights(frame)
    finite = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    sums = np.bincount(
        pix[finite],
        weights=weights[finite] * values[finite],
        minlength=npix,
    ).astype(float)
    weight_sums = np.bincount(pix[finite], weights=weights[finite], minlength=npix).astype(float)
    mean = np.full(npix, np.nan, dtype=float)
    good = weight_sums > 0.0
    mean[good] = sums[good] / weight_sums[good]
    return mean


def _categorical_pixel_fractions(
    frame: pd.DataFrame,
    column: str,
    nside: int,
) -> tuple[list[NDArray[np.float64]], list[str]]:
    pix, npix, _ = pixelize_radec(frame["ra"].to_numpy(), frame["dec"].to_numpy(), nside)
    values = frame[column].astype(str).to_numpy()
    weights = _catalog_weights(frame)
    finite = np.isfinite(weights) & (weights > 0.0) & pd.notna(frame[column]).to_numpy()
    categories = sorted(set(values[finite]))
    if len(categories) <= 1:
        return [], []
    total = np.bincount(pix[finite], weights=weights[finite], minlength=npix).astype(float)
    templates: list[NDArray[np.float64]] = []
    names: list[str] = []
    for category in categories[:-1]:
        mask = finite & (values == category)
        sums = np.bincount(pix[mask], weights=weights[mask], minlength=npix).astype(float)
        fraction = np.full(npix, np.nan, dtype=float)
        good = total > 0.0
        fraction[good] = sums[good] / total[good]
        templates.append(fraction)
        names.append(f"{column}={category}")
    return templates, names


def _catalog_weights(frame: pd.DataFrame) -> NDArray[np.float64]:
    if "weight" in frame:
        return frame["weight"].to_numpy(dtype=float)
    return np.ones(len(frame), dtype=float)


def _standardize_template(
    values: NDArray[np.float64],
    valid: NDArray[np.bool_],
    weights: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    arr = np.asarray(values, dtype=float).copy()
    selected = valid & np.isfinite(arr)
    if np.sum(selected) < 4:
        return None
    selected_weights = np.clip(weights[selected], 1e-12, None)
    mean = float(np.average(arr[selected], weights=selected_weights))
    arr[~np.isfinite(arr)] = mean
    centered = arr - mean
    variance = float(np.average(centered[selected] ** 2, weights=selected_weights))
    if not np.isfinite(variance) or variance <= 1e-20:
        return None
    return centered / np.sqrt(variance)


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


def _parse_external_template_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path_text = spec.split("=", 1)
    elif ":" in spec:
        name, path_text = spec.split(":", 1)
    else:
        path = Path(spec).expanduser()
        return path.stem, path
    name = name.strip()
    path = Path(path_text.strip()).expanduser()
    if not name:
        raise ValueError(f"External template spec has an empty name: {spec}")
    return name, path


def _load_template_array(path: Path) -> NDArray[np.float64]:
    if not path.exists():
        raise FileNotFoundError(f"External template not found: {path}")
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".npy"):
        return np.asarray(np.load(path, allow_pickle=False), dtype=float)
    if suffixes.endswith(".npz"):
        with np.load(path, allow_pickle=False) as loaded:
            key = _template_key_for_npz(loaded)
            return np.asarray(loaded[key], dtype=float)
    if suffixes.endswith(".fits") or suffixes.endswith(".fits.gz"):
        try:
            import healpy as hp
        except ImportError as exc:  # pragma: no cover - depends on optional healpy
            raise RuntimeError("healpy is required to read external FITS HEALPix maps.") from exc
        return np.asarray(hp.read_map(path, verbose=False), dtype=float)
    raise ValueError(f"Unsupported external template format: {path}")


def _template_key_for_npz(loaded: np.lib.npyio.NpzFile) -> str:
    preferred = ("template", "map", "values", "delta")
    for key in preferred:
        if key in loaded.files:
            return key
    one_dimensional = [key for key in loaded.files if np.asarray(loaded[key]).ndim == 1]
    if not one_dimensional:
        raise ValueError("NPZ template archive must contain at least one 1D array.")
    return one_dimensional[0]
