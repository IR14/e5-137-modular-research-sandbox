"""Plotting helpers for maps and dipole summaries."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from cosmo_gradient.coords import unit_to_radec
from cosmo_gradient.maps import SkyMap


def plot_overdensity_map(sky_map: SkyMap, output_path: Path, title: str) -> Path:
    """Plot a pixel-center scatter view of the overdensity map."""
    import matplotlib.pyplot as plt

    ra, dec = unit_to_radec(sky_map.pixel_vectors)
    delta = sky_map.delta
    valid = sky_map.valid & np.isfinite(delta)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vmax = np.nanpercentile(np.abs(delta[valid]), 95) if np.any(valid) else 1.0
    scatter = ax.scatter(
        ra[valid],
        dec[valid],
        c=delta[valid],
        s=12,
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
        linewidths=0,
    )
    ax.set_xlim(360, 0)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("DEC [deg]")
    ax.set_title(title)
    fig.colorbar(scatter, ax=ax, label="overdensity delta")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_amplitude_by_redshift(results: pd.DataFrame, output_path: Path) -> Path:
    """Plot fitted dipole amplitude versus redshift bin center."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for tracer, group in results.groupby("tracer"):
        z_mid = 0.5 * (group["z_min"] + group["z_max"])
        yerr = group["amplitude_std"] if "amplitude_std" in group else None
        ax.errorbar(z_mid, group["amplitude"], yerr=yerr, marker="o", capsize=3, label=tracer)
    ax.set_xlabel("redshift bin center")
    ax.set_ylabel("dipole amplitude")
    ax.set_title("First-pass dipole amplitude by redshift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_axis_by_redshift(results: pd.DataFrame, output_path: Path) -> Path:
    """Plot fitted axis RA and DEC versus redshift."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, (ax_ra, ax_dec) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for tracer, group in results.groupby("tracer"):
        z_mid = 0.5 * (group["z_min"] + group["z_max"])
        ax_ra.plot(z_mid, group["ra_deg"], marker="o", label=tracer)
        ax_dec.plot(z_mid, group["dec_deg"], marker="o", label=tracer)
    ax_ra.set_ylabel("RA [deg]")
    ax_dec.set_ylabel("DEC [deg]")
    ax_dec.set_xlabel("redshift bin center")
    ax_ra.set_title("First-pass dipole axis by redshift")
    ax_ra.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def plot_null_distribution(null_amplitudes: np.ndarray, observed: float, output_path: Path) -> Path:
    """Plot permutation null distribution for one fitted amplitude."""
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(null_amplitudes, bins=30, color="#8394ad", edgecolor="white")
    ax.axvline(observed, color="#b33a3a", linewidth=2, label="observed")
    ax.set_xlabel("dipole amplitude under null")
    ax.set_ylabel("permutations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path
