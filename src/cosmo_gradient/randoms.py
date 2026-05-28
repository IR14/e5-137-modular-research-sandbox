"""Random-catalog helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from cosmo_gradient.config import DESIConfig
from cosmo_gradient.io.desi import standardize_catalog


def stack_random_catalogs(
    frames: Sequence[pd.DataFrame],
    config: DESIConfig,
    tracer: str,
    min_redshift: float,
    max_redshift: float,
) -> pd.DataFrame:
    """Standardize and concatenate random-catalog tables."""
    standardized = [
        standardize_catalog(
            frame,
            config=config,
            tracer=tracer,
            min_redshift=min_redshift,
            max_redshift=max_redshift,
            is_random=True,
        )
        for frame in frames
    ]
    return pd.concat(standardized, ignore_index=True)


def expected_random_paths(
    data_dir: Path,
    tracer: str,
    region: str,
    indices: Sequence[int],
    template: str,
) -> list[Path]:
    """Return expected local random-catalog paths."""
    return [
        data_dir / template.format(tracer=tracer, region=region, index=index)
        for index in indices
    ]
