"""Configuration models and YAML loading."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PathConfig:
    """Filesystem layout used by the pipeline."""

    data_raw: Path = Path("data/raw")
    additional_raw_roots: list[Path] = field(default_factory=list)
    data_interim: Path = Path("data/interim")
    data_processed: Path = Path("data/processed")
    data_external: Path = Path("data/external")
    outputs: Path = Path("outputs")
    figures: Path = Path("outputs/figures")
    tables: Path = Path("outputs/tables")
    reports: Path = Path("outputs/reports")

    def resolve_from(self, base_dir: Path) -> PathConfig:
        """Return a copy where relative paths are resolved from *base_dir*."""
        values = {}
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name == "additional_raw_roots":
                roots = []
                for path_value in value:
                    path = Path(path_value)
                    roots.append(path if path.is_absolute() else base_dir / path)
                values[field_name] = roots
                continue
            path = Path(value)
            values[field_name] = path if path.is_absolute() else base_dir / path
        return PathConfig(**values)

    def ensure_dirs(self) -> None:
        """Create all project directories if needed."""
        for field_name in self.__dataclass_fields__:
            value = getattr(self, field_name)
            if field_name == "additional_raw_roots":
                continue
            value.mkdir(parents=True, exist_ok=True)

    def raw_search_roots(self) -> list[Path]:
        """Return raw-data roots in lookup order."""
        return [self.data_raw, *self.additional_raw_roots]


@dataclass(frozen=True)
class DESIConfig:
    """DESI DR1 LSS catalog location and column conventions."""

    release: str = "dr1"
    spectrograph: str = "iron"
    lss_version: str = "v1.5"
    base_url: str = (
        "https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/"
        "LSS/iron/LSScats/v1.5"
    )
    default_region: str = "NGC"
    random_indices: list[int] = field(default_factory=lambda: [0])
    tracer_aliases: dict[str, str] = field(default_factory=lambda: {"ELG": "ELG_LOPnotqso"})
    file_templates: dict[str, str] = field(
        default_factory=lambda: {
            "data": "{tracer}_{region}_clustering.dat.fits",
            "random": "{tracer}_{region}_{index}_clustering.ran.fits",
        }
    )
    columns: dict[str, list[str]] = field(
        default_factory=lambda: {
            "ra": ["RA", "ra"],
            "dec": ["DEC", "dec"],
            "redshift": ["Z", "z", "Z_not4clus"],
            "object_weight": ["WEIGHT", "weight"],
            "weight_components": ["WEIGHT_COMP", "WEIGHT_ZFAIL", "WEIGHT_SYS", "WEIGHT_FKP"],
            "metadata": [
                "PHOTSYS",
                "NTILE",
                "FRAC_TLOBS_TILES",
                "WEIGHT_COMP",
                "WEIGHT_ZFAIL",
                "WEIGHT_SYS",
                "WEIGHT_SN",
                "WEIGHT_RF",
                "WEIGHT_FKP",
                "NX",
            ],
        }
    )


@dataclass(frozen=True)
class AnalysisConfig:
    """Scientific analysis options."""

    mode: str = "synthetic"
    tracers: list[str] = field(default_factory=lambda: ["LRG", "ELG", "QSO"])
    regions: list[str] = field(default_factory=lambda: ["NGC"])
    z_bins: dict[str, list[float]] = field(
        default_factory=lambda: {
            "LRG": [0.4, 0.6, 0.8, 1.1],
            "ELG": [0.8, 1.1, 1.4, 1.6],
            "QSO": [0.8, 1.2, 1.6, 2.1, 3.5],
        }
    )
    min_redshift: float = 0.01
    max_redshift: float = 6.0
    nside: int = 16
    min_random_per_pixel: float = 1.0
    random_seed: int = 20260524


@dataclass(frozen=True)
class SyntheticConfig:
    """Synthetic catalog generation options."""

    n_data_per_tracer: int = 20_000
    n_random_per_tracer: int = 160_000
    dipole_amplitude: dict[str, float] = field(
        default_factory=lambda: {"LRG": 0.05, "ELG": 0.035, "QSO": 0.025}
    )
    dipole_axis_ra_deg: float = 215.0
    dipole_axis_dec_deg: float = 25.0
    z_min: float = 0.4
    z_max: float = 3.5
    apply_survey_like_mask: bool = True


@dataclass(frozen=True)
class DipoleConfig:
    """Dipole fitting and null-test options."""

    bootstrap_samples: int = 80
    null_permutations: int = 120
    poisson_mocks: int = 0
    block_null_mocks: int = 0
    block_null_regions: int = 12
    jackknife_regions: int = 12


@dataclass(frozen=True)
class ReportConfig:
    """Report generation options."""

    filename: str = "first_pass_report.md"


@dataclass(frozen=True)
class ProjectConfig:
    """Top-level project configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    desi: DESIConfig = field(default_factory=DESIConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    synthetic: SyntheticConfig = field(default_factory=SyntheticConfig)
    dipole: DipoleConfig = field(default_factory=DipoleConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    config_path: Path | None = None
    base_dir: Path = Path(".")


def _dataclass_from_mapping(cls: type, mapping: Mapping[str, Any]) -> Any:
    allowed = cls.__dataclass_fields__
    values = {}
    for key, value in mapping.items():
        if key in allowed:
            values[key] = value
    return cls(**values)


def load_config(path: str | Path = "configs/default.yaml") -> ProjectConfig:
    """Load project configuration from YAML.

    Relative paths inside the YAML are resolved from the directory containing
    ``pyproject.toml`` when it can be inferred, otherwise from the config file's
    parent directory.
    """
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - exercised only in incomplete envs
        raise RuntimeError("PyYAML is required to load YAML configuration files.") from exc

    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    base_dir = _find_project_root(config_path.parent)
    paths = _dataclass_from_mapping(PathConfig, raw.get("paths", {})).resolve_from(base_dir)
    config = ProjectConfig(
        paths=paths,
        desi=_dataclass_from_mapping(DESIConfig, raw.get("desi", {})),
        analysis=_dataclass_from_mapping(AnalysisConfig, raw.get("analysis", {})),
        synthetic=_dataclass_from_mapping(SyntheticConfig, raw.get("synthetic", {})),
        dipole=_dataclass_from_mapping(DipoleConfig, raw.get("dipole", {})),
        report=_dataclass_from_mapping(ReportConfig, raw.get("report", {})),
        config_path=config_path,
        base_dir=base_dir,
    )
    return config


def _find_project_root(start: Path) -> Path:
    """Find the nearest parent containing ``pyproject.toml``."""
    for candidate in [start, *start.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    return start
