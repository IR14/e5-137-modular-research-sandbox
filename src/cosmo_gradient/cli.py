"""Command-line interface for the cosmo-gradient pipeline."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from cosmo_gradient.clustered_mocks import run_lognormal_mock_calibration
from cosmo_gradient.config import load_config
from cosmo_gradient.diagnostics import (
    parse_axis_specs,
    write_injection_recovery,
    write_template_group_audit,
)
from cosmo_gradient.dipole import jackknife_region_table
from cosmo_gradient.external_templates import build_brick_templates, build_pixweight_templates
from cosmo_gradient.fastspecfit import (
    inventory_local_tables,
    run_fastspecfit_gradient_validation,
    write_fastspecfit_preflight_report,
)
from cosmo_gradient.logging_utils import configure_logging
from cosmo_gradient.maps import load_sky_map
from cosmo_gradient.mock_calibration import run_mock_calibration
from cosmo_gradient.multipoles import write_map_multipole_diagnostics
from cosmo_gradient.official_mocks import (
    run_official_mock_ensemble,
    write_official_mock_download_queue,
    write_official_mock_manifest,
)
from cosmo_gradient.pipeline import (
    make_plots,
    prepare_downloads,
    run_combined_regions,
    run_combined_template_systematics,
    run_first_pass,
    run_nside_robustness_grid,
    run_systematics_audit,
    run_template_systematics,
    write_report,
)
from cosmo_gradient.reports import write_axis_diagnostics, write_master_science_report
from cosmo_gradient.shell_mocks import run_shell_lognormal_calibration
from cosmo_gradient.statistics import write_look_elsewhere_summary
from cosmo_gradient.storage import check_storage_root

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cosmo-gradient",
        description="DESI DR1 LSS first-pass directional-gradient checks.",
    )
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config.")
    parser.add_argument("--log-level", default="INFO", help="Python logging level.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare", help="Create a dry-run DESI download manifest.")
    prepare.add_argument("--tracer", action="append", dest="tracers", help="Tracer to include.")
    prepare.add_argument("--region", action="append", dest="regions", help="Region to include, e.g. NGC.")
    prepare.add_argument("--random-index", action="append", type=int, dest="random_indices")

    storage = subparsers.add_parser(
        "check-storage",
        help="Check whether a local or external data root is mounted, writable, and large enough.",
    )
    storage.add_argument(
        "--path",
        required=True,
        help="Candidate data root, e.g. /Volumes/COSMO_T7/datasets.",
    )
    storage.add_argument("--min-free-gib", type=float, default=100.0)

    run = subparsers.add_parser("run-first-pass", help="Run maps, dipoles, null tests, plots, and report.")
    run.add_argument("--mode", choices=["synthetic", "real"], help="Override analysis mode.")
    run.add_argument("--tracer", action="append", dest="tracers", help="Tracer to run.")
    run.add_argument("--region", action="append", dest="regions", help="Region to run, e.g. NGC or SGC.")
    run.add_argument("--random-index", action="append", type=int, dest="random_indices")
    run.add_argument("--nside", type=int, help="Override HEALPix nside for this run.")
    run.add_argument("--null-permutations", type=int, help="Override permutation null count.")
    run.add_argument("--poisson-mocks", type=int, help="Override Poisson mock null count.")
    run.add_argument("--block-null-mocks", type=int, help="Override block sign-flip null count.")
    run.add_argument("--block-null-regions", type=int, help="Override block sign-flip sky-region count.")
    run.add_argument("--bootstrap-samples", type=int, help="Override bootstrap sample count.")
    run.add_argument("--jackknife-regions", type=int, help="Override jackknife sky-region count.")
    run.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")

    combined = subparsers.add_parser(
        "run-combined-regions",
        help="Run maps and dipoles after summing count maps over multiple regions.",
    )
    combined.add_argument("--mode", choices=["synthetic", "real"], help="Override analysis mode.")
    combined.add_argument("--tracer", action="append", dest="tracers", help="Tracer to run.")
    combined.add_argument("--region", action="append", dest="regions", help="Region to combine.")
    combined.add_argument("--random-index", action="append", type=int, dest="random_indices")
    combined.add_argument("--nside", type=int, help="Override HEALPix nside for this run.")
    combined.add_argument("--null-permutations", type=int, help="Override permutation null count.")
    combined.add_argument("--poisson-mocks", type=int, help="Override Poisson mock null count.")
    combined.add_argument("--block-null-mocks", type=int, help="Override block sign-flip null count.")
    combined.add_argument("--block-null-regions", type=int, help="Override block sign-flip sky-region count.")
    combined.add_argument("--bootstrap-samples", type=int, help="Override bootstrap sample count.")
    combined.add_argument("--jackknife-regions", type=int, help="Override jackknife sky-region count.")
    combined.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")
    combined.add_argument("--output", default="combined_region_dipoles.csv")

    nside = subparsers.add_parser("robustness-nside", help="Run first-pass analysis over several nsides.")
    nside.add_argument("--mode", choices=["synthetic", "real"], help="Override analysis mode.")
    nside.add_argument("--tracer", action="append", dest="tracers", help="Tracer to run.")
    nside.add_argument("--region", action="append", dest="regions", help="Region to run, e.g. NGC or SGC.")
    nside.add_argument("--random-index", action="append", type=int, dest="random_indices")
    nside.add_argument("--null-permutations", type=int, help="Override permutation null count.")
    nside.add_argument("--poisson-mocks", type=int, help="Override Poisson mock null count.")
    nside.add_argument("--block-null-mocks", type=int, help="Override block sign-flip null count.")
    nside.add_argument("--block-null-regions", type=int, help="Override block sign-flip sky-region count.")
    nside.add_argument("--bootstrap-samples", type=int, help="Override bootstrap sample count.")
    nside.add_argument("--jackknife-regions", type=int, help="Override jackknife sky-region count.")
    nside.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")
    nside.add_argument(
        "--nside",
        action="append",
        type=int,
        dest="nsides",
        help="HEALPix nside value. Repeat for a grid. Defaults to 8, 16, 32.",
    )

    audit = subparsers.add_parser("systematics-audit", help="Run split tests over survey columns.")
    audit.add_argument("--mode", choices=["synthetic", "real"], default="real")
    audit.add_argument("--tracer", required=True, help="Tracer to audit, e.g. ELG.")
    audit.add_argument("--region", required=True, help="Region to audit, e.g. SGC.")
    audit.add_argument("--random-index", action="append", type=int, dest="random_indices")
    audit.add_argument("--nside", type=int, help="Override HEALPix nside.")
    audit.add_argument("--null-permutations", type=int, help="Override permutation null count.")
    audit.add_argument("--poisson-mocks", type=int, help="Override Poisson mock null count.")
    audit.add_argument("--block-null-mocks", type=int, help="Override block sign-flip null count.")
    audit.add_argument("--block-null-regions", type=int, help="Override block sign-flip sky-region count.")
    audit.add_argument("--bootstrap-samples", type=int, help="Override bootstrap sample count.")
    audit.add_argument("--jackknife-regions", type=int, help="Override jackknife sky-region count.")
    audit.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")
    audit.add_argument("--split-column", action="append", dest="split_columns")
    audit.add_argument("--ra-sectors", type=int, default=4)

    template = subparsers.add_parser(
        "template-systematics",
        help="Regress survey-template maps from overdensity maps and refit dipoles.",
    )
    template.add_argument("--mode", choices=["synthetic", "real"], default="real")
    template.add_argument("--tracer", required=True, help="Tracer to audit, e.g. ELG.")
    template.add_argument("--region", required=True, help="Region to audit, e.g. SGC.")
    template.add_argument("--random-index", action="append", type=int, dest="random_indices")
    template.add_argument("--nside", type=int, help="Override HEALPix nside.")
    template.add_argument("--null-permutations", type=int, help="Override permutation null count.")
    template.add_argument("--block-null-mocks", type=int, help="Override block sign-flip null count.")
    template.add_argument("--block-null-regions", type=int, help="Override block sign-flip sky-region count.")
    template.add_argument("--bootstrap-samples", type=int, help="Override bootstrap sample count.")
    template.add_argument("--jackknife-regions", type=int, help="Override jackknife sky-region count.")
    template.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")
    template.add_argument("--template-column", action="append", dest="template_columns")
    template.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )

    combined_template = subparsers.add_parser(
        "combined-template-systematics",
        help="Regress external templates after summing count maps over regions.",
    )
    combined_template.add_argument("--mode", choices=["synthetic", "real"], default="real")
    combined_template.add_argument("--tracer", action="append", dest="tracers")
    combined_template.add_argument("--region", action="append", dest="regions")
    combined_template.add_argument("--random-index", action="append", type=int, dest="random_indices")
    combined_template.add_argument("--nside", type=int, help="Override HEALPix nside.")
    combined_template.add_argument("--null-permutations", type=int)
    combined_template.add_argument("--block-null-mocks", type=int)
    combined_template.add_argument("--block-null-regions", type=int)
    combined_template.add_argument("--bootstrap-samples", type=int)
    combined_template.add_argument("--jackknife-regions", type=int)
    combined_template.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")
    combined_template.add_argument("--external-template", action="append", dest="external_templates")
    combined_template.add_argument("--output-prefix", default="combined_template_systematics")

    external = subparsers.add_parser(
        "build-external-templates",
        help="Convert DESI/Legacy Survey systematics FITS files to nside RING .npz maps.",
    )
    external.add_argument("--nside", type=int, help="Output HEALPix nside.")
    external.add_argument(
        "--pixweight",
        default="data/external/pixweight-1-dark.fits",
        help="DESI pixweight FITS file.",
    )
    external.add_argument(
        "--brick",
        action="append",
        dest="brick_paths",
        help="Legacy Survey brick summary FITS file. Repeat for north/south.",
    )
    external.add_argument(
        "--pixweight-column",
        action="append",
        dest="pixweight_columns",
        help="Pixweight column to convert. Defaults cover EBV, STARDENS, depth, and seeing.",
    )
    external.add_argument(
        "--brick-column",
        action="append",
        dest="brick_columns",
        help="Brick column to convert. Defaults cover COSKY sky brightness proxies.",
    )

    group_audit = subparsers.add_parser(
        "template-group-audit",
        help="Audit which external template groups suppress a saved sky map dipole.",
    )
    group_audit.add_argument("--map", required=True, help="Input .npz sky map path.")
    group_audit.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")
    group_audit.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        required=True,
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )

    injection = subparsers.add_parser(
        "injection-recovery",
        help="Inject known dipoles into a saved sky map and measure recovery after regression.",
    )
    injection.add_argument("--map", required=True, help="Input .npz sky map path.")
    injection.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")
    injection.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        required=True,
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )
    injection.add_argument(
        "--amplitude",
        action="append",
        type=float,
        dest="amplitudes",
        help="Injected dipole amplitude. Repeat for a grid.",
    )
    injection.add_argument(
        "--axis",
        action="append",
        dest="axes",
        help="Injected axis as name=ra,dec or ra,dec. Repeat for multiple axes.",
    )

    calibration = subparsers.add_parser(
        "mock-calibration",
        help="Calibrate dipole false positives and injection detection efficiency with mocks.",
    )
    calibration.add_argument("--map", required=True, help="Input .npz sky map path.")
    calibration.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")
    calibration.add_argument("--null-mocks", type=int, default=1000)
    calibration.add_argument("--injection-mocks", type=int, default=300)
    calibration.add_argument("--seed", type=int, default=20260525)
    calibration.add_argument("--detection-alpha", type=float, default=0.05)
    calibration.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        required=True,
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )
    calibration.add_argument(
        "--amplitude",
        action="append",
        type=float,
        dest="amplitudes",
        help="Injected dipole amplitude. Repeat for a grid.",
    )
    calibration.add_argument(
        "--axis",
        action="append",
        dest="axes",
        help="Injected axis as name=ra,dec or ra,dec. Repeat for multiple axes.",
    )

    lognormal = subparsers.add_parser(
        "lognormal-mock-calibration",
        help="Calibrate dipole amplitudes with mask-matched clustered lognormal mocks.",
    )
    lognormal.add_argument("--map", required=True, help="Input .npz sky map path.")
    lognormal.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")
    lognormal.add_argument("--mocks", type=int, default=1000)
    lognormal.add_argument("--seed", type=int, default=20260525)
    lognormal.add_argument("--smoothing-deg", type=float, default=8.0)
    lognormal.add_argument("--cl-slope", type=float, default=1.4)
    lognormal.add_argument("--lmax", type=int)
    lognormal.add_argument(
        "--sigma",
        action="append",
        dest="sigmas",
        help="Lognormal clustered-field RMS. Use 'auto' to estimate from the map.",
    )
    lognormal.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        default=[],
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )

    official = subparsers.add_parser(
        "prepare-official-mocks",
        help="Create a dry-run manifest for official DESI DR1 EZmock/AbacusSummit files.",
    )
    official.add_argument("--family", choices=["EZmock", "AbacusSummit"], default="EZmock")
    official.add_argument("--program", default="dark")
    official.add_argument("--flavor", help="Mock flavor, e.g. ffa or complete.")
    official.add_argument("--tracer", action="append", dest="tracers", default=[])
    official.add_argument("--region", action="append", dest="regions", default=[])
    official.add_argument("--realization", action="append", type=int, dest="realizations")
    official.add_argument("--random-index", action="append", type=int, dest="random_indices")
    official.add_argument(
        "--output",
        default="desi_dr1_official_mock_download_manifest.md",
        help="Manifest filename under data/raw.",
    )

    official_queue = subparsers.add_parser(
        "queue-official-mocks",
        help="Write a TSV download queue for official DESI DR1 mock files.",
    )
    official_queue.add_argument("--family", choices=["EZmock", "AbacusSummit"], default="EZmock")
    official_queue.add_argument("--program", default="dark")
    official_queue.add_argument("--flavor", help="Mock flavor, e.g. ffa or complete.")
    official_queue.add_argument("--tracer", action="append", dest="tracers", default=[])
    official_queue.add_argument("--region", action="append", dest="regions", default=[])
    official_queue.add_argument("--realization", action="append", type=int, dest="realizations")
    official_queue.add_argument("--random-index", action="append", type=int, dest="random_indices")
    official_queue.add_argument("--mocks-root", default="data/raw/mocks")
    official_queue.add_argument("--use-mirror", action="store_true")
    official_queue.add_argument(
        "--output",
        default="desi_dr1_official_mock_download_queue.tsv",
        help="Queue filename under data/raw.",
    )

    official_ensemble = subparsers.add_parser(
        "official-mock-ensemble",
        help="Run the dipole/template estimator on downloaded official DESI mock realizations.",
    )
    official_ensemble.add_argument("--family", choices=["EZmock", "AbacusSummit"], default="EZmock")
    official_ensemble.add_argument("--program", default="dark")
    official_ensemble.add_argument("--flavor", help="Mock flavor, e.g. ffa or complete.")
    official_ensemble.add_argument("--tracer", required=True)
    official_ensemble.add_argument("--region", action="append", dest="regions", required=True)
    official_ensemble.add_argument("--realization", action="append", type=int, dest="realizations")
    official_ensemble.add_argument(
        "--random-index",
        action="append",
        type=int,
        dest="random_indices",
    )
    official_ensemble.add_argument("--z-min", type=float, required=True)
    official_ensemble.add_argument("--z-max", type=float, required=True)
    official_ensemble.add_argument("--observed-map", required=True)
    official_ensemble.add_argument("--output-prefix", required=True)
    official_ensemble.add_argument(
        "--mocks-root",
        action="append",
        dest="mocks_roots",
        help=(
            "Root containing official mocks. Repeat to combine local repo data "
            "with an external SSD."
        ),
    )
    official_ensemble.add_argument("--nside", type=int)
    official_ensemble.add_argument(
        "--weight-mode",
        choices=["desi", "uniform", "no_fkp"],
        default="desi",
    )
    official_ensemble.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        default=[],
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )
    official_ensemble.add_argument(
        "--require-complete",
        action="store_true",
        help=(
            "Fail if any requested realization is missing instead of processing "
            "the available subset."
        ),
    )

    shell_mock = subparsers.add_parser(
        "shell-lognormal-calibration",
        help="Run tuned redshift-shell lognormal mocks from real DESI data/random catalogs.",
    )
    shell_mock.add_argument("--tracer", required=True)
    shell_mock.add_argument("--region", action="append", dest="regions", required=True)
    shell_mock.add_argument("--z-min", type=float, required=True)
    shell_mock.add_argument("--z-max", type=float, required=True)
    shell_mock.add_argument("--output-prefix", required=True)
    shell_mock.add_argument("--random-index", action="append", type=int, dest="random_indices")
    shell_mock.add_argument("--nside", type=int)
    shell_mock.add_argument("--weight-mode", choices=["desi", "uniform", "no_fkp"], default="desi")
    shell_mock.add_argument("--mocks", type=int, default=1000)
    shell_mock.add_argument("--shells", type=int, default=6)
    shell_mock.add_argument("--sigma", default="auto")
    shell_mock.add_argument("--radial-corr", type=float, default=0.05)
    shell_mock.add_argument("--smoothing-deg", type=float, default=8.0)
    shell_mock.add_argument("--cl-slope", type=float, default=1.4)
    shell_mock.add_argument("--lmax", type=int)
    shell_mock.add_argument("--seed", type=int, default=20260525)
    shell_mock.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        default=[],
        help="External per-pixel template as name=path. Repeat for multiple maps.",
    )

    subparsers.add_parser("plot", help="Regenerate plots from outputs/tables/first_pass_dipoles.csv.")
    subparsers.add_parser("report", help="Regenerate markdown report from results.")

    summarize = subparsers.add_parser(
        "summarize-look-elsewhere",
        help="Apply multiple-testing corrections to a first-pass result CSV.",
    )
    summarize.add_argument("--input", required=True, help="Input CSV with null_p_value results.")
    summarize.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix under outputs/tables and outputs/reports.",
    )
    summarize.add_argument(
        "--description",
        default="",
        help="Short markdown description included in the report.",
    )
    diagnostics = subparsers.add_parser(
        "axis-diagnostics",
        help="Write CMB/reference, region, and cross-tracer axis diagnostics for a result CSV.",
    )
    diagnostics.add_argument("--input", required=True, help="Input result CSV.")
    diagnostics.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")

    map_jackknife = subparsers.add_parser(
        "map-jackknife",
        help="Write leave-one-RA-region-out diagnostics for a saved sky map.",
    )
    map_jackknife.add_argument("--map", required=True, help="Input .npz sky map path.")
    map_jackknife.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")
    map_jackknife.add_argument("--regions", type=int, default=12)

    multipole = subparsers.add_parser(
        "multipole-diagnostics",
        help="Compare dipole-only and dipole+quadrupole fits for a saved sky map.",
    )
    multipole.add_argument("--map", required=True, help="Input .npz sky map path.")
    multipole.add_argument("--output-prefix", required=True, help="Output prefix under outputs.")

    master = subparsers.add_parser("master-report", help="Write a consolidated science report.")
    master.add_argument("--all-results", required=True, help="All-tracer result CSV.")
    master.add_argument("--combined-results", help="Combined-region result CSV.")
    master.add_argument("--nside-stability", help="Nside stability CSV.")
    master.add_argument("--weight-summary", help="Weight-mode summary CSV.")
    master.add_argument("--template-summary", help="Template-regression summary CSV.")
    master.add_argument("--multipole-summary", help="Multipole diagnostics summary CSV.")
    master.add_argument("--systematics", help="Systematics audit CSV.")
    master.add_argument("--focused-null", help="Focused null-diagnostic CSV.")
    master.add_argument("--jackknife-raw", help="Raw-map jackknife CSV.")
    master.add_argument("--jackknife-corrected", help="Template-corrected map jackknife CSV.")
    master.add_argument("--axis-reference", help="Reference-axis diagnostics CSV.")
    master.add_argument("--region-consistency", help="Region consistency diagnostics CSV.")
    master.add_argument("--tracer-consistency", help="Cross-tracer consistency diagnostics CSV.")
    master.add_argument("--output", default="outputs/reports/master_first_pass_science_report.md")

    inventory_fsf = subparsers.add_parser(
        "inventory-fastspecfit",
        help="Scan local FITS/Parquet metadata for DESI LSS and FastSpecFit/FastPhot candidates.",
    )
    inventory_fsf.add_argument(
        "--root",
        action="append",
        dest="roots",
        required=True,
        help="Root to scan. Repeat for repo data and external SSD roots.",
    )
    inventory_fsf.add_argument(
        "--output-prefix",
        default="outputs/tables/desi_local_fastspecfit_inventory",
        help="Output prefix; .csv and .md are written.",
    )
    inventory_fsf.add_argument("--max-files", type=int, help="Limit files inspected for a quick preflight.")

    gradient = subparsers.add_parser(
        "fastspecfit-gradient",
        help="Join one DESI LSS catalog to a FastSpecFit-like VAC and fit a population-residual dipole.",
    )
    gradient.add_argument("--lss", required=True, help="LSS clustering catalog FITS/Parquet path.")
    gradient.add_argument("--vac", required=True, help="FastSpecFit/FastPhot-like VAC FITS/Parquet path.")
    gradient.add_argument("--random", action="append", dest="random_paths", default=[])
    gradient.add_argument("--observable", default="DN4000", help="Population observable, e.g. DN4000.")
    gradient.add_argument("--z-min", type=float)
    gradient.add_argument("--z-max", type=float)
    gradient.add_argument("--nside", type=int, default=16)
    gradient.add_argument("--max-rows", type=int, help="Debug limit after LSS filtering.")
    gradient.add_argument(
        "--external-template",
        action="append",
        dest="external_templates",
        default=[],
        help="Per-pixel template as name=path. Must match nside.",
    )
    gradient.add_argument("--min-objects-per-pixel", type=float, default=5.0)
    gradient.add_argument(
        "--block-null-mocks",
        type=int,
        default=500,
        help="Spatial block-shuffle null realizations for population residual dipoles.",
    )
    gradient.add_argument(
        "--block-nside",
        type=int,
        default=2,
        help="Coarse HEALPix nside used for spatial block shuffling.",
    )
    gradient.add_argument("--seed", type=int, default=20260527)
    gradient.add_argument(
        "--output-prefix",
        default="outputs/tables/desi_fastspecfit_gradient_validation",
        help="Output prefix for CSV/MD and figure names.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    config = load_config(args.config)

    if args.command == "prepare":
        manifest = prepare_downloads(
            config,
            tracers=args.tracers,
            regions=args.regions,
            random_indices=args.random_indices,
        )
        LOGGER.info("Wrote dry-run manifest: %s", manifest)
        print(manifest)
        return 0
    if args.command == "check-storage":
        status = check_storage_root(Path(args.path), min_free_gib=args.min_free_gib)
        print(f"path={status.path}")
        print(f"exists={status.exists}")
        print(f"writable={status.writable}")
        print(f"total_gib={status.total_gib:.1f}")
        print(f"free_gib={status.free_gib:.1f}")
        print(f"message={status.message}")
        return 0 if status.writable else 1
    if args.command == "run-first-pass":
        results = run_first_pass(
            config,
            mode=args.mode,
            tracers=args.tracers,
            regions=args.regions,
            random_indices=args.random_indices,
            nside=args.nside,
            null_permutations=args.null_permutations,
            poisson_mocks=args.poisson_mocks,
            block_null_mocks=args.block_null_mocks,
            block_null_regions=args.block_null_regions,
            bootstrap_samples=args.bootstrap_samples,
            jackknife_regions=args.jackknife_regions,
            weight_mode=args.weight_mode,
        )
        print(config.paths.tables / "first_pass_dipoles.csv")
        print(f"rows={len(results)}")
        return 0
    if args.command == "run-combined-regions":
        results = run_combined_regions(
            config,
            mode=args.mode,
            tracers=args.tracers,
            regions=args.regions,
            random_indices=args.random_indices,
            nside=args.nside,
            null_permutations=args.null_permutations,
            poisson_mocks=args.poisson_mocks,
            block_null_mocks=args.block_null_mocks,
            block_null_regions=args.block_null_regions,
            bootstrap_samples=args.bootstrap_samples,
            jackknife_regions=args.jackknife_regions,
            weight_mode=args.weight_mode,
            results_filename=args.output,
        )
        print(config.paths.tables / args.output)
        print(f"rows={len(results)}")
        return 0
    if args.command == "robustness-nside":
        grid, stability = run_nside_robustness_grid(
            config,
            nsides=args.nsides or [8, 16, 32],
            mode=args.mode,
            tracers=args.tracers,
            regions=args.regions,
            random_indices=args.random_indices,
            null_permutations=args.null_permutations,
            poisson_mocks=args.poisson_mocks,
            block_null_mocks=args.block_null_mocks,
            block_null_regions=args.block_null_regions,
            bootstrap_samples=args.bootstrap_samples,
            jackknife_regions=args.jackknife_regions,
            weight_mode=args.weight_mode,
        )
        print(config.paths.tables / "nside_robustness_grid.csv")
        print(config.paths.tables / "nside_axis_stability.csv")
        print(f"grid_rows={len(grid)} stability_rows={len(stability)}")
        return 0
    if args.command == "systematics-audit":
        results = run_systematics_audit(
            config,
            tracer=args.tracer,
            region=args.region,
            mode=args.mode,
            random_indices=args.random_indices,
            nside=args.nside,
            null_permutations=args.null_permutations,
            poisson_mocks=args.poisson_mocks,
            block_null_mocks=args.block_null_mocks,
            block_null_regions=args.block_null_regions,
            bootstrap_samples=args.bootstrap_samples,
            jackknife_regions=args.jackknife_regions,
            weight_mode=args.weight_mode,
            split_columns=args.split_columns,
            ra_sectors=args.ra_sectors,
        )
        print(config.paths.tables / f"systematics_audit_{args.tracer}_{args.region}.csv")
        print(f"rows={len(results)}")
        return 0
    if args.command == "template-systematics":
        results = run_template_systematics(
            config,
            tracer=args.tracer,
            region=args.region,
            mode=args.mode,
            random_indices=args.random_indices,
            nside=args.nside,
            null_permutations=args.null_permutations,
            block_null_mocks=args.block_null_mocks,
            block_null_regions=args.block_null_regions,
            bootstrap_samples=args.bootstrap_samples,
            jackknife_regions=args.jackknife_regions,
            weight_mode=args.weight_mode,
            template_columns=args.template_columns,
            external_templates=args.external_templates,
        )
        print(config.paths.tables / f"template_systematics_{args.tracer}_{args.region}.csv")
        print(f"rows={len(results)}")
        return 0
    if args.command == "combined-template-systematics":
        results = run_combined_template_systematics(
            config,
            mode=args.mode,
            tracers=args.tracers,
            regions=args.regions,
            random_indices=args.random_indices,
            nside=args.nside,
            null_permutations=args.null_permutations,
            block_null_mocks=args.block_null_mocks,
            block_null_regions=args.block_null_regions,
            bootstrap_samples=args.bootstrap_samples,
            jackknife_regions=args.jackknife_regions,
            weight_mode=args.weight_mode,
            external_templates=args.external_templates,
            output_prefix=args.output_prefix,
        )
        safe_regions = "_".join(args.regions or config.analysis.regions)
        print(config.paths.tables / f"{args.output_prefix}_{safe_regions}.csv")
        print(f"rows={len(results)}")
        return 0
    if args.command == "build-external-templates":
        nside = args.nside or config.analysis.nside
        output_dir = config.paths.data_external / "templates"
        pixweight_columns = args.pixweight_columns or [
            "EBV",
            "STARDENS",
            "GALDEPTH_G",
            "GALDEPTH_R",
            "GALDEPTH_Z",
            "PSFSIZE_G",
            "PSFSIZE_R",
            "PSFSIZE_Z",
        ]
        brick_columns = args.brick_columns or ["cosky_g", "cosky_r", "cosky_z"]
        manifest_frames = []
        pixweight_path = config.base_dir / args.pixweight
        if pixweight_path.exists():
            manifest_frames.append(
                build_pixweight_templates(
                    pixweight_path=pixweight_path,
                    output_dir=output_dir,
                    nside_out=nside,
                    columns=pixweight_columns,
                )
            )
        default_bricks = [
            config.paths.data_external / "survey-bricks-dr9-north.fits.gz",
            config.paths.data_external / "survey-bricks-dr9-south.fits.gz",
        ]
        brick_paths = [config.base_dir / path for path in (args.brick_paths or [])] or [
            path for path in default_bricks if path.exists()
        ]
        if brick_paths:
            manifest_frames.append(
                build_brick_templates(
                    brick_paths=brick_paths,
                    output_dir=output_dir,
                    nside_out=nside,
                    columns=brick_columns,
                )
            )
        if not manifest_frames:
            raise FileNotFoundError("No external FITS inputs were found for template building.")
        import pandas as pd

        manifest = pd.concat(manifest_frames, ignore_index=True)
        manifest_path = config.paths.data_external / f"external_template_manifest_nside{nside}.csv"
        manifest.to_csv(manifest_path, index=False)
        print(manifest_path)
        print(f"rows={len(manifest)}")
        return 0
    if args.command == "template-group-audit":
        output_csv = config.paths.tables / f"{args.output_prefix}.csv"
        output_report = config.paths.reports / f"{args.output_prefix}.md"
        frame = write_template_group_audit(
            map_path=config.base_dir / args.map,
            external_templates=args.external_templates,
            output_csv=output_csv,
            output_report=output_report,
        )
        print(output_csv)
        print(output_report)
        print(f"rows={len(frame)}")
        return 0
    if args.command == "injection-recovery":
        output_csv = config.paths.tables / f"{args.output_prefix}.csv"
        output_report = config.paths.reports / f"{args.output_prefix}.md"
        frame = write_injection_recovery(
            map_path=config.base_dir / args.map,
            external_templates=args.external_templates,
            output_csv=output_csv,
            output_report=output_report,
            amplitudes=args.amplitudes or [0.01, 0.02, 0.03, 0.05],
            axes=parse_axis_specs(args.axes),
        )
        print(output_csv)
        print(output_report)
        print(f"rows={len(frame)}")
        return 0
    if args.command == "mock-calibration":
        outputs = run_mock_calibration(
            map_path=config.base_dir / args.map,
            external_templates=args.external_templates,
            output_prefix=config.paths.tables / args.output_prefix,
            amplitudes=args.amplitudes or [0.01, 0.02, 0.03, 0.05],
            axes=parse_axis_specs(args.axes),
            null_mocks=args.null_mocks,
            injection_mocks=args.injection_mocks,
            seed=args.seed,
            detection_alpha=args.detection_alpha,
        )
        print(outputs.null_csv)
        print(outputs.injections_csv)
        print(outputs.summary_csv)
        print(outputs.report)
        return 0
    if args.command == "lognormal-mock-calibration":
        outputs = run_lognormal_mock_calibration(
            map_path=config.base_dir / args.map,
            external_templates=args.external_templates,
            output_prefix=config.paths.tables / args.output_prefix,
            mocks=args.mocks,
            sigmas=args.sigmas or ["auto"],
            smoothing_deg=args.smoothing_deg,
            cl_slope=args.cl_slope,
            lmax=args.lmax,
            seed=args.seed,
        )
        print(outputs.null_csv)
        print(outputs.summary_csv)
        print(outputs.report)
        return 0
    if args.command == "prepare-official-mocks":
        manifest = write_official_mock_manifest(
            output_path=config.paths.data_raw / args.output,
            family=args.family,
            tracers=args.tracers or ["ELG"],
            regions=args.regions or ["NGC", "SGC"],
            realizations=args.realizations or ([1] if args.family == "EZmock" else [0]),
            random_indices=args.random_indices or [0],
            program=args.program,
            flavor=args.flavor,
        )
        print(manifest)
        return 0
    if args.command == "queue-official-mocks":
        queue = write_official_mock_download_queue(
            output_path=config.paths.data_raw / args.output,
            root=config.base_dir / args.mocks_root,
            family=args.family,
            tracers=args.tracers or ["ELG"],
            regions=args.regions or ["NGC", "SGC"],
            realizations=args.realizations or ([1] if args.family == "EZmock" else [0]),
            random_indices=args.random_indices or [0],
            program=args.program,
            flavor=args.flavor,
            use_mirror=args.use_mirror,
        )
        print(queue)
        return 0
    if args.command == "official-mock-ensemble":
        outputs = run_official_mock_ensemble(
            config=config,
            family=args.family,
            tracer=args.tracer,
            regions=args.regions,
            realizations=args.realizations or ([1] if args.family == "EZmock" else [0]),
            random_indices=args.random_indices or [0],
            z_min=args.z_min,
            z_max=args.z_max,
            observed_map_path=config.base_dir / args.observed_map,
            output_prefix=config.paths.tables / args.output_prefix,
            external_templates=args.external_templates,
            program=args.program,
            flavor=args.flavor,
            nside=args.nside,
            weight_mode=args.weight_mode,
            mocks_roots=[
                config.base_dir / root for root in (args.mocks_roots or ["data/raw/mocks"])
            ],
            allow_partial=not args.require_complete,
        )
        print(outputs.mocks_csv)
        print(outputs.summary_csv)
        print(outputs.report)
        if outputs.null_distribution_png is not None:
            print(outputs.null_distribution_png)
        return 0
    if args.command == "shell-lognormal-calibration":
        outputs = run_shell_lognormal_calibration(
            config=config,
            tracer=args.tracer,
            regions=args.regions,
            z_min=args.z_min,
            z_max=args.z_max,
            output_prefix=config.paths.tables / args.output_prefix,
            external_templates=args.external_templates,
            random_indices=args.random_indices,
            nside=args.nside,
            weight_mode=args.weight_mode,
            mocks=args.mocks,
            shells=args.shells,
            sigma=args.sigma,
            radial_corr=args.radial_corr,
            smoothing_deg=args.smoothing_deg,
            cl_slope=args.cl_slope,
            lmax=args.lmax,
            seed=args.seed,
        )
        print(outputs.null_csv)
        print(outputs.summary_csv)
        print(outputs.report)
        return 0
    if args.command == "inventory-fastspecfit":
        roots = [config.base_dir / Path(root) for root in args.roots]
        output_prefix = config.base_dir / Path(args.output_prefix)
        frame = inventory_local_tables(
            roots=roots,
            output_prefix=output_prefix,
            max_files=args.max_files,
        )
        population_count = int(
            (frame["population_columns"].fillna("").astype(str) != "").sum()
        )
        if population_count == 0:
            preflight_report = config.paths.reports / "desi_fastspecfit_gradient_validation.md"
            write_fastspecfit_preflight_report(
                preflight_report,
                inventory_frame=frame,
                roots=roots,
            )
            print(preflight_report)
        inventory_report = (
            output_prefix.parent.parent / "reports" / f"{output_prefix.name}.md"
            if output_prefix.parent.name == "tables"
            else output_prefix.with_suffix(".md")
        )
        print(output_prefix.with_suffix(".csv"))
        print(inventory_report)
        print(f"rows={len(frame)} population_candidates={population_count}")
        return 0
    if args.command == "fastspecfit-gradient":
        output_prefix = config.base_dir / Path(args.output_prefix)
        result = run_fastspecfit_gradient_validation(
            lss_path=config.base_dir / Path(args.lss),
            vac_path=config.base_dir / Path(args.vac),
            output_prefix=output_prefix,
            random_paths=[config.base_dir / Path(path) for path in args.random_paths],
            observable=args.observable,
            z_min=args.z_min,
            z_max=args.z_max,
            nside=args.nside,
            max_rows=args.max_rows,
            external_templates=args.external_templates,
            min_objects_per_pixel=args.min_objects_per_pixel,
            block_null_mocks=args.block_null_mocks,
            block_nside=args.block_nside,
            seed=args.seed,
        )
        print(result.table_path)
        print(result.report_path)
        for figure in result.figure_paths:
            print(figure)
        print(
            f"joined={result.n_joined} used={result.n_used} "
            f"amp={result.dipole.amplitude:.6g} "
            f"axis=({result.dipole.ra_deg:.3f}, {result.dipole.dec_deg:.3f})"
        )
        return 0
    if args.command == "plot":
        outputs = make_plots(config)
        for path in outputs:
            print(path)
        return 0
    if args.command == "report":
        print(write_report(config))
        return 0
    if args.command == "summarize-look-elsewhere":
        output_csv = config.paths.tables / f"{args.output_prefix}.csv"
        output_report = config.paths.reports / f"{args.output_prefix}.md"
        axis_output_csv = config.paths.tables / f"{args.output_prefix}_axis_separations.csv"
        corrected, separations = write_look_elsewhere_summary(
            results_path=config.base_dir / args.input,
            output_csv=output_csv,
            output_report=output_report,
            axis_output_csv=axis_output_csv,
            description=args.description,
        )
        print(output_csv)
        print(output_report)
        print(axis_output_csv)
        print(f"rows={len(corrected)} axis_rows={len(separations)}")
        return 0
    if args.command == "axis-diagnostics":
        paths = write_axis_diagnostics(
            results_path=config.base_dir / args.input,
            output_prefix=args.output_prefix,
            tables_dir=config.paths.tables,
            reports_dir=config.paths.reports,
        )
        for path in paths:
            print(path)
        return 0
    if args.command == "map-jackknife":
        sky_map = load_sky_map(str(config.base_dir / args.map))
        table = jackknife_region_table(sky_map, n_regions=args.regions)
        output_csv = config.paths.tables / f"{args.output_prefix}.csv"
        output_report = config.paths.reports / f"{args.output_prefix}.md"
        table.to_csv(output_csv, index=False)
        _write_map_jackknife_report(output_report, table, Path(args.map))
        print(output_csv)
        print(output_report)
        print(f"rows={len(table)}")
        return 0
    if args.command == "multipole-diagnostics":
        paths = write_map_multipole_diagnostics(
            map_path=config.base_dir / args.map,
            output_prefix=args.output_prefix,
            tables_dir=config.paths.tables,
            reports_dir=config.paths.reports,
        )
        for path in paths:
            print(path)
        return 0
    if args.command == "master-report":
        output = write_master_science_report(
            config=config,
            all_results_path=config.base_dir / args.all_results,
            output_path=config.base_dir / args.output,
            combined_results_path=(config.base_dir / args.combined_results if args.combined_results else None),
            nside_stability_path=(config.base_dir / args.nside_stability if args.nside_stability else None),
            weight_summary_path=(config.base_dir / args.weight_summary if args.weight_summary else None),
            template_summary_path=(config.base_dir / args.template_summary if args.template_summary else None),
            multipole_summary_path=(config.base_dir / args.multipole_summary if args.multipole_summary else None),
            systematics_path=(config.base_dir / args.systematics if args.systematics else None),
            focused_null_path=(config.base_dir / args.focused_null if args.focused_null else None),
            jackknife_raw_path=(config.base_dir / args.jackknife_raw if args.jackknife_raw else None),
            jackknife_corrected_path=(
                config.base_dir / args.jackknife_corrected if args.jackknife_corrected else None
            ),
            axis_reference_path=(config.base_dir / args.axis_reference if args.axis_reference else None),
            region_consistency_path=(config.base_dir / args.region_consistency if args.region_consistency else None),
            tracer_consistency_path=(config.base_dir / args.tracer_consistency if args.tracer_consistency else None),
        )
        print(output)
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2

def _write_map_jackknife_report(output_report: Path, table, map_path: Path) -> None:
    lines = [
        "# Map jackknife diagnostics",
        "",
        f"- Map: `{map_path}`",
        "",
        "Rows are leave-one-RA-region-out dipole fits sorted by axis shift. Large shifts",
        "identify sky regions that strongly influence the fitted axis.",
        "",
    ]
    if len(table):
        lines.append(table.head(12).to_markdown(index=False))
    else:
        lines.append("No jackknife rows were produced.")
    output_report.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
