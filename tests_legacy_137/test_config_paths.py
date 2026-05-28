from pathlib import Path

from cosmo_gradient.config import PathConfig
from cosmo_gradient.storage import check_storage_root


def test_path_config_resolves_additional_raw_roots(tmp_path):
    config = PathConfig(
        data_raw=Path("data/raw"),
        additional_raw_roots=[Path("external/raw"), tmp_path / "absolute_raw"],
    ).resolve_from(tmp_path)

    assert config.data_raw == tmp_path / "data/raw"
    assert config.additional_raw_roots == [
        tmp_path / "external/raw",
        tmp_path / "absolute_raw",
    ]
    assert config.raw_search_roots() == [
        tmp_path / "data/raw",
        tmp_path / "external/raw",
        tmp_path / "absolute_raw",
    ]


def test_check_storage_root_reports_writable_path(tmp_path):
    status = check_storage_root(tmp_path, min_free_gib=0.0)

    assert status.exists
    assert status.writable
    assert status.free_bytes > 0
