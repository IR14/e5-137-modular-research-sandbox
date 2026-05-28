from cosmo_gradient.official_mocks import (
    _resolve_official_mock_local_path,
    build_official_mock_entries,
    official_mock_local_path,
    write_official_mock_download_queue,
)


def test_build_ezmock_manifest_entries_use_public_names():
    entries = build_official_mock_entries(
        family="EZmock",
        tracers=["ELG"],
        regions=["NGC"],
        realizations=[1],
        random_indices=[0],
    )

    filenames = {entry.filename for entry in entries}
    assert "ELG_LOP_ffa_NGC_clustering.dat.fits" in filenames
    assert "ELG_LOP_ffa_NGC_0_clustering.ran.fits" in filenames
    assert all("mocks/EZmock/dark/v1/mock1" in entry.url for entry in entries)


def test_build_abacus_manifest_entries_default_to_complete():
    entries = build_official_mock_entries(
        family="AbacusSummit",
        tracers=["QSO"],
        regions=["SGC"],
        realizations=[0],
        random_indices=[0],
    )

    filenames = {entry.filename for entry in entries}
    assert "QSO_complete_SGC_clustering.dat.fits" in filenames
    assert "QSO_complete_SGC_0_clustering.ran.fits" in filenames
    assert all("mocks/AbacusSummit/dark/v4.2/mock0" in entry.url for entry in entries)


def test_official_mock_local_path_preserves_mock_tree(tmp_path):
    entry = build_official_mock_entries(
        family="EZmock",
        tracers=["ELG"],
        regions=["SGC"],
        realizations=[12],
        random_indices=[0],
    )[0]

    path = official_mock_local_path(tmp_path, entry)

    assert path == tmp_path / "EZmock" / "dark" / "v1" / "mock12" / entry.filename


def test_write_official_mock_download_queue(tmp_path):
    queue = write_official_mock_download_queue(
        output_path=tmp_path / "queue.tsv",
        root=tmp_path / "mocks",
        family="EZmock",
        tracers=["ELG"],
        regions=["NGC", "SGC"],
        realizations=[1, 2],
        random_indices=[0],
    )

    lines = queue.read_text(encoding="utf-8").strip().splitlines()

    assert len(lines) == 8
    assert lines[0].startswith("https://data.desi.lbl.gov/public/dr1/survey/catalogs/dr1/mocks/")
    assert "\t" in lines[0]
    assert "EZmock/dark/v1/mock1/ELG_LOP_ffa_NGC_clustering.dat.fits" in lines[0]


def test_resolve_official_mock_local_path_searches_roots(tmp_path):
    entry = build_official_mock_entries(
        family="EZmock",
        tracers=["ELG"],
        regions=["NGC"],
        realizations=[101],
        random_indices=[0],
    )[0]
    first = tmp_path / "repo" / "mocks"
    second = tmp_path / "external" / "mocks"
    expected = official_mock_local_path(second, entry)
    expected.parent.mkdir(parents=True)
    expected.write_text("placeholder", encoding="utf-8")

    resolved = _resolve_official_mock_local_path([first, second], entry)

    assert resolved == expected
