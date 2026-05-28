import numpy as np

from cosmo_gradient.clustered_mocks import (
    estimate_excess_lognormal_sigma,
    lognormal_mock_map,
    run_lognormal_mock_calibration,
)
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, pixel_vectors, save_sky_map


def test_lognormal_mock_map_preserves_selection_shape():
    rng = np.random.default_rng(11)
    nside = 4
    vectors = pixel_vectors(nside)
    npix = len(vectors)
    random_counts = np.linspace(1000.0, 3000.0, npix)
    base = SkyMap(
        nside=nside,
        backend="equal_angle",
        data_counts=np.ones(npix),
        random_counts=random_counts,
        alpha=0.02,
        delta=np.zeros(npix),
        valid=np.ones(npix, dtype=bool),
        pixel_vectors=vectors,
    )

    mock = lognormal_mock_map(base, rng, sigma=0.05, smoothing_deg=20.0, lmax=8)
    fit = fit_dipole_map(mock)

    assert mock.data_counts.shape == random_counts.shape
    assert np.isfinite(mock.delta[mock.valid]).all()
    assert fit.amplitude >= 0.0


def test_lognormal_mock_calibration_writes_summary(tmp_path):
    nside = 4
    vectors = pixel_vectors(nside)
    npix = len(vectors)
    rng = np.random.default_rng(13)
    random_counts = np.full(npix, 1500.0)
    expected = 0.03 * random_counts
    data_counts = rng.poisson(expected).astype(float)
    delta = data_counts / expected - 1.0
    sky_map = SkyMap(
        nside=nside,
        backend="equal_angle",
        data_counts=data_counts,
        random_counts=random_counts,
        alpha=0.03,
        delta=delta,
        valid=np.ones(npix, dtype=bool),
        pixel_vectors=vectors,
    )
    map_path = tmp_path / "map.npz"
    save_sky_map(str(map_path), sky_map)

    outputs = run_lognormal_mock_calibration(
        map_path=map_path,
        external_templates=[],
        output_prefix=tmp_path / "lognormal_test",
        mocks=4,
        sigmas=[0.02],
        smoothing_deg=20.0,
        lmax=8,
        seed=3,
    )

    assert outputs.null_csv.exists()
    assert outputs.summary_csv.exists()
    assert outputs.report.exists()
    assert estimate_excess_lognormal_sigma(sky_map) >= 0.0
