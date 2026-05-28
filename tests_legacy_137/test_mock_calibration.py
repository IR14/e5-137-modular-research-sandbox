import numpy as np

from cosmo_gradient.coords import radec_to_unit
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, pixel_vectors
from cosmo_gradient.mock_calibration import poisson_mock_map


def test_poisson_mock_map_can_inject_positive_dipole_trend():
    rng = np.random.default_rng(7)
    nside = 4
    vectors = pixel_vectors(nside)
    npix = len(vectors)
    base = SkyMap(
        nside=nside,
        backend="healpy",
        data_counts=np.ones(npix),
        random_counts=np.full(npix, 5000.0),
        alpha=1.0,
        delta=np.zeros(npix),
        valid=np.ones(npix, dtype=bool),
        pixel_vectors=vectors,
    )
    axis = radec_to_unit([130.0], [20.0])[0]

    mock = poisson_mock_map(base, rng, amplitude=0.08, axis_vector=axis)
    fit = fit_dipole_map(mock)

    assert np.dot(fit.vector, axis) > 0.04
