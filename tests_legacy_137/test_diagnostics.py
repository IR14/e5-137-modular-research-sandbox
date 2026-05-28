import numpy as np

from cosmo_gradient.coords import angular_separation_deg, radec_to_unit
from cosmo_gradient.diagnostics import inject_dipole, parse_axis_specs
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.maps import SkyMap, pixel_vectors


def test_parse_axis_specs_accepts_named_and_unnamed_axes():
    axes = parse_axis_specs(["cmb=168.0,-7.0", "215.0,25.0"])

    assert axes == [("cmb", 168.0, -7.0), ("axis_2", 215.0, 25.0)]


def test_injected_dipole_is_recovered_on_zero_full_sky_map():
    nside = 4
    vectors = pixel_vectors(nside)
    npix = len(vectors)
    sky_map = SkyMap(
        nside=nside,
        backend="healpy",
        data_counts=np.ones(npix),
        random_counts=np.ones(npix),
        alpha=1.0,
        delta=np.zeros(npix),
        valid=np.ones(npix, dtype=bool),
        pixel_vectors=vectors,
    )
    axis = radec_to_unit([135.0], [25.0])[0]

    injected = inject_dipole(sky_map, amplitude=0.05, axis_vector=axis)
    fit = fit_dipole_map(injected)

    assert np.isclose(fit.amplitude, 0.05, rtol=0.05)
    assert float(angular_separation_deg(fit.vector, axis)) < 1.0
