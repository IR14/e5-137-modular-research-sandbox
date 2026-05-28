import numpy as np
import pandas as pd

from cosmo_gradient.coords import angular_separation_deg, radec_to_unit
from cosmo_gradient.dipole import fit_dipole_map
from cosmo_gradient.io.desi import _sample_dipole_modulated_sky, _sample_masked_sky
from cosmo_gradient.maps import build_overdensity_map


def test_weighted_overdensity_is_zero_for_scaled_randoms():
    data = pd.DataFrame(
        {
            "ra": [10.0, 20.0, 210.0, 220.0],
            "dec": [5.0, 6.0, -5.0, -6.0],
            "z": [0.5, 0.5, 0.5, 0.5],
            "weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    randoms = pd.concat([data] * 10, ignore_index=True)
    sky_map = build_overdensity_map(data, randoms, nside=4, min_random_per_pixel=0.1)

    valid_with_counts = sky_map.valid & (sky_map.data_counts > 0)
    np.testing.assert_allclose(sky_map.delta[valid_with_counts], 0.0, atol=1e-12)


def test_recovers_known_synthetic_dipole_axis():
    rng = np.random.default_rng(42)
    axis = radec_to_unit([135.0], [25.0])[0]
    data = _sample_dipole_modulated_sky(
        n_rows=30000,
        rng=rng,
        axis=axis,
        amplitude=0.25,
        apply_mask=False,
    )
    randoms = _sample_masked_sky(180000, rng=rng, apply_mask=False)
    data["z"] = 0.8
    randoms["z"] = 0.8
    data["weight"] = 1.0
    randoms["weight"] = 1.0

    sky_map = build_overdensity_map(data, randoms, nside=8, min_random_per_pixel=10.0)
    fit = fit_dipole_map(sky_map)
    separation = angular_separation_deg(fit.vector, axis)

    assert fit.amplitude > 0.12
    assert separation < 15.0
