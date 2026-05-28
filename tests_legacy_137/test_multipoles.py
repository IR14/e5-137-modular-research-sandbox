import numpy as np

from cosmo_gradient.coords import radec_to_unit
from cosmo_gradient.maps import SkyMap
from cosmo_gradient.multipoles import map_multipole_diagnostics


def test_quadrupole_marginalization_recovers_dipole_component():
    rng = np.random.default_rng(123)
    vectors = radec_to_unit(rng.uniform(0.0, 360.0, 800), rng.uniform(-70.0, 70.0, 800))
    axis = radec_to_unit([40.0], [10.0])[0]
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]
    delta = 0.04 * (vectors @ axis) + 0.2 * (x * x - 1.0 / 3.0) - 0.1 * y * z
    sky_map = SkyMap(
        nside=8,
        backend="test",
        data_counts=np.ones(len(delta)),
        random_counts=np.ones(len(delta)),
        alpha=1.0,
        delta=delta,
        valid=np.ones(len(delta), dtype=bool),
        pixel_vectors=vectors,
    )

    summary, _ = map_multipole_diagnostics(sky_map)
    marginal = summary.loc[summary["model"] == "dipole_plus_quadrupole"].iloc[0]

    assert np.isclose(marginal["dipole_amplitude"], 0.04, atol=1e-6)
    assert marginal["amplitude_ratio_to_dipole_only"] < 1.0
