import numpy as np

from cosmo_gradient.coords import angular_separation_deg, radec_to_unit, unit_to_radec


def test_radec_to_unit_cardinal_axes():
    vectors = radec_to_unit([0.0, 90.0, 0.0], [0.0, 0.0, 90.0])

    np.testing.assert_allclose(vectors[0], [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(vectors[1], [0.0, 1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(vectors[2], [0.0, 0.0, 1.0], atol=1e-12)


def test_unit_to_radec_roundtrip():
    ra = np.array([12.0, 180.0, 359.0])
    dec = np.array([-30.0, 10.0, 70.0])
    vectors = radec_to_unit(ra, dec)
    out_ra, out_dec = unit_to_radec(vectors)

    np.testing.assert_allclose(out_ra, ra, atol=1e-10)
    np.testing.assert_allclose(out_dec, dec, atol=1e-10)


def test_angular_separation_is_ninety_degrees():
    a = radec_to_unit([0.0], [0.0])[0]
    b = radec_to_unit([90.0], [0.0])[0]

    assert np.isclose(angular_separation_deg(a, b), 90.0)
