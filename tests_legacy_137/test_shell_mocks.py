import numpy as np

from cosmo_gradient.shell_mocks import (
    ShellCube,
    cube_to_sky_map,
    estimate_shell_sigma,
    shell_lognormal_mock_counts,
)


def test_shell_lognormal_mock_counts_preserve_cube_shape():
    rng = np.random.default_rng(5)
    cube = _small_cube()

    counts = shell_lognormal_mock_counts(
        cube,
        rng=rng,
        sigma=0.05,
        radial_corr=0.1,
        smoothing_deg=30.0,
        lmax=8,
    )
    sky_map = cube_to_sky_map(cube, counts)

    assert counts.shape == cube.random_counts.shape
    assert np.isfinite(sky_map.delta[sky_map.valid]).all()
    assert estimate_shell_sigma(cube) >= 0.0


def _small_cube() -> ShellCube:
    nside = 4
    npix = 12 * nside * nside
    random_counts = np.vstack(
        [
            np.full(npix, 800.0),
            np.full(npix, 1000.0),
            np.full(npix, 900.0),
        ]
    )
    shell_alpha = np.array([0.02, 0.025, 0.018])
    data_counts = shell_alpha[:, None] * random_counts
    return ShellCube(
        tracer="ELG",
        regions=("NGC", "SGC"),
        z_min=0.8,
        z_max=1.1,
        z_edges=np.array([0.8, 0.9, 1.0, 1.1]),
        nside=nside,
        backend="healpy",
        data_counts=data_counts,
        random_counts=random_counts,
        shell_alpha=shell_alpha,
        data_rows=np.array([100, 120, 90]),
        random_rows=np.array([5000, 6000, 5500]),
    )
