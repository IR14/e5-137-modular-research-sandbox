import numpy as np

from cosmo_gradient.systematics import combine_template_matrices, load_external_template_maps


def test_load_external_npz_template_standardizes_valid_pixels(tmp_path):
    path = tmp_path / "dust_template.npz"
    np.savez(path, template=np.array([1.0, 2.0, 3.0, 4.0, 5.0]))

    valid = np.array([True, True, True, True, False])
    matrix, names = load_external_template_maps(
        [f"dust={path}"],
        valid=valid,
        expected_npix=5,
    )

    assert names == ["external:dust"]
    assert matrix.shape == (5, 1)
    np.testing.assert_allclose(np.mean(matrix[valid, 0]), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.std(matrix[valid, 0]), 1.0, atol=1e-12)


def test_combine_template_matrices_preserves_empty_row_count():
    empty = np.empty((5, 0), dtype=float)
    combined, names = combine_template_matrices((empty, []))

    assert combined.shape == (5, 0)
    assert names == []
