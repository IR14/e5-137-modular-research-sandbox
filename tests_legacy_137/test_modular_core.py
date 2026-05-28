from __future__ import annotations

import numpy as np

from cosmo_gradient.modular_core import (
    compile_cpp_kernel,
    fused_matmul_mod137_threshold,
    fused_two_layer_mod137_predict,
    gf137_thread_count,
)


def test_cpp_fused_matmul_kernel_matches_numpy_reference():
    compile_cpp_kernel(force=True)
    x = np.array([[0, 1, 2], [2, 2, 1], [1, 0, 2]], dtype=np.uint8)
    weights = np.array([[3, 80], [50, 2], [136, 12]], dtype=np.uint8)

    expected = (((x.astype(np.uint32) @ weights.astype(np.uint32)) % 137) >= 42).astype(
        np.uint8
    )

    assert np.array_equal(fused_matmul_mod137_threshold(x, weights), expected)


def test_cpp_two_layer_kernel_matches_numpy_reference():
    x = np.array([[0, 1, 2, 1], [2, 2, 1, 0], [1, 0, 2, 2]], dtype=np.uint8)
    w1 = np.array(
        [
            [3, 80, 1],
            [50, 2, 7],
            [136, 12, 3],
            [1, 20, 100],
        ],
        dtype=np.uint8,
    )
    b1 = np.array([5, 9, 11], dtype=np.uint8)
    w2 = np.array([13, 127, 22], dtype=np.uint8)
    b2 = np.array([6], dtype=np.uint8)
    output_threshold = 57

    hidden_residue = (x.astype(np.uint32) @ w1.astype(np.uint32) + b1.astype(np.uint32)) % 137
    hidden = (hidden_residue >= 42).astype(np.uint32)
    output_residue = (hidden @ w2.astype(np.uint32) + int(b2[0])) % 137
    expected = (output_residue >= output_threshold).astype(np.uint8)

    got = fused_two_layer_mod137_predict(
        x,
        w1,
        b1,
        w2,
        b2,
        hidden_threshold=42,
        output_threshold=output_threshold,
        greater_is_prime=True,
    )

    assert np.array_equal(got, expected)


def test_cpp_thread_pool_reports_positive_worker_count():
    assert gf137_thread_count() >= 1
