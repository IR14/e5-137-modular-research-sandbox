from __future__ import annotations

import numpy as np

from cosmo_gradient.modular_quantum import (
    compile_vqp_kernel,
    float_grover_search,
    vqp_axis_count,
    vqp_cnot,
    vqp_grover_search,
    vqp_hadamard,
    vqp_init,
    vqp_measure,
    vqp_modulus,
    vqp_threshold,
)


def test_vqp_kernel_compiles_and_reports_constants():
    assert compile_vqp_kernel(force=True).exists()
    assert vqp_modulus() == 137
    assert vqp_axis_count() == 26
    assert vqp_threshold() == 42


def test_vqp_initial_state_is_compact_gf137_matrix():
    state = vqp_init(4, seed=137)

    assert state.shape == (4, 26)
    assert state.dtype == np.uint8
    assert int(state.max()) < 137


def test_vqp_hadamard_and_cnot_keep_state_in_field():
    state = vqp_init(3, seed=26)
    mixed = vqp_hadamard(state, 1)
    cnot_state, triggered = vqp_cnot(mixed, 0, 2)

    assert triggered in {0, 1}
    assert cnot_state.shape == (3, 26)
    assert int(cnot_state.max()) < 137


def test_vqp_grover_toy_search_recovers_target():
    measured, state = vqp_grover_search(qubits=5, target=17, iterations=3, seed=20260529)

    assert measured == 17
    assert vqp_measure(state) == 17


def test_float_grover_baseline_recovers_same_small_target():
    measured, state = float_grover_search(qubits=4, target=9, iterations=3)

    assert measured == 9
    assert state.shape == (16,)
