from __future__ import annotations

import math

import numpy as np

from cosmo_gradient.modular_biology import (
    axis_count as dna_axis_count,
    correction_budget,
    semantic_reference_vector_count,
    semantic_vector_count_for_tokens,
    simulate_mitosis,
    simulate_semantic_context,
)
from cosmo_gradient.modular_core import fused_matmul_mod137_threshold
from cosmo_gradient.modular_crypto import (
    axis_count as crypto_axis_count,
    generate_key,
    mod_inverse,
    recover_symbol,
    replicate_symbol,
)
from cosmo_gradient.modular_quantum import (
    vqp_axis_count,
    vqp_grover_search,
    vqp_measure,
    vqp_modulus,
)
from cosmo_gradient.theory import (
    CODATA_ALPHA_INV,
    D_STRING,
    ELECTRON_MASS_MEV,
    F26,
    N_TOPOLOGICAL,
    dark_matter_closure_check,
    delta_phi,
    fourth_neutrino_prediction,
    gce_resonance_audit,
    phase9_hadron_upgrade_audit,
    phase9_proton_upgrade_audit,
    schwinger_s,
    tau_prime_prediction,
    vacuum_compression_operator,
)


def test_field_and_lepton_core():
    n = N_TOPOLOGICAL
    d = D_STRING
    q = vacuum_compression_operator()
    s = schwinger_s()
    t5 = 2.0 / (math.e * n)
    c_e = 1.0 / (1.0 - s * (n + t5))
    electron_ev = 10.0 * math.e**11 * math.cos(2 / n) ** 2 * c_e
    muon_ratio = CODATA_ALPHA_INV * (1.5 + q)
    f_tau = 2 * (d - n) + 2 * math.exp(-2) + 2 * math.exp(-7)
    tau_ratio = CODATA_ALPHA_INV * (n**2 + q * f_tau)
    tau_prime = tau_prime_prediction()

    assert 137 % 2 == 1 and 137 % 3 == 2 and 137 % 5 == 2 and 137 % 13 == 7
    assert (42 * mod_inverse(42)) % 137 == 1
    assert np.isclose(q, 0.008860611874290873)
    assert np.isclose(t5, 0.14715177646857694)
    assert np.isclose(electron_ev, 510998.95069, rtol=2.0e-7)
    assert np.isclose(muon_ratio, 206.768221426689)
    assert np.isclose(tau_ratio, 3477.2282035579738)
    assert tau_prime.i5_topological == 42
    assert tau_prime.generation_coefficient == 130
    assert np.isclose(tau_prime.mass_gev, 47.417730830364825)


def test_hadron_phase_gate():
    n = N_TOPOLOGICAL
    q = vacuum_compression_operator()
    s = schwinger_s()
    delta = delta_phi()
    neutron_excess_ev = (
        (F26 / CODATA_ALPHA_INV)
        * 10.0
        * math.e**5
        * math.cos(1 / n)
        * (1.0 + (math.pi + delta) * s)
    )
    pi0_ratio = CODATA_ALPHA_INV * (2.0 - delta - q * (1.0 + 1.0 / n + 1.0 / math.pi))
    pi0_mass_mev = ELECTRON_MASS_MEV * pi0_ratio
    magnetic_anomaly = n * delta * (1.0 - n * s)
    pion_audit = phase9_hadron_upgrade_audit()
    proton_audit = phase9_proton_upgrade_audit()

    assert np.isclose(delta, 0.05900558383835652)
    assert np.isclose(neutron_excess_ev, 1_293_297.182830104)
    assert np.isclose(pi0_mass_mev, 134.97656207710023)
    assert np.isclose(magnetic_anomaly, 0.2933146620750762)
    assert 0.29 < magnetic_anomaly < 0.30
    assert not pion_audit.literal_replacement_improves
    assert pion_audit.modulated_q_improves
    assert abs(pion_audit.modulated_q_ppm) < 0.5
    assert proton_audit.phi_gap == 6
    assert proton_audit.complexity_score < 12
    assert proton_audit.passes_threshold


def test_cosmo_vacuum_limits():
    n = N_TOPOLOGICAL
    d = D_STRING
    q = vacuum_compression_operator()
    s = schwinger_s()
    delta = delta_phi()
    t5 = 2.0 / (math.e * n)
    e_lambda = n**3 - d / n + 5.0 * math.pi / 9.0
    w0 = -1.0 + t5 / n + d * q - delta / n
    wa = -n * delta * (math.pi + s - d * q + q / math.pi)
    h0_planck = 67.4
    h0_sh0es = 73.04
    h0_tension_fraction = (h0_sh0es - h0_planck) / h0_planck

    assert np.isclose(e_lambda, 121.54532925199433)
    assert np.isclose(w0, -0.7519948527423932)
    assert np.isclose(wa, -0.8600649695978589)
    assert 0.08 < h0_tension_fraction < 0.09
    assert abs(h0_tension_fraction - delta) > 0.02


def test_topological_fault_tolerance():
    key = generate_key(b"super-matrix")
    shares = replicate_symbol(136, key)
    corrupted = shares.copy()
    corrupted[:10] = (corrupted[:10] + np.arange(1, 11, dtype=np.uint8)) % 137
    recovered, votes = recover_symbol(corrupted, key)

    dna_result = simulate_mitosis("ATGCGATTACA", cycles=100, damage_axes=correction_budget(), seed=137)
    semantic_result = simulate_semantic_context(cycles=100, damage_axes=correction_budget(), seed=117)
    measured, vqp_state = vqp_grover_search(qubits=5, target=17, iterations=3, seed=20260529)
    x = np.array([[0, 1, 2], [2, 2, 1]], dtype=np.uint8)
    weights = np.array([[3, 80], [50, 2], [136, 12]], dtype=np.uint8)
    expected = (((x.astype(np.uint32) @ weights.astype(np.uint32)) % 137) >= 42).astype(
        np.uint8
    )

    assert crypto_axis_count() == dna_axis_count() == vqp_axis_count() == 26
    assert recovered == 136 and votes >= 16
    assert dna_result["final"] == dna_result["initial"]
    assert semantic_reference_vector_count() == semantic_vector_count_for_tokens(120000) == 117
    assert semantic_result["final"] == semantic_result["initial"]
    assert vqp_modulus() == 137 and measured == vqp_measure(vqp_state) == 17
    assert np.array_equal(fused_matmul_mod137_threshold(x, weights), expected)


def test_dark_matter_gamma_phenomenology():
    neutrino = fourth_neutrino_prediction()
    closure = dark_matter_closure_check()
    gce = gce_resonance_audit()
    tokens_per_vector = 120_000 / semantic_reference_vector_count()
    compact_ratio = semantic_reference_vector_count() / neutrino.mass_gev
    nearest_integer_relation = round(neutrino.mass_gev * 75.0)

    assert np.isclose(neutrino.mass_gev, 1.5566463427697075)
    assert closure.geV_scale_cold_candidate
    assert closure.conditionally_viable_if_fully_sterile
    assert not closure.relic_density_calculable_from_mass_only
    assert gce.in_gce_photon_energy_band
    assert not gce.in_common_gce_dm_mass_window
    assert gce.sigma_v_linear_cm3_s < 1.0e-28
    assert gce.sigma_v_quadratic_cm3_s < 1.0e-31
    assert np.isclose(tokens_per_vector, 1025.6410256410256)
    assert compact_ratio > 75.0
    assert nearest_integer_relation == 117
