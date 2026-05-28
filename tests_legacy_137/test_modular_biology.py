from __future__ import annotations

import numpy as np
import pytest

from cosmo_gradient.modular_biology import (
    axis_count,
    base_code,
    compile_biology_kernel,
    correction_budget,
    corrupt_random_axes,
    decode_dna,
    encode_dna,
    repair_dna,
    replicate_dna,
    semantic_context_sequence,
    semantic_reference_token_count,
    semantic_reference_vector_count,
    semantic_vector_count_for_tokens,
    simulate_mitosis,
    simulate_semantic_context,
)


SEQUENCES = [
    "A",
    "T",
    "G",
    "C",
    "ATGC",
    "GATTACA",
    "CCGGTTAA",
    "ATATGCGC",
    "GGGGCCCC",
    "TACGATCGATCG",
]


def test_biology_kernel_compiles_and_reports_config():
    assert compile_biology_kernel(force=True).exists()
    assert axis_count() == 26
    assert correction_budget() == 10
    assert semantic_reference_token_count() == 120000
    assert semantic_reference_vector_count() == 117
    assert semantic_vector_count_for_tokens(120000) == 117
    assert semantic_vector_count_for_tokens(1) == 1
    assert len(semantic_context_sequence()) == 117


@pytest.mark.parametrize("base", ["A", "T", "G", "C"])
def test_base_encoder_roundtrip(base: str):
    assert decode_dna(encode_dna(base)) == base
    assert base_code(base) in {0, 1, 2, 3}


@pytest.mark.parametrize("sequence", SEQUENCES)
def test_dna_encode_decode_roundtrip(sequence: str):
    assert decode_dna(encode_dna(sequence)) == sequence


@pytest.mark.parametrize("symbol", range(20))
def test_replicated_repair_returns_original_sequence_for_repeated_symbols(symbol: int):
    base = "ATGC"[symbol % 4]
    sequence = base * (1 + symbol % 5)

    repaired, votes, min_votes = repair_dna(replicate_dna(sequence))

    assert repaired == sequence
    assert np.all(votes == axis_count())
    assert min_votes == axis_count()


@pytest.mark.parametrize("damage_axes", range(0, 11))
def test_repair_tolerates_random_damage_up_to_configured_budget(damage_axes: int):
    sequence = "ATGCGATTACAGGCTA"
    shares = replicate_dna(sequence)
    corrupted = corrupt_random_axes(shares, damage_axes, seed=20260529 + damage_axes)

    repaired, _votes, min_votes = repair_dna(corrupted)

    assert repaired == sequence
    assert min_votes >= axis_count() - damage_axes


@pytest.mark.parametrize("damage_axes", range(11, 16))
def test_adversarial_damage_boundary_above_budget(damage_axes: int):
    sequence = "A" * 8
    wrong = "T" * 8
    shares = replicate_dna(sequence)
    wrong_shares = replicate_dna(wrong)
    adversarial = shares.copy()
    adversarial[:, :damage_axes] = wrong_shares[:, :damage_axes]

    repaired, _votes, min_votes = repair_dna(adversarial)

    if damage_axes <= 13:
        assert repaired == sequence
        assert min_votes == axis_count() - damage_axes
    else:
        assert repaired == wrong
        assert min_votes == damage_axes


@pytest.mark.parametrize("damage_axes", range(1, 16))
def test_mitosis_simulation_returns_valid_sequence_for_noise_levels(damage_axes: int):
    sequence = "ATGCGATTACAGGCTA"

    result = simulate_mitosis(sequence, cycles=1000, damage_axes=damage_axes, seed=137 + damage_axes)

    assert len(result["final"]) == len(sequence)
    assert set(str(result["final"])).issubset({"A", "T", "G", "C"})
    assert result["cycles"] == 1000
    assert result["damage_axes"] == damage_axes
    if damage_axes <= correction_budget():
        assert result["final"] == sequence
        assert result["failed_cycles"] == 0
        assert result["accuracy"] == 1.0
        semantic = simulate_semantic_context(cycles=1000, damage_axes=damage_axes, seed=4200 + damage_axes)
        assert semantic["vectors"] == 117
        assert semantic["tokens"] == 120000
        assert semantic["final"] == semantic["initial"]
        assert semantic["failed_cycles"] == 0
        assert semantic["accuracy"] == 1.0


@pytest.mark.parametrize("position", range(15))
def test_encoded_residues_are_gf137_bounded_by_position(position: int):
    sequence = ("ATGC" * 5)[:20]
    encoded = encode_dna(sequence)
    pair = encoded[2 * position : 2 * position + 2]

    assert pair.shape == (2,)
    assert np.all(pair < 137)


def test_invalid_base_is_rejected():
    with pytest.raises(ValueError):
        encode_dna("ATXG")
