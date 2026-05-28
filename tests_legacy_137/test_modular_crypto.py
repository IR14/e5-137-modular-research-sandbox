from __future__ import annotations

import time

import numpy as np
from cosmo_gradient.modular_crypto import (
    axis_count,
    compile_crypto_kernel,
    correctable_axes,
    decrypt,
    encrypt,
    generate_key,
    mod_inverse,
    recover_symbol,
    replicate_symbol,
)


def test_crypto_kernel_compiles_and_reports_topology():
    assert compile_crypto_kernel(force=True).exists()
    assert axis_count() == 26
    assert correctable_axes() == 10


def test_gf137_inverse_table_for_nonzero_residues():
    for value in [1, 2, 3, 5, 13, 42, 136]:
        inverse = mod_inverse(value)
        assert 1 <= inverse < 137
        assert (value * inverse) % 137 == 1


def test_key_generation_is_deterministic_and_gf137_bounded():
    key_a = generate_key(b"e5137-seed")
    key_b = generate_key(b"e5137-seed")

    assert np.array_equal(key_a, key_b)
    assert key_a.shape == (26,)
    assert np.all(key_a < 137)
    assert not np.array_equal(key_a, generate_key(b"other-seed"))


def test_symbol_replication_roundtrip_without_noise():
    key = generate_key(b"raid-symbol")

    shares = replicate_symbol(42, key)
    recovered, votes = recover_symbol(shares, key)

    assert recovered == 42
    assert votes == 26


def test_symbol_recovery_tolerates_ten_corrupted_axes():
    key = generate_key(b"noise-immune")
    shares = replicate_symbol(136, key)
    corrupted = shares.copy()
    corrupted[:10] = (corrupted[:10] + np.arange(1, 11, dtype=np.uint8)) % 137

    recovered, votes = recover_symbol(corrupted, key)

    assert recovered == 136
    assert votes >= 16


def test_encrypt_decrypt_roundtrip_for_all_byte_values():
    key = generate_key(b"byte-roundtrip")
    plaintext = bytes(range(256))

    cipher = encrypt(plaintext, key)
    decrypted = decrypt(cipher, key)

    assert decrypted == plaintext


def test_ciphertext_is_gf137_residue_stream_and_key_sensitive():
    plaintext = b"DESI null-result meets GF(137) toy protocol."
    key_a = generate_key(b"alice")
    key_b = generate_key(b"bob")

    cipher_a = encrypt(plaintext, key_a)
    cipher_b = encrypt(plaintext, key_b)

    assert cipher_a.dtype == np.uint8
    assert cipher_a.shape == (2 * len(plaintext),)
    assert np.all(cipher_a < 137)
    assert not np.array_equal(cipher_a, cipher_b)
    assert decrypt(cipher_a, key_a) == plaintext
    assert decrypt(cipher_a, key_b) != plaintext


def test_encryption_benchmark_smoke_run_is_positive():
    key = generate_key(b"speed-smoke")
    payload = bytes((index * 17 + 3) % 256 for index in range(4096))

    start = time.perf_counter()
    cipher = encrypt(payload, key)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    assert cipher.size == 2 * len(payload)
    assert elapsed_ms > 0.0
