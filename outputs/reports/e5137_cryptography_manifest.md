# E-5-137 Modular Cryptography Manifest

## Status

This is an experimental toy protocol for deterministic GF(137) arithmetic,
redundant symbol replication, and benchmarkable native C++ bindings. It is not
a production cryptographic system, not a post-quantum security claim, and has
not been externally audited.

## Components

- C++ kernel: `src/modular_crypto/e5137_protocol.cpp`
- Python bindings: `src/cosmo_gradient/modular_crypto.py`
- Tests: `tests/test_modular_crypto.py`
- Native library: `build/modular_crypto/libe5137crypto.dylib`
- Compile log: `build/modular_crypto/compile.log`

## Topology

- GF modulus: 137
- Semantic axes: 26
- Designed random-axis correction budget: 10 of 26 axes
- RAID rule: each GF(137) symbol is masked across 26 axes; recovery performs
  inverse masking followed by majority vote.
- Byte encoding: each byte is represented as two GF(137) residues, so all
  values 0..255 are round-trippable.

## Benchmark

Payload: 1,000,000 bytes.

- Encrypt time: 78.034 ms
- Decrypt time: 76.811 ms
- Encrypt throughput: 12.815 MB/s
- Decrypt throughput: 13.019 MB/s
- Cipher residues: 2,000,000 GF(137) symbols
- Expansion factor: 2.0x as stored uint8 residues

## Error-Correction Smoke Test

Symbol `42` was replicated over 26 axes. After corrupting 10 axes, recovery
returned `42` with `16` agreeing axes. This demonstrates the
configured redundancy mechanism under random-axis corruption; it is not a proof
against adversarial manipulation.

## Security Guardrails

- The stream mask is deterministic and not a standard cipher construction.
- No IND-CPA/CCA proof is provided.
- No side-channel analysis is provided.
- The module is suitable for experiments, reproducibility tests, and teaching
  GF(137)/redundancy mechanics only.
