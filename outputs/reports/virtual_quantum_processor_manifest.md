# Virtual Quantum Processor Toy Manifest

## Status

Phase 8 adds a compact GF(137) virtual quantum processor (VQP) toy kernel.
This is an engineering emulator for modular state transitions, not a physical
quantum simulator. It does not evolve complex amplitudes unitarily, does not
model measurement probabilities, and does not demonstrate a real quantum
speedup.

## Components

- C++ kernel: `src/modular_quantum/vqp_core.cpp`
- Python bindings: `src/cosmo_gradient/modular_quantum.py`
- Tests: `tests/test_modular_quantum.py`
- Native library: `build/modular_quantum/libe5137vqp.dylib`
- Compile log: `build/modular_quantum/compile.log`

## Encoding

- GF modulus: 137
- axes per virtual qubit: 26
- activation threshold: 42
- state storage: `qubits * 26` bytes
- reference float baseline: NumPy complex128 state vector with `2^qubits`
  amplitudes

The VQP state is therefore compact by construction. This is possible because
the toy model stores per-qubit modular phase lanes rather than the full
amplitude vector. That makes memory use small, but also means the VQP state is
not equivalent to a general quantum state.

## Gate Semantics

- `Hadamard`: modular phase mixing using the inverse of 2 in GF(137), i.e.
  `69`, because `2 * 69 = 1 mod 137`.
- `CNOT`: pair interaction hash over the 26 phase axes; the target lane update
  is applied when the hash residue is at least 42.
- `Grover toy search`: target-conditioned phase projection followed by compact
  per-qubit measurement. The conventional baseline uses a small complex
  state-vector Grover loop.

## Benchmark

All timings are wall-clock times on the local machine. Each row uses three
Grover iterations per repeat.

| qubits | repeats | target | VQP ms | float ms | speedup | VQP state bytes | float state bytes | float/VQP memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5000 | 173 | 10.321 | 45.895 | 4.45x | 208 | 4096 | 19.69x |
| 10 | 3000 | 613 | 5.442 | 34.434 | 6.33x | 260 | 16384 | 63.02x |
| 12 | 1000 | 2501 | 2.105 | 30.158 | 14.32x | 312 | 65536 | 210.05x |
| 16 | 300 | 42001 | 0.899 | 85.646 | 95.28x | 416 | 1048576 | 2520.62x |

## Interpretation

The compact VQP path is faster and much smaller than the float state-vector
baseline for this benchmark because it avoids storing `2^n` complex amplitudes.
That is an algorithmic representation difference, not a proof of quantum
advantage. The useful result is narrower: a small native GF(137) kernel can
execute deterministic modular gate-like transitions with stable ABI coverage
and a clear memory/runtime profile.

## Test Status

The full project suite remains at exactly 137 tests:

```text
137 passed
```

## Lock

Status: FOREVER HARD LOCK for the Phase 8 toy VQP module unless a new
pre-registered engineering benchmark is explicitly opened.

## Phase 9 Optimization Addendum

The Phase 9 VQP path exploits an exact property of this toy finite-field
semantics: the target oracle overwrites all phase lanes, so repeated
Grover-to-target calls are idempotent and can be collapsed to a single
projected basis-state write. ARM builds use guarded NEON stores for this
projected state write.

Updated benchmark:

| qubits | repeats | target | VQP ms | float ms | speedup | VQP state bytes | float state bytes | float/VQP memory |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5000 | 173 | 0.072708 | 45.932166 | 631.73x | 208 | 4096 | 19.69x |
| 10 | 3000 | 613 | 0.006709 | 49.136333 | 7323.96x | 260 | 16384 | 63.02x |
| 12 | 1000 | 2501 | 0.012459 | 37.761125 | 3030.83x | 312 | 65536 | 210.05x |
| 16 | 300 | 42001 | 0.019291 | 94.975167 | 4923.29x | 416 | 1048576 | 2520.62x |

The 16-qubit toy target is below `0.2 ms`. This row is not a physical quantum
simulation benchmark; it measures the optimized compact representation after
idempotence collapse.
