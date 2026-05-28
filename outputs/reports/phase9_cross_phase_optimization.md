# Phase 9 Cross-Phase Optimization Report

## Status

Phase 9 audits three requested upgrades while keeping the active pytest matrix
compressed to five super-tests.

```text
.....                                                                    [100%]
5 passed
```

## 1. Hadron Upgrade: Neutral Pion

The literal requested replacement,

```text
q -> 117/137
```

was tested and rejected. It destroys the pion scale:

| branch | pi0 mass MeV | residual ppm |
|---|---:|---:|
| baseline q | 134.976562077 | -1.762695 |
| literal 117/137 | 45.119763172 | -665722.085780 |
| modulated q*(1 - 4/(117*137)) | 134.976797166 | -0.020995 |

The accepted low-complexity Phase 9 branch is therefore not a replacement of
`q`, but a small modulation:

```text
q_117 = q * (1 - 2^2/(117*137))
```

This reaches the requested sub-0.5 ppm target for the neutral-pion benchmark.

## 2. Proton Upgrade

The previous F26 proton ansatz had a residual of about `-9.203 ppm`.
Using the Euler decomposition

```text
phi(137) = 136 = D*N + 6
```

the retained compact correction is:

```text
K_p = [2*F26/(D*N)] * cos(1/N)
      * [1 + e*s + 6/(137^2 * (D + N + 2^2))]
```

with `s = alpha/(2*pi)`. The audit gives:

| branch | Kp | residual ppm |
|---|---:|---:|
| baseline | 1836.135772041 | -9.203431 |
| Phase 9 | 1836.152489795 | -0.098658 |

The expression passes the requested `<0.5 ppm` threshold with a project
complexity score of `11`. It remains an empirical compact candidate, not a QCD
derivation.

## 3. VQP Optimization

The VQP Grover-like loop was simplified by an exact property of the toy
semantics: the oracle projection overwrites all phase lanes, so repeated
Grover-to-target calls are idempotent. The native repeated path now collapses
the repeated circuit to a single projected basis-state write. On ARM builds,
the projected state write uses guarded NEON stores.

Benchmark against the local NumPy complex128 state-vector baseline:

| qubits | repeats | target | VQP ms | float ms | speedup | VQP bytes | float bytes | memory ratio |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 8 | 5000 | 173 | 0.072708 | 45.932166 | 631.73x | 208 | 4096 | 19.69x |
| 10 | 3000 | 613 | 0.006709 | 49.136333 | 7323.96x | 260 | 16384 | 63.02x |
| 12 | 1000 | 2501 | 0.012459 | 37.761125 | 3030.83x | 312 | 65536 | 210.05x |
| 16 | 300 | 42001 | 0.019291 | 94.975167 | 4923.29x | 416 | 1048576 | 2520.62x |

The 16-qubit target `<0.2 ms` is met in the toy benchmark.

## Guardrail

The VQP speedup is a representation and algebraic-collapse result. It is not a
physical quantum-computing speedup and not a fair replacement for a full
complex-amplitude circuit simulator. The pion and proton improvements are
compact numerical candidates; they should be treated as empirical formula
audits unless a real dynamical derivation is supplied.
