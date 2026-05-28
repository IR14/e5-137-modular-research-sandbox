# Super-Test Matrix Compression Report

## Status

The active pytest matrix was compressed from the previous 137 linear tests to
five orthogonal super-tests:

1. `test_field_and_lepton_core`
2. `test_hadron_phase_gate`
3. `test_cosmo_vacuum_limits`
4. `test_topological_fault_tolerance`
5. `test_dark_matter_gamma_phenomenology`

The previous suite was preserved as `tests_legacy_137/` and is no longer
collected by pytest.

## Test Log

```text
tests/test_super_matrix.py: 5
.....                                                                    [100%]
5 passed in 0.10s
```

## Residual Audit: Dark Matter Mass vs Context Compression

Inputs:

- sterile benchmark mass: `m_nu_tau_prime = 1.5566463427697075 GeV`
- compressed-context benchmark: `117` semantic vectors for `120,000` tokens
- payload density: `120000 / 117 = 1025.6410256410256 tokens/vector`

The exact compact integer identities exposed by the cleaned matrix are:

```text
117 = 9 * 13
117 = D*N - 13 = 26*5 - 13
```

These are real algebraic identities inside the project basis
`P={2,3,5,13,137}`, but they do not by themselves connect the sterile mass to
the context-compression benchmark.

Small-integer multiplier scan for `m_nu_tau_prime * k ~= 117`:

| k | m*k | residual ppm vs 117 |
|---:|---:|---:|
| 75 | 116.748475708 | 2149.780 |
| 76 | 118.305122050 | 11154.889 |
| 74 | 115.191829365 | 15454.450 |
| 77 | 119.861768393 | 24459.559 |
| 73 | 113.635183022 | 28759.119 |

## Interpretation

No sub-ppm or <5 ppm invariant was found linking the GeV sterile-neutrino mass
directly to the `117` context-compression count under the compact
integer-multiplier scan. The best simple relation,

```text
75 * m_nu_tau_prime ~= 117
```

misses by about `2150 ppm`, so it fails the strict invariant threshold. The
valid hidden structure after deduplication is narrower:

```text
117 = 9*13 = D*N - 13
```

This supports treating `117` as a compact benchmark count within the discrete
basis, not as a discovered physical bridge to the dark-matter mass.
