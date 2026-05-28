# DNA Fault-Tolerance Toy Simulation Manifest

## Status

This is a synthetic fault-tolerance simulation over nucleotide symbols encoded
as GF(137) residues. It is not a biological immortality result and does not
model real DNA repair biochemistry, mutation spectra, selection, telomere
shortening, cancer risk, or cell senescence.

## Components

- C++ kernel: `src/modular_biology/dna_fault_tolerance.cpp`
- Python bindings: `src/cosmo_gradient/modular_biology.py`
- Tests: `tests/test_modular_biology.py`
- Native library: `build/modular_biology/libe5137dna.dylib`
- Compile log: `build/modular_biology/compile.log`

## Encoding

- Alphabet: `A`, `T`, `G`, `C`
- GF modulus: 137
- Axes per nucleotide: 26
- Configured correction budget: 10 randomly corrupted axes
- Residues per nucleotide: 2, one primary residue and one hydrogen-bond proxy
- Hydrogen-bond proxy uses `delta_phi_code = round((cos(1/5)-cos(2/5))*1000) = 59`

## Random-Noise Mitosis Simulation

Sequence length: 44 bases. Cycles per row: 1000.

| damaged axes per base | failed cycles | final accuracy | final equals initial |
|---:|---:|---:|:---|
| 1 | 0 | 1.000000 | True |
| 2 | 0 | 1.000000 | True |
| 3 | 0 | 1.000000 | True |
| 4 | 0 | 1.000000 | True |
| 5 | 0 | 1.000000 | True |
| 6 | 0 | 1.000000 | True |
| 7 | 0 | 1.000000 | True |
| 8 | 0 | 1.000000 | True |
| 9 | 0 | 1.000000 | True |
| 10 | 0 | 1.000000 | True |
| 11 | 0 | 1.000000 | True |
| 12 | 0 | 1.000000 | True |
| 13 | 1 | 0.977273 | False |
| 14 | 35 | 0.340909 | False |
| 15 | 55 | 0.295455 | False |

## Adversarial Boundary Check

The redundancy guarantee is a majority-vote statement, not a biological law. If
a coherent adversary replaces enough axes with a consistent wrong code, recovery
fails once the wrong code has more votes than the original.

| adversarial axes | recovered original | recovered wrong base | majority votes |
|---:|:---|:---|---:|
| 11 | True | False | 15 |
| 12 | True | False | 14 |
| 13 | True | False | 13 |
| 14 | False | True | 14 |
| 15 | False | True | 15 |

## Interpretation

Within this toy channel model, random corruption of up to 10 axes is
corrected exactly in the 1000-cycle simulation. This should be read as an
engineering property of the synthetic redundancy code only. It does not imply
biological immortality or a physical limit for real genomes.

## Semantic Compression Appendix

This appendix maps a long-context compression benchmark onto the same synthetic
redundancy channel. The external reference point is the reported embedding-space
capacity result that optimized memory vectors can represent up to about 1500
tokens per vector, together with a popular-summary benchmark that a
120,000-token book can fall in a 100--200 vector range.

The code fixes the project benchmark at:

- reference tokens: 120,000
- semantic super-vectors: 117
- axes per semantic vector: 26
- cycles per row: 1000

Numerical note: the `117` value is treated here as a benchmark constant for the
compression simulation. The literal arithmetic `1/cos^2(2/5)` is about
`1.17875`, not `1.5278`; therefore this report does not claim that 117 has been
derived from the stated cosine factor. The test below only asks whether a
117-symbol compressed-context chain survives the same GF(137) repair channel.

| damaged axes per semantic vector | failed cycles | final accuracy | final equals initial |
|---:|---:|---:|:---|
| 1 | 0 | 1.000000 | True |
| 2 | 0 | 1.000000 | True |
| 3 | 0 | 1.000000 | True |
| 4 | 0 | 1.000000 | True |
| 5 | 0 | 1.000000 | True |
| 6 | 0 | 1.000000 | True |
| 7 | 0 | 1.000000 | True |
| 8 | 0 | 1.000000 | True |
| 9 | 0 | 1.000000 | True |
| 10 | 0 | 1.000000 | True |
| 11 | 0 | 1.000000 | True |
| 12 | 0 | 1.000000 | True |
| 13 | 8 | 0.931624 | False |
| 14 | 70 | 0.384615 | False |
| 15 | 162 | 0.230769 | False |

Interpretation: the 117-vector chain has the same expected behavior as the DNA
toy channel. Random corruption at the configured `1..10` axis budget is repaired
exactly in the 1000-cycle smoke simulation. Beyond the majority margin,
degradation appears quickly. This is a context-redundancy benchmark, not a
claim that present LLMs can recover arbitrary 120,000-token text from 117
vectors without an explicit trained decoder.
