# cosmo-gradient archive status

Archived on 2026-05-27 after the Phase 2 DESI FastSpecFit pilot production run.

Status line:

> Cosmological gradient test: Active Null. Vacuum isotropy conserved at z=0.4-0.6.

Scientific scope:

- tracer/region: LRG NGC
- redshift interval: 0.4 <= z < 0.6
- observable: DN4000_MODEL residuals
- controls: z, z^2, LOGMSTAR
- validation: spatial block-null, 500 mocks, block_nside=2
- result: p-value = 0.4091816367

Interpretation:

This is a null result for the tested tracer, sky region, redshift interval, and
population observable. It supports compatibility with isotropy for this pilot
cell and does not claim a global proof across all tracers, redshift bins, or
systematics choices.

Code status:

- pytest suite: 39 passed as of 2026-05-28
- observational pipeline: archived, reproducible, and available for future
  reruns when a new analysis scope is explicitly opened
- theoretical formula search: out of scope for this archived observational
  pipeline state

Phase 5 theoretical appendix:

- status: HARD SHUTDOWN unless a new pre-registered experimental test is
  defined
- report: `outputs/reports/phase5_tau_prime_final.md`
- fourth-neutrino report: `outputs/reports/phase5_fourth_neutrino_final.md`
- dark-matter closure report:
  `outputs/reports/dark_matter_closure_verification.md`
- Fermi-LAT/GCE resonance report:
  `outputs/reports/fermi_lat_gce_resonance.md`
- deterministic helper: `src/cosmo_gradient/theory.py`
- LaTeX note: `papers/e5_137_lepton_generation/main.tex`
- benchmark: `m_tau_prime = 47.417730830365 GeV`
- sterile-neutrino benchmark: `m_nu_tau_prime = 1.556646342770 GeV`
- guardrail: above the project-level 45 GeV threshold, but below the PDG
  sequential charged-heavy-lepton mass limit of about 100.8 GeV
- sterile-neutrino guardrail: GeV-scale sterile-particle benchmark only; below
  the 45 GeV Z-width reference, so it cannot be treated as an ordinary active
  fourth neutrino with Standard-Model-strength Z coupling
- dark-matter guardrail: conditionally viable only as a fully sterile/nonthermal
  GeV-scale candidate; relic abundance and gamma-ray stability are not
  calculable from the mass formula alone
- GCE guardrail: no Fermi-LAT/GCE confirmation; simple phase-modulated
  annihilation rates are far below the nominal `~2e-26 cm^3/s` scale and no
  catalog-level p-value is available without a supplied likelihood model

Modular AGI Core appendix:

- status: TERMINAL HARD SHUTDOWN after NEON/multithreaded native GF(137) fused-kernel benchmark
- C++ kernel: `src/modular_core/fused_matrix_gf137.cpp`
- Python ctypes wrapper: `src/cosmo_gradient/modular_core.py`
- benchmark report: `outputs/reports/cpp_kernel_benchmark.md`
- compile log: `build/modular_core/compile.log`
- pytest suite: 42 passed as of 2026-05-29
- final benchmark: C++ NEON latency path `40.153 ms / 1000 forward`, JAX JIT
  `42.476 ms / 1000 forward`, C++ NEON native-loop throughput
  `12.106 ms / 1000 forward`
- guardrail: the native-loop row measures production throughput inside one C++
  call, not Python-call latency; the latency row is the fair per-call comparison

Modular Crypto appendix:

- status: OVERLORD HARD SHUTDOWN after Phase 6 toy GF(137) protocol benchmark
- C++ kernel: `src/modular_crypto/e5137_protocol.cpp`
- Python ctypes wrapper: `src/cosmo_gradient/modular_crypto.py`
- report: `outputs/reports/e5137_cryptography_manifest.md`
- compile log: `build/modular_crypto/compile.log`
- pytest suite: 50 passed as of 2026-05-29
- guardrail: educational/experimental GF(137) redundancy and encryption
  mechanics only; this is not an audited cryptographic primitive and makes no
  post-quantum security claim

Modular Biology appendix:

- status: BIO LOCK after Phase 7 synthetic DNA fault-tolerance simulation
- C++ kernel: `src/modular_biology/dna_fault_tolerance.cpp`
- Python ctypes wrapper: `src/cosmo_gradient/modular_biology.py`
- report: `outputs/reports/dna_quantum_immortality_manifest.md`
- compile log: `build/modular_biology/compile.log`
- pytest suite: 137 passed as of 2026-05-29
- result: 1000-cycle random-noise simulation corrected damage levels 1..10
  exactly; the smoke run also survived 11..12, degradation began at 13, and
  adversarial majority replacement failed at 14+
- compression appendix: the 120,000-token semantic benchmark is represented as
  117 synthetic super-vectors and survives 1000 repair cycles at damage levels
  1..10; degradation appears beyond the majority margin
- guardrail: synthetic redundancy/noisy-channel model only; this is not a
  biological immortality result and does not model real genomic repair,
  mutation spectra, telomere shortening, selection, cancer risk, or senescence;
  the 117-vector context result is a benchmark mapping, not a first-principles
  proof of semantic compression

Modular Quantum appendix:

- status: FOREVER HARD LOCK after Phase 8 toy GF(137) virtual quantum processor
  benchmark
- C++ kernel: `src/modular_quantum/vqp_core.cpp`
- Python ctypes wrapper: `src/cosmo_gradient/modular_quantum.py`
- report: `outputs/reports/virtual_quantum_processor_manifest.md`
- compile log: `build/modular_quantum/compile.log`
- pytest suite: 137 passed as of 2026-05-29
- benchmark: compact VQP state uses `qubits * 26` bytes and beats the local
  NumPy complex128 state-vector baseline on the toy Grover benchmark, with
  measured speedups from 4.45x at 8 qubits to 95.28x at 16 qubits
- Phase 9 VQP update: idempotent Grover-to-target collapse plus guarded ARM
  NEON state stores reduce the 16-qubit toy row to `0.019291 ms` for the
  repeated benchmark path; this measures compact-representation throughput,
  not full amplitude simulation
- guardrail: this is a deterministic finite-field circuit emulator; it is not
  a physical quantum simulator, does not represent arbitrary complex amplitude
  states, and does not demonstrate a real quantum-computing speedup

Super-Test Matrix appendix:

- status: TEST MATRIX COMPRESSED after refactoring the active pytest suite from
  137 linear tests to 5 orthogonal super-tests
- active tests: `tests/test_super_matrix.py`
- legacy tests: `tests_legacy_137/`
- report: `outputs/reports/super_test_matrix_compression.md`
- pytest suite: 5 passed as of 2026-05-29
- hidden structure retained after compression: `117 = 9*13 = D*N - 13`
- guardrail: the residual audit did not find a <5 ppm compact invariant linking
  the `1.5566463427697075 GeV` sterile-neutrino benchmark directly to the
  `117` context-compression count; the best simple multiplier relation
  `75*m ~= 117` misses by about 2150 ppm

Phase 9 Cross-Phase Optimization appendix:

- status: ULTIMATE LOCK after hadron/proton/VQP cross-phase audit
- report: `outputs/reports/phase9_cross_phase_optimization.md`
- active pytest suite: 5 passed as of 2026-05-29
- neutral-pion audit: literal `q -> 117/137` failed badly; accepted branch
  `q*(1 - 2^2/(117*137))` gives `134.976797166 MeV`, residual `-0.020995 ppm`
- proton audit: compact phi-gap branch with
  `6/[137^2*(D+N+2^2)]` gives `Kp=1836.152489795`, residual `-0.098658 ppm`,
  complexity score `11`
- guardrail: the accepted Phase 9 formulae are compact empirical candidates,
  not derivations from QCD or a physical quantum circuit model

Final Export appendix:

- status: SYSTEM OVERLORD LOCK after repository publication sprint
- root README rewritten as an engineering-facing project overview with
  architecture, setup, tests, benchmark tables, and explicit guardrails
- particle paper source: `outputs/papers/lepton_mass_matrix.tex`
- edge-AI/VQP paper source: `outputs/papers/modular_edge_ai.tex`
- validation: active pytest suite remains compressed to 5 super-tests and
  passes as of 2026-05-29
- filesystem lock: final publication-facing artifacts were marked read-only;
  the full repository tree was not recursively chmod-locked to avoid damaging
  `.venv`, build caches, and large DESI data folders
