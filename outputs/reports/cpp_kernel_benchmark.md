# Modular AGI Core Benchmark

## Scope

This is a small simulator comparing a conventional float32 MLP baseline with a
network whose weights and matrix products live in `Z/137Z`. It is not an AGI
demonstration. The classification task is intentionally difficult: previous
base-e digit audits did not find a robust prime/composite separator.

## Dataset

Balanced dataset: 500 primes and 500 composites; features are first 50 fractional base-e digits; split is 70/10/20 train/validation/test.

Sampling range: [1000, 5000], seed: 20260528.

## Modular Layer

```python
class ModularLayer(NN.Module):
    def forward_residue(self, x):
        residue = x @ W + b
        return residue % 137

    def forward(self, x):
        return (forward_residue(x) >= 42)
```

The optimized NumPy path uses the same residues but computes the small integer
matrix products through exact float32 BLAS and a fast floor reduction:

```python
y = x_float32 @ W_float32 + b_float32
residue = y - floor(y / 137) * 137
```

The sums in this benchmark are below `2^24`, so float32 represents the integer
intermediates exactly.

The C++ path stores only the compact `uint8` residues and fuses matrix
multiplication, Barrett reduction modulo 137, and threshold activation in one
native loop. The current C++ backend uses a persistent standard-library worker
pool to split rows across CPU threads without paying thread-spawn cost on every
forward pass. On ARM64, the `H=32` benchmark shape uses a specialized NEON path
for hidden accumulation, modulo-thresholding, and active output-weight summation.

The `cpp_neon_uint8_native_loop` row is a production-throughput check: it runs
all timed iterations inside one native C++ call and therefore removes the
per-call Python/ctypes boundary from the measurement.

Backend availability in this run:

- `numba_available = True`
- `mlx_available = True`
- `torch_available = True`
- `jax_available = True`
- `cpp_kernel_available = True`
- `cpp_compile_log = /Users/i.mikhailov/Desktop/work/cosmo/cosmo_genesis_gradient/build/modular_core/compile.log`

The float32 baseline remains a NumPy MLP so that the baseline does not depend
on optional runtime installation.

## Metrics

| model                                            |   memory_kb |   forward_1000_ms |   train_accuracy |   validation_accuracy |   test_accuracy | notes                                                                                                                                                                                        |
|:-------------------------------------------------|------------:|------------------:|-----------------:|----------------------:|----------------:|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| float32_mlp_torch_available_but_numpy_used       |     6.50391 |           58.7784 |         0.605714 |                  0.45 |           0.405 | Torch is available, but this script keeps both models in NumPy for comparable timing.                                                                                                        |
| gf137_modular_network_compact_uint8              |     1.62793 |          582.904  |         0.472857 |                  0.53 |           0.505 | GF(137) weights stored as uint8; activation is residue >= I5=42; readout threshold=49, greater_is_prime=False.                                                                               |
| gf137_modular_network_fast_exact                 |     8.13184 |           55.5445 |         0.472857 |                  0.53 |           0.505 | Exact GF(137) residues via float32 BLAS plus floor modulo; active memory includes uint8 storage and float32 compute cache.                                                                   |
| gf137_modular_network_cpp_fused_uint8            |     1.62793 |           40.153  |         0.472857 |                  0.53 |           0.505 | C++ ctypes fused kernel; shape-specialized ARM NEON fast path for H=32; threads=4. Compile log: /Users/i.mikhailov/Desktop/work/cosmo/cosmo_genesis_gradient/build/modular_core/compile.log. |
| gf137_modular_network_cpp_neon_uint8_native_loop |     1.62793 |           12.1055 |         0.472857 |                  0.53 |           0.505 | C++ NEON throughput path; one native call runs all 1000 iterations; threads=4. This measures production throughput rather than Python-call latency.                                          |
| gf137_modular_network_numba_fused                |     3.25391 |          480.585  |         0.472857 |                  0.53 |           0.505 | Fused Numba CPU kernel; matmul, modulo, threshold, and readout in one loop.                                                                                                                  |
| gf137_modular_network_mlx_gpu                    |     6.50391 |          240.195  |         0.472857 |                  0.53 |           0.505 | MLX GPU path on Apple Silicon; includes GPU synchronization each timed pass.                                                                                                                 |
| gf137_modular_network_torch_mps_eager            |     6.50391 |          145.539  |         0.472857 |                  0.53 |           0.505 | Torch eager path using floor modulo; synchronization included for MPS.                                                                                                                       |
| gf137_modular_network_torch_mps_compile          |     6.50391 |          118.264  |         0.472857 |                  0.53 |           0.505 | Torch compile reduce-overhead path.                                                                                                                                                          |
| gf137_modular_network_jax_jit_cpu                |     6.50391 |           42.4758 |         0.472857 |                  0.53 |           0.505 | JAX JIT CPU path; JAX reports CPU only in this environment.                                                                                                                                  |

## Winners

- Lowest memory footprint: `gf137_modular_network_compact_uint8`, `gf137_modular_network_cpp_fused_uint8`, `gf137_modular_network_cpp_neon_uint8_native_loop` (1.62793 KB)
- Fastest 1000 forward passes, throughput mode included: `gf137_modular_network_cpp_neon_uint8_native_loop` (12.1055 ms)
- Fastest Python-call latency path: `gf137_modular_network_cpp_fused_uint8` (40.153 ms)
- Highest test accuracy: `gf137_modular_network_compact_uint8` (0.505)

## Interpretation

The modular model is much smaller because its weights are stored as `uint8`
residues. The multithreaded C++ fused path keeps that compact storage and
removes most of the Python/NumPy integer-loop overhead, but the fastest backend
in this run may still be a JIT/BLAS path if vectorization dominates threading
overhead for this small dense workload. Accuracy should be read cautiously: if
the models are near chance, that is consistent with the earlier null result that
base-e fractional digits do not expose a clean primality invariant.

## What Would Be Faster Than NumPy Here?

For this exact GF(137) workload, the next realistic speed steps are not more
Python wrappers but a SIMD/tiled/multithreaded C++ kernel or a custom XLA-style
integer kernel. This script now records all optional backends that are
importable in the active environment.
