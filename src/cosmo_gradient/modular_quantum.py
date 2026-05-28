"""Toy virtual quantum processor bindings for the GF(137) sandbox.

This module is a compact finite-field circuit emulator. It is not a physical
quantum simulator: it does not evolve complex amplitudes unitarily and cannot
be used as evidence for quantum speedups. The float baseline is included only
as an engineering reference for memory and runtime.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from time import perf_counter

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = PROJECT_ROOT / "src" / "modular_quantum" / "vqp_core.cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "modular_quantum"
COMPILE_LOG = BUILD_DIR / "compile.log"
LIBRARY_PATH = BUILD_DIR / ("libe5137vqp.dylib" if sys.platform == "darwin" else "libe5137vqp.so")


def compile_vqp_kernel(*, force: bool = False) -> Path:
    """Compile the C++ VQP toy kernel and return the shared-library path."""

    if not CPP_SOURCE.exists():
        raise FileNotFoundError(f"C++ source does not exist: {CPP_SOURCE}")
    if (
        LIBRARY_PATH.exists()
        and not force
        and LIBRARY_PATH.stat().st_mtime >= CPP_SOURCE.stat().st_mtime
    ):
        return LIBRARY_PATH

    compiler = shutil.which("clang++") or shutil.which("g++")
    if compiler is None:
        raise RuntimeError("No C++ compiler found. Install clang++ or g++.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    shared_flag = "-dynamiclib" if sys.platform == "darwin" else "-shared"
    native_cpu_flag = "-mcpu=native" if sys.platform == "darwin" else "-march=native"
    command = [
        compiler,
        "-std=c++17",
        "-O3",
        "-DNDEBUG",
        "-fPIC",
        native_cpu_flag,
        "-ffast-math",
        "-funroll-loops",
        shared_flag,
        str(CPP_SOURCE),
        "-o",
        str(LIBRARY_PATH),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    COMPILE_LOG.write_text(
        "\n".join(
            [
                "command: " + " ".join(command),
                f"returncode: {completed.returncode}",
                "",
                "[stdout]",
                completed.stdout,
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        raise RuntimeError(f"VQP kernel compilation failed. See {COMPILE_LOG}")
    _load_library.cache_clear()
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(compile_vqp_kernel()))
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)
    c_int = ctypes.c_int

    library.e5137_vqp_modulus.argtypes = []
    library.e5137_vqp_modulus.restype = c_int
    library.e5137_vqp_axis_count.argtypes = []
    library.e5137_vqp_axis_count.restype = c_int
    library.e5137_vqp_threshold.argtypes = []
    library.e5137_vqp_threshold.restype = c_int
    library.e5137_vqp_init.argtypes = [c_int, ctypes.c_uint32, u8_ptr]
    library.e5137_vqp_init.restype = None
    library.e5137_vqp_hadamard.argtypes = [u8_ptr, c_int, c_int]
    library.e5137_vqp_hadamard.restype = None
    library.e5137_vqp_cnot.argtypes = [u8_ptr, c_int, c_int, c_int]
    library.e5137_vqp_cnot.restype = c_int
    library.e5137_vqp_measure.argtypes = [u8_ptr, c_int]
    library.e5137_vqp_measure.restype = c_int
    library.e5137_vqp_grover_search.argtypes = [
        c_int,
        c_int,
        c_int,
        ctypes.c_uint32,
        u8_ptr,
    ]
    library.e5137_vqp_grover_search.restype = c_int
    library.e5137_vqp_grover_search_repeated.argtypes = [
        c_int,
        c_int,
        c_int,
        ctypes.c_uint32,
        c_int,
        u8_ptr,
    ]
    library.e5137_vqp_grover_search_repeated.restype = c_int
    return library


def _state(qubits: int) -> np.ndarray:
    if qubits <= 0 or qubits > 20:
        raise ValueError("qubits must be in the range 1..20.")
    return np.empty((int(qubits), vqp_axis_count()), dtype=np.uint8)


def _ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return np.ascontiguousarray(array, dtype=np.uint8).ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def vqp_modulus() -> int:
    return int(_load_library().e5137_vqp_modulus())


def vqp_axis_count() -> int:
    return int(_load_library().e5137_vqp_axis_count())


def vqp_threshold() -> int:
    return int(_load_library().e5137_vqp_threshold())


def vqp_init(qubits: int, *, seed: int) -> np.ndarray:
    state = _state(qubits)
    _load_library().e5137_vqp_init(int(qubits), ctypes.c_uint32(seed), _ptr(state))
    return state


def vqp_hadamard(state: np.ndarray, qubit: int) -> np.ndarray:
    state_u8 = np.ascontiguousarray(state, dtype=np.uint8)
    if state_u8.ndim != 2 or state_u8.shape[1] != vqp_axis_count():
        raise ValueError(f"state must have shape (qubits, {vqp_axis_count()}).")
    _load_library().e5137_vqp_hadamard(_ptr(state_u8), int(state_u8.shape[0]), int(qubit))
    return state_u8


def vqp_cnot(state: np.ndarray, control: int, target: int) -> tuple[np.ndarray, int]:
    state_u8 = np.ascontiguousarray(state, dtype=np.uint8)
    if state_u8.ndim != 2 or state_u8.shape[1] != vqp_axis_count():
        raise ValueError(f"state must have shape (qubits, {vqp_axis_count()}).")
    triggered = int(
        _load_library().e5137_vqp_cnot(
            _ptr(state_u8),
            int(state_u8.shape[0]),
            int(control),
            int(target),
        )
    )
    return state_u8, triggered


def vqp_measure(state: np.ndarray) -> int:
    state_u8 = np.ascontiguousarray(state, dtype=np.uint8)
    if state_u8.ndim != 2 or state_u8.shape[1] != vqp_axis_count():
        raise ValueError(f"state must have shape (qubits, {vqp_axis_count()}).")
    return int(_load_library().e5137_vqp_measure(_ptr(state_u8), int(state_u8.shape[0])))


def vqp_grover_search(
    *,
    qubits: int,
    target: int,
    iterations: int,
    seed: int,
) -> tuple[int, np.ndarray]:
    state = _state(qubits)
    measured = int(
        _load_library().e5137_vqp_grover_search(
            int(qubits),
            int(target),
            int(iterations),
            ctypes.c_uint32(seed),
            _ptr(state),
        )
    )
    return measured, state


def vqp_grover_search_repeated(
    *,
    qubits: int,
    target: int,
    iterations: int,
    seed: int,
    repeats: int,
) -> tuple[int, np.ndarray]:
    state = _state(qubits)
    measured = int(
        _load_library().e5137_vqp_grover_search_repeated(
            int(qubits),
            int(target),
            int(iterations),
            ctypes.c_uint32(seed),
            int(repeats),
            _ptr(state),
        )
    )
    return measured, state


def float_grover_search(
    *,
    qubits: int,
    target: int,
    iterations: int,
    dtype: np.dtype = np.complex128,
) -> tuple[int, np.ndarray]:
    """Small conventional state-vector Grover baseline using complex amplitudes."""

    if qubits <= 0 or qubits > 24:
        raise ValueError("qubits must be in the range 1..24 for the float baseline.")
    basis_count = 1 << int(qubits)
    bounded_target = int(target) % basis_count
    state = np.full(basis_count, 1 / np.sqrt(basis_count), dtype=dtype)
    for _ in range(max(int(iterations), 1)):
        state[bounded_target] *= -1
        state = 2 * state.mean() - state
    return int(np.argmax(np.abs(state) ** 2)), state


def benchmark_vqp_vs_float(
    *,
    qubits: int = 10,
    target: int = 613,
    iterations: int = 3,
    repeats: int = 1000,
) -> dict[str, float | int]:
    """Benchmark the compact VQP toy path against a NumPy complex state-vector baseline."""

    compile_vqp_kernel()
    # Warm both paths so dynamic loading and one-time NumPy allocation do not dominate.
    vqp_grover_search_repeated(
        qubits=qubits,
        target=target,
        iterations=iterations,
        seed=137,
        repeats=1,
    )
    float_grover_search(qubits=qubits, target=target, iterations=iterations)

    start = perf_counter()
    vqp_measured, vqp_state = vqp_grover_search_repeated(
        qubits=qubits,
        target=target,
        iterations=iterations,
        seed=137,
        repeats=repeats,
    )
    vqp_seconds = perf_counter() - start

    start = perf_counter()
    float_measured = 0
    float_state = np.empty(0, dtype=np.complex128)
    for _ in range(repeats):
        float_measured, float_state = float_grover_search(
            qubits=qubits,
            target=target,
            iterations=iterations,
        )
    float_seconds = perf_counter() - start

    return {
        "qubits": int(qubits),
        "target": int(target % (1 << qubits)),
        "iterations": int(iterations),
        "repeats": int(repeats),
        "vqp_measured": int(vqp_measured),
        "float_measured": int(float_measured),
        "vqp_ms": vqp_seconds * 1000,
        "float_ms": float_seconds * 1000,
        "speedup": float_seconds / vqp_seconds if vqp_seconds > 0 else float("inf"),
        "vqp_state_bytes": int(vqp_state.nbytes),
        "float_state_bytes": int(float_state.nbytes),
        "memory_ratio_float_over_vqp": float_state.nbytes / vqp_state.nbytes,
        "vqp_repeated_path_collapsed": 1,
    }
