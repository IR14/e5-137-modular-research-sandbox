"""ctypes bindings for the native GF(137) fused matrix kernel."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = PROJECT_ROOT / "src" / "modular_core" / "fused_matrix_gf137.cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "modular_core"
COMPILE_LOG = BUILD_DIR / "compile.log"
LIBRARY_PATH = BUILD_DIR / ("libgf137.dylib" if sys.platform == "darwin" else "libgf137.so")


def compile_cpp_kernel(*, force: bool = False) -> Path:
    """Compile the C++ fused kernel and return the shared-library path."""

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
        "-pthread",
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
        raise RuntimeError(f"C++ kernel compilation failed. See {COMPILE_LOG}")
    _load_library.cache_clear()
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(compile_cpp_kernel()))
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)
    c_int = ctypes.c_int

    library.fused_matmul_mod137_threshold.argtypes = [
        u8_ptr,
        u8_ptr,
        u8_ptr,
        c_int,
        c_int,
        c_int,
    ]
    library.fused_matmul_mod137_threshold.restype = None

    library.fused_matmul_mod137_bias_threshold.argtypes = [
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        c_int,
        c_int,
        c_int,
        ctypes.c_uint8,
    ]
    library.fused_matmul_mod137_bias_threshold.restype = None

    library.fused_two_layer_mod137_predict.argtypes = [
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        c_int,
        c_int,
        c_int,
        ctypes.c_uint8,
        ctypes.c_uint8,
        c_int,
    ]
    library.fused_two_layer_mod137_predict.restype = None

    library.gf137_thread_count.argtypes = []
    library.gf137_thread_count.restype = c_int

    library.fused_two_layer_mod137_predict_repeated.argtypes = [
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        u8_ptr,
        c_int,
        c_int,
        c_int,
        ctypes.c_uint8,
        ctypes.c_uint8,
        c_int,
        c_int,
    ]
    library.fused_two_layer_mod137_predict_repeated.restype = None
    return library


def _uint8_contiguous(array: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=np.uint8)
    if result.ndim == 0:
        raise ValueError(f"{name} must be at least one-dimensional.")
    return result


def _pointer(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def fused_matmul_mod137_threshold(x: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Run `(x @ weights % 137) >= 42` through the native C++ kernel."""

    x_u8 = _uint8_contiguous(x, "x")
    weights_u8 = _uint8_contiguous(weights, "weights")
    if x_u8.ndim != 2 or weights_u8.ndim != 2:
        raise ValueError("x and weights must both be 2D arrays.")
    if x_u8.shape[1] != weights_u8.shape[0]:
        raise ValueError(f"Shape mismatch: x={x_u8.shape}, weights={weights_u8.shape}.")

    m, k = x_u8.shape
    n = weights_u8.shape[1]
    y = np.empty((m, n), dtype=np.uint8)
    _load_library().fused_matmul_mod137_threshold(
        _pointer(x_u8),
        _pointer(weights_u8),
        _pointer(y),
        m,
        k,
        n,
    )
    return y


def fused_matmul_mod137_bias_threshold(
    x: np.ndarray,
    weights: np.ndarray,
    bias: np.ndarray,
    *,
    threshold: int = 42,
) -> np.ndarray:
    """Run a biased GF(137) matrix product with a configurable threshold."""

    x_u8 = _uint8_contiguous(x, "x")
    weights_u8 = _uint8_contiguous(weights, "weights")
    bias_u8 = _uint8_contiguous(bias, "bias").reshape(-1)
    if x_u8.ndim != 2 or weights_u8.ndim != 2:
        raise ValueError("x and weights must both be 2D arrays.")
    if x_u8.shape[1] != weights_u8.shape[0]:
        raise ValueError(f"Shape mismatch: x={x_u8.shape}, weights={weights_u8.shape}.")
    if bias_u8.shape[0] != weights_u8.shape[1]:
        raise ValueError(f"Bias length {bias_u8.shape[0]} does not match output width.")

    m, k = x_u8.shape
    n = weights_u8.shape[1]
    y = np.empty((m, n), dtype=np.uint8)
    _load_library().fused_matmul_mod137_bias_threshold(
        _pointer(x_u8),
        _pointer(weights_u8),
        _pointer(bias_u8),
        _pointer(y),
        m,
        k,
        n,
        ctypes.c_uint8(threshold),
    )
    return y


def fused_two_layer_mod137_predict(
    x: np.ndarray,
    hidden_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
    *,
    hidden_threshold: int = 42,
    output_threshold: int = 42,
    greater_is_prime: bool = True,
) -> np.ndarray:
    """Run the two-layer modular benchmark network through the native kernel."""

    x_u8 = _uint8_contiguous(x, "x")
    w1_u8 = _uint8_contiguous(hidden_weights, "hidden_weights")
    b1_u8 = _uint8_contiguous(hidden_bias, "hidden_bias").reshape(-1)
    w2_u8 = _uint8_contiguous(output_weights, "output_weights").reshape(-1)
    b2_u8 = _uint8_contiguous(output_bias, "output_bias").reshape(-1)
    if x_u8.ndim != 2 or w1_u8.ndim != 2:
        raise ValueError("x and hidden_weights must both be 2D arrays.")
    if x_u8.shape[1] != w1_u8.shape[0]:
        raise ValueError(f"Shape mismatch: x={x_u8.shape}, hidden_weights={w1_u8.shape}.")
    hidden_width = w1_u8.shape[1]
    if b1_u8.shape[0] != hidden_width:
        raise ValueError("hidden_bias length does not match hidden width.")
    if w2_u8.shape[0] != hidden_width:
        raise ValueError("output_weights length does not match hidden width.")
    if b2_u8.shape[0] != 1:
        raise ValueError("output_bias must contain exactly one value.")

    m, k = x_u8.shape
    y = np.empty(m, dtype=np.uint8)
    _load_library().fused_two_layer_mod137_predict(
        _pointer(x_u8),
        _pointer(w1_u8),
        _pointer(b1_u8),
        _pointer(w2_u8),
        _pointer(b2_u8),
        _pointer(y),
        m,
        k,
        hidden_width,
        ctypes.c_uint8(hidden_threshold),
        ctypes.c_uint8(output_threshold),
        int(greater_is_prime),
    )
    return y


def fused_two_layer_mod137_predict_into(
    x: np.ndarray,
    hidden_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
    out: np.ndarray,
    *,
    hidden_threshold: int = 42,
    output_threshold: int = 42,
    greater_is_prime: bool = True,
) -> np.ndarray:
    """Run the two-layer native kernel into a caller-owned uint8 output buffer."""

    x_u8 = _uint8_contiguous(x, "x")
    w1_u8 = _uint8_contiguous(hidden_weights, "hidden_weights")
    b1_u8 = _uint8_contiguous(hidden_bias, "hidden_bias").reshape(-1)
    w2_u8 = _uint8_contiguous(output_weights, "output_weights").reshape(-1)
    b2_u8 = _uint8_contiguous(output_bias, "output_bias").reshape(-1)
    out_u8 = _uint8_contiguous(out, "out").reshape(-1)
    if x_u8.ndim != 2 or w1_u8.ndim != 2:
        raise ValueError("x and hidden_weights must both be 2D arrays.")
    if x_u8.shape[1] != w1_u8.shape[0]:
        raise ValueError(f"Shape mismatch: x={x_u8.shape}, hidden_weights={w1_u8.shape}.")
    hidden_width = w1_u8.shape[1]
    if b1_u8.shape[0] != hidden_width:
        raise ValueError("hidden_bias length does not match hidden width.")
    if w2_u8.shape[0] != hidden_width:
        raise ValueError("output_weights length does not match hidden width.")
    if b2_u8.shape[0] != 1:
        raise ValueError("output_bias must contain exactly one value.")
    if out_u8.shape[0] != x_u8.shape[0]:
        raise ValueError("out length does not match input row count.")

    m, k = x_u8.shape
    _load_library().fused_two_layer_mod137_predict(
        _pointer(x_u8),
        _pointer(w1_u8),
        _pointer(b1_u8),
        _pointer(w2_u8),
        _pointer(b2_u8),
        _pointer(out_u8),
        m,
        k,
        hidden_width,
        ctypes.c_uint8(hidden_threshold),
        ctypes.c_uint8(output_threshold),
        int(greater_is_prime),
    )
    return out_u8


def gf137_thread_count() -> int:
    """Return the native worker count selected by the C++ kernel."""

    return int(_load_library().gf137_thread_count())


def fused_two_layer_mod137_predict_repeated(
    x: np.ndarray,
    hidden_weights: np.ndarray,
    hidden_bias: np.ndarray,
    output_weights: np.ndarray,
    output_bias: np.ndarray,
    out: np.ndarray,
    iterations: int,
    *,
    hidden_threshold: int = 42,
    output_threshold: int = 42,
    greater_is_prime: bool = True,
) -> np.ndarray:
    """Run repeated two-layer inference inside one native C++ call."""

    x_u8 = _uint8_contiguous(x, "x")
    w1_u8 = _uint8_contiguous(hidden_weights, "hidden_weights")
    b1_u8 = _uint8_contiguous(hidden_bias, "hidden_bias").reshape(-1)
    w2_u8 = _uint8_contiguous(output_weights, "output_weights").reshape(-1)
    b2_u8 = _uint8_contiguous(output_bias, "output_bias").reshape(-1)
    out_u8 = _uint8_contiguous(out, "out").reshape(-1)
    if x_u8.ndim != 2 or w1_u8.ndim != 2:
        raise ValueError("x and hidden_weights must both be 2D arrays.")
    if x_u8.shape[1] != w1_u8.shape[0]:
        raise ValueError(f"Shape mismatch: x={x_u8.shape}, hidden_weights={w1_u8.shape}.")
    hidden_width = w1_u8.shape[1]
    if b1_u8.shape[0] != hidden_width:
        raise ValueError("hidden_bias length does not match hidden width.")
    if w2_u8.shape[0] != hidden_width:
        raise ValueError("output_weights length does not match hidden width.")
    if b2_u8.shape[0] != 1:
        raise ValueError("output_bias must contain exactly one value.")
    if out_u8.shape[0] != x_u8.shape[0]:
        raise ValueError("out length does not match input row count.")

    m, k = x_u8.shape
    _load_library().fused_two_layer_mod137_predict_repeated(
        _pointer(x_u8),
        _pointer(w1_u8),
        _pointer(b1_u8),
        _pointer(w2_u8),
        _pointer(b2_u8),
        _pointer(out_u8),
        m,
        k,
        hidden_width,
        ctypes.c_uint8(hidden_threshold),
        ctypes.c_uint8(output_threshold),
        int(greater_is_prime),
        int(iterations),
    )
    return out_u8
