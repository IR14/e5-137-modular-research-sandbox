"""Toy DNA fault-tolerance bindings for the E-5-137 GF(137) sandbox.

This module models a redundant noisy channel over nucleotide symbols. It is not
a biological immortality model and does not claim real genomic error rates.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CPP_SOURCE = PROJECT_ROOT / "src" / "modular_biology" / "dna_fault_tolerance.cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "modular_biology"
COMPILE_LOG = BUILD_DIR / "compile.log"
LIBRARY_PATH = BUILD_DIR / ("libe5137dna.dylib" if sys.platform == "darwin" else "libe5137dna.so")


def compile_biology_kernel(*, force: bool = False) -> Path:
    """Compile the C++ DNA fault-tolerance toy kernel."""

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
        raise RuntimeError(f"DNA kernel compilation failed. See {COMPILE_LOG}")
    _load_library.cache_clear()
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(compile_biology_kernel()))
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)
    int_ptr = ctypes.POINTER(ctypes.c_int)

    library.e5137_dna_axis_count.argtypes = []
    library.e5137_dna_axis_count.restype = ctypes.c_int
    library.e5137_dna_correction_budget.argtypes = []
    library.e5137_dna_correction_budget.restype = ctypes.c_int
    library.e5137_semantic_reference_token_count.argtypes = []
    library.e5137_semantic_reference_token_count.restype = ctypes.c_int
    library.e5137_semantic_reference_vector_count.argtypes = []
    library.e5137_semantic_reference_vector_count.restype = ctypes.c_int
    library.e5137_semantic_vector_count_for_tokens.argtypes = [ctypes.c_int]
    library.e5137_semantic_vector_count_for_tokens.restype = ctypes.c_int
    library.e5137_semantic_build_context.argtypes = [ctypes.c_int, ctypes.c_char_p]
    library.e5137_semantic_build_context.restype = None
    library.e5137_dna_base_code.argtypes = [ctypes.c_char]
    library.e5137_dna_base_code.restype = ctypes.c_int
    library.e5137_dna_code_base.argtypes = [ctypes.c_uint8]
    library.e5137_dna_code_base.restype = ctypes.c_char
    library.e5137_dna_encode.argtypes = [ctypes.c_char_p, ctypes.c_int, u8_ptr]
    library.e5137_dna_encode.restype = None
    library.e5137_dna_decode.argtypes = [u8_ptr, ctypes.c_int, ctypes.c_char_p]
    library.e5137_dna_decode.restype = ctypes.c_int
    library.e5137_dna_replicate.argtypes = [ctypes.c_char_p, ctypes.c_int, u8_ptr]
    library.e5137_dna_replicate.restype = ctypes.c_int
    library.e5137_dna_repair.argtypes = [u8_ptr, ctypes.c_int, ctypes.c_char_p, int_ptr]
    library.e5137_dna_repair.restype = ctypes.c_int
    library.e5137_dna_simulate.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_char_p,
        int_ptr,
    ]
    library.e5137_dna_simulate.restype = ctypes.c_int
    library.e5137_semantic_simulate_hobbit_context.argtypes = [
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_uint32,
        ctypes.c_char_p,
        int_ptr,
    ]
    library.e5137_semantic_simulate_hobbit_context.restype = ctypes.c_int
    return library


def _dna_bytes(sequence: str | bytes | bytearray) -> bytes:
    data = bytes(sequence, "ascii") if isinstance(sequence, str) else bytes(sequence)
    if not data:
        raise ValueError("DNA sequence must not be empty.")
    invalid = sorted({chr(base) for base in data if chr(base).upper() not in {"A", "T", "G", "C"}})
    if invalid:
        raise ValueError(f"Invalid DNA bases: {invalid}")
    return data.upper()


def _u8(array: np.ndarray, name: str) -> np.ndarray:
    result = np.ascontiguousarray(array, dtype=np.uint8)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return result


def _ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def axis_count() -> int:
    return int(_load_library().e5137_dna_axis_count())


def correction_budget() -> int:
    return int(_load_library().e5137_dna_correction_budget())


def semantic_reference_token_count() -> int:
    return int(_load_library().e5137_semantic_reference_token_count())


def semantic_reference_vector_count() -> int:
    return int(_load_library().e5137_semantic_reference_vector_count())


def semantic_vector_count_for_tokens(token_count: int) -> int:
    return int(_load_library().e5137_semantic_vector_count_for_tokens(int(token_count)))


def semantic_context_sequence(vector_count: int | None = None) -> str:
    count = semantic_reference_vector_count() if vector_count is None else int(vector_count)
    if count <= 0:
        raise ValueError("vector_count must be positive.")
    out = ctypes.create_string_buffer(count)
    _load_library().e5137_semantic_build_context(count, out)
    return out.raw.decode("ascii")


def base_code(base: str) -> int:
    data = _dna_bytes(base)
    if len(data) != 1:
        raise ValueError("base must contain exactly one nucleotide.")
    return int(_load_library().e5137_dna_base_code(ctypes.c_char(data)))


def encode_dna(sequence: str | bytes | bytearray) -> np.ndarray:
    data = _dna_bytes(sequence)
    out = np.empty(2 * len(data), dtype=np.uint8)
    _load_library().e5137_dna_encode(data, len(data), _ptr(out))
    return out


def decode_dna(residues: np.ndarray) -> str:
    encoded = _u8(residues, "residues").reshape(-1)
    if encoded.size % 2 != 0:
        raise ValueError("Encoded DNA residue vector must have even length.")
    out = ctypes.create_string_buffer(encoded.size // 2)
    decoded_len = int(_load_library().e5137_dna_decode(_ptr(encoded), int(encoded.size), out))
    return out.raw[:decoded_len].decode("ascii")


def replicate_dna(sequence: str | bytes | bytearray) -> np.ndarray:
    data = _dna_bytes(sequence)
    shares = np.empty((len(data), axis_count()), dtype=np.uint8)
    ok = int(_load_library().e5137_dna_replicate(data, len(data), _ptr(shares.reshape(-1))))
    if ok != 1:
        raise ValueError("DNA replication failed.")
    return shares


def repair_dna(shares: np.ndarray) -> tuple[str, np.ndarray, int]:
    share_array = _u8(shares, "shares")
    if share_array.ndim != 2 or share_array.shape[1] != axis_count():
        raise ValueError(f"shares must have shape (n, {axis_count()}).")
    if np.any(share_array >= 137):
        raise ValueError("shares must contain GF(137) residues, i.e. < 137.")
    out = ctypes.create_string_buffer(share_array.shape[0])
    votes = np.empty(share_array.shape[0], dtype=np.int32)
    min_votes = int(
        _load_library().e5137_dna_repair(
            _ptr(np.ascontiguousarray(share_array).reshape(-1)),
            int(share_array.shape[0]),
            out,
            votes.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
    )
    return out.raw.decode("ascii"), votes, min_votes


def corrupt_random_axes(shares: np.ndarray, damage_axes: int, *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    corrupted = np.array(shares, dtype=np.uint8, copy=True)
    n_axes = corrupted.shape[1]
    for row in range(corrupted.shape[0]):
        axes = rng.choice(n_axes, size=min(max(damage_axes, 0), n_axes), replace=False)
        deltas = rng.integers(1, 137, size=len(axes), dtype=np.uint16)
        corrupted[row, axes] = ((corrupted[row, axes].astype(np.uint16) + deltas) % 137).astype(
            np.uint8
        )
    return corrupted


def simulate_mitosis(
    sequence: str | bytes | bytearray,
    *,
    cycles: int,
    damage_axes: int,
    seed: int,
) -> dict[str, object]:
    data = _dna_bytes(sequence)
    out = ctypes.create_string_buffer(len(data))
    failed_cycles = np.zeros(1, dtype=np.int32)
    ok = int(
        _load_library().e5137_dna_simulate(
            data,
            len(data),
            int(cycles),
            int(damage_axes),
            ctypes.c_uint32(seed),
            out,
            failed_cycles.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
    )
    if ok != 1:
        raise ValueError("DNA mitosis simulation failed.")
    final_sequence = out.raw.decode("ascii")
    matches = sum(a == b for a, b in zip(data.decode("ascii"), final_sequence, strict=True))
    return {
        "initial": data.decode("ascii"),
        "final": final_sequence,
        "cycles": int(cycles),
        "damage_axes": int(damage_axes),
        "failed_cycles": int(failed_cycles[0]),
        "accuracy": matches / len(data),
    }


def simulate_semantic_context(
    *,
    cycles: int,
    damage_axes: int,
    seed: int,
) -> dict[str, object]:
    """Run the 117-vector compressed-context toy benchmark through the repair channel."""

    vector_count = semantic_reference_vector_count()
    out = ctypes.create_string_buffer(vector_count)
    failed_cycles = np.zeros(1, dtype=np.int32)
    ok = int(
        _load_library().e5137_semantic_simulate_hobbit_context(
            int(cycles),
            int(damage_axes),
            ctypes.c_uint32(seed),
            out,
            failed_cycles.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
        )
    )
    if ok != 1:
        raise ValueError("Semantic context simulation failed.")
    initial = semantic_context_sequence(vector_count)
    final = out.raw.decode("ascii")
    matches = sum(a == b for a, b in zip(initial, final, strict=True))
    return {
        "initial": initial,
        "final": final,
        "tokens": semantic_reference_token_count(),
        "vectors": vector_count,
        "cycles": int(cycles),
        "damage_axes": int(damage_axes),
        "failed_cycles": int(failed_cycles[0]),
        "accuracy": matches / vector_count,
    }
