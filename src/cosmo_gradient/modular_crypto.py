"""Experimental ctypes bindings for the E-5-137 GF(137) toy crypto module.

This module is an engineering sandbox, not an audited cryptographic library.
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
CPP_SOURCE = PROJECT_ROOT / "src" / "modular_crypto" / "e5137_protocol.cpp"
BUILD_DIR = PROJECT_ROOT / "build" / "modular_crypto"
COMPILE_LOG = BUILD_DIR / "compile.log"
LIBRARY_PATH = BUILD_DIR / ("libe5137crypto.dylib" if sys.platform == "darwin" else "libe5137crypto.so")


def compile_crypto_kernel(*, force: bool = False) -> Path:
    """Compile the C++ E-5-137 toy protocol and return the shared library path."""

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
        raise RuntimeError(f"E-5-137 crypto kernel compilation failed. See {COMPILE_LOG}")
    _load_library.cache_clear()
    return LIBRARY_PATH


@lru_cache(maxsize=1)
def _load_library() -> ctypes.CDLL:
    library = ctypes.CDLL(str(compile_crypto_kernel()))
    u8_ptr = ctypes.POINTER(ctypes.c_uint8)

    library.e5137_axis_count.argtypes = []
    library.e5137_axis_count.restype = ctypes.c_int
    library.e5137_correctable_axes.argtypes = []
    library.e5137_correctable_axes.restype = ctypes.c_int
    library.e5137_mod_inverse.argtypes = [ctypes.c_uint8]
    library.e5137_mod_inverse.restype = ctypes.c_uint8
    library.e5137_generate_key.argtypes = [u8_ptr, ctypes.c_int, u8_ptr]
    library.e5137_generate_key.restype = None
    library.e5137_replicate_symbol.argtypes = [ctypes.c_uint8, u8_ptr, u8_ptr]
    library.e5137_replicate_symbol.restype = None
    library.e5137_recover_symbol.argtypes = [u8_ptr, u8_ptr, u8_ptr]
    library.e5137_recover_symbol.restype = ctypes.c_int
    library.e5137_encrypt.argtypes = [u8_ptr, ctypes.c_int, u8_ptr, u8_ptr]
    library.e5137_encrypt.restype = None
    library.e5137_decrypt.argtypes = [u8_ptr, ctypes.c_int, u8_ptr, u8_ptr]
    library.e5137_decrypt.restype = None
    return library


def _as_u8(array: np.ndarray | bytes | bytearray, name: str) -> np.ndarray:
    if isinstance(array, bytes | bytearray):
        result = np.frombuffer(array, dtype=np.uint8)
    else:
        result = np.asarray(array, dtype=np.uint8)
    result = np.ascontiguousarray(result, dtype=np.uint8).reshape(-1)
    if result.size == 0:
        raise ValueError(f"{name} must not be empty.")
    return result


def _ptr(array: np.ndarray) -> ctypes.POINTER(ctypes.c_uint8):
    return array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))


def axis_count() -> int:
    return int(_load_library().e5137_axis_count())


def correctable_axes() -> int:
    return int(_load_library().e5137_correctable_axes())


def mod_inverse(value: int) -> int:
    return int(_load_library().e5137_mod_inverse(ctypes.c_uint8(value % 137)))


def generate_key(seed: bytes | bytearray | np.ndarray) -> np.ndarray:
    seed_u8 = _as_u8(seed, "seed")
    key = np.empty(axis_count(), dtype=np.uint8)
    _load_library().e5137_generate_key(_ptr(seed_u8), int(seed_u8.size), _ptr(key))
    return key


def _key26(key: np.ndarray) -> np.ndarray:
    key_u8 = _as_u8(key, "key")
    if key_u8.size != axis_count():
        raise ValueError(f"key must contain exactly {axis_count()} residues.")
    if np.any(key_u8 >= 137):
        raise ValueError("key residues must be in GF(137), i.e. < 137.")
    return key_u8


def replicate_symbol(symbol: int, key: np.ndarray) -> np.ndarray:
    key_u8 = _key26(key)
    shares = np.empty(axis_count(), dtype=np.uint8)
    _load_library().e5137_replicate_symbol(ctypes.c_uint8(symbol % 137), _ptr(key_u8), _ptr(shares))
    return shares


def recover_symbol(shares: np.ndarray, key: np.ndarray) -> tuple[int, int]:
    shares_u8 = _as_u8(shares, "shares")
    key_u8 = _key26(key)
    if shares_u8.size != axis_count():
        raise ValueError(f"shares must contain exactly {axis_count()} residues.")
    if np.any(shares_u8 >= 137):
        raise ValueError("shares must be GF(137) residues, i.e. < 137.")
    symbol = np.empty(1, dtype=np.uint8)
    votes = int(_load_library().e5137_recover_symbol(_ptr(shares_u8), _ptr(key_u8), _ptr(symbol)))
    return int(symbol[0]), votes


def encrypt(plaintext: bytes | bytearray | np.ndarray, key: np.ndarray) -> np.ndarray:
    plaintext_u8 = _as_u8(plaintext, "plaintext")
    key_u8 = _key26(key)
    cipher = np.empty(2 * plaintext_u8.size, dtype=np.uint8)
    _load_library().e5137_encrypt(_ptr(plaintext_u8), int(plaintext_u8.size), _ptr(key_u8), _ptr(cipher))
    return cipher


def decrypt(cipher: np.ndarray, key: np.ndarray) -> bytes:
    cipher_u8 = _as_u8(cipher, "cipher")
    key_u8 = _key26(key)
    if cipher_u8.size % 2 != 0:
        raise ValueError("cipher length must be even because bytes are encoded as two GF(137) residues.")
    if np.any(cipher_u8 >= 137):
        raise ValueError("cipher must contain GF(137) residues, i.e. < 137.")
    plaintext = np.empty(cipher_u8.size // 2, dtype=np.uint8)
    _load_library().e5137_decrypt(_ptr(cipher_u8), int(cipher_u8.size), _ptr(key_u8), _ptr(plaintext))
    return plaintext.tobytes()

