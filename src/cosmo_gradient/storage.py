"""Storage checks for large external data roots."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StorageStatus:
    """Writable storage status for a candidate data root."""

    path: Path
    exists: bool
    writable: bool
    total_bytes: int
    used_bytes: int
    free_bytes: int
    message: str

    @property
    def free_gib(self) -> float:
        """Free space in GiB."""
        return self.free_bytes / 1024**3

    @property
    def total_gib(self) -> float:
        """Total space in GiB."""
        return self.total_bytes / 1024**3


def check_storage_root(path: Path, min_free_gib: float = 1.0) -> StorageStatus:
    """Check that *path* exists, is writable, and has enough free space."""
    root = path.expanduser()
    if not root.exists():
        return StorageStatus(
            path=root,
            exists=False,
            writable=False,
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
            message=f"path does not exist: {root}",
        )
    try:
        usage = shutil.disk_usage(root)
    except OSError as error:
        return StorageStatus(
            path=root,
            exists=True,
            writable=False,
            total_bytes=0,
            used_bytes=0,
            free_bytes=0,
            message=f"cannot read disk usage for {root}: {error}",
        )

    writable = _can_write(root)
    free_gib = usage.free / 1024**3
    if not writable:
        message = f"path is not writable: {root}"
    elif free_gib < min_free_gib:
        message = f"only {free_gib:.1f} GiB free; need at least {min_free_gib:.1f} GiB"
    else:
        message = f"ok: {free_gib:.1f} GiB free"
    return StorageStatus(
        path=root,
        exists=True,
        writable=writable and free_gib >= min_free_gib,
        total_bytes=usage.total,
        used_bytes=usage.used,
        free_bytes=usage.free,
        message=message,
    )


def _can_write(path: Path) -> bool:
    test_dir = path / ".cosmo_gradient_write_test"
    test_file = test_dir / "probe.txt"
    try:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file.write_text("ok\n", encoding="utf-8")
        test_file.unlink(missing_ok=True)
        os.rmdir(test_dir)
    except OSError:
        return False
    return True
