#!/usr/bin/env python3
"""Stage and verify the liquid-water benchmark from a les_fit checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

FILES = {
    "data-benchmark/train-H2O_RPBE-D3.xyz": (
        "data/train-H2O_RPBE-D3.xyz",
        "5e55bc3e3e39f09d40b105542429b6ebac7c6d33f966b36c8ffacc9d2593d3ff",
    ),
    "data-benchmark/test-H2O_RPBE-D3.xyz": (
        "data/test-H2O_RPBE-D3.xyz",
        "63246aa6ec1e5c1f806236a05c0ce3181f54b7a2c622d4520985b59d5dc40aef",
    ),
    "MLIPs/MACE-LES/water/mace-r-4.5-nl-0/H20_stagetwo.model": (
        "pretrained/H20_stagetwo.model",
        "dbca3dc1a0446e015760d9730bc2048da5004ea01a7f7160007ba6a2bd336e5a",
    ),
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    configured_work_root = os.environ.get("MACE_FNO_WORK_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--les-fit-root", type=Path, required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            Path(configured_work_root) / "les_water" / "source"
            if configured_work_root
            else None
        ),
        help="Staging directory; defaults to MACE_FNO_WORK_ROOT/les_water/source",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def copy_verified(
    source: Path,
    destination: Path,
    expected_sha256: str,
    force: bool,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    source_digest = sha256sum(source)
    if source_digest != expected_sha256:
        raise ValueError(
            f"checksum mismatch for {source}: "
            f"{source_digest} != {expected_sha256}"
        )
    if destination.exists() and not force:
        if destination.is_file() and sha256sum(destination) == expected_sha256:
            print(f"verified existing {destination}")
            return
        raise FileExistsError(f"refusing to overwrite {destination}; pass --force")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    shutil.copy2(source, temporary)
    temporary.replace(destination)
    print(f"staged {destination}")


def main() -> None:
    args = parse_arguments()
    if args.output_root is None:
        raise ValueError("set MACE_FNO_WORK_ROOT or pass --output-root")
    source_root = args.les_fit_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    for relative_source, (relative_destination, checksum) in FILES.items():
        copy_verified(
            source_root / relative_source,
            output_root / relative_destination,
            checksum,
            args.force,
        )


if __name__ == "__main__":
    main()
