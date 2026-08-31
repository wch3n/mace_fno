#!/usr/bin/env python3
"""Stage and verify the Au2-MgO benchmark from a les_fit checkout."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path


FILES = {
    "data-benchmark/train-Au-MgO-Al.xyz": (
        "data/train-Au-MgO-Al.xyz",
        "81180e01728f187ea3b9a6384ffb42fc553fb827f46fd2615aa62e85eb3f17de",
    ),
    "data-benchmark/test-Au-MgO-Al.xyz": (
        "data/test-Au-MgO-Al.xyz",
        "f96a856c91e183aa271676dcac50d7bd9590cbdd43abcd2819856c6992029f8f",
    ),
    "MLIPs/MACE-LES/Au-MgO-Al/mace-r5.5-nl-0/Au2-MgO_stagetwo.model": (
        "model/Au2-MgO_r5.5_nl0_stagetwo.model",
        "8b3c5b7d106f4ad62c1f4b4487683424e879723a4622fb2a4881693b06b7d46f",
    ),
    "data-benchmark/analysis-Au-MgO/dft-optimized-struct/1-doped.xyz": (
        "wetting/1-doped.xyz",
        "19a8d8768b8b618d5a06edb1966c2b3691cd9620b40f25493e5663e092fd868b",
    ),
    "data-benchmark/analysis-Au-MgO/dft-optimized-struct/3-doped.xyz": (
        "wetting/3-doped.xyz",
        "a09ba8de2b8fd000a7eb2b30fd5c3bd721c9a03daa12e076fb3a31c27bfe9576",
    ),
    "data-benchmark/analysis-Au-MgO/dft-optimized-struct/1-undoped.xyz": (
        "wetting/1-undoped.xyz",
        "4361bf9e4f7685965af17084ee3fd24888dbdfc7662907f58e16a5ba48165ae8",
    ),
    "data-benchmark/analysis-Au-MgO/dft-optimized-struct/3-undoped.xyz": (
        "wetting/3-undoped.xyz",
        "77fa635a0bc642f9b81d2219dc21a9fa4e5867e235e1d0059c609e87cbd03b85",
    ),
}


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_arguments() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[2]
    configured_work_root = os.environ.get("MACE_FNO_WORK_ROOT")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--les-fit-root", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=project_root / "data" / "les_au_mgo",
    )
    parser.add_argument(
        "--work-root",
        type=Path,
        default=(
            Path(configured_work_root) / "les_au_mgo"
            if configured_work_root
            else None
        ),
        help="Runtime Au2-MgO directory; defaults to MACE_FNO_WORK_ROOT/les_au_mgo",
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
            f"checksum mismatch for {source}: {source_digest} != {expected_sha256}"
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
    if args.work_root is None:
        raise ValueError("set MACE_FNO_WORK_ROOT or pass --work-root")
    source_root = args.les_fit_root.resolve()
    data_root = args.data_root.resolve()
    work_root = args.work_root.resolve()
    destinations = {
        "data/train-Au-MgO-Al.xyz": data_root / "train-Au-MgO-Al.xyz",
        "data/test-Au-MgO-Al.xyz": data_root / "test-Au-MgO-Al.xyz",
        "model/Au2-MgO_r5.5_nl0_stagetwo.model": (
            work_root / "pretrained" / "Au2-MgO_r5.5_nl0_stagetwo.model"
        ),
        "wetting/1-doped.xyz": work_root / "wetting_switch" / "inputs" / "1-doped.xyz",
        "wetting/3-doped.xyz": work_root / "wetting_switch" / "inputs" / "3-doped.xyz",
        "wetting/1-undoped.xyz": (
            work_root / "wetting_switch" / "inputs" / "1-undoped.xyz"
        ),
        "wetting/3-undoped.xyz": (
            work_root / "wetting_switch" / "inputs" / "3-undoped.xyz"
        ),
    }
    for relative_source, (destination_key, checksum) in FILES.items():
        copy_verified(
            source_root / relative_source,
            destinations[destination_key],
            checksum,
            args.force,
        )


if __name__ == "__main__":
    main()
