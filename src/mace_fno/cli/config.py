"""Train frozen MACE plus a 2D, hybrid 2.5D, or periodic 3D FNO residual.

The input should be an extended XYZ file containing reference total energies
and forces. Validation data are never used for gradients, and a separate test
file is evaluated only before and after optimization. Training and evaluation
use true multi-configuration batches for MACE and the particle-mesh residual.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mace-model", type=Path, required=True)
    parser.add_argument("--train-file", type=Path, required=True)
    parser.add_argument(
        "--validation-file",
        type=Path,
        help="Optional validation XYZ; otherwise split the training file",
    )
    parser.add_argument(
        "--validation-indices-file",
        type=Path,
        help=(
            "Optional zero-based training-set indices to use for validation; "
            "mutually exclusive with --validation-file"
        ),
    )
    parser.add_argument(
        "--test-file",
        type=Path,
        help="Held-out XYZ evaluated only before and after optimization",
    )
    parser.add_argument("--train-cache", type=Path)
    parser.add_argument("--validation-cache", type=Path)
    parser.add_argument("--test-cache", type=Path)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Ignore and replace compatible preprocessed sample caches",
    )
    parser.add_argument("--energy-key", default="REF_energy")
    parser.add_argument("--forces-key", default="REF_forces")
    parser.add_argument(
        "--head",
        help="MACE head name (needed only when a multi-head checkpoint is ambiguous)",
    )
    parser.add_argument(
        "--num-atoms",
        type=int,
        help="Keep one fixed-cell subset selected by its atom count",
    )
    parser.add_argument(
        "--allow-periodic-z",
        action="store_true",
        help=(
            "Accept 3D-periodic input with a 2D/2.5D residual that remains "
            "nonperiodic in z"
        ),
    )
    parser.add_argument(
        "--skip-cell-mismatch",
        action="store_true",
        help=(
            "Skip structures outside the reference in-plane cell (2D/2.5D) "
            "or complete cell (3D)"
        ),
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--grid", type=int, default=32)
    parser.add_argument("--modes", type=int, default=8)
    parser.add_argument(
        "--spatial-scheme",
        choices=("auto", "2d", "2.5d", "3d"),
        default="auto",
        help=(
            "Mesh periodicity; auto preserves legacy behavior (2D without "
            "--z-grid, otherwise 2.5D)"
        ),
    )
    parser.add_argument(
        "--cell-mode",
        choices=("fixed", "isotropic", "anisotropic"),
        default="fixed",
        help=(
            "Cell treatment for the FNO residual. Isotropic accepts positive "
            "uniform scalings of a cubic 3D reference cell. Anisotropic accepts "
            "finite 3D cells and conditions the nonlinear operator on volume "
            "and the normalized lattice metric."
        ),
    )
    parser.add_argument(
        "--z-grid",
        type=int,
        default=0,
        help="Number of explicit z layers for the 2.5D or 3D scheme",
    )
    parser.add_argument(
        "--z-modes",
        type=int,
        default=0,
        help="Retained z Fourier modes for 3D; zero uses --modes",
    )
    parser.add_argument(
        "--z-extent",
        type=float,
        help="Physical width in angstrom of the nonperiodic z window",
    )
    parser.add_argument(
        "--z-center",
        choices=("mean", "cell"),
        default="mean",
        help="Centre the finite z window on each graph's mean height or cell centre",
    )
    parser.add_argument(
        "--lateral-interlacing",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "For 2.5D, average one mesh origin or a 2x2 set of half-grid "
            "origins to suppress lateral particle-mesh egg-box errors"
        ),
    )
    parser.add_argument(
        "--volume-interlacing",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "For 3D, average one mesh origin or a 2x2x2 set of half-grid "
            "origins to suppress particle-mesh egg-box errors"
        ),
    )
    parser.add_argument(
        "--interlacing-training",
        choices=("full", "random"),
        default="full",
        help=(
            "With 3D interlacing, either average every mesh origin during "
            "optimization or sample one origin per training batch; evaluation "
            "always averages all origins"
        ),
    )
    parser.add_argument(
        "--planar-symmetry",
        choices=("none", "c4", "d4"),
        default="none",
        help=(
            "For square 2.5D cells, group-average the field operator to enforce "
            "fourfold rotations (c4) or rotations and reflections (d4)"
        ),
    )
    parser.add_argument(
        "--spectral-symmetry",
        choices=("none", "eqgino", "cubic_adaptive", "metric_eqgino"),
        default="none",
        help=(
            "Use EqGINO radial sharing for cubic cells, or a cubic-adaptive "
            "EqGINO core plus a gated anisotropic branch for mixed cell shapes, "
            "or physical reciprocal-metric radial weights for arbitrary cells"
        ),
    )
    parser.add_argument(
        "--spectral-groups",
        type=int,
        default=1,
        help=(
            "Block-diagonal channel groups in the EqGINO spectral contraction; "
            "one retains dense channel mixing"
        ),
    )
    parser.add_argument(
        "--metric-hidden-channels",
        type=int,
        default=16,
        help="Hidden width of the physical-|k| radial network in metric EqGINO",
    )
    parser.add_argument(
        "--z-kernel-size",
        type=int,
        default=3,
        help="Odd nonperiodic z-CNN kernel size for the nonlinear 2.5D FNO",
    )
    parser.add_argument(
        "--z-mixing",
        choices=("local", "global"),
        default="local",
        help=(
            "Use a zero-padded local z CNN or channel-wise dense global "
            "nonperiodic z mixing"
        ),
    )
    parser.add_argument("--channels", type=int, default=4)
    parser.add_argument("--source-hidden-channels", type=int, default=64)
    parser.add_argument("--fno-hidden-channels", type=int, default=32)
    parser.add_argument("--fno-layers", type=int, default=4)
    parser.add_argument(
        "--architecture", choices=("linear", "nonlinear"), default="nonlinear"
    )
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument(
        "--output-initialization-scale",
        type=float,
        default=0.0,
        help=(
            "Scale the nonlinear FNO's random final-projection initialization; "
            "zero preserves the exact frozen-MACE start, while a small positive "
            "value enables upstream gradients from the first step"
        ),
    )
    parser.add_argument(
        "--output-warmup-steps",
        type=int,
        default=0,
        help=(
            "For a nonlinear FNO, train only the final output projection for "
            "this many initial optimizer steps before unfreezing the complete "
            "residual branch"
        ),
    )
    parser.add_argument(
        "--output-warmup-learning-rate",
        type=float,
        default=0.0,
        help=(
            "Learning rate during output-projection warm-up; zero uses the "
            "main --learning-rate"
        ),
    )
    parser.add_argument(
        "--lr-scheduler",
        choices=("none", "plateau"),
        default="none",
        help="Optionally reduce the learning rate when validation stalls",
    )
    parser.add_argument("--lr-decay-factor", type=float, default=0.5)
    parser.add_argument("--lr-patience-evals", type=int, default=4)
    parser.add_argument("--minimum-learning-rate", type=float, default=1.0e-6)
    parser.add_argument(
        "--early-stopping-patience-evals",
        type=int,
        default=0,
        help=(
            "Stop after this many non-improving validation checks at the minimum "
            "learning rate; zero disables early stopping"
        ),
    )
    parser.add_argument("--energy-weight", type=float, default=1.0)
    parser.add_argument("--force-weight", type=float, default=10.0)
    parser.add_argument("--energy-scale", type=float, default=1.0)
    parser.add_argument("--force-scale", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument(
        "--spectral-diagnostic-samples",
        type=int,
        default=0,
        help=(
            "Number of fixed validation structures used for the geometry-aware "
            "2D/2.5D/3D low-k diagnostic at every validation check; zero disables it"
        ),
    )
    parser.add_argument(
        "--spectral-diagnostic-max-mode",
        type=int,
        default=1,
        help="Largest integer reciprocal component used by the low-k probe",
    )
    parser.add_argument(
        "--spectral-diagnostic-fit-shells",
        type=int,
        default=3,
        help="Number of smallest physical reciprocal shells used for the low-k fit",
    )
    parser.add_argument(
        "--spectral-diagnostic-relative-amplitude",
        type=float,
        default=0.05,
        help="Fourier probe amplitude relative to the deposited-field RMS",
    )
    parser.add_argument(
        "--spectral-diagnostic-field-batch-size",
        type=int,
        default=32,
        help="Direct-field probes evaluated together by the FNO diagnostic",
    )
    parser.add_argument(
        "--spectral-diagnostic-z-profiles",
        type=int,
        choices=(1, 2, 3),
        default=3,
        help=(
            "For the selected 2.5D checkpoint, use monopole only, "
            "monopole+dipole, or monopole+dipole+quadrupole z probes; "
            "routine validation always uses the cheaper monopole probe"
        ),
    )
    parser.add_argument(
        "--spectral-diagnostic-depth",
        choices=("fast", "deep"),
        default="fast",
        help=(
            "Use one final spectral probe (fast), or run a final finite-amplitude "
            "convergence audit in addition to cheap validation probes (deep)"
        ),
    )
    parser.add_argument(
        "--spectral-diagnostic-amplitudes",
        type=float,
        nargs="+",
        default=(0.025, 0.05, 0.1),
        help="Relative field amplitudes for the deep selected-checkpoint audit",
    )
    parser.add_argument(
        "--spectral-diagnostic-relative-span-tolerance",
        type=float,
        default=0.05,
        help="Maximum relative curvature span considered amplitude-converged",
    )
    parser.add_argument(
        "--spectral-diagnostic-output",
        type=Path,
        help="Optional JSON file updated with the validation low-k diagnostic history",
    )
    parser.add_argument(
        "--evaluation-scope",
        choices=("all", "validation-test"),
        default="all",
        help="Skip full-train and null-control diagnostics in routine repeats",
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="True minibatches accumulated before each optimizer update",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Configurations evaluated together in one MACE and FNO forward pass",
    )
    parser.add_argument(
        "--evaluation-batch-size",
        type=int,
        default=0,
        help="Evaluation batch size; zero uses --batch-size",
    )
    parser.add_argument(
        "--random-residual-initialization",
        action="store_true",
        help="Do not initialize the combined model to the frozen-MACE baseline",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--dtype",
        choices=("float32", "float64"),
        default="float32",
        help=(
            "Model compute dtype; float32 retains and accumulates MACE atomic "
            "reference energies in float64"
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args(argv)
