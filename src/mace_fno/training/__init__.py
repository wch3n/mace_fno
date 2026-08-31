"""Reusable building blocks for frozen-MACE residual training."""

from .checkpoint import load_residual_state_dict, residual_state_dict
from .data import (
    CACHE_FORMAT_VERSION,
    batch_graphs,
    clone_graph,
    collate_samples,
    has_reference_labels,
    load_or_create_samples,
    load_samples,
    reference_energy,
    reference_forces,
    sample_cache_metadata,
    save_sample_cache,
    split_samples,
)
from .evaluation import (
    ensure_frozen_residual_targets,
    evaluate,
    print_metrics,
    validation_objective,
)
from .initialization import (
    configure_output_projection_warmup,
    finish_output_projection_warmup,
    initialize_scaled_residual_output,
    initialize_zero_residual,
)
from .runtime import choose_device, elapsed_since
from .spectral_diagnostic import (
    low_k_response_diagnostic,
    planar_2d_response_diagnostic,
    periodic_3d_response_diagnostic,
    slab_2p5d_response_diagnostic,
)

__all__ = [
    "CACHE_FORMAT_VERSION",
    "batch_graphs",
    "choose_device",
    "clone_graph",
    "collate_samples",
    "configure_output_projection_warmup",
    "elapsed_since",
    "ensure_frozen_residual_targets",
    "evaluate",
    "finish_output_projection_warmup",
    "has_reference_labels",
    "initialize_scaled_residual_output",
    "initialize_zero_residual",
    "load_or_create_samples",
    "load_residual_state_dict",
    "load_samples",
    "low_k_response_diagnostic",
    "planar_2d_response_diagnostic",
    "periodic_3d_response_diagnostic",
    "print_metrics",
    "reference_energy",
    "reference_forces",
    "residual_state_dict",
    "sample_cache_metadata",
    "save_sample_cache",
    "slab_2p5d_response_diagnostic",
    "split_samples",
    "validation_objective",
]
