"""Shared translations between resolved model settings and constructor names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .configuration import ModelConfig

# Checkpoints and configuration use these setting names. The model constructor
# distinguishes FNO options from those of the local backbone using a prefix.
_MODEL_ARGUMENTS = {
    "grid_shape": "grid_shape",
    "n_modes": "n_modes",
    "channels": "channels",
    "source_hidden_channels": "source_hidden_channels",
    "fno_hidden_channels": "fno_hidden_channels",
    "fno_layers": "fno_layers",
    "architecture": "fno_architecture",
    "spatial_scheme": "spatial_scheme",
    "cell_mode": "cell_mode",
    "z_grid_size": "z_grid_size",
    "z_extent": "z_extent",
    "z_center": "z_center",
    "lateral_interlacing": "fno_lateral_interlacing",
    "volume_interlacing": "fno_volume_interlacing",
    "interlacing_training": "fno_interlacing_training",
    "z_kernel_size": "fno_z_kernel_size",
    "z_mixing": "fno_z_mixing",
    "planar_symmetry": "fno_planar_symmetry",
    "spectral_symmetry": "fno_spectral_symmetry",
    "spectral_groups": "fno_spectral_groups",
    "spectral_rank": "fno_spectral_rank",
    "metric_hidden_channels": "fno_metric_hidden_channels",
    "metric_parameterization": "fno_metric_parameterization",
}


def configured_model_settings(configuration: ModelConfig) -> dict[str, Any]:
    """Resolve scalar training dimensions into the stored mesh/mode layout."""
    settings = {
        name: getattr(configuration, name)
        for name in _MODEL_ARGUMENTS
        if name not in {"grid_shape", "n_modes", "z_grid_size"}
    }
    modes = (configuration.modes, configuration.modes)
    settings.update(
        grid_shape=(configuration.grid, configuration.grid),
        n_modes=(configuration.resolved_z_modes, *modes)
        if configuration.spatial_scheme == "3d"
        else modes,
        z_grid_size=configuration.z_grid or None,
    )
    return settings


def model_constructor_arguments(settings: Mapping[str, Any]) -> dict[str, Any]:
    """Translate supplied settings without adding checkpoint fallback values."""
    arguments = {
        argument: settings[name]
        for name, argument in _MODEL_ARGUMENTS.items()
        if name in settings
    }
    if settings["spatial_scheme"] == "3d":
        arguments["n_modes"] = settings["n_modes"][1:]
        arguments["fno_z_modes"] = settings["n_modes"][0]
    return arguments


def checkpoint_model_settings(configuration: ModelConfig) -> dict[str, Any]:
    """Preserve the checkpoint conventions for inactive and direct-linear options."""
    settings = configured_model_settings(configuration)
    if configuration.spatial_scheme != "2.5d":
        settings.update(
            z_extent=None,
            z_center=None,
            lateral_interlacing=1,
            z_kernel_size=None,
            z_mixing=None,
            planar_symmetry="none",
        )
    elif configuration.architecture == "linear" and configuration.z_mixing != "full":
        settings["z_mixing"] = "spectral"
    if configuration.spatial_scheme != "3d":
        settings.update(volume_interlacing=1, interlacing_training="full")
    if configuration.spatial_scheme not in {"2.5d", "3d"}:
        settings.update(
            spectral_symmetry="none", spectral_groups=1, metric_hidden_channels=16
        )
    return settings
