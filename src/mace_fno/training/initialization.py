"""Initialization and staged-unfreezing policies for residual models."""

from __future__ import annotations

import torch

from ..coupling import MACEFNOResidual


def initialize_zero_residual(model: MACEFNOResidual) -> None:
    """Make the initial combined prediction exactly equal to MACE."""
    operator = model.long_range.field_operator
    projection = getattr(operator.fno, "projection_output", None)
    if projection is None:
        for parameter in operator.parameters():
            parameter.data.zero_()
    else:
        # Zeroing all factors of a stacked linear map also zeros its gradients.
        # Only the last projection must vanish to start at the MACE baseline.
        projection.weight.data.zero_()


def initialize_scaled_residual_output(
    model: MACEFNOResidual,
    scale: float,
) -> None:
    """Scale a stacked operator's output without freezing upstream layers."""
    if scale < 0.0:
        raise ValueError("output initialization scale must be non-negative")
    operator = model.long_range.field_operator
    projection = getattr(operator.fno, "projection_output", None)
    if projection is None:
        raise ValueError("scaled output initialization requires an output projection")
    projection.weight.data.mul_(float(scale))


def configure_output_projection_warmup(
    model: MACEFNOResidual,
    warmup_steps: int,
) -> tuple[list[torch.nn.Parameter], list[torch.nn.Parameter]]:
    """Freeze all but the final projection during an optional warm-up.

    The first returned list contains every parameter that was trainable before
    warm-up, allowing later unfreezing without rebuilding the optimizer. The
    second contains the projection parameters active during warm-up.
    """
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if warmup_steps == 0:
        return parameters, []
    operator = model.long_range.field_operator
    projection = getattr(operator.fno, "projection_output", None)
    if projection is None:
        raise ValueError("output-projection warm-up requires an output projection")
    projection_parameters = list(projection.parameters())
    if not projection_parameters:
        raise ValueError("the FNO has no output-projection parameters")
    projection_ids = {id(parameter) for parameter in projection_parameters}
    for parameter in parameters:
        parameter.requires_grad_(id(parameter) in projection_ids)
    return parameters, projection_parameters


def finish_output_projection_warmup(
    parameters: list[torch.nn.Parameter],
) -> None:
    """Unfreeze every residual parameter captured before warm-up."""
    for parameter in parameters:
        parameter.requires_grad_(True)
