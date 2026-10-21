"""Per-channel field normalization shared by the FNO adapters."""

from torch import Tensor


def _fit_channel_normalization(
    inputs: Tensor,
    targets: Tensor,
    input_scale: Tensor,
    output_scale: Tensor,
    *,
    minimum_scale: float,
) -> None:
    """Write RMS scales into existing buffers without changing their identity."""
    if inputs.ndim != input_scale.ndim or targets.shape != inputs.shape:
        shape = (
            "(batch, channels, nx, ny)"
            if input_scale.ndim == 4
            else "(batch, channels, nz, nx, ny)"
        )
        raise ValueError(f"inputs and targets must have matching {shape} shapes")
    dimensions = (0, *range(2, inputs.ndim))
    input_rms = inputs.square().mean(dim=dimensions, keepdim=True).sqrt()
    output_rms = targets.square().mean(dim=dimensions, keepdim=True).sqrt()
    input_scale.copy_(input_rms.clamp_min(minimum_scale))
    output_scale.copy_(output_rms.clamp_min(minimum_scale))
