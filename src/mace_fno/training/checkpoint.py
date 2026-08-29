"""Residual-only checkpoint helpers."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from ..coupling import MACEFNOResidual


def residual_state_dict(model: MACEFNOResidual) -> dict[str, torch.Tensor]:
    """Return learned state without duplicating the frozen MACE checkpoint."""
    prefix = "backbone.mace_model."
    return {
        key: value.detach().cpu()
        for key, value in model.state_dict().items()
        if not key.startswith(prefix)
    }


def load_residual_state_dict(
    model: MACEFNOResidual,
    state: Mapping[str, torch.Tensor],
) -> None:
    """Restore residual-only state while retaining the frozen backbone."""
    complete_state = model.state_dict()
    unexpected = set(state) - set(complete_state)
    if unexpected:
        raise KeyError(f"residual checkpoint contains unexpected keys: {sorted(unexpected)}")
    complete_state.update(
        {
            key: value.to(
                device=complete_state[key].device,
                dtype=complete_state[key].dtype,
            )
            for key, value in state.items()
        }
    )
    model.load_state_dict(complete_state)
