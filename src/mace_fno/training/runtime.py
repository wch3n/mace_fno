"""Runtime helpers shared by training and audit commands."""

from __future__ import annotations

from time import perf_counter

import torch


def choose_device(requested: str) -> torch.device:
    """Resolve an explicit or automatic PyTorch device request."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def elapsed_since(start: float, device: torch.device) -> float:
    """Return a stage wall time after completing queued CUDA work."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return perf_counter() - start
