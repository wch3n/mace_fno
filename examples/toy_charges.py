"""Evaluate the analytic particle-mesh prototype and its conservative forces."""

from __future__ import annotations

import torch

from mace_fno import ParticleMeshLongRange, direct_planar_coulomb_energy


torch.set_default_dtype(torch.float64)

cell = torch.diag(torch.tensor((10.0, 12.0, 20.0)))
positions = torch.tensor(
    ((1.37, 2.11, 0.0), (7.08, 8.63, 0.0), (4.29, 3.54, 0.0)),
    requires_grad=True,
)
charges = torch.tensor((1.0, -0.4, -0.6))
max_modes = (4, 4)

model = ParticleMeshLongRange((64, 64), max_modes=max_modes)
mesh_energy = model(positions, charges, cell)
forces = -torch.autograd.grad(mesh_energy, positions)[0]
reference_energy = direct_planar_coulomb_energy(
    positions.detach(), charges, cell, max_modes=max_modes
)

print(f"mesh energy:     {mesh_energy.item(): .10f}")
print(f"direct energy:   {reference_energy.item(): .10f}")
print(f"absolute error:  {(mesh_energy - reference_energy).abs().item(): .3e}")
print("forces:")
print(forces)

