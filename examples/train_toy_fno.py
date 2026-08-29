"""Train the learned FNO against the analytic planar Coulomb field operator."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mace_fno import (
    FNOFieldOperator,
    ParticleMeshEnergy,
    ParticleMeshLongRange,
    generate_planar_coulomb_fields,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--test-samples", type=int, default=48)
    parser.add_argument("--atoms", type=int, default=6)
    parser.add_argument("--grid", type=int, default=16)
    parser.add_argument("--max-mode", type=int, default=3)
    parser.add_argument("--hidden-channels", type=int, default=12)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument(
        "--architecture",
        choices=("linear", "nonlinear"),
        default="linear",
        help="Use the linear identification baseline or a nonlinear FNO",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Defaults to 3e-2 for linear and 3e-3 for nonlinear FNO",
    )
    parser.add_argument(
        "--gradient-weight",
        type=float,
        default=0.25,
        help="Weight of normalized periodic-gradient error in the field loss",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--checkpoint", type=Path)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
    return torch.device(requested)


def normalized_rmse(
    operator: FNOFieldOperator, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    with torch.no_grad():
        residual = (operator(inputs) - targets) / operator.output_scale
        return residual.square().mean().sqrt().item()


def relative_l2(
    operator: FNOFieldOperator, inputs: torch.Tensor, targets: torch.Tensor
) -> float:
    with torch.no_grad():
        residual_norm = torch.linalg.vector_norm(operator(inputs) - targets)
        target_norm = torch.linalg.vector_norm(targets)
        return (residual_norm / target_norm).item()


def relative_energy_mae(
    operator: FNOFieldOperator,
    densities: torch.Tensor,
    potentials: torch.Tensor,
    point_area: torch.Tensor,
) -> float:
    with torch.no_grad():
        predicted = operator(densities)
        predicted_energy = 0.5 * (densities * predicted).sum(dim=(1, 2, 3)) * point_area
        target_energy = 0.5 * (densities * potentials).sum(dim=(1, 2, 3)) * point_area
        return (
            (predicted_energy - target_energy).abs().mean()
            / target_energy.abs().mean()
        ).item()


def normalized_field_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    output_scale: torch.Tensor,
    gradient_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Combine normalized field MSE with a periodic H1-like gradient loss."""
    prediction_normalized = prediction / output_scale
    target_normalized = target / output_scale
    value_loss = (prediction_normalized - target_normalized).square().mean()

    def periodic_gradient(field: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        nx, ny = field.shape[-2:]
        derivative_x = 0.5 * nx * (
            torch.roll(field, shifts=-1, dims=-2)
            - torch.roll(field, shifts=1, dims=-2)
        )
        derivative_y = 0.5 * ny * (
            torch.roll(field, shifts=-1, dims=-1)
            - torch.roll(field, shifts=1, dims=-1)
        )
        return derivative_x, derivative_y

    prediction_dx, prediction_dy = periodic_gradient(prediction_normalized)
    target_dx, target_dy = periodic_gradient(target_normalized)
    gradient_error = (
        (prediction_dx - target_dx).square().mean()
        + (prediction_dy - target_dy).square().mean()
    )
    gradient_reference = (
        target_dx.square().mean() + target_dy.square().mean()
    ).detach().clamp_min(1.0e-12)
    gradient_loss = gradient_error / gradient_reference
    total = value_loss + gradient_weight * gradient_loss
    return total, value_loss, gradient_loss


def held_out_force_error(
    operator: FNOFieldOperator,
    cell: torch.Tensor,
    grid_shape: tuple[int, int],
    max_modes: tuple[int, int],
) -> float:
    positions = torch.tensor(
        (
            (1.37, 2.11, 0.0),
            (7.08, 8.63, 0.0),
            (4.29, 3.54, 0.0),
            (8.41, 1.22, 0.0),
        ),
        dtype=cell.dtype,
        device=cell.device,
        requires_grad=True,
    )
    charges = torch.tensor(
        (1.0, -0.4, -0.6, 0.0), dtype=cell.dtype, device=cell.device
    )
    analytic = ParticleMeshLongRange(grid_shape, max_modes=max_modes).to(cell.device)
    learned = ParticleMeshEnergy(grid_shape, operator).to(cell.device)

    reference_energy = analytic(positions, charges, cell)
    reference_force = -torch.autograd.grad(reference_energy, positions)[0]
    learned_energy = learned(positions, charges, cell)
    learned_force = -torch.autograd.grad(learned_energy, positions)[0]
    return (
        torch.linalg.vector_norm(learned_force - reference_force)
        / torch.linalg.vector_norm(reference_force)
    ).item()


def main() -> None:
    args = parse_arguments()
    if args.steps < 1 or args.samples < 1 or args.test_samples < 1:
        raise ValueError("steps, samples, and test-samples must be positive")

    torch.manual_seed(args.seed)
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    device = choose_device(args.device)
    dtype = torch.float32
    grid_shape = (args.grid, args.grid)
    max_modes = (args.max_mode, args.max_mode)
    fno_modes = (args.max_mode + 1, args.max_mode + 1)
    cell = torch.diag(
        torch.tensor((10.0, 12.0, 20.0), device=device, dtype=dtype)
    )

    train_density, train_potential = generate_planar_coulomb_fields(
        args.samples,
        args.atoms,
        1,
        cell,
        grid_shape,
        max_modes,
        seed=args.seed,
    )
    test_density, test_potential = generate_planar_coulomb_fields(
        args.test_samples,
        args.atoms,
        1,
        cell,
        grid_shape,
        max_modes,
        seed=args.seed + 1,
    )

    operator = FNOFieldOperator(
        channels=1,
        n_modes=fno_modes,
        hidden_channels=args.hidden_channels,
        n_layers=args.layers,
        architecture=args.architecture,
    ).to(device)
    operator.fit_normalization(train_density, train_potential)
    learning_rate = args.learning_rate
    if learning_rate is None:
        learning_rate = 3.0e-2 if args.architecture == "linear" else 3.0e-3
    optimizer = torch.optim.Adam(operator.parameters(), lr=learning_rate)

    initial_rmse = normalized_rmse(operator, test_density, test_potential)
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed + 2)
    for step in range(args.steps):
        indices = torch.randint(
            args.samples,
            (min(args.batch_size, args.samples),),
            generator=generator,
            device=device,
        )
        density_batch = train_density[indices]
        potential_batch = train_potential[indices]
        prediction = operator(density_batch)
        loss, value_loss, gradient_loss = normalized_field_loss(
            prediction,
            potential_batch,
            operator.output_scale,
            args.gradient_weight,
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        if step == 0 or (step + 1) % max(1, args.steps // 5) == 0:
            print(
                f"step {step + 1:4d}/{args.steps}: "
                f"loss={loss.item():.6e}, value={value_loss.item():.6e}, "
                f"gradient={gradient_loss.item():.6e}"
            )

    final_rmse = normalized_rmse(operator, test_density, test_potential)
    test_relative_l2 = relative_l2(operator, test_density, test_potential)
    point_area = cell[0, 0] * cell[1, 1] / (args.grid * args.grid)
    energy_error = relative_energy_mae(
        operator, test_density, test_potential, point_area
    )
    force_error = held_out_force_error(operator, cell, grid_shape, max_modes)

    print(f"device:                    {device}")
    print(f"architecture:              {args.architecture}")
    print(f"learning rate:             {learning_rate:.6e}")
    print(f"initial normalized RMSE:   {initial_rmse:.6e}")
    print(f"final normalized RMSE:     {final_rmse:.6e}")
    print(f"test relative potential L2:{test_relative_l2: .6e}")
    print(f"test relative energy MAE:  {energy_error: .6e}")
    print(f"held-out relative force L2:{force_error: .6e}")

    if args.checkpoint is not None:
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": operator.state_dict(),
                "grid_shape": grid_shape,
                "max_modes": max_modes,
                "fno_modes": fno_modes,
                "architecture": args.architecture,
                "hidden_channels": args.hidden_channels,
                "n_layers": args.layers,
                "cell": cell.detach().cpu(),
                "seed": args.seed,
            },
            args.checkpoint,
        )
        print(f"checkpoint:                {args.checkpoint}")


if __name__ == "__main__":
    main()
