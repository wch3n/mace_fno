"""Include every trainable extended-LES readout in the pinned MACE optimizer."""

import json
import logging
from pathlib import Path

import torch


def install(output_dir):
    from mace.tools import scripts_utils

    output_dir = Path(output_dir)
    original_options = scripts_utils.get_params_options
    original_optimizer = scripts_utils.get_optimizer
    extra_parameters = {}

    def options(args, model):
        result = original_options(args, model)
        covered = set()
        for group in result["params"]:
            group["params"] = list(group["params"])
            for parameter in group["params"]:
                assert id(parameter) not in covered, "Duplicate optimizer parameter"
                covered.add(id(parameter))
        missing = {
            name: p
            for name, p in model.named_parameters()
            if p.requires_grad and id(p) not in covered
        }
        allowed = (
            "les_u_readouts",
            "les_quad_1o_readouts",
            "les_quad_2e_readouts",
            "les_kappa_readouts",
            "les_alpha_readouts",
            "les_alpha_1o_readouts",
            "les_alpha_2e_readouts",
        )
        assert all(name.split(".")[0] in allowed for name in missing), list(missing)
        for prefix in allowed:
            selected = {
                name: p for name, p in missing.items() if name.split(".")[0] == prefix
            }
            if selected:
                result["params"].append(
                    dict(
                        name=prefix,
                        params=list(selected.values()),
                        weight_decay=0.0,
                        lr=args.lr,
                    )
                )
                extra_parameters.update(selected)
        ids = [id(p) for g in result["params"] for p in g["params"]]
        assert len(ids) == len(set(ids))
        assert all(id(p) in set(ids) for p in model.parameters() if p.requires_grad)
        record = dict(
            all_trainable_parameters_covered=True,
            total_trainable_parameters=sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            formerly_omitted_parameters={
                name: p.numel() for name, p in missing.items()
            },
            formerly_omitted_count=sum(p.numel() for p in missing.values()),
            groups=[
                dict(name=g["name"], parameters=sum(p.numel() for p in g["params"]))
                for g in result["params"]
            ],
        )
        (output_dir / "optimizer_coverage.json").write_text(
            json.dumps(record, indent=2) + "\n"
        )
        logging.info(
            "Optimizer includes all trainable parameters; added %d extended-LES parameters",
            record["formerly_omitted_count"],
        )
        return result

    def optimizer(args, param_options):
        result = original_optimizer(args, param_options)
        before = {}
        complete = False

        def pre_step(opt, args, kwargs):
            if not complete:
                before.update(
                    {name: p.detach().clone() for name, p in extra_parameters.items()}
                )

        def post_step(opt, args, kwargs):
            nonlocal complete
            if complete:
                return
            report = {}
            for name, p in extra_parameters.items():
                assert p.grad is not None and torch.isfinite(p.grad).all(), name
                change = float((p.detach() - before[name]).abs().max())
                report[name] = dict(
                    max_abs_update=change, gradient_max_abs=float(p.grad.abs().max())
                )
            (output_dir / "first_response_updates.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
            before.clear()
            complete = True

        result.register_step_pre_hook(pre_step)
        result.register_step_post_hook(post_step)
        return result

    scripts_utils.get_params_options = options
    scripts_utils.get_optimizer = optimizer
