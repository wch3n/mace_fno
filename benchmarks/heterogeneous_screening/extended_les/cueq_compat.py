"""Preserve extended-LES settings through the pinned MACE backend converters.

The upstream extractor omits les_arguments and keep_last_layer_irreps for
MACELES. Patch only this process; installed MACE/LES packages stay unchanged.
"""

from copy import deepcopy
from functools import wraps


def _normalise_irreps(initializer):
    @wraps(initializer)
    def initialise(self, irreps_in, *args, **kwargs):
        # cuEquivariance Irreps are not iterable in e3nn's expected format.
        return initializer(self, str(irreps_in), *args, **kwargs)

    return initialise


def install(les_arguments):
    from mace.cli import convert_cueq_e3nn, convert_e3nn_cueq
    from mace.modules.blocks import LinearLesReadoutBlock, NonLinearLesReadoutBlock
    from mace.tools import scripts_utils

    original = scripts_utils.extract_config_mace_model
    arguments = deepcopy(les_arguments)
    for cls in (LinearLesReadoutBlock, NonLinearLesReadoutBlock):
        cls.__init__ = _normalise_irreps(cls.__init__)

    def extract(model):
        config = original(model)
        if model.__class__.__name__ == "MACELES":
            for attr, key in (
                ("use_dipoles", "use_dipole"),
                ("use_quads", "use_quad"),
                ("use_induced_charges", "use_induced_charge"),
                ("use_induced_dipoles", "use_induced_dipole"),
                ("use_anisotropic_polarizability", "use_anisotropic_polarizability"),
            ):
                assert getattr(model, attr) == arguments.get(key, False), (attr, key)
            config["les_arguments"] = deepcopy(arguments)
            config["keep_last_layer_irreps"] = True
        return config

    scripts_utils.extract_config_mace_model = extract
    convert_e3nn_cueq.extract_config_mace_model = extract
    convert_cueq_e3nn.extract_config_mace_model = extract
