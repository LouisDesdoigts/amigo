"""AMIGO modelling tools; numerical and plotting modules load on demand."""

import importlib
import importlib.metadata

__version__ = importlib.metadata.version("amigo")

__all__ = [
    "core_models",
    "optical_models",
    "vis_models",
    "discos",
    "detector_models",
    "ramp_models",
    "read_models",
    "model_fits",
    "files",
    "optimisation",
    "stats",
    "pipelines",
    "plotting",
    "misc",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(f".{name}", __name__)
    globals()[name] = module
    return module
