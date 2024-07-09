import importlib.metadata

__version__ = importlib.metadata.version("amigo")

from . import core_models

# from . import detector_layers

from . import mask_models
from . import read_models
from . import detector_models
from . import jitter_models
from . import optical_models
from . import model_fits
from . import files
from . import fisher
from . import fitting
from . import interferometry
from . import modelling
from . import ramp_models
from . import pipelines
from . import plotting
from . import stats

__all__ = [
    core_models,
    # detector_layers,
    model_fits,
    mask_models,
    read_models,
    detector_models,
    files,
    fisher,
    fitting,
    interferometry,
    jitter_models,
    modelling,
    ramp_models,
    optical_models,
    pipelines,
    plotting,
    stats,
]
