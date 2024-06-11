import importlib.metadata

__version__ = importlib.metadata.version("amigo")

from . import core
from . import detector_layers
from . import detectors
from . import files
from . import fisher
from . import fitting
from . import interferometry
from . import jitter
from . import modelling
from . import non_linear
from . import optics
from . import pipelines
from . import plotting
from . import stats

__all__ = [
    core,
    detector_layers,
    detectors,
    files,
    fisher,
    fitting,
    interferometry,
    jitter,
    modelling,
    non_linear,
    optics,
    pipelines,
    plotting,
    stats,
]
