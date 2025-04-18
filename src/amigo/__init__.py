import importlib.metadata

__version__ = importlib.metadata.version("amigo")

from . import core_models
from . import optical_models
from . import vis_models
from . import vis_analysis
from . import detector_models
from . import ramp_models
from . import read_models
from . import model_fits
from . import files
from . import fisher
from . import fitting
from . import stats
from . import pipelines
from . import plotting
from . import misc

__all__ = [
    core_models,
    optical_models,
    vis_models,
    vis_analysis,
    detector_models,
    ramp_models,
    read_models,
    model_fits,
    files,
    fisher,
    fitting,
    stats,
    pipelines,
    plotting,
    misc,
]
