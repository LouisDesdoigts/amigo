import importlib.metadata

__version__ = importlib.metadata.version("amigo")

from . import core
from . import BFE
from . import CNN
from . import files
from . import FIM
from . import fitting
from . import interferometry
from . import misc
from . import optical_layers
from . import detector_layers
from . import stats

__all__ = [
    core,
    BFE,
    CNN,
    files,
    FIM,
    fitting,
    interferometry,
    misc,
    optical_layers,
    detector_layers,
    stats,
]
