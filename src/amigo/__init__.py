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

# pipelines.py - requires other random shit to install, make this an optional install
# dependency so we dont have to fuck around with CRDS paths in the amigo install

# plotting.py is old