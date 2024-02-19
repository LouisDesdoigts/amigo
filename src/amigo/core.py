import os
import equinox as eqx
import dLuxWebbpsf as dlw
from jax import Array
import zodiax as zdx
import jax.numpy as np
from jax import vmap
import dLux.utils as dlu
import dLux as dl
from jax.scipy.stats import multivariate_normal as mvn
from .detector_layers import Rotate, ApplySensitivities
from .optical_layers import DynamicAMI

# Get the directory of the current script for relative imports
current_dir = os.path.dirname(os.path.realpath(__file__))

class Exposure(zdx.Base):
    data: Array
    covariance: Array
    support: Array = eqx.field(static=True)  # Make this explicit as an indexer
    nints: int
    filter: str
    star: str
    key: str
    # TODO: Store per data set WFS measurements?

    def __init__(self, data, covariance, support, nints, filter, star, key):
        self.data = data
        self.covariance = covariance
        self.support = support
        self.nints = nints
        self.filter = filter
        self.star = star
        self.key = key

    @property
    def ngroups(self):
        return len(self.data)

    def to_vec(self, image):
        return image[..., *self.support]

    def loglike_vec(self, ramp):
        ramp_vec = self.nints * self.to_vec(ramp)
        data_vec = self.nints * self.to_vec(self.data)
        cov_vec = self.nints * self.to_vec(self.covariance)
        return vmap(mvn.logpdf, (-1, -1, -1))(ramp_vec, data_vec, cov_vec)

    def loglike_im(self, ramp):
        loglike_vec = self.loglike_vec(ramp)
        return (np.nan * np.ones_like(ramp[0])).at[self.support].set(loglike_vec)


class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    def __init__(
        self,
        radial_orders=np.arange(6),
        pupil_mask=None,
        opd=None,
        normalise=False,
        oversample=3,
    ):
        self.wf_npixels = 1024
        self.diameter = 6.603464
        self.psf_npixels = 80
        self.oversample = oversample
        self.psf_pixel_scale = 0.065524085

        # Get the primary mirror

        import pkg_resources

        # data = pkg_resources.resource_string(__name__, 'static/primary.npy')
        file_path = pkg_resources.resource_filename(__name__, 'src/amigo/data/primary.npy')
        transmission = np.load(file_path)
        # Construct the path to the file you want to load
        # primary_path = os.path.join(current_dir, 'relative/path/to/your/file')
        # primary_path = os.path.join(current_dir, "src/amigo/files/static/primary.npy")
        # transmission = np.load(primary_path)
        primary = dlw.JWSTAberratedPrimary(
            transmission,
            opd=np.zeros_like(transmission),
            radial_orders=radial_orders,
            AMI=True,
        )

        if opd is None:
            opd = np.zeros_like(transmission)
        primary = primary.set("opd", opd)
        primary = primary.multiply("basis", 1e-9)  # Normalise to nm

        if pupil_mask is None:
            pupil_mask = DynamicAMI(f2f=0.80, normalise=normalise)

        # Set the layers
        self.layers = dlu.list2dictionary(
            [
                ("pupil", primary),
                ("InvertY", dl.Flip(0)),
                ("pupil_mask", pupil_mask),
            ],
            ordered=True,
        )


class SUB80Ramp(dl.detectors.LayeredDetector):
    def __init__(
        self,
        angle=-0.56126717,
        SRF=np.ones((4, 4)),
        FF=np.ones((80, 80)),
        downsample=True,
    ):
        self.layers = dlu.list2dictionary(
            [
                ("rotate", Rotate(angle)),
                ("sensitivity", ApplySensitivities(FF, SRF, downsample=downsample)),
            ],
            ordered=False,
        )


class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )
