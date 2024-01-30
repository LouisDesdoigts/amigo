import equinox as eqx
import dLuxWebbpsf as dlw
from jax import Array
import zodiax as zdx
import jax.numpy as np
from jax import vmap
import dLux.utils as dlu
import dLux as dl
from optical_layers import DynamicAMI
from jax.scipy.stats import multivariate_normal as mvn
from detector_layers import Rotate, ApplyPRF
from stats import get_covariance_matrix


class Exposure(zdx.Base):
    data: Array
    support: Array = eqx.field(static=True)  # Make this explicit as an indexer
    read_noise: Array
    nints: int
    filter: str
    star: str
    key: str

    def __init__(self, data, support, read_noise, nints, filter, star, key):
        self.data = data
        self.support = support
        self.read_noise = read_noise
        self.nints = nints
        self.filter = filter
        self.star = star
        self.key = key

    @property
    def ngroups(self):
        return len(self.data)

    def to_vec(self, image):
        return image[..., *self.support]

    def covariance(self, total_bias):
        # This works on _image ramps_ and outputs _image ramped shaped cov mats_
        return get_covariance_matrix(self.data, total_bias, self.read_noise)

    def loglike_vec(self, ramp, total_bias):
        cov = self.nints * self.to_vec(self.covariance(total_bias))
        ramp_vec = self.nints * self.to_vec(ramp)
        data_vec = self.nints * self.to_vec(self.data)
        return vmap(mvn.logpdf, (-1, -1, -1))(ramp_vec, data_vec, cov)

    def loglike_im(self, ramp, total_bias):
        loglike_vec = self.loglike_vec(ramp, total_bias)
        return (np.nan * np.ones_like(ramp[0])).at[self.support].set(loglike_vec)


class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    def __init__(
        self,
        radial_orders=np.arange(6),
        pupil_mask=None,
        opd=None,
        normalise=False,
    ):
        self.wf_npixels = 1024
        self.diameter = 6.603464
        self.psf_npixels = 80
        self.oversample = 4
        self.psf_pixel_scale = 0.065524085

        # Get the primary mirror
        transmission = np.load("files/static/primary.npy")
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
        PRF=np.ones((4, 4)),
        FF=np.ones((80, 80)),
    ):
        self.layers = dlu.list2dictionary(
            [
                ("rotate", Rotate(angle)),
                ("PRF", ApplyPRF(FF, PRF)),
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
