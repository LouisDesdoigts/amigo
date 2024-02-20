<<<<<<< HEAD
=======
import pkg_resources
>>>>>>> 0bb49b827597d30514134394c09f4b356c03b780
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
from .files import prep_data, get_wss_ops
<<<<<<< HEAD
from astropy.io import fits
import pkg_resources as pkg
from .files import full_to_SUB80
=======

>>>>>>> 0bb49b827597d30514134394c09f4b356c03b780

class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be 
    modelled.

    """
    data: Array # Make this static too?
    covariance: Array = eqx.field(static=True)
    support: Array = eqx.field(static=True)
    opd: Array = eqx.field(static=True)
    nints: int = eqx.field(static=True)
    ngroups: int = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    star: str = eqx.field(static=True)
    key: str = eqx.field(static=True)

    # def __init__(self, data, covariance, support, file):
    def __init__(self, file, read_noise=None, opd=None, key_fn=None):

        if key_fn is None:
            key_fn = lambda file: "_".join(file[0].header["FILENAME"].split("_")[:3])

        if opd is None:
            opd = get_wss_ops([file])[0]

        if read_noise is None:
<<<<<<< HEAD
            file_path = pkg.resource_filename(__name__, "data/SUB80_readnoise.py")
            read_noise = np.load(file_path)
=======
            read_noise_path = "data/jwst_niriss_readnoise_0005.fits"
            file_path = pkg.resource_filename(__name__, read_noise_path)
            read_noise = full_to_SUB80(np.asarray(fits.open(file_path)[1].data, float))
>>>>>>> 0bb49b827597d30514134394c09f4b356c03b780
        
        data, covariance, support = prep_data(file, read_noise=read_noise)

        self.nints = file[0].header["NINTS"]
        self.ngroups = file[0].header["NGROUPS"]
        self.filter = file[0].header["FILTER"]
        self.star = file[0].header["TARGPROP"]
        self.data = data
        self.covariance = covariance
        self.support = support
        self.key = key_fn(file)
        self.opd = opd
    
    @property
    def summary(self):
        return (
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"Filter {self.filter}\n"
            f"nints {self.nints}\n"
            f"ngroups {self.ngroups}\n"
        )

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
        normalise=True,
        psf_npixels=80,
        oversample=4,
        pixel_scale = 0.065524085,
        diameter = 6.603464,
        wf_npixels = 1024,
    ):
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale

        # Get the primary mirror transmission
        file_path = pkg.resource_filename(__name__, 'data/primary.npy')
        transmission = np.load(file_path)
        # Create the primary
        primary = dlw.JWSTAberratedPrimary(
            transmission,
            opd=np.zeros_like(transmission),
            radial_orders=radial_orders,
            AMI=True,
        )

        # Load the values into the primary
        file_path = pkg.resource_filename(__name__, 'data/FDA_coeffs.npy')

        if opd is None:
            opd = np.zeros_like(transmission)
        primary = primary.set("opd", opd)
        primary = primary.multiply("basis", 1e-9)  # Normalise to nm
        primary = primary.set("coefficients", np.load(file_path))

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

from astropy.io import fits
import pkg_resources as pkg
from .files import full_to_SUB80

class SUB80Ramp(dl.detectors.LayeredDetector):
    def __init__(
        self,
        angle=-0.56126717,
        oversample=4,
        SRF=None,
        FF=None,
        downsample=False,
        npixels_in=80,

    ):
        # Load the FF
        if FF is None:
<<<<<<< HEAD
            file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.fits")
            FF = np.load(file_path)
            if npixels_in != 80:
                pad = (npixels_in - 80) // 2
                SUB80 = np.pad(FF, pad, constant_values=1)
            return SUB80
            # FF = full_to_SUB80(full_FF, npix_out=npixels_in, fill=1.)
=======
            file_path = pkg.resource_filename(__name__, "data/jwst_niriss_flat_0277.fits")
            full_FF = np.asarray(fits.open(file_path)[1].data, float)
            FF = full_to_SUB80(full_FF, npix_out=npixels_in, fill=1.)
>>>>>>> 0bb49b827597d30514134394c09f4b356c03b780
        
        if SRF is None:
            SRF = np.ones((oversample, oversample))

        layers = [
            ("rotate", Rotate(angle)),
            ("sensitivity", ApplySensitivities(FF, SRF)),
        ]

        if downsample:
            layers.append(("downsample", dl.Downsample(oversample)))

        self.layers = dlu.list2dictionary(layers, ordered=True)



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
