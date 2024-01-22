import equinox as eqx
import dLuxWebbpsf as dlw
from jax import Array
import zodiax as zdx
from sources import planck
import jax.numpy as np
from jax import vmap
import dLux.utils as dlu
import dLux as dl
import jax
from optical_layers import DynamicAMI

from jax import vmap
import zodiax as zdx
from jax.scipy.stats import multivariate_normal as mvn


def get_read_cov(read_noise, ngroups):
    # Bind the read noise function
    # (This is just an outer product?, find a friendlier syntax)
    identity = np.eye(ngroups)
    raw_read_fn = lambda i, j: identity * read_noise[i, j] ** 2
    read_fn = vmap(vmap(raw_read_fn, (0, None), (2)), (None, 0), (2))

    # Get the read noise covariance matrix
    pix_idx = np.arange(read_noise.shape[-1])
    return read_fn(pix_idx, pix_idx)


def build_covariance_matrix(electrons):
    # Can this jut be done by using this as an index?
    Is = np.arange(len(electrons))
    IJs = np.array(np.meshgrid(Is, Is))
    inds = np.min(IJs, (0))
    cov = vmap(vmap(lambda ind: electrons[ind], 0), 1)(inds)
    return cov


def get_covariance_matrix(data_ramp, bias, one_on_fs, read_noise):
    total_bias = bias[None, ...] + vmap(model_amplifier)(one_on_fs)
    # cov_mat_inds = build_covariance_matrix_inds(len(data_ramp))
    electron_cov = build_covariance_matrix(data_ramp - total_bias)
    read_cov = get_read_cov(read_noise, len(data_ramp))
    return electron_cov + read_cov


def get_loglike(psf_ramp, data_ramp, cov, support):
    eval_fn = vmap(mvn.logpdf, (-1, -1, -1))
    return eval_fn(
        psf_ramp[..., *support], data_ramp[..., *support], cov[..., *support]
    )


def get_loglike_vec(psf_ramp, data_ramp, bias, one_on_fs, read_noise, support, nints):
    cov = get_covariance_matrix(data_ramp, bias, one_on_fs, read_noise)
    # return get_loglike(psf_ramp, data_ramp, cov, support)
    return get_loglike(nints * psf_ramp, nints * data_ramp, nints * cov, support)


def get_loglike_im(psf_ramp, data, bias, one_on_fs, read_noise, support, nints):
    loglike_vec = get_loglike_vec(
        psf_ramp, data, bias, one_on_fs, read_noise, support, nints
    )
    return (np.nan * np.ones((80, 80))).at[support].set(loglike_vec)

def exposure_covariance(tel, exposure, read_noise):
    return get_covariance_matrix(
        exposure.data,
        tel.biases[exposure.key],
        tel.OneOnFs[exposure.key],
        read_noise,
    )

def exposure_loglike_vec(tel, exposure, read_noise):
    return get_loglike_vec(
        tel.model_exposure(exposure),
        exposure.data,
        tel.biases[exposure.key],
        tel.OneOnFs[exposure.key],
        read_noise,
        exposure.support,
        exposure.nints,
    )

def exposure_loglike_im(tel, exposure, read_noise):
    return get_loglike_im(
        tel.model_exposure(exposure),
        exposure.data,
        tel.biases[exposure.key],
        tel.OneOnFs[exposure.key],
        read_noise,
        exposure.support,
        exposure.nints,
    )


def model_amplifier(coeffs, axis=0):
    """
    Models the amplifier noise as a polynomial along one axis of the detector.
    Assumes Detector is square and coeffs has shape (npix, order + 1).
    """
    # Evaluation function
    xs = np.linspace(-1, 1, coeffs.shape[0])
    eval_fn = lambda coeffs: np.polyval(coeffs, xs)

    # Vectorise over columns and groups in the data
    vals = vmap(eval_fn, 0)(coeffs)

    if axis == 0:
        return np.rot90(vals)
    return vals


class Exposure(zdx.Base):
    data: Array
    support: Array = eqx.field(static=True)  # Make this explicit as an indexer
    nints: int
    filter: str
    star: str
    key: str

    def __init__(self, data, support, nints, filter, star, key, coeffs=None):
        self.data = data
        self.support = support
        self.nints = nints
        self.filter = filter
        self.star = star
        self.key = key

    @property
    def ngroups(self):
        return len(self.data)


class Star(zdx.Base):
    """
    For a consistent Teff/visibilities across all observations of a star
    """

    Teff: Array
    # visibilities: Array

    def __init__(self, Teff):
        self.Teff = np.asarray(Teff, float)
        # self.visibilities = visibilities

    def weights(self, wavels):
        return planck(wavels, self.Teff)


class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    def __init__(self, radial_orders=np.arange(6), normalise=False):
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

        primary = primary.set("opd", np.zeros((1024, 1024)))
        primary = primary.multiply("basis", 1e-9)  # Normalise to nm

        # Set the layers
        self.layers = dlu.list2dictionary(
            [
                ("pupil", primary),
                ("InvertY", dl.Flip(0)),
                ("pupil_mask", DynamicAMI(f2f=0.80, normalise=normalise)),
            ],
            ordered=True,
        )


class SUB80Ramp(dl.detectors.BaseDetector):
    ngroups: int
    oversample: int
    N: int  # Number of times to apply the BFE

    rotation: float  # deg
    PRF: jax.Array  # osamp x osamp
    FF: jax.Array  # npix x npix
    BFE: object  # BFE layer
    bias: jax.Array  # npix x npix
    OneOnF: object

    def __init__(
        self,
        ngroups,
        BFE,
        oversample=4,
        rotation=-0.56126717,
        PRF=np.ones((4, 4)),
        FF=np.ones((80, 80)),
        bias=np.zeros((80, 80)),
        N=1,
    ):
        self.ngroups = ngroups
        self.oversample = oversample
        self.rotation = rotation
        self.PRF = PRF
        self.FF = FF
        self.BFE = BFE
        self.bias = bias
        # self.OneOnF = AmplifierNoiseRamp(1) # Dummy ngroups
        self.OneOnF = np.zeros((ngroups, 80, 2))
        self.N = N

    def model_photons(self, PSF):
        # Apply rotation
        data = dlw.utils.rotate(PSF.data, dlu.deg2rad(self.rotation), order=3)

        # Apply intra-pixel sensitivities
        bc_shape = (80, self.oversample, 80, self.oversample)  # 'broadcast' shape
        prf_data = data.reshape(bc_shape) * self.PRF[None, :, None, :]

        # Re-cast back to PSF object so pixel scale is tracked through the downsample
        PSF = PSF.set("data", prf_data.reshape(data.shape)).downsample(self.oversample)

        # Now we can apply the inter-pixel sensitivities
        return PSF * self.FF

    def model(self, PSF, return_psf=False, return_electrons=False):
        if return_psf and return_electrons:
            raise ValueError("Cant return PSF and electrons")

        # For now just model full ramp and take the last group
        optical_PSF = self.model_photons(PSF)  # Un-grouped PSF
        electron_PSF = self.ramp_model(optical_PSF)
        if return_electrons:
            return electron_PSF.data

        read_PSF = self.model_read(electron_PSF)
        if return_psf:
            return read_PSF
        return read_PSF.data

    def ramp_model(self, PSF):
        """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
        should have shape (npix, npix) and return shape (ngroups, npix, npix)"""

        if PSF.data.ndim != 2:
            raise ValueError(f"Expected 2D PSF - got data shape: {PSF.data.shape}")
        lin_ramp = (np.arange(self.ngroups) + 1) / self.ngroups
        electrons = PSF.data[None, ...] * lin_ramp[..., None, None]

        # Now apply the BFE
        for i in range(self.N):
            electrons = eqx.filter_vmap(self.BFE.apply_array)(electrons)

        return PSF.set("data", electrons)

    def model_read(self, PSF):
        """Applies the read-noise model the PSF-ramp"""
        if PSF.data.ndim != 3:
            raise ValueError(f"Expected ramp data - got data shape: {PSF.data.shape}")

        # Add the pixel bias
        PSF += self.bias[None, ...]

        # One On F correction
        return PSF + vmap(model_amplifier)(self.OneOnF)


class Modeller(dl.Telescope):
    """
    Holds everything and deals with interfacing between them
    """

    # Foundation model parameters
    optics: object
    detector: object

    # Meta-data, optionally optimised (ie Teff, filter weights)
    stars: dict
    filters: dict

    # Per-exposure parameters
    positions: dict
    fluxes: dict
    aberrations: dict
    biases: dict
    OneOnFs: dict

    def __init__(
        self,
        optics,
        detector,
        stars,
        filters,
        positions,
        fluxes,
        aberrations,
        biases,
        OneOnFs,
    ):
        self.optics = optics
        self.detector = detector
        self.stars = stars
        self.filters = filters
        self.source = None  # Dummy source for now

        # These are unique to each exposure so are held in a dictionary
        self.positions = positions
        self.fluxes = fluxes
        self.aberrations = aberrations
        self.biases = biases
        self.OneOnFs = OneOnFs

    def model_exposure(self, exposure, return_electrons=False):
        # Get wavelengths and weights
        wavels, weights = self.filters[exposure.filter]
        weights *= self.stars[exposure.star].weights(wavels)
        weights /= weights.sum()

        # Get optical PSF (foundation NRM contained within optics)
        key = exposure.key
        pos = dlu.arcsec2rad(self.positions[key])
        optics = self.optics.set("coefficients", self.aberrations[key])
        PSF = optics.propagate(wavels, pos, weights, return_psf=True)
        PSF = PSF.multiply("data", self.fluxes[key])

        # Apply ramp detector
        detector = self.detector.set(
            ["bias", "OneOnF", "ngroups"],
            [self.biases[key], self.OneOnFs[key], exposure.ngroups],
        )
        # return detector.ramp_model(PSF)
        return detector.model(PSF, return_electrons=return_electrons)

    # def exposure_covariance(self, exposure, read_noise):
    #     return get_covariance_matrix(
    #         exposure.data,
    #         self.biases[exposure.key],
    #         self.OneOnFs[exposure.key],
    #         read_noise,
    #     )

    # def exposure_loglike_vec(self, exposure, read_noise):
    #     return get_loglike_vec(
    #         self.model_exposure(exposure),
    #         exposure.data,
    #         self.biases[exposure.key],
    #         self.OneOnFs[exposure.key],
    #         read_noise,
    #         exposure.support,
    #         exposure.nints,
    #     )

    # def exposure_loglike_im(self, exposure, read_noise):
    #     return get_loglike_im(
    #         self.model_exposure(exposure),
    #         exposure.data,
    #         self.biases[exposure.key],
    #         self.OneOnFs[exposure.key],
    #         read_noise,
    #         exposure.support,
    #         exposure.nints,
    #     )
