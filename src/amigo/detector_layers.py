import jax
import jax.numpy as np
import dLux as dl
import dLuxWebbpsf as dlw
import dLux.utils as dlu
import pkg_resources as pkg
from jax import vmap
from jax.scipy.signal import convolve
from jax import Array
from jax.scipy.stats import multivariate_normal
from abc import abstractmethod

PSF = dl.psfs.PSF
DetectorLayer = dl.detector_layers.DetectorLayer


class SUB80Ramp(dl.detectors.LayeredDetector):

    def __init__(
        self,
        EDM=None,
        angle=-0.56126717,
        oversample=4,
        SRF=None,
        FF=None,
        # downsample=False,
        npixels_in=80,
        anisotropy=True,
        jitter=True,
        dark_current=0.0,
        ipc=True,
        one_on_fs=np.zeros((2, 80, 2)),
    ):
        layers = [("rotate", Rotate(angle))]

        if anisotropy:
            compression = np.array([0.99580676, 1.00343162])
            anisotropy = PixelAnisotropy(order=3).set("compression", compression)
            layers.append(("anisotropy", anisotropy))

        if jitter:
            layers.append(("jitter", GaussianJitter(2.5e-7, kernel_size=19, kernel_oversample=3)))

        # Load the FF
        if FF is None:
            file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.npy")
            FF = np.load(file_path)
            if npixels_in != 80:
                pad = (npixels_in - 80) // 2
                FF = np.pad(FF, pad, constant_values=1)

        if SRF is None:
            SRF = np.ones((oversample, oversample))

        layers.append(("sensitivity", ApplySensitivities(FF, SRF)))

        if EDM is None:
            EDM = NullEDM(oversample)
        layers.append(("EDM", EDM))

        if ipc:
            file_path = pkg.resource_filename(__name__, "data/SUB80_ipc.npy")
            ipc = np.load(file_path)
        else:
            ipc = np.array([[1.0]])

        layers.append(("read", DarkCurrent(dark_current)))
        layers.append(("IPC", IPC(ipc)))
        layers.append(("amplifier", Amplifier(one_on_fs)))

        self.layers = dlu.list2dictionary(layers, ordered=True)

    def __getattr__(self, key: str):
        if key in self.layers.keys():
            return self.layers[key]
        for layer in list(self.layers.values()):
            if hasattr(layer, key):
                return getattr(layer, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


# Amplifier/ramp modelling
def model_amplifier(coeffs, axis=0):
    """
    Models the amplifier noise as a polynomial along one axis of the detector.
    Assumes Detector is square and coeffs has shape (npix, order + 1).
    """

    def read_fn(coeffs):
        # Evaluation function
        xs = np.linspace(-1, 1, coeffs.shape[0])
        eval_fn = lambda coeffs: np.polyval(coeffs, xs)

        # Vectorise over each column
        vals = vmap(eval_fn, 0)(coeffs)

        if axis == 0:
            return np.rot90(vals)
        return vals

    # vmap over each group
    return vmap(read_fn)(coeffs)


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def model_dark_current(dark_current, ngroups):
    """Models the dark current as a constant background value added cumulatively to
    each group. For now we assume that the dark current is a float."""
    return (dark_current * (np.arange(ngroups) + 1))[..., None, None]


class ApplySensitivities(dl.layers.detector_layers.DetectorLayer):

    FF: jax.Array
    SRF: jax.Array

    def __init__(
        self,
        FF,
        SRF,
    ):
        self.FF = FF
        self.SRF = SRF

    @property
    def sensitivity_map(self):
        oversample = self.SRF.shape[0]
        npix = self.FF.shape[1]
        bc_sens_map = self.SRF[None, :, None, :] * self.FF[:, None, :, None]
        return bc_sens_map.reshape((npix * oversample, npix * oversample))

    def apply(self, PSF):
        return PSF * self.sensitivity_map


class Rotate(dl.layers.detector_layers.DetectorLayer):
    """
    Applies cubic spline interpolator for rotation of the PSF
    """

    angle: float

    def __init__(self, angle):
        self.angle = angle

    def apply(self, PSF):
        psf = dlw.utils.rotate(PSF.data, dlu.deg2rad(self.angle), order=3)
        return PSF.set("data", psf)


class BaseJitter(dl.layers.detector_layers.DetectorLayer):
    """
    Base class for jitter layers. Apply method applies a convolution.
    """

    kernel_size: int

    def __init__(self: DetectorLayer, kernel_size: int):
        """
        Constructor for the BaseJitter class.

        Parameters
        ----------
        kernel_size : odd
            The size of the convolution kernel in pixels to use.
        """
        self.kernel_size = kernel_size
        super().__init__()

    def apply(self: DetectorLayer, psf: PSF) -> PSF:
        """
        Applies the layer to the Image.

        Parameters
        ----------
        psf : PSF
            The PSF object to operate on.

        Returns
        -------
        psf : PSF
            The convolved PSF object.
        """
        kernel = self.generate_kernel(dlu.rad2arcsec(psf.pixel_scale))
        # kernel = self.generate_kernel(psf.pixel_scale)

        return psf.convolve(kernel)

    @abstractmethod
    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the convolution kernel to be applied.
        """
        pass


class GaussianJitter(BaseJitter):
    """
    Convolves the image with a multivariate Gaussian kernel parameterised by the
    magnitude, shear and angle.

    Attributes
    ----------
    kernel_size : int
        The size in pixels of the convolution kernel to use.
    r : float, arcseconds
        The magnitude of the jitter.
    shear : float
        The shear of the jitter.
    phi : float, degrees
        The angle of the jitter.
    kernel_size : int, odd
        The size of the convolution kernel in pixels to use.
    kernel_oversample : int
        The oversampling factor for the kernel generation.
    """

    r: float | Array
    shear: float | Array = None
    phi: float | Array = None
    kernel_oversample: int

    def __init__(
        self: BaseJitter,
        r: float | Array,
        shear: float | Array = 0,
        phi: float | Array = 0,
        kernel_size: int = 11,
        kernel_oversample: int = 1,
    ):
        """
        Constructor for the ApplyJitter class.

        Parameters
        ----------
        r : float
            The jitter magnitude, defined as the determinant of the covariance
            matrix of the multivariate Gaussian kernel. This is the product of the
            standard deviations of the minor and major axes of the kernel, given in
            arcseconds.
        shear : float, [0, 1)
            A measure of how asymmetric the jitter is. Defined as one minus the ratio between
            the standard deviations of the minor/major axes of the multivariate
            Gaussian kernel. It must lie on the interval [0, 1). A shear of 0
            corresponds to a symmetric jitter, while as shear approaches one the
            jitter kernel becomes more linear.
        phi : float
            The angle of the jitter in degrees.
        kernel_size : int = 10
            The size of the convolution kernel in pixels to use.
        """

        # checking for odd kernel size
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd integer")

        # checking shear is valid
        if shear >= 1 or shear < 0:
            raise ValueError("shear must lie on the interval [0, 1)")

        super().__init__(kernel_size)

        self.r = np.asarray(r, float)
        self.shear = np.asarray(shear, float)
        self.phi = np.asarray(phi, float)
        self.kernel_oversample = kernel_oversample

    @property
    def covariance_matrix(self):
        """
        Generates the covariance matrix for the multivariate normal distribution.

        Returns
        -------
        covariance_matrix : Array
            The covariance matrix.
        """
        # Compute the rotation angle
        rot_angle = np.radians(self.phi)

        # Construct the rotation matrix
        R = np.array(
            [
                [np.cos(rot_angle), -np.sin(rot_angle)],
                [np.sin(rot_angle), np.cos(rot_angle)],
            ]
        )

        # calculating the eigenvalues (lambda1 > lambda2)
        lambda1 = (self.r / (1 - self.shear)) ** 0.25
        lambda2 = lambda1 * (1 - self.shear)

        # Construct the skew matrix
        base_matrix = np.array(
            [
                [lambda1**2, 0],
                [0, lambda2**2],
            ]
        )

        # Compute the covariance matrix
        covariance_matrix = np.dot(np.dot(R, base_matrix), R.T)

        return covariance_matrix

    def generate_kernel(self, pixel_scale: float) -> Array:
        """
        Generates the normalised multivariate Gaussian kernel.

        Parameters
        ----------
        pixel_scale : float
            The pixel scale of the image in arcseconds per pixel.

        Returns
        -------
        kernel : Array
            The normalised Gaussian kernel.
        """
        # Generate distribution
        extent = pixel_scale * self.kernel_size  # kernel size in arcseconds
        x = np.linspace(0, extent, self.kernel_oversample * self.kernel_size) - 0.5 * extent
        xs, ys = np.meshgrid(x, x)
        pos = np.dstack((xs, ys))

        kernel = dlu.downsample(
            multivariate_normal.pdf(pos, mean=np.array([0.0, 0.0]), cov=self.covariance_matrix),
            self.kernel_oversample,
        )

        return kernel / np.sum(kernel)


def arr2pix(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return pscale * (coords - shift)


def pix2arr(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return (coords / pscale) + shift


from dLuxWebbpsf.utils.interpolation import _map_coordinates


class PixelAnisotropy(dl.layers.detector_layers.DetectorLayer):
    transform: dl.CoordTransform
    order: int

    def __init__(self, order=3):
        self.transform = dl.CoordTransform(compression=np.ones(2))
        self.order = int(order)

    def __getattr__(self, key):
        if hasattr(self.transform, key):
            return getattr(self.transform, key)
        raise AttributeError(f"PixelAnisotropy has no attribute {key}")

    def apply(self, PSF):
        npix = PSF.data.shape[0]
        transformed = self.transform.apply(dlu.pixel_coords(npix, npix * PSF.pixel_scale))
        coords = np.roll(pix2arr(transformed, PSF.pixel_scale), 1, axis=0)
        interp_fn = lambda x: _map_coordinates(x, coords, order=3, mode="constant", cval=0.0)
        return PSF.set("data", interp_fn(PSF.data))


class Ramp(dl.PSF):
    pass


class DownsampleRamp(dl.detector_layers.Downsample):

    def apply(self, ramp):
        dsample_fn = lambda x: dlu.downsample(x, self.kernel_size, mean=False)
        return ramp.set("data", vmap(dsample_fn)(ramp))


class EmptyLayer(dl.detector_layers.Downsample):

    def apply(self, ramp):
        return ramp


class IPC(dl.detector_layers.DetectorLayer):
    ipc: Array

    def __init__(self, ipc):
        self.ipc = np.array(ipc, float)

    def apply(self, ramp):
        conv_fn = lambda x: convolve(x, self.ipc, mode="same")
        return ramp.set("data", vmap(conv_fn)(ramp.data))


class Amplifier(dl.detector_layers.DetectorLayer):
    one_on_fs: Array

    def __init__(self, one_on_fs):
        self.one_on_fs = np.array(one_on_fs, float)

    def apply(self, ramp):
        return ramp.add("data", model_amplifier(self.one_on_fs))


class DarkCurrent(dl.detector_layers.DetectorLayer):
    dark_current: Array

    def __init__(self, dark_current):
        self.dark_current = np.array(dark_current, float)

    def apply(self, ramp):
        dark_current = model_dark_current(self.dark_current, len(ramp.data))
        return ramp.add("data", dark_current)


class BuildRamp(dl.detector_layers.DetectorLayer):
    ngroups: int

    def __init__(self, ngroups):
        self.ngroups = int(ngroups)

    def apply(self, psf):
        return Ramp(model_ramp(psf.data, self.ngroups), psf.pixel_scale)


class NullEDM(dl.detector_layers.DetectorLayer):
    downsample: int
    ngroups: int
    flux: float
    filter: str

    def __init__(self, downsample=4, ngroups=2, flux=1.0, filter="F430M"):
        self.ngroups = int(ngroups)
        self.flux = np.asarray(flux, float)
        self.downsample = int(downsample)
        self.filter = str(filter)

    def apply(self, psf):
        downsampled = dlu.downsample(psf.data * self.flux, self.downsample, mean=False)
        return Ramp(model_ramp(downsampled, self.ngroups), psf.pixel_scale)


# class BleedModel(dl.detector_layers.DetectorLayer):
#     pass
