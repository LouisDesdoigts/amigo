"""
This code is taken wholesale from Max Charles on a branch of dLuxToliman, but has been 
placed here for temporary simplicity.
"""

from jax.scipy.stats import multivariate_normal
from abc import abstractmethod
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from jax import Array


DetectorLayer = dl.detector_layers.DetectorLayer
PSF = dl.psfs.PSF


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
