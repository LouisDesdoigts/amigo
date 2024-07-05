import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
import equinox as eqx
import pkg_resources as pkg
from jax import Array, vmap
from dLux.utils.zernikes import scale_coords, eval_radial, eval_azimuthal


# This is here to add diameter as in input
def zernike_fast(n: int, m: int, c: Array, k: Array, coordinates: Array, diameter) -> Array:
    """
    Calculates the Zernike polynomial using the pre-calculated c and k parameters, such
    that this function is jittable.

    Parameters
    ----------
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.

    Returns
    -------
    zernike : Array
        The Zernike polynomial.
    """
    coordinates = scale_coords(coordinates, diameter / 2)
    polar_coordinates = dlu.cart2polar(coordinates)
    rho = polar_coordinates[0]
    theta = polar_coordinates[1]
    aperture = rho <= 1.0
    return aperture * eval_radial(rho, n, c, k) * eval_azimuthal(theta, n, m)


def polike_fast(
    nsides: int, n: int, m: int, c: Array, k: Array, coordinates: Array, diameter
) -> Array:
    """
    Calculates the Zernike polynomial on an n-sided aperture using the pre-calculated
    c and k parameters, such that this function is jittable.

    Parameters
    ----------
    nsides : int
        The number of sides of the aperture.
    n : int
        The radial order of the Zernike polynomial.
    m : int
        The azimuthal order of the Zernike polynomial.
    c : Array
        The normalisation coefficients of the Zernike polynomial.
    k : Array
        The powers of the Zernike polynomial.
    coordinates : Array
        The Cartesian coordinates to calculate the Zernike polynomial upon.

    Returns
    -------
    polike : Array
        The Zernike polynomial on an n-sided aperture.
    """
    if nsides < 3:
        raise ValueError(f"nsides must be >= 3, not {nsides}.")
    coordinates = scale_coords(coordinates, diameter / 2)
    alpha = np.pi / nsides
    phi = dlu.cart2polar(coordinates)[1] + alpha
    wedge = np.floor((phi + alpha) / (2.0 * alpha))
    u_alpha = phi - wedge * (2 * alpha)
    r_alpha = np.cos(alpha) / np.cos(u_alpha)
    return 1 / r_alpha * zernike_fast(n, m, c, k, coordinates / r_alpha)


def get_noll_indices(radial_orders: Array | list = None, noll_indices: Array | list = None):
    if radial_orders is not None:
        radial_orders = np.array(radial_orders)

        if (radial_orders < 0).any():
            raise ValueError("Radial orders must be >= 0")

        noll_indices = []
        for order in radial_orders:
            start = dlu.triangular_number(order)
            stop = dlu.triangular_number(order + 1)
            noll_indices.append(np.arange(start, stop) + 1)
        noll_indices = np.concatenate(noll_indices)

    elif noll_indices is None:
        raise ValueError("Must specify either radial_orders or noll_indices")

    if noll_indices is not None:
        noll_indices = np.array(noll_indices, dtype=int)

    return noll_indices


def gen_coords(npix, diameter, holes_positions, transformation=None):

    # No transformations applied
    if transformation is None:
        coords = dlu.pixel_coords(npix, diameter)
        return vmap(dlu.translate_coords, (None, 0))(coords, holes_positions)

    # We use the rotation transformation as a way to check if we are dealing with
    # a single transformation of a _vectorise_ transformation
    if transformation.rotation.ndim == 0:
        # Generate coordinates
        coords = dlu.pixel_coords(npix, diameter)

        # Apply transformations
        coords = dlu.translate_coords(coords, transformation.translation)
        coords = dlu.rotate_coords(coords, transformation.rotation)
        coords = dlu.compress_coords(coords, transformation.compression)
        coords = dlu.shear_coords(coords, transformation.shear)

        # Shift the coordinates for each hole
        return vmap(dlu.translate_coords, (None, 0))(coords, holes_positions)

    # We are dealing with a vectorised transformation so vectorise the operations
    if transformation.rotation.ndim == 1:
        # Generate coordinates
        make_coords = lambda x: dlu.pixel_coords(npix, diameter)
        coords = eqx.filter_vmap(make_coords)(np.arange(7))

        # Shift the coordinates for each hole
        coords = vmap(dlu.translate_coords, (0, 0))(coords, holes_positions)

        # Apply transformations
        coords = vmap(dlu.translate_coords, (0, 0))(coords, transformation.translation)
        coords = vmap(dlu.rotate_coords, (0, 0))(coords, transformation.rotation)
        coords = vmap(dlu.compress_coords, (0, 0))(coords, transformation.compression)
        return vmap(dlu.shear_coords, (0, 0))(coords, transformation.shear)

    raise ValueError("Invalid transformation")


def calc_mask(coords, f2f, pixel_scale):
    hex_fn = lambda coords: dlu.soft_reg_polygon(coords, f2f / np.sqrt(3), 6, pixel_scale)
    return vmap(hex_fn)(coords).sum(0)


def calc_basis(coords, f2f, radial_orders, oversize=1.1, polike=False):
    noll_inds = get_noll_indices(np.arange(radial_orders))

    if polike:
        basis_fn = lambda coords: dlu.polike_basis(
            6, noll_inds, coords, oversize * 2 * f2f / np.sqrt(3)
        )
    else:
        basis_fn = lambda coords: dlu.zernike_basis(
            noll_inds, coords, oversize * 2 * f2f / np.sqrt(3)
        )
    return vmap(basis_fn)(coords)


def calc_dynamic_factors(radial_orders):
    noll_inds = get_noll_indices(np.arange(radial_orders))
    values = {"c": [], "k": [], "n": [], "m": []}
    for ind in noll_inds:
        n, m = dlu.noll_indices(ind)
        c, k = dlu.zernike_factors(ind)
        values["n"].append(n)
        values["m"].append(m)
        values["c"].append(c)
        values["k"].append(k)
    return values


def get_initial_holes(diameter=6.603464, npixels=1024, x_shift=21, y_shift=-13):
    file_path = pkg.resource_filename(__name__, "data/AMI_holes.npy")
    # file_path = "../amigo/data/AMI_holes.npy"
    shift = np.array([x_shift, y_shift]) * (diameter / npixels)
    return np.load(file_path) + shift[None, :]


class StaticAMI(dl.layers.optical_layers.TransmissiveLayer):

    def __init__(
        self,
        holes=None,
        diameter=6.603464,
        npixels=1024,
        f2f=0.80,
        transformation=None,
        normalise=True,
    ):

        # Get the holes
        if holes is None:
            holes = get_initial_holes(diameter, npixels)

        # Generate coordinates and mask
        coords = gen_coords(npixels, diameter, holes, transformation)
        self.transmission = calc_mask(coords, f2f, diameter / npixels)
        self.normalise = bool(normalise)


class AberratedStaticAMI(StaticAMI):
    abb_basis: Array
    abb_coeffs: Array
    amp_basis: Array
    amp_coeffs: Array

    def __init__(
        self,
        holes=None,
        diameter=6.603464,
        npixels=1024,
        f2f=0.80,
        transformation=None,
        normalise=True,
        aberration_orders=None,
        amplitude_orders=None,
        oversize=1.1,
        polike=False,
    ):
        # Get the holes
        if holes is None:
            holes = get_initial_holes(diameter, npixels)

        # Generate coordinates and mask
        coords = gen_coords(npixels, diameter, holes, transformation)
        self.transmission = calc_mask(coords, f2f, diameter / npixels)
        self.normalise = bool(normalise)

        # Calculate the aberration basis functions
        if aberration_orders is not None:
            self.abb_basis = 1e-9 * calc_basis(coords, f2f, aberration_orders, oversize, polike)
            self.abb_coeffs = np.zeros(self.abb_basis.shape[:-2])
        else:
            self.abb_basis = None
            self.abb_coeffs = None

        # Calculate the amplitude basis functions
        if amplitude_orders is not None:
            self.amp_basis = calc_basis(coords, f2f, amplitude_orders, oversize, polike)
            self.amp_coeffs = np.zeros(self.amp_basis.shape[:-2])
        else:
            self.amp_basis = None
            self.amp_coeffs = None

    def calc_transmission(self):
        transmission = self.transmission
        if self.amp_basis is not None:
            transmission *= 1 + dlu.eval_basis(self.amp_basis, self.amp_coeffs)
        return transmission

    def calc_aberrations(self):
        if self.abb_basis is not None:
            return dlu.eval_basis(self.abb_basis, self.abb_coeffs)
        return np.zeros_like(self.transmission)

    def apply(self, wavefront):
        wavefront *= self.calc_transmission()
        wavefront += self.calc_aberrations()
        if self.normalise:
            return wavefront.normalise()
        return wavefront


class DynamicAMIStaticAbb(dl.layers.optical_layers.OpticalLayer):
    holes: Array
    f2f: Array
    transformation: dl.CoordTransform
    normalise: bool
    abb_basis: Array
    abb_coeffs: Array
    amp_basis: Array
    amp_coeffs: Array

    def __init__(
        self,
        holes=None,
        diameter=6.603464,
        npixels=1024,
        f2f=0.80,
        transformation=None,
        normalise=True,
        aberration_orders=None,
        amplitude_orders=None,
        oversize=1.1,
        polike=False,
        unique_holes=False,
    ):
        # Get the holes
        if holes is None:
            holes = get_initial_holes(diameter, npixels)
        self.holes = holes
        self.f2f = np.asarray(f2f, float)

        # Unique set of transformations per hole
        if unique_holes:
            make_transform = lambda x: dl.CoordTransform((0.0, 0.0), 0.0, (1.0, 1.0), (0.0, 0.0))
            self.transformation = eqx.filter_vmap(make_transform)(np.arange(7))

        # Single transformation for all holes
        else:
            self.transformation = dl.CoordTransform((0.0, 0.0), 0.0, (1.0, 1.0), (0.0, 0.0))

        # Normalisation
        self.normalise = bool(normalise)

        # Get the coords
        coords = gen_coords(npixels, diameter, self.holes, self.transformation)

        # Aberrations
        if aberration_orders is not None:
            self.abb_basis = 1e-9 * calc_basis(
                coords, self.f2f, aberration_orders, oversize, polike
            )
            self.abb_coeffs = np.zeros(self.abb_basis.shape[:-2])
        else:
            self.abb_basis = None
            self.abb_coeffs = None

        # Amplitudes
        if amplitude_orders is not None:
            self.amp_basis = calc_basis(coords, self.f2f, amplitude_orders, oversize, polike)
            self.amp_coeffs = np.zeros(self.amp_basis.shape[:-2])
        else:
            self.amp_basis = None
            self.amp_coeffs = None

    def calc_mask(self, npixels, diameter):
        coords = gen_coords(npixels, diameter, self.holes, self.transformation)
        return calc_mask(coords, self.f2f, diameter / npixels)

    def calc_transmission(self):
        if self.amp_basis is not None:
            return 1 + dlu.eval_basis(self.amp_basis, self.amp_coeffs)
        return 1.0

    def calc_aberrations(self):
        if self.abb_basis is not None:
            return dlu.eval_basis(self.abb_basis, self.abb_coeffs)
        return 0.0

    def calculate(self, npixels, diameter):
        mask = self.calc_mask(npixels, diameter)
        transmission = self.calc_transmission()
        aberrations = self.calc_aberrations()
        return mask, transmission, aberrations

    def apply(self, wavefront):
        mask, transmission, aberrations = self.calculate(wavefront.npixels, wavefront.diameter)
        wavefront *= mask * transmission
        wavefront += aberrations
        if self.normalise:
            return wavefront.normalise()
        return wavefront

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        else:
            raise AttributeError(f"Interpolator has no attribute {key}")


class DynamicAMIDynamicAbb(dl.layers.optical_layers.OpticalLayer):
    holes: Array
    f2f: Array
    transformation: dl.CoordTransform
    normalise: bool
    abb_factors: Array | dict
    abb_coeffs: Array
    amp_factors: Array | dict
    amp_coeffs: Array
    oversize: float
    polike: bool

    def __init__(
        self,
        holes=None,
        diameter=6.603464,
        npixels=1024,
        f2f=0.80,
        transformation=None,
        normalise=True,
        aberration_orders=None,
        amplitude_orders=None,
        oversize=1.1,
        polike=False,
        unique_holes=False,
    ):
        # Get the holes
        if holes is None:
            holes = get_initial_holes(diameter, npixels)
        self.holes = holes
        self.f2f = np.asarray(f2f, float)

        # Unique set of transformations per hole
        if unique_holes:
            make_transform = lambda x: dl.CoordTransform((0.0, 0.0), 0.0, (1.0, 1.0), (0.0, 0.0))
            self.transformation = eqx.filter_vmap(make_transform)(np.arange(7))

        # Single transformation for all holes
        else:
            self.transformation = dl.CoordTransform((0.0, 0.0), 0.0, (1.0, 1.0), (0.0, 0.0))

        # Normalisation
        self.normalise = bool(normalise)

        # Polike or not
        self.polike = bool(polike)

        # Oversize
        self.oversize = float(oversize)

        # Aberrations
        if aberration_orders is not None:
            self.abb_factors = calc_dynamic_factors(aberration_orders)
            self.abb_coeffs = np.zeros((7, len(self.abb_factors["n"])))
        else:
            self.abb_factors = None
            self.abb_coeffs = None

        # Amplitudes
        if amplitude_orders is not None:
            self.amp_factors = calc_dynamic_factors(amplitude_orders)
            self.amp_coeffs = np.zeros((7, len(self.amp_factors["n"])))
        else:
            self.amp_factors = None
            self.amp_coeffs = None

    def calc_mask(self, coords, npixels, diameter):
        return calc_mask(coords, self.f2f, diameter / npixels)

    def calc_transmission(self, coords, full_basis_fn):
        if self.amp_factors is not None:
            basis_fn = lambda coords: full_basis_fn(
                self.amp_factors["n"],
                self.amp_factors["m"],
                self.amp_factors["c"],
                self.amp_factors["k"],
                coords,
            )
            return 1 + dlu.eval_basis(vmap(basis_fn)(coords), self.amp_coeffs)
        return np.zeros_like(coords[-2:])

    def calc_aberrations(self, coords, full_basis_fn):
        if self.abb_factors is not None:
            basis_fn = lambda coords: full_basis_fn(
                self.amp_factors["n"],
                self.amp_factors["m"],
                self.amp_factors["c"],
                self.amp_factors["k"],
                coords,
            )
            return dlu.eval_basis(1e-9 * vmap(basis_fn)(coords), self.abb_coeffs)
        return np.zeros_like(coords[-2:])

    def calculate(self, npixels, diameter):

        # Get coordinates
        coords = gen_coords(npixels, diameter, self.holes, self.transformation)

        # Get mask
        mask = self.calc_mask(coords, npixels, diameter)

        # Get the zernike or polike fn
        if self.polike:
            zernike_fn = polike_fast
        else:
            zernike_fn = zernike_fast

        # Basis function, note we cant vmap this fn because c, and k have different
        # shapes for different zernikes
        full_basis_fn = lambda ns, ms, cs, ks, coords: np.array(
            [
                zernike_fn(n, m, c, k, coords, 2 * self.oversize * self.f2f / np.sqrt(3))
                for n, m, c, k in zip(ns, ms, cs, ks)
            ]
        )

        # Calculate the transmission
        transmission = self.calc_transmission(coords, full_basis_fn)

        # Calculate the aberrations
        aberrations = self.calc_aberrations(coords, full_basis_fn)

        return mask, transmission, aberrations

    def apply(self, wavefront):
        mask, transmission, aberrations = self.calculate(wavefront.npixels, wavefront.diameter)
        wavefront *= mask * transmission
        wavefront += aberrations
        if self.normalise:
            return wavefront.normalise()
        return wavefront

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        else:
            raise AttributeError(f"Interpolator has no attribute {key}")
