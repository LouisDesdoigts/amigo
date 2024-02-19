import dLux as dl
import dLux.utils as dlu
import itertools
import jax.numpy as np
from jax import vmap
from jax.lax import dynamic_slice
import jax.tree_util as jtu
import equinox as eqx


def triu_indices(N, order=2):
    """
    Generates the upper triangle indices of an n-dimensional array.

    Args:
        arr (ndarray): The input array.

    Returns:
        tuple: A tuple of n arrays containing the indices of the upper triangle.
    """
    # Get the shape of the array
    shape = (N,) * order

    # Generate all the combinations of indices
    indices = itertools.product(*[range(s) for s in shape])

    # Filter the indices to keep only the upper triangle
    upper_triangle_indices = filter(
        lambda x: all(x[i] <= x[i + 1] for i in range(len(x) - 1)), indices
    )

    # Convert the filtered indices to arrays
    triangle_indices = zip(*upper_triangle_indices)
    triangle_indices = [np.array(indices) for indices in triangle_indices]

    return np.array(tuple(triangle_indices))


vmap2d_im = lambda fn: vmap(vmap(fn, (0,)), (1,))

# This allows for vmapping across a 2d array using indices. This is useful here as
# a way to deal with edge effect or to deal with oversampling, ie pad the array and
# then vmap using _indexes_, or to only vmap over the non-oversampled pixel basis.
vmap2d_inds = lambda fn: vmap(vmap(fn, (0, None)), (None, 0))


def base_inds(ksize, oversample):
    """
    We need a way to index the bleeding kernels in a way that it spatially coherent,
    since each pixel is bled into by all of the kernels it is part of. This function
    builds this spatially coherent indexes that can be summed over later.

    The output should have shape (N, 4), where N is ksize^2. Each vector of shape (4,)
    has the first two values as the _image (or array) indexes_ and the last two values
    are the _kernel indexes_.

    A pixels final values can be found by summing the bleeding kernel values at
    these indexes.
    """
    vals = np.arange(ksize)
    array_inds = np.array([np.tile(vals, ksize), np.repeat(vals, ksize)]).T
    kernel_inds = oversample * np.flipud(array_inds)
    return np.concatenate([array_inds, kernel_inds], 1)


def shift_inds(inds, i, j):
    """
    Shifts the indexes that are returned from the `base_inds` function to (i, j), ie
    the pixel that we want to find the final bleeding value of.
    """
    return inds + np.array([i, j, 0, 0])[None, :]


def build_pixel(kernels, inds, i, j, oversample):
    """
    Builds the final value of an oversampled pixel located at (i, j) by summing the
    bleeding kernels that contribute to it.
    This should return an (oversample, oversample) shaped array
    """
    size = (1, 1, oversample, oversample)
    slice_fn = lambda arr, idx: np.squeeze(dynamic_slice(arr, idx, size))
    return vmap(slice_fn, (None, 0))(kernels, shift_inds(inds, i, j)).sum(0)


def get_basis_inds(ksize, order):
    """
    Builds the of pixel basis vectors of a specific order
    """
    if order == 0:
        return np.ones(ksize**2).astype(int)
    return triu_indices(ksize**2, order)


def kernel_basis(pixel_kernel, inds):
    vals = np.array([pixel_kernel[ind] for ind in inds])
    if inds.ndim == 1:
        # No need to multiply for 1d coefficients, its just indexing
        return vals
    return np.prod(vals, axis=0)


def kernel_bleeding(basis, coeffs):
    if coeffs.ndim == 1:
        # No basis dot here here since zeroth order term basis is just ones
        bleed_vec = coeffs
    else:
        bleed_vec = vmap(lambda c: np.dot(basis, c))(coeffs)

    # Subtract the mean to conserve charge
    return bleed_vec - bleed_vec.mean()


def build_bleed_kernels(pixel_kernels, coeffs, inds):
    """ """
    # Build the basis, shape (npix, npix, nbasis)
    basis_fn = lambda kern: kernel_basis(kern, inds)
    basis = vmap2d_im(basis_fn)(pixel_kernels)

    # Calculate bleeding, shape (npix, npix, nbasis)
    bleed_fn = lambda basis: kernel_bleeding(basis, coeffs)
    return vmap2d_im(bleed_fn)(basis)


def calc_bleeding(bleed_kernels, npix, ksize, oversample):
    # Build the indexes and function to calculate final pixel values
    inds = base_inds(ksize, oversample)
    pixel_fn = lambda i, j: build_pixel(bleed_kernels, inds, i, j, oversample)

    # Build the final pixel values
    # I still dont _fully_ understand why this needs the -1. Without it the output
    # is shifted by one pixel.
    bleeding = vmap2d_inds(pixel_fn)(*2 * (np.arange(npix) - 1,))

    # Oversample == 1 is a special case and does not need to be transposed/reshaped
    if oversample != 1:
        # Thank you Chat-GPT for this transpose trick, don't ask me why it works
        bleeding = bleeding.transpose(0, 2, 1, 3).reshape(2 * (npix * oversample))

    return bleeding


def get_pixel_kernel(arr, i, j, ksize):
    """
    Dynamically returns a kernel of size (ksize, ksize) centred on the pixel (i,j).
    ksize is kernel size, and must be odd.
    """
    k = ksize // 2
    return dynamic_slice(arr, (i - k, j - k), (ksize, ksize))


def build_pixel_kernels(image, npix, k, oksize, oversample):
    # Build the pixels kernels
    padded_im = np.pad(image, (k, k))
    kern_fn = lambda i, j: get_pixel_kernel(padded_im, i, j, oksize)
    return vmap2d_inds(kern_fn)(*2 * (oversample * np.arange(npix) + k,))
    # kernels should now have shape (npix, npix, oksize, oksize)


def apply_BFE(
    image,
    coeffs,
    ksize,
    inds,
    oversample,
    return_bleed_kernels=False,
    return_bleed=False,
):
    """
    Applies the Brighter-Fatter effect model to a image.

    Coefficients should be a dictionary of arrays, with the keys as the integer order of
    the respective coefficients.

    ksize determines the range of the bleeding kernel, in the non oversampled pixel
    basis. It must be odd.

    order is the polynomial order of the model. It must be >= 1, but is strongly
    recommended to be 2.

    oversample is the oversampling factor of the pixels. It must be odd and can be 1
    for non-oversampled pixels.

    bleed_kernels is a flag that, if True, will return the bleeding kernels instead of
    the charge-bled image.
    """
    # Check inputs
    if ksize % 2 != 1:
        raise ValueError("ksize must be odd")
    if oversample % 2 != 1:
        raise ValueError("oversample must be odd")

    # Shapes and sizes
    npix = image.shape[0] // oversample
    oksize = ksize * oversample
    k = (oksize - 1) // 2

    # Build the pixel kernels, shape (npix, npix, nbasis)
    pixel_kernels = build_pixel_kernels(image, npix, k, oksize, oversample)
    pixel_kernels = pixel_kernels.reshape(npix, npix, oksize**2)

    # Build the bleeding kernels, shape (npix, npix, nbasis)
    bleed_fn = lambda coeffs, inds: build_bleed_kernels(pixel_kernels, coeffs, inds)
    bleed_kernels = jtu.tree_map(bleed_fn, coeffs, inds)

    # Reshape from (npix, npix, nbasis) to (npix, npix, oksize, oksize)
    reshape_fn = lambda kern: kern.reshape(npix, npix, oksize, oksize)
    bleed_kernels = jtu.tree_map(reshape_fn, bleed_kernels)

    # Return the bleeding kernels if requested (as a dict)
    if return_bleed_kernels:
        return bleed_kernels

    # Calculate final pixel bleeding values
    image_bleed_fn = lambda kern: calc_bleeding(kern, npix, ksize, oversample)
    bleeding = jtu.tree_map(image_bleed_fn, bleed_kernels)

    # Return just the bleeding if requested (as a dict)
    if return_bleed:
        return bleeding

    # Calculate the final bled image (as an array)
    return image + np.array(jtu.tree_leaves(bleeding)).sum(0)


def map_to_str(order):
    if order < 0 or order > 5:
        raise ValueError("Order must be between 0 and 5")
    order_dict = {
        0: "constant",
        1: "linear",
        2: "quadratic",
        3: "cubic",
        4: "quartic",
        5: "quintic",
        # Add more if needed
    }
    return order_dict.get(order, "unknown")


class PolyBFE(dl.detector_layers.DetectorLayer):
    ksize: int
    oversample: int
    inds: dict[int] = eqx.field(static=True)
    coeffs: dict

    def __init__(self, ksize, oversample, orders):
        self.ksize = int(ksize)
        # self.orders = list(orders)
        self.oversample = int(oversample)

        # Check inputs
        if self.ksize % 2 != 1:
            raise ValueError("ksize must be odd")

        if self.oversample % 2 != 1:
            raise ValueError("oversample must be odd")

        # Set up coefficients
        oksize = self.ksize * self.oversample
        N = oksize**2

        coeffs = {}
        for order in orders:
            if order == 0:
                coeffs = {map_to_str(0): np.zeros((oksize**2))}
            else:
                N_coeffs = triu_indices(oksize**2, order).shape[1]
                coeffs[map_to_str(order)] = np.zeros((N, N_coeffs))
        self.coeffs = coeffs

        # Pre calculate the basis indexes
        self.inds = {
            map_to_str(order): get_basis_inds(oksize, order) for order in orders
        }
    
    def __getattr__(self, key):
        if key in self.coeffs:
            return self.coeffs[key]
        raise AttributeError(f"Attribute {key} not found")

    def apply(self, PSF):
        new_data = apply_BFE(
            PSF.data, self.coeffs, self.ksize, self.inds, self.oversample
        )
        return PSF.set("data", new_data)

    def apply_array(self, image):
        image = dlu.downsample(image, self.oversample, mean=False)
        return apply_BFE(image, self.coeffs, self.ksize, self.inds, self.oversample)
