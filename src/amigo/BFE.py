import dLux as dl
import itertools
import jax.numpy as np
from jax import vmap
from jax.lax import dynamic_slice


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


def coeffs_to_vec(coeffs, order):
    """
    Turns the coefficient dictionary into a vector that can be applied to the basis.
    Assumes the coeffs input is a dictionary of arrays.
    """
    # Note we use list-comp rather than tree-map to ensure the order is correct
    return np.concatenate([coeffs[str(i)] for i in range(order + 1)], axis=1)


def kernel_basis(kernel, order):
    """
    Builds the set of 'basis vectors' of the polynomial,
    The basis vectors are the independent variables of the polynomial ie for equation:
    f(x) = a + bx + cx^2 + ..., this returns as a vector [1, x, x^2, ...], where x is
    the values of each individual pixel in the kernel.
    """
    # Get the data vector
    vals = kernel.flatten()

    # The zeroth order basis is just a constant
    basis = [np.ones_like(vals)]

    # Iterate over orders and append the components
    for i in range(1, order + 1):
        inds = triu_indices(len(vals), i)
        prods = np.prod(np.array([vals[ind] for ind in inds]), axis=0)
        basis.append(prods)
    return np.concatenate(basis)


def get_kernel(arr, i, j, ksize):
    """
    Dynamically returns a kernel of size (ksize, ksize) centred on the pixel (i,j).
    ksize is kernel size, and must be odd.
    """
    k = ksize // 2
    return dynamic_slice(arr, (i - k, j - k), (ksize, ksize))


def kernel_bleeding(basis, coeffs, ksize):
    """
    Calculate the bleeding of each pixel kernel, returning the kernel in its
    'original' shape.
    """
    bleed_vec = vmap(lambda c: np.dot(basis, c))(coeffs).reshape(ksize, ksize)
    return bleed_vec - bleed_vec.mean()  # Subtract the mean to conserve charge


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


def build_bleed_kernels(array, coeffs, ksize, order, oversample):
    """
    Builds the individual 'bleeding' kernels for each pixel, describing how the
    charge migrates between each pixel. Output is 4 dimensional, with shape
    (npix, npix, ksize * oversample, ksize * oversample).
    """
    # Array is oversampled
    npix = array.shape[0] // oversample

    # Build the bleeding kernels
    oksize = ksize * oversample
    k = (oksize - 1) // 2
    kern_fn = lambda i, j: get_kernel(np.pad(array, (k, k)), i, j, oksize)
    Is = oversample * np.arange(npix) + k
    kernels = vmap2d_inds(kern_fn)(Is, Is)
    # kernels should now have shape (npix, npix, oksize, oksize)

    # Build the polynomial basis
    basis = vmap2d_im(lambda kernel: kernel_basis(kernel, order))(kernels)
    # basis should now have shape (npix, npix, nbasis)

    # Calculate the bleeding
    bleed_fn = lambda basis: kernel_bleeding(basis, coeffs, oksize)
    bleeding = vmap2d_im(bleed_fn)(basis)
    # bleeding should now have shape (npix, npix, oksize, oksize)
    return bleeding


def build_pixel(kernels, inds, i, j, oversample):
    """
    Builds the final value of an oversampled pixel located at (i, j) by summing the
    bleeding kernels that contribute to it.
    This should return an (oversample, oversample) shaped array
    """
    size = (1, 1, oversample, oversample)
    slice_fn = lambda arr, idx: np.squeeze(dynamic_slice(arr, idx, size))
    return vmap(slice_fn, (None, 0))(kernels, shift_inds(inds, i, j)).sum(0)


def apply_BFE(
    image,
    coeffs,
    ksize,
    order,
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

    # Get the coefficients in a vectors
    coeffs_vec = coeffs_to_vec(coeffs, order)

    # Build the bleeding kernels
    bleed_kernels = build_bleed_kernels(image, coeffs_vec, ksize, order, oversample)

    # Return the bleeding kernels if requested
    if return_bleed_kernels:
        return bleed_kernels

    # Build the indexes and function to calculate final pixel values
    inds = base_inds(ksize, oversample)
    pixel_fn = lambda i, j: build_pixel(bleed_kernels, inds, i, j, oversample)

    # Build the final pixel values
    # I still dont _fully_ understand why this needs the -1. Without it the output
    # is shifted by one pixel.
    Is = np.arange(image.shape[0] // oversample) - 1
    bleeding = vmap2d_inds(pixel_fn)(Is, Is)

    # Oversample == 1 is a special case and does not need to be transposed/reshaped
    if oversample != 1:
        # Thank you Chat-GPT for this transpose trick, don't ask me why it works
        bleeding = bleeding.transpose(0, 2, 1, 3).reshape(image.shape)

    # Return just the bleeding if requested
    if return_bleed:
        return bleeding

    # Calculate the final bleed image
    return image + bleeding


class PolyBFE(dl.detector_layers.DetectorLayer):
    order: int
    ksize: int
    oversample: int
    coeffs: dict

    def __init__(self, ksize, oversample, order):
        self.ksize = int(ksize)
        self.order = int(order)
        self.oversample = int(oversample)

        # Check inputs
        if self.ksize % 2 != 1:
            raise ValueError("ksize must be odd")

        if self.oversample % 2 != 1:
            raise ValueError("oversample must be odd")

        # Doing the BFE
        oksize = self.ksize * self.oversample

        N = oksize**2
        coeffs = {"0": np.zeros((N, oksize**2))}
        for i in range(1, order + 1):
            N_coeffs = triu_indices(oksize**2, i).shape[1]
            coeffs[str(i)] = np.zeros((N, N_coeffs))

        self.coeffs = coeffs

    def apply(self, PSF):
        new_data = apply_BFE(
            PSF.data, self.coeffs, self.ksize, self.order, self.oversample
        )
        return PSF.set("data", new_data)

    def apply_array(self, image):
        return apply_BFE(image, self.coeffs, self.ksize, self.order, self.oversample)
