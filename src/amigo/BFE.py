import dLux as dl
import jax
import itertools
import jax.numpy as np
from jax import vmap


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


vmap2d_ij = lambda fn: vmap(vmap(fn, (0, None)), (None, 0))
vmap2d_im = lambda fn: vmap(vmap(fn, (0,)), (1,))


class PolyBFE(dl.layers.detector_layers.DetectorLayer):
    order: int
    ksize: int
    coeffs: dict

    def __init__(self, ksize, order=3):
        self.ksize = int(ksize)
        self.order = int(order)
        if self.ksize % 2 != 1:
            raise ValueError("ksize must be odd")

        N = self.ksize**2
        coeffs = {"0": np.zeros((N, N))}
        for i in range(self.order):
            N_coeffs = triu_indices(self.ksize**2, i + 1).shape[1]
            idx = str(i + 1)
            coeffs[idx] = np.zeros((N, N_coeffs))
        self.coeffs = coeffs

    @property
    def k(self):
        return self.ksize // 2

    def arr_kernel(self, arr, i, j):
        """Returns a square of size (ksize, ksize) centred on the pixel (i,j)."""
        padded = np.pad(arr, (self.k, self.k))
        start = (i - self.k, j - self.k)
        size = (self.ksize, self.ksize)
        return jax.lax.dynamic_slice(padded, start, size)

    def build_basis(self, arr_kernel):
        """Builds the polynomial basis from an (ksize, ksize) shaped array kernel.
        Note we only support order = 2 for now, and this is the function that would
        need to modified to support other orders."""

        # Get our data vector
        x = arr_kernel.flatten()
        components = [np.ones_like(x)]  # Zeroth order is just a constant

        # Iterate over orders and append the components
        for i in range(1, self.order + 1):
            inds = triu_indices(self.ksize**2, i)
            prods = np.prod(np.array([x[ind] for ind in inds]), axis=0)
            components.append(prods)
        return np.concatenate(components)

    def eval_pixel(self, basii, arr_kernels, i, j):
        """Dimensionality is hard - Here we want to evaluate the 'migration' kernel of
        a single pixel, given the polynomial basis for that pixel, and the coefficients
        for the charge migration. Given this we can vectorise this function over each
        pixel in the input array to get the full migration kernel. It is easier to
        vectorise over a 1d array, and our basis vectors/coefficients are given in 1d,
        so we evaluate it in 1d here, and reshape the output to (ksize, ksize) so we
        have the correct output dimensions to apply to the image. We can also apply the
        flux conservation easily here."""
        basis = basii[j, i]
        arr_kernel = arr_kernels[i, j]

        # migrated_charge = vmap(lambda c: np.dot(basis, c))(self.coeffs)
        migrated_charge = vmap(lambda c: np.dot(basis, c))(self.coeffs_vec)
        migrated_charge -= migrated_charge.mean()
        migrated_charge = migrated_charge.reshape((self.ksize, self.ksize))
        return arr_kernel + migrated_charge

    @property
    def coeffs_vec(self):
        arrs = []
        for i in range(self.order + 1):
            arrs.append(self.coeffs[str(i)])
        return np.concatenate(arrs, axis=1)
        # print(arr.shape)
        # keys = sorted(self.coeffs.keys())
        # print(keys)
        # return np.concatenate([c.flatten() for c in self.coeffs.values()])

    def build(self, array):
        # Build the array kernels - individual shape (ksize, ksize),
        # output shape (N, N, ksize, ksize). These are the array kernels that the
        # polynomials are applied to
        Is = np.arange(array.shape[0])
        arr_kernels = vmap2d_ij(lambda i, j: self.arr_kernel(array, i, j))(Is, Is)

        # Build the polynomial basis
        basis = vmap2d_im(self.build_basis)(arr_kernels)

        # Evaluate the polynomial basis
        eval_pixel_fn = lambda i, j: self.eval_pixel(basis, arr_kernels, i, j)
        out = vmap2d_ij(eval_pixel_fn)(Is, Is)

        # eval_fn = vmap(vmap(self.eval_pixel, (0, 0)), (1, 1))
        # out = eval_fn(basis, arr_kernels)
        return out

    # Get indexes
    def build_indexs(self, i, j):
        """Built this like a year ago, literally no idea what is happening here."""
        vals = np.arange(self.ksize)
        xs, ys = np.tile(vals, self.ksize), np.repeat(vals, self.ksize)
        out = np.array([xs, ys]).T
        inv_out = np.flipud(out)
        out_shift = out + np.array([i, j])
        return np.concatenate([out_shift, inv_out], 1)

    def build_and_mean(self, array, i, j):
        indexes = self.build_indexs(j, i)
        return vmap(lambda x, i: x[tuple(i)], in_axes=(None, 0))(array, indexes).mean()

    def apply_BFE(self, array):
        kernels = self.build(array)
        kernels = np.pad(kernels, ((self.k, self.k), (self.k, self.k), (0, 0), (0, 0)))

        # Bind the kernels to the function for a simple vmap signature
        mean_fn = lambda i, j: self.build_and_mean(kernels, i, j)

        # Take the mean of the migrated charge
        Is = np.arange(array.shape[0]) + 1
        return vmap2d_ij(mean_fn)(Is, Is)

    def apply(self, PSF):
        # Pad the input to avoid edge artifacts
        pad = 1 + self.k
        array = np.pad(PSF.data, (pad, pad))
        BFEd = self.apply_BFE(array)
        return PSF.set("data", BFEd[pad:-pad, pad:-pad])
