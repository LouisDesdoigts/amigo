import jax
import equinox as eqx
import zodiax as zdx
import jax.numpy as np
import jax.nn as nn
import jax.random as jr
import jax.tree as jtu
import dLux as dl
import dLux.utils as dlu
from jax import vmap
from jax.lax import dynamic_slice as dyn_slice
from .optical_models import gen_powers, distort_coords
from .core_models import NNWrapper
from .misc import interp_ramp


class Ramp(dl.PSF):
    pass


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def quadratic_SRF(a, oversample, norm=True):
    """
    norm will normalise the SRF to have a mean of 1
    """
    coords = dlu.pixel_coords(oversample, 2)
    # quad = 1 - np.sum((a * coords) ** 2, axis=0)
    # quad = 1 - a * np.sum(coords**2, axis=0)
    quad = 1 - a * np.hypot(*coords)
    if norm:
        quad -= quad.mean() - 1
    return quad


def broadcast_subpixel(pixels, subpixel):
    npix = pixels.shape[1]
    oversample = subpixel.shape[0]
    bc_sens_map = subpixel[None, :, None, :] * pixels[:, None, :, None]
    return bc_sens_map.reshape((npix * oversample, npix * oversample))


class PixelSensitivity(zdx.Base):
    FF: jax.Array
    SRF: jax.Array

    def __init__(self, FF=np.ones((80, 80)), SRF=0.1):
        self.FF = np.array(FF, float)
        self.SRF = np.array(SRF, float)

    @property
    def sensitivity(self):
        """Return the oversampled (240, 240) pixel sensitivities"""
        return broadcast_subpixel(self.FF, quadratic_SRF(self.SRF, 3))


def to_edges(box):
    # Format: [center_x, center_y, length]
    return np.array(
        [
            [box[0] - box[-1] / 2, box[0] + box[-1] / 2],  # x: [left, right]
            [box[1] - box[-1] / 2, box[1] + box[-1] / 2],  # y: [bottom, top]
        ]
    )


def calc_overlap(small, large):
    # Edges format: [left, right] or [bottom, top]
    return np.maximum(0.0, np.minimum(small[1], large[1]) - np.maximum(small[0], large[0]))


# def overlap_fraction(large_box, small_box):
#     # Compute the edges of the small and large squares.
#     small_edges = to_edges(small_box)
#     large_edges = to_edges(large_box)

#     # Compute the overlapping length in the x and y directions.
#     overlap_x = calc_overlap(small_edges[0], large_edges[0])
#     overlap_y = calc_overlap(small_edges[1], large_edges[1])

#     # Calculate the overlapping area and return the fraction relative to the small square's area.
#     overlap_area = overlap_x * overlap_y
#     small_area = small_box[-1] ** 2
#     return overlap_area / small_area


# def kernels_to_array(oversampled_array):
#     npix, _, n, _ = oversampled_array.shape
#     return oversampled_array.transpose(0, 2, 1, 3).reshape(npix * n, npix * n)


# def array_to_kernels(full_res_array, npix, n):
#     return full_res_array.reshape(npix, n, npix, n).transpose(0, 2, 1, 3)


def fill_array(outer, inner):
    n = (len(outer) - len(inner)) // 2
    return outer.at[n:-n, n:-n].set(inner)


def overlap_fn(cen, size=1 / 3):
    large_box = np.array([0.0, 0.0, 1.0])
    small_box = np.array([cen[0], cen[1], size])

    # Compute the edges of the small and large squares.
    small_edges = to_edges(small_box)
    large_edges = to_edges(large_box)

    # Compute the overlapping length in the x and y directions.
    overlap_x = calc_overlap(small_edges[0], large_edges[0])
    overlap_y = calc_overlap(small_edges[1], large_edges[1])

    # Calculate the overlapping area and return the fraction relative to the small square's area.
    overlap_area = overlap_x * overlap_y
    small_area = small_box[-1] ** 2
    return overlap_area / small_area


# def calc_kernels(coords):
#     # coords shape: (npix, npix, 2, k_size, k_szie)
#     shape = coords.shape
#     npix, oversample = shape[0], shape[-1]

#     #
#     full_coords = vmap(kernels_to_array, 2)(coords)  # (n*npix, n*npix)

#     # Create an empty coordinates array to pad
#     empty_coords = np.tile(dlu.pixel_coords(3, 1), (npix + 2, npix + 2))
#     padded = vmap(fill_array)(empty_coords, full_coords)

#     # Define the convolution function
#     k_size = 3  # Hard set the kernel size for now
#     n = k_size * oversample

#     # cens = dlu.pixel_coords(npix + 1, npix + 1)
#     rel_cen = dlu.pixel_coords(3, 3)
#     # ones = np.ones((3, 3, 2))
#     # rel_cen_kerns = ones[..., None, None] * rel_cen[None, None, ...]
#     # rel_cen_im = vmap(kernels_to_array, 2)(rel_cen_kerns)

#     def kern_fn(i, j):
#         # Get the grid of neighbouring coordinates
#         coords_window = dyn_slice(padded, (0, i, j), (2, n, n))
#         coords_kerns = vmap(array_to_kernels, (0, None, None))(coords_window, 3, k_size)
#         box_coord_kerns = coords_kerns + rel_cen[..., None, None]

#         box_coords = vmap(kernels_to_array)(box_coord_kerns)
#         box_coords_vec = box_coords.reshape(2, -1).T
#         fractions = vmap(overlap_fn)(box_coords_vec)
#         return fractions.reshape(n, n)

#     # Apply the convolution
#     indices = k_size * np.indices((npix, npix)).reshape(2, -1)
#     return vmap(kern_fn)(*indices).reshape(npix, npix, n, n)


def apply_kernels_stride(illuminance, kernels, stride=3):
    """
    Convolves the Illuminance with the kernels.

    Kernels should have shape (k, k, 80, 80)
    Illuminance should have shape (240, 240)
    Their size ratio should be stride
    k needs to at least the oversample factor
    """
    shape = kernels.shape
    ksize, npix = shape[0], shape[-1]
    k = ksize // stride

    # Assume illuminance has shape (npix, npix)
    padding = ((k, k), (k, k))
    illum = np.pad(illuminance, padding, mode="constant")

    # Get the kernel vector
    kernels_vec = kernels.reshape(ksize, ksize, -1)

    # Define the convolution function
    def conv_fn(i, j, kernel):
        return np.sum(dyn_slice(illum, (i, j), (ksize, ksize)) * kernel)

    # Apply the convolution
    indices = stride * np.indices((npix, npix)).reshape(2, -1)
    # print(indices.shape)

    convd_vec = vmap(conv_fn, (-1, -1, -1))(*indices, kernels_vec)
    return convd_vec.reshape(shape[2:])


def get_spatial_kernels(full=True, normalize=True):
    """
    Returns 3x3 2D gradient kernels: axis-aligned and optionally diagonal.
    First-order gradients are restricted to affect only the central column/row.
    Diagonals affect corners only.

    Args:
        full (bool): Include diagonal first-order gradients if True.
        normalize (bool): Normalize central differences (e.g., 0.5 instead of 1 step).

    Returns:
        dict[str, np.ndarray]: Gradient kernel dictionary.
    """
    s = 0.5 if normalize else 1.0

    # First-order axis-aligned
    kernels = {
        # Central
        "central": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        # Central x-gradient: only center row, center column
        "central_x": np.array(
            [
                [0.0, 0.0, 0.0],
                [-s, 0.0, s],
                [0.0, 0.0, 0.0],
            ]
        ),
        # Central y-gradient: only center column, center row
        "central_y": np.array(
            [
                [0.0, -s, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, s, 0.0],
            ]
        ),
        # Forward x-gradient
        "forward_x": np.array(
            [
                [0.0, 0.0, 0.0],
                [-1.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        # Backward x-gradient
        "backward_x": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        # Forward y-gradient
        "forward_y": np.array(
            [
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        # Backward y-gradient
        "backward_y": np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        # Second-order x-gradient: central row only
        "second_xx": np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, -2.0, 1.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        # Second-order y-gradient: central column only
        "second_yy": np.array(
            [
                [0.0, 1.0, 0.0],
                [0.0, -2.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        # Laplacian: standard 5-point stencil
        "laplacian": np.array(
            [
                [0.0, 1.0, 0.0],
                [1.0, -4.0, 1.0],
                [0.0, 1.0, 0.0],
            ]
        ),
    }

    if full:
        # Diagonal central gradients
        kernels.update(
            {
                "central_diag_se": np.array(
                    [
                        [s, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, -s],
                    ]
                ),
                "central_diag_sw": np.array(
                    [
                        [0.0, 0.0, s],
                        [0.0, 0.0, 0.0],
                        [-s, 0.0, 0.0],
                    ]
                ),
                "forward_diag_se": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
                "backward_diag_se": np.array(
                    [
                        [1.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
                "forward_diag_sw": np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [0.0, -1.0, 0.0],
                        [1.0, 0.0, 0.0],
                    ]
                ),
                "backward_diag_sw": np.array(
                    [
                        [0.0, 0.0, 1.0],
                        [0.0, -1.0, 0.0],
                        [0.0, 0.0, 0.0],
                    ]
                ),
            }
        )

    return kernels


def array_to_kernels(array, npix, k_size):
    reshaped = array.reshape(npix, k_size, npix, k_size)
    return np.transpose(reshaped, (1, 3, 0, 2))


def kernels_to_array(kernels):
    k_size, _, npix, _ = kernels.shape
    permuted = np.transpose(kernels, (2, 0, 3, 1))
    return permuted.reshape(npix * k_size, npix * k_size)


def calc_kernels(coords, sensitivity):
    # coords shape: (2, k_size, k_size, npix, npix)
    shape = coords.shape
    oversample, npix = shape[1], shape[-1]
    full_coords = vmap(kernels_to_array)(coords)

    # Create an empty coordinates array to pad
    empty_coords = np.tile(dlu.pixel_coords(3, 1), (npix + 2, npix + 2))
    padded = vmap(fill_array)(empty_coords, full_coords)

    # Define the convolution function
    k_size = 3  # Hard set the kernel size for now
    n = k_size * oversample  # (9, 9)

    # TODO: Cache this guy?
    rel_cen = dlu.pixel_coords(3, 3)

    # Sensitivity is a 2D array of shape (npix * ksize, npix * ksize)
    n_pad = sensitivity.shape[0] + 2 * k_size
    ones = np.ones((n_pad, n_pad))
    padded_sens = ones.at[k_size:-k_size, k_size:-k_size].set(sensitivity)

    def kern_fn(i, j):
        coords_window = dyn_slice(padded, (0, i, j), (2, n, n))
        coords_kerns = vmap(array_to_kernels, (0, None, None))(coords_window, 3, k_size)
        box_coord_kerns = coords_kerns + rel_cen[:, None, None, ...]
        box_coords = vmap(kernels_to_array)(box_coord_kerns)
        box_coords_vec = box_coords.reshape(2, -1).T
        fractions = vmap(overlap_fn)(box_coords_vec).reshape(n, n)
        return fractions * dyn_slice(padded_sens, (i, j), (n, n))

    # Apply the convolution
    indices = k_size * np.indices((npix, npix)).reshape(2, -1)
    kernels_vec = vmap(kern_fn, out_axes=-1)(*indices)
    return kernels_vec.reshape(n, n, npix, npix)


class PolyKernelModel(zdx.Base):
    knots: jax.Array
    powers: jax.Array
    spatial_encoder: eqx.nn.Conv2d
    bleed_encoder: NNWrapper

    def __init__(self, key=jr.key(0)):
        self.knots = dlu.pixel_coords(3, 1)

        # Coordinate distortion set up
        self.knots = dlu.pixel_coords(3, 1)
        self.powers = np.array(gen_powers(3))
        n_features = np.array(self.powers).size

        # Default convolutional layers
        def Conv2d(in_channels=1, out_channels=1, key=jr.key(0)):
            return eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                key=key,
                use_bias=False,
            )

        # Spatial gradient encoder
        key, subkey = jr.split(key, 2)
        kernels = np.array(jtu.leaves(get_spatial_kernels()))[:, None]
        spatial_encoder = Conv2d(in_channels=1, out_channels=16, key=subkey)
        spatial_encoder = eqx.tree_at(lambda conv: conv.weight, spatial_encoder, kernels)
        layers = [eqx.nn.Lambda(lambda x: x[None]), spatial_encoder]
        self.spatial_encoder = eqx.nn.Sequential(layers)

        # Convolution layers: Feature extraction from charge/bias distribution
        keys = jr.split(key, 3)
        layers = [
            Conv2d(in_channels=16, out_channels=16, key=keys[0]),
            eqx.nn.Lambda(nn.relu),
            Conv2d(in_channels=16, out_channels=16, key=keys[1]),
            eqx.nn.Lambda(nn.relu),
            Conv2d(in_channels=16, out_channels=16, key=keys[1]),
            eqx.nn.Lambda(nn.relu),
            Conv2d(in_channels=16, out_channels=n_features, key=keys[2]),
        ]

        # Construct the encoder
        self.bleed_encoder = NNWrapper(eqx.nn.Sequential(layers))

    def __getattr__(self, key):
        if hasattr(self.bleed_encoder, key):
            return getattr(self.bleed_encoder, key)
        raise AttributeError(f"KernelModel has no attribute {key}")

    def predict_coords(self, charge):
        """Predicts the distorted coordinates."""
        # Predict the distortion coefficients
        spatial_features = self.spatial_encoder(charge)
        coeffs = self.bleed_encoder(spatial_features)
        coeffs_vec = coeffs.reshape(len(coeffs), -1)  # (n_coeffs, 6400)

        # Calculate the distorted coordinates
        distort_fn = lambda coeffs: distort_coords(
            self.knots, coeffs.reshape(self.powers.shape), self.powers
        )
        return vmap(distort_fn, -1, -1)(coeffs_vec).reshape(2, 3, 3, *charge.shape)

    def predict_kernels(self, charge, sensitivity):
        """Predict spatially adaptive transposed convolution kernels."""
        coords = self.predict_coords(charge)
        kernels = calc_kernels(coords, sensitivity)
        return kernels

    def __call__(self, charge, sensitivity):
        return self.predict_kernels(charge, sensitivity)


class NonLinearRamp(zdx.Base):
    """ "Dynamic Filter Recurrent Neural Network (DFRNN) to model charge diffusion and
    bleeding."""

    norm: int
    bleed: bool
    time_steps: int = eqx.field(static=True)
    use_charge: bool
    ff_model: PixelSensitivity
    kernel_model: PolyKernelModel

    def __init__(
        self,
        key=jr.key(0),
        time_steps=8,
        norm=2**15,
        SRF=0.1,
        use_charge=True,
        bleed=True,
    ):
        self.norm = norm
        self.bleed = bleed
        self.time_steps = time_steps
        self.use_charge = use_charge
        self.kernel_model = PolyKernelModel(key=key)
        self.ff_model = PixelSensitivity(SRF=SRF)

    def __getattr__(self, key):
        if hasattr(self.kernel_model, key):
            return getattr(self.kernel_model, key)
        elif hasattr(self.ff_model, key):
            return getattr(self.ff_model, key)
        raise AttributeError(f"NonLinearRamp has no attribute {key}")

    def predict_kernels(self, charge):
        # NOTE: This is just for visualisation purposes, and preform redundant calcs
        # it should not be used during normal operation
        return self.kernel_model(charge, self.ff_model.sensitivity)

    def predict_coords(self, charge):
        # NOTE: This is just for visualisation purposes, and preform redundant calcs
        # it should not be used during normal operation
        return self.kernel_model(charge, self.ff_model.sensitivity)

    def build_ramp(self, illuminance, charge):
        # Normalise by the time-steps and apply the sensitivity
        illuminance /= self.time_steps
        sensitivity = self.ff_model.sensitivity

        # Dont use the bias if requested
        if not self.use_charge:
            charge = np.zeros_like(charge)

        # Evolve the charge
        charges = [charge]
        if self.bleed:
            for _ in range(self.time_steps):
                kernels = self.kernel_model(charge, sensitivity)
                charge += apply_kernels_stride(illuminance, kernels)
                charges.append(charge)
        else:
            illum = dlu.downsample(illuminance * sensitivity, 3)
            charges = [illum for _ in range(self.time_steps)]

        if self.use_charge:
            return np.array(charges)
        return np.array(charges) + charge  # NOTE THIS IS CURRENTLY BROKEN

    def evolve_illuminance(self, illuminance, charge, ngroups):
        # Normalise the Illuminance and charge
        illuminance = illuminance / self.norm
        charge = charge / self.norm

        # Evolve the charge
        ramp = self.build_ramp(illuminance, charge)
        return self.norm * interp_ramp(ramp, ngroups)
