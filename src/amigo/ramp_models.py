import jax
import zodiax as zdx
import jax.numpy as np
import jax.random as jr
import jax.nn as nn
import equinox as eqx
from jax import vmap
import dLux as dl
from jax.lax import dynamic_slice as dyn_slice
import dLux.utils as dlu
from .optical_models import gen_powers, distort_coords
from .misc import interp_ramp
from .core_models import build_wrapper, WrapperHolder


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
    quad = 1 - np.sum((a * coords) ** 2, axis=0)
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


def overlap_fraction(large_box, small_box):
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


def kernels_to_array(oversampled_array):
    npix, _, n, _ = oversampled_array.shape
    return oversampled_array.transpose(0, 2, 1, 3).reshape(npix * n, npix * n)


def array_to_kernels(full_res_array, npix, n):
    return full_res_array.reshape(npix, n, npix, n).transpose(0, 2, 1, 3)


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


def calc_kernels(coords):
    # coords shape: (npix, npix, 2, k_size, k_szie)
    shape = coords.shape
    npix, oversample = shape[0], shape[-1]

    #
    full_coords = vmap(kernels_to_array, 2)(coords)

    # Create an empty coordinates array to pad
    empty_coords = np.tile(dlu.pixel_coords(3, 1), (npix + 2, npix + 2))
    padded = vmap(fill_array)(empty_coords, full_coords)

    # Define the convolution function
    k_size = 3  # Hard set the kernel size for now
    n = k_size * oversample

    # cens = dlu.pixel_coords(npix + 1, npix + 1)
    rel_cen = dlu.pixel_coords(3, 3)
    # ones = np.ones((3, 3, 2))
    # rel_cen_kerns = ones[..., None, None] * rel_cen[None, None, ...]
    # rel_cen_im = vmap(kernels_to_array, 2)(rel_cen_kerns)

    def kern_fn(i, j):
        # Get the grid of neighbouring coordinates
        coords_window = dyn_slice(padded, (0, i, j), (2, n, n))
        coords_kerns = vmap(array_to_kernels, (0, None, None))(coords_window, 3, k_size)
        box_coord_kerns = coords_kerns + rel_cen[..., None, None]

        box_coords = vmap(kernels_to_array)(box_coord_kerns)
        box_coords_vec = box_coords.reshape(2, -1).T
        fractions = vmap(overlap_fn)(box_coords_vec)
        return fractions.reshape(n, n)

    # Apply the convolution
    indices = k_size * np.indices((npix, npix)).reshape(2, -1)
    return vmap(kern_fn)(*indices).reshape(npix, npix, n, n)


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


class DFRNN(eqx.Module):
    """ "Dynamic Filter Recurrent Neural Network (DFRNN) to model charge diffusion and
    bleeding."""

    kernel_model: None
    use_bias: bool
    time_steps: int = eqx.field(static=True)

    def __init__(self, key=jr.key(0), order=2, time_steps=8, use_bias=True):
        self.kernel_model = DFN(order=order, key=key)
        self.time_steps = time_steps
        self.use_bias = use_bias

    def __call__(self, bias, illuminance, sensitivity, bleed=True):
        # Normalise by the time-steps
        illuminance /= self.time_steps

        illum = dlu.downsample(sensitivity * illuminance, 3, mean=False)
        sensitivity = dlu.downsample(sensitivity, 3, mean=True)

        # Get the initial charge
        if self.use_bias:
            charge = bias - np.mean(bias)
        else:
            charge = np.zeros_like(bias)

        # Evolve the charge
        charges, diffusions = [charge], []
        for _ in range(self.time_steps):

            if bleed:
                # Calculate the transfer kernels
                norm_charge = charge - np.mean(charge)
                kernels = self.kernel_model(norm_charge[None]).T

                # Apply the kernels
                new_charge = sensitivity * apply_kernels_stride(illuminance, kernels)
                charge += new_charge

                diffusions.append(illum - new_charge)
            else:
                new_charge = sensitivity * dlu.downsample(illuminance, 3, mean=False)
                charge += new_charge
                diffusions.append(np.zeros_like(charge))

            # Add the new charge
            charges.append(charge)

        if self.use_bias:
            ramp = np.array(charges) + np.median(bias)
        else:
            ramp = np.array(charges) + bias
        return ramp, np.array(diffusions)


class DFN(eqx.Module):
    """
    Dynamic Filter Network (DFN) to predict transposed convolution kernels
    based on charge/bias distribution.
    """

    encoder: eqx.Module  # CNN to encode charge/bias distribution
    # dense: eqx.Module  # CNN to encode charge/bias distribution
    distort_fn: callable  # Function to distort the kernels

    def __init__(self, order=3, key=jr.key(0)):

        # Coordinate distortion set up
        knots = dlu.pixel_coords(3, 1)
        powers = np.array(gen_powers(order))
        n_features = np.array(powers).size
        self.distort_fn = lambda coeffs: distort_coords(
            knots, coeffs.reshape(powers.shape), powers
        )

        # Default convolutional layers
        def Conv2d(in_channels=1, out_channels=1):
            return lambda key: eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                key=key,
                use_bias=False,
            )

        # Convolution layers: Feature extraction from charge/bias distribution
        conv_layers = [
            Conv2d(in_channels=1, out_channels=32),
            Conv2d(in_channels=32, out_channels=32),
            Conv2d(in_channels=32, out_channels=n_features),
        ]

        # Initialize them with keys, and add relu activation
        keys = jr.split(key, len(conv_layers))
        layers = []
        for i, layer in enumerate(conv_layers):
            layers.append(layer(keys[i]))
            if i < len(conv_layers) - 1:
                layers.append(eqx.nn.Lambda(nn.relu))

        # Construct the encoder
        self.encoder = eqx.nn.Sequential(layers)

        # # Dense layer to predict the distortion coefficients
        # self.dense = eqx.nn.MLP(
        #     in_size=8,
        #     out_size=n_features,
        #     width_size=16,
        #     depth=4,
        #     activation=jax.nn.relu,
        #     use_bias=False,
        #     use_final_bias=False,
        #     key=jr.key(42),
        # )

    def __call__(self, charge_bias):
        """Predict spatially adaptive transposed convolution kernels."""

        # TODO: Try this but without adding the base coordinates back to allow for shifts?
        def distort_coords(coords, coeffs, pows):
            pow_base = np.multiply(*(coords[:, None, ...] ** pows[..., None, None]))
            distortion = np.sum(coeffs[..., None, None] * pow_base[None, ...], axis=1)
            return coords + distortion

        #
        features = self.encoder(charge_bias)
        coeffs_vec = features.reshape(len(features), -1).T

        # #
        # features_vec = self.encoder(charge_bias)
        # coeffs = vmap(self.dense)(features_vec)
        # coeffs_vec = coeffs.reshape(len(coeffs), -1).T

        #
        npix = charge_bias.shape[-1]
        coords = vmap(self.distort_fn)(1 * coeffs_vec).reshape(npix, npix, 2, 3, 3)
        kernels = calc_kernels(coords)
        return kernels


class RNNRamp(WrapperHolder):
    sensitivity_model: PixelSensitivity
    bleed: bool
    norm: int

    def __init__(self, conv_rnn=DFRNN(), gain_model=PixelSensitivity(), norm=2**14, bleed=True):

        values, structure = build_wrapper(conv_rnn)
        self.values = values
        self.structure = structure
        self.norm = norm
        self.sensitivity_model = gain_model
        self.bleed = bleed

    def __getattr__(self, key):
        if hasattr(self.sensitivity_model, key):
            return getattr(self.sensitivity_model, key)
        raise AttributeError(f"RNNRamp has no attribute {key}")

    def evolve_ramp(self, illuminance, bias, ngroups, return_bleed=False):
        # Normalise the Illuminance and charge
        illuminance = illuminance / self.norm
        bias = bias / self.norm

        # Evolve the ramp
        sensitivity = self.sensitivity_model.sensitivity
        charge_ramp, bleed = self.build(bias, illuminance, sensitivity, self.bleed)

        # Interpolate the ramps to the number of groups
        ramp = self.norm * interp_ramp(charge_ramp, ngroups)

        if return_bleed:
            return ramp, bleed
        return ramp
