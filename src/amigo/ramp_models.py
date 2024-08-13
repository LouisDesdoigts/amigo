import jax
import jax.numpy as np
import jax.random as jr
import equinox as eqx
import dLux as dl
import dLux.utils as dlu
from jax import vmap


def build_image_basis(image):
    ygrads, xgrads = np.gradient(image)
    rgrads = np.hypot(xgrads, ygrads)

    yygrads = np.gradient(ygrads)[0]
    xxgrads = np.gradient(xgrads)[1]
    xxyygrads = yygrads + xxgrads

    xyrgrads = np.hypot(xxgrads, yygrads)

    return np.array([image, rgrads, xyrgrads, xxyygrads])


def build_basis(image, powers=[1], norm=1.0):
    image /= norm
    safe_pow = lambda x, p: np.where(x < 0, -np.abs(np.pow(-x, p)), np.pow(x, p))
    images = [safe_pow(image, pow) for pow in powers]
    basis = [build_image_basis(im) for im in images]
    return np.concatenate(basis)


def model_ramp(psf, ngroups):
    """Applies an 'up the ramp' model of the input 'optical' PSF. Input PSF.data
    should have shape (npix, npix) and return shape (ngroups, npix, npix)"""
    lin_ramp = (np.arange(ngroups) + 1) / ngroups
    return psf[None, ...] * lin_ramp[..., None, None]


def calc_rfield(layers):
    """
    Calculates the receptive field of a CNN.

    Only works for 2d convolutions. Assumes all strides and kernels are square.

    Equations from here: https://theaisummer.com/receptive-field/
    """
    conv_layers = [layer for layer in layers if isinstance(layer, eqx.nn.Conv2d)]
    vals = []
    for i in range(len(conv_layers)):
        size_fn = lambda layer: layer.stride[0] * layer.dilation[0]
        mult = np.array([size_fn(layer) for layer in conv_layers[:i]]).prod()
        vals.append((conv_layers[i].kernel_size[0] - 1) * mult)
    return int(np.array(vals).sum() + 1)


def build_conv_layers(
    in_channels=4,
    conv_hidden=64,
    n_connect=16,
    key=jr.PRNGKey(0),
):
    keys = jr.split(key, (3,))
    layers = [
        eqx.nn.Conv2d(
            in_channels=in_channels,
            out_channels=conv_hidden,
            kernel_size=3,
            stride=1,
            dilation=(2, 2),
            padding=(2, 2),
            use_bias=False,
            key=keys[0],
        ),
        eqx.nn.Conv2d(
            in_channels=conv_hidden,
            out_channels=conv_hidden // 2,
            kernel_size=3,
            stride=2,
            dilation=(2, 2),
            padding=(2, 2),
            use_bias=False,
            key=keys[1],
        ),
        eqx.nn.Conv2d(
            in_channels=conv_hidden // 2,
            out_channels=n_connect,
            kernel_size=3,
            stride=2,
            dilation=(1, 1),
            padding=(1, 1),
            use_bias=False,
            key=keys[2],
        ),
    ]
    return layers


def build_dense_layers(
    n_connect=4,
    n_hidden=9,
    n_terms=4,
    key=jr.PRNGKey(1),
):
    keys = jr.split(key, (2,))
    dense_layers = [
        eqx.nn.Linear(
            in_features=n_connect,
            out_features=n_hidden,
            key=keys[0],
            use_bias=False,
        ),
        eqx.nn.Linear(
            in_features=n_hidden,
            out_features=n_terms,
            key=keys[1],
            use_bias=False,
        ),
    ]
    return dense_layers


class Ramp(dl.PSF):
    pass


class SimpleRamp(dl.detectors.BaseDetector):

    def apply(self, psf, flux, exposure, oversample):
        # lin_ramp = (np.arange(exposure.ngroups) + 1) / exposure.ngroups
        image = dlu.downsample(psf.data * flux, oversample, mean=False)
        # ramp = image[None, ...] * lin_ramp[..., None, None]
        ramp = model_ramp(image, exposure.ngroups)
        return Ramp(ramp, psf.pixel_scale)

    def model(self, psf):
        raise NotImplementedError


class PolyConv(eqx.Module):
    conv: None
    dense: None

    def __init__(self, conv_layers, dense_layers, init_scale=1):
        from .core_models import NNWrapper

        conv = NNWrapper(conv_layers)
        dense = NNWrapper(dense_layers)

        self.conv = conv.multiply("values", init_scale)
        self.dense = dense.multiply("values", init_scale)

    def apply(self, psf, flux, exposure, oversample):
        return Ramp(self.eval_ramp(psf.data, flux, exposure.ngroups), psf.pixel_scale)

    def eval_ramp(self, psf, flux, ngroups):
        # Get the input coordinates for the polynomial - arbitrary norm
        sample_ratio = flux / 2e6

        # Get the group flux coordinates and regular ramp
        groups = flux * (np.arange(ngroups) + 1) / ngroups
        ramp = groups[:, None, None] * dlu.downsample(psf, 4, mean=False)[None, ...]

        # Predict the polynomial coefficients
        norm_fn = vmap(lambda arr: arr / np.max(np.abs(arr)))
        conv_coeffs = self.conv(norm_fn(build_image_basis(psf)))

        def apply_linear(x, layers):
            for layer in layers[:-1]:
                x = jax.nn.relu(layer(x))
            return layers[-1](x)

        vmap_fn = lambda fn: lambda x: vmap(vmap(fn, 1, 1), 2, 2)(x)
        layers = [vmap_fn(layer) for layer in self.dense._layers]
        coeffs = apply_linear(conv_coeffs, layers)
        coeffs = 2e6 * coeffs

        # We dont want a constant term, so start from 1
        pows = np.arange(0, len(coeffs)) + 1
        sample_points = sample_ratio * (np.arange(ngroups) + 1) / ngroups

        # Regular polynomial
        eval_points = sample_points[:, None] ** pows[None, :]
        bleed_ramp = np.sum(coeffs[None, ...] * eval_points[..., None, None], axis=1)

        # Calculate the polynomial
        return ramp + bleed_ramp
