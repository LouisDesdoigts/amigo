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
    #
    true_layers = []
    for layer in layers:
        if isinstance(layer, eqx.Module):
            true_layers.append(layer)

    vals = []
    for i, layer in enumerate(true_layers):
        if isinstance(layer, eqx.nn.Conv) or isinstance(layer, eqx.nn.Pool):

            def size_fn(layer):
                if hasattr(layer, "stride") and hasattr(layer, "dilation"):
                    return layer.stride[0] * layer.dilation[0]
                return layer.stride[0]

        else:
            continue
        mult = np.array([size_fn(layer) for layer in true_layers[:i]]).prod()
        vals.append((true_layers[i].kernel_size[0] - 1) * mult)
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


class PolyBias(eqx.Module):
    psf_conv: None
    bias_conv: None
    mixing_conv: None

    def __init__(self, conv_layers, bias_layers, mixing_layers, init_scale=1):
        # from .core_models import NNWrapper
        from amigo.core_models import NNWrapper

        self.psf_conv = NNWrapper(conv_layers).multiply("values", init_scale)
        self.bias_conv = NNWrapper(bias_layers).multiply("values", init_scale)
        self.mixing_conv = NNWrapper(mixing_layers).multiply("values", init_scale)

    def apply(self, psf, flux, bias, exposure, oversample):
        return Ramp(self.eval_ramp(psf.data, flux, bias, exposure.ngroups), psf.pixel_scale)

    def eval_ramp(self, psf, flux, bias, ngroups):
        # Get the input coordinates for the polynomial - arbitrary norm
        sample_ratio = flux / 2e6

        # psf conv net
        norm_psf = psf / np.max(np.abs(psf))
        psf_conv_out = self.psf_conv(norm_psf[None, ...])

        # Bias conv net
        norm_bias = bias / 15e3
        norm_bias -= np.mean(norm_bias)
        bias_conv_out = 0.1 * self.bias_conv(norm_bias[None, ...])

        # Mixing conv net
        mixing_in = np.concatenate([psf_conv_out, bias_conv_out], axis=0)
        coeffs = self.mixing_conv(mixing_in)

        # Re-normalise coefficients
        coeffs = 2e6 * coeffs

        # We dont want a constant term, so start from 1
        pows = np.arange(0, len(coeffs)) + 1
        sample_points = sample_ratio * (np.arange(ngroups) + 1) / ngroups

        # Regular polynomial
        eval_points = sample_points[:, None] ** pows[None, :]
        bleed_ramp = np.sum(coeffs[None, ...] * eval_points[..., None, None], axis=1)

        # Get the group flux coordinates and regular ramp
        groups = flux * (np.arange(ngroups) + 1) / ngroups
        ramp = groups[:, None, None] * dlu.downsample(psf, 4, mean=False)[None, ...]

        # Calculate the polynomial
        return ramp + bleed_ramp


def build_pooled_layers(width, depth, poly_order=4, seed=0, pooling="avg"):
    key = jr.PRNGKey(seed)
    if pooling == "avg":
        pooling_layer = eqx.nn.AvgPool2d(kernel_size=2, stride=(2, 2))
    elif pooling == "max":
        pooling_layer = eqx.nn.MaxPool2d(kernel_size=2, stride=(2, 2))
    else:
        raise ValueError("Pooling must be 'avg' or 'max'")

    keys = jr.split(key, (depth + 1,))
    widths = np.linspace(width, poly_order, depth).astype(int)
    widths = np.concatenate([np.array([1]), widths])

    layers = []
    for i in range(depth):

        layers.append(
            eqx.nn.Conv2d(
                in_channels=int(widths[i]),
                out_channels=int(widths[i + 1]),
                kernel_size=3,
                padding=(1, 1),
                use_bias=False,
                key=keys[i],
            )
        )
    return layers, pooling_layer


class PredictivePoly(eqx.Module):

    def eval_poly(self, coeffs, psf, flux, ngroups, oversample):
        # The value at which the x-evaluation point is 1
        x_max = 2**16

        # Downsample psf - (80, 80)
        # NOTE: We got concretization errors if we use oversample
        # x = dlu.downsample(psf, oversample, mean=False)
        x = dlu.downsample(psf, 4, mean=False)
        x_norm = 1 / x.max()

        # Get the flux normalisation for the x-evaluation points
        maxed_psf = x * x_norm * x_max
        max_flux = maxed_psf.sum()

        # For flux == max flux we want sample points to be [0, 1)
        latent_x_pts = (np.arange(ngroups) + 1) / ngroups
        x_pts = flux * latent_x_pts / max_flux

        # Get the polynomial evaluation points
        pows = np.arange(0, len(coeffs)) + 1
        eval_points = x_pts[:, None] ** pows[None, :]

        # Get the bleed ramp
        latent_bleed_ramp = np.sum(coeffs[None, ...] * eval_points[..., None, None], axis=1)
        bleed_ramp = max_flux * latent_bleed_ramp

        # Get the group flux coordinates and regular ramp
        groups = flux * (np.arange(ngroups) + 1) / ngroups
        ramp = groups[:, None, None] * x[None, ...]
        return ramp, bleed_ramp

    def eval_poly_norm(self, coeffs, psf, flux, ngroups, oversample):

        # Up the scale to keep outputs normalised
        coeffs = 1e1 * coeffs

        # The value at which the x-evaluation point is 1
        x_max = 2**16

        # Downsample psf - (80, 80)
        # NOTE: We got concretization errors if we use oversample
        # x = dlu.downsample(psf, oversample, mean=False)
        x = dlu.downsample(psf, 4, mean=False)
        x_norm = 1 / x.max()
        norm_x = x * x_norm

        # print(norm_x.shape)
        maxed_psf = norm_x * x_max
        max_flux = maxed_psf.sum()

        single_x_pts = (np.arange(ngroups) + 1) / ngroups
        x_pts = flux * single_x_pts / max_flux
        true_x_pts = norm_x[None, ...] * x_pts[:, None, None]

        # Get the bleed ramp
        pows = np.arange(0, len(coeffs)) + 1
        eval_points = true_x_pts[:, None, ...] ** pows[None, :, None, None]
        ys = np.sum(coeffs[None, ...] * eval_points, axis=1)
        bleed_ramp = ys * x_max

        # Get the group flux coordinates and regular ramp
        groups = flux * (np.arange(ngroups) + 1) / ngroups
        ramp = groups[:, None, None] * x[None, ...]
        return ramp, bleed_ramp


class MinimalConv(PredictivePoly):
    conv: None
    pool: None
    # norm: bool = eqx.field(static=True)

    # def __init__(self, conv_layers, pooling_layer, init_scale=1, norm=False):
    def __init__(self, conv_layers, pooling_layer, init_scale=1):
        from amigo.core_models import NNWrapper

        self.conv = NNWrapper(conv_layers).multiply("values", init_scale)
        self.pool = pooling_layer
        # self.norm = norm

    @property
    def FoR(self):
        layers = self.conv._layers
        true_layers = [layer for layer in layers]
        true_layers.insert(len(layers) // 2, self.pool)
        true_layers.append(self.pool)
        return calc_rfield(true_layers)

    def apply(self, psf, flux, exposure, oversample):
        return Ramp(self.eval_ramp(psf.data, flux, exposure.ngroups, oversample), psf.pixel_scale)

    def calc_conv(self, psf):
        layers = self.conv._layers

        x = psf[None, ...]
        if len(layers) == 2:
            # Handle special case of depth of 1
            x = jax.nn.relu(self.pool(layers[0](x)))
        else:
            for i, layer in enumerate(layers[:-1]):
                if i == len(layers) // 2:
                    x = jax.nn.relu(self.pool(layer(x)))
                else:
                    x = jax.nn.relu(layer(x))
        return self.pool(layers[-1](x))

    def eval_ramp(self, psf, flux, ngroups, oversample):
        # coeffs = self.calc_conv(psf / np.max(np.abs(psf)))
        coeffs = self.calc_conv(psf / np.max(psf))
        # if self.norm:
        #     ramp, bleed_ramp = self.eval_poly_norm(coeffs, psf, flux, ngroups, oversample)
        # else:
        ramp, bleed_ramp = self.eval_poly(coeffs, psf, flux, ngroups, oversample)
        # ramp, bleed_ramp = self.eval_poly_norm(coeffs, psf, flux, ngroups, oversample)
        return ramp + bleed_ramp

    # def eval_ramp(self, psf, flux, ngroups, oversample):
    #     # coeffs = self.calc_conv(psf / np.max(np.abs(psf)))
    #     coeffs = self.calc_conv(psf / np.max(psf))
    #     ramp, bleed_ramp = self.eval_poly_norm(coeffs, psf, flux, ngroups, oversample)
    #     return ramp + bleed_ramp


# from .misc import interp_ramp


# class SUB80Ramp(dl.detectors.BaseDetector):

#     def apply(self, psf, flux, exposure, oversample):
#         # lin_ramp = (np.arange(exposure.ngroups) + 1) / exposure.ngroups
#         image = dlu.downsample(psf.data * flux, oversample, mean=False)
#         # ramp = image[None, ...] * lin_ramp[..., None, None]
#         ramp = model_ramp(image, exposure.ngroups)
#         return Ramp(ramp, psf.pixel_scale)

#     def model(self, psf):
#         raise NotImplementedError


# class SUB80Ramp(LayeredDetector):
#     jitter_model:
#     resample_model:
#     sensitivity_model:
#     ramp_model:
#     read_layers:
#     resample_layers:


#     def apply(self, psf):
#         for layer in list(self.layers.values()):
#             if layer is None:
#                 continue
#             psf = layer.apply(psf)
#         return psf

#     def model_ramp(self, illuminance, ngroups):

#         # Get the evolved gain and diffusion terms
#         gain, diffusion, latent_paths = self.evolve(
#             illuminance, n_samples=10, return_paths=True
#         )

#         # Interpolate the sample into ramps
#         gain_ramp = interp_ramp(gain + 1, ngroups)
#         diffusion_ramp = interp_ramp(diffusion, ngroups)

#         # Apply diffusion and gain
#         pixel_gain = gain_ramp * self.sensitivity_map[None, ...]
#         base_ramp = model_ramp(illuminance, ngroups)
#         full_ramp = gain_ramp * (base_ramp + diffusion_ramp)

#         # Downsample
#         ramp = vmap(lambda x: dlu.downsample(x, 4, mean=False))(full_ramp)

#         # Return the ramp
#         return ramp, latent_paths

#     def predict_slopes(self, illuminance, ngroups):
#         ramp = self.predict_ramp(illuminance, ngroups)
#         return np.diff(ramp, axis=0)

#     def apply(self, psf, exposure, return_paths=False):
#         # out = self.predict_ramp(psf.data * flux, exposure.ngroups, return_paths=return_paths)
#         out = self.predict_ramp(psf.data, exposure.ngroups, return_paths=return_paths)
#         if return_paths:
#             ramp, latent_paths = out
#             return Ramp(ramp, psf.pixel_scale), latent_paths
#         return Ramp(out, psf.pixel_scale)
