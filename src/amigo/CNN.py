import jax
import jax.numpy as np
import equinox as eqx
from typing import List, Callable
import dLux.utils as dlu
import jax.random as jr


def calc_rfield(layers):
    """
    Calculates the receptive field of a CNN.

    Only works for 2d convolutions. Assumes all strides and kernels are square.

    Equations from here: https://theaisummer.com/receptive-field/
    """
    conv_layers = [layer for layer in layers if isinstance(layer, eqx.nn.Conv2d)]
    # for layer in conv_layers:
    vals = []
    for i in range(len(conv_layers)):
        size_fn = lambda layer: layer.stride[0] * layer.dilation[0]
        mult = np.array([size_fn(layer) for layer in conv_layers[:i]]).prod()
        vals.append((conv_layers[i].kernel_size[0] - 1) * mult)
    return int(np.array(vals).sum() + 1)


class Expand(eqx.Module):
    def __call__(self, x):
        return np.expand_dims(x, 0)


class Squeeze(eqx.Module):
    def __call__(self, x):
        return np.squeeze(x)


class SpatialGrads(eqx.Module):

    def __call__(self, image):
        return np.array(np.gradient(image))


class SpatialCurvature(eqx.Module):

    def image_to_grads(self, image):
        ygrads, xgrads = np.gradient(image)
        yygrads = np.gradient(ygrads)[0]
        xxgrads = np.gradient(xgrads)[1]
        return yygrads + xxgrads

    def __call__(self, image):
        return jax.vmap(self.image_to_grads)(image)


class Downsample(eqx.Module):
    oversample: int = eqx.field(static=True)

    def __init__(self, oversample):
        self.oversample = oversample

    def __call__(self, x):
        return dlu.downsample(x, self.oversample, mean=False)


class ConvBFE(eqx.Module):
    """
    A CNN To calculate the charge bleeding.

    Output should be normalised such that the sum is zero, ie conserve charge.

    Norm fact is the value that everything is scaled by at the input/output of the
    network. This keeps the network values in a reasonable range, and allows for the
    inputs/outputs to be in the same range as the input data. This will need to be
    larger for deeper well depth data.
    """

    layers: List[Callable]
    oversample: int = eqx.field(static=True)
    pad: int = eqx.field(static=True)

    def __init__(self, layers, oversample, pad):

        self.layers = layers
        self.oversample = oversample
        self.pad = pad

    @property
    def field_of_regard(self):
        return calc_rfield(self.layers)

    def image_to_grads(self, image):
        ygrads, xgrads = np.gradient(image)
        yygrads = np.gradient(ygrads)[0]
        xxgrads = np.gradient(xgrads)[1]
        output = np.array([xgrads, ygrads, xxgrads, yygrads])
        return output / np.array([1e2, 1e2, 1e2, 1e2])[:, None, None]

    def __call__(self, image):
        # Re add back to the original image
        initial_charge = dlu.downsample(image, self.oversample, mean=False)
        # return initial_charge

        x = self.image_to_grads(image)
        for layer in self.layers:
            x = layer(x)
        # bleeding = np.squeeze(x) * 5e4
        bleeding = np.squeeze(x) * 1e3

        k = self.pad

        cent_bleed = bleeding[k:-k, k:-k]
        bleeding -= cent_bleed.mean()
        return initial_charge + bleeding

    def apply_array(self, x):
        """This only exists to match PolyBFE methods"""
        return self(x)
