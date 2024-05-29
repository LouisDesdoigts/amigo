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


class NonLinDetector(eqx.Module):
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


# import jax
# import jax.numpy as np
# import jax.random as jr
# import equinox as eqx
# from amigo.CNN import ConvBFE, Expand, Squeeze, calc_rfield

# kernel_size = 3
# oversample = model.optics.oversample
# N = kernel_size * oversample


# Conv2d = lambda **kwargs: eqx.nn.Conv2d(
#     **kwargs,
#     kernel_size=kernel_size,
# )

# key = jr.PRNGKey(0)
# subkeys = jr.split(key, 100)

# widths = [
#     16,
#     16,
#     4,
#     4
# ]

# layers = [
#     # Expand(), # Dont need expand, handled by the image2grads fn
#     Conv2d(
#         in_channels=4,
#         out_channels=widths[0],
#         stride=(1, 1),
#         padding=1,
#         key=subkeys[0],
#     ),
#     jax.nn.relu,
#     Conv2d(
#         in_channels=widths[0],
#         out_channels=widths[1],
#         stride=(2, 2),
#         padding=1,
#         key=subkeys[1],
#     ),
#     jax.nn.relu,
#     Conv2d(
#         in_channels=widths[1],
#         out_channels=widths[2],
#         stride=(1, 1),
#         padding=1,
#         key=subkeys[2],
#     ),
#     jax.nn.relu,
#     Conv2d(
#         in_channels=widths[2],
#         out_channels=widths[3],
#         stride=(2, 2),
#         padding=1,
#         key=subkeys[3],
#     ),
#     jax.nn.relu,
#     Conv2d(
#         in_channels=widths[3],
#         out_channels=1,
#         dilation=(2, 2),
#         padding=2,
#         key=subkeys[4],
#     ),
#     Squeeze(),
# ]
# print(f"Field of Regard: {calc_rfield(layers)}")


# ramp = model_fn(model, base_exposures[0], to_BFE=True)
# # x =
# x = np.zeros((4,) + ramp.shape[1:])
# # x = jr.normal(jr.PRNGKey(1), x.shape)
# print(f"Input shape: {x.shape}")
# print()

# for layer in layers:
#     x = layer(x)
#     if isinstance(layer, (eqx.nn.Conv, eqx.nn.Pool, Squeeze, Expand)):
#         string = "\n" + str(type(layer)).split(".")[-1][:-2] + f" -> {x.shape}"
#         if isinstance(layer, eqx.nn.Conv):
#             string += (
#                 # f"  Pad: {layer.padding[0][0]}\n"
#                 f"\n  Stride: {layer.stride[0]}\n"
#                 f"  Dilation: {layer.dilation[0]}"
#             )
#         print(string)
#         # print(f)


# import jax.tree_util as jtu

# def leaf_fn(leaf):
#     if hasattr(leaf, 'size'):
#         return leaf.size
#     else:
#         return 0

# vals = np.array(jtu.tree_leaves(jtu.tree_map(leaf_fn, layers))).sum()
# vals


# k = (model.optics.psf_npixels - 80) // 2

# convBFE = ConvBFE(layers, oversample, pad=k)
# # model = model.set("BFE", convBFE)

# psf = ramp[-1]
# im1 = dlu.downsample(psf, oversample, mean=False)[k:-k, k:-k]
# im2 = convBFE(psf)[k:-k, k:-k]
# bleeding = im1 - im2
# print()
# print(im1.sum(), im2.sum())
# print(im1.sum() / im2.sum())
# print(bleeding.sum())

# plt.figure(figsize=(15, 4))
# plt.subplot(1, 3, 1)
# plt.imshow(im1)
# plt.colorbar()

# plt.subplot(1, 3, 2)
# plt.imshow(im2, cmap="inferno")
# plt.colorbar()

# plt.subplot(1, 3, 3)
# v = np.abs(bleeding).max()
# plt.imshow(bleeding, cmap="seismic", vmin=-v, vmax=v)
# plt.colorbar()
# plt.show()
