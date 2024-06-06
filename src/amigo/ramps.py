import jax
import jax.numpy as np
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import dLux as dl
import dLux.utils as dlu
from jax import vmap
from .detector_layers import model_ramp, Ramp


class SimpleRamp(dl.detectors.BaseDetector):

    def apply(self, psf, flux, exposure, oversample):
        return model_ramp(dlu.downsample(psf * flux, oversample, mean=False), exposure.ngroups)

    def model(self, psf):
        raise NotImplementedError


def build_image_basis(image):
    ygrads, xgrads = np.gradient(image)
    rgrads = np.hypot(xgrads, ygrads)

    yygrads = np.gradient(ygrads)[0]
    xxgrads = np.gradient(xgrads)[1]
    xxyygrads = yygrads + xxgrads

    xyrgrads = np.hypot(xxgrads, yygrads)

    return np.array([image, rgrads, xyrgrads, xxyygrads])


def build_basis(image, powers=[1, 2], norm=1900.0):
    image /= norm
    safe_pow = lambda x, p: np.where(x < 0, -np.abs(np.pow(-x, p)), np.pow(x, p))
    images = [safe_pow(image, pow) for pow in powers]
    basis = [build_image_basis(im) for im in images]
    return np.concatenate(basis)


class PolyNonLin(dl.detectors.BaseDetector):
    coeffs: jax.Array
    conv: None
    ksize: int = eqx.field(static=True)
    orders: list = eqx.field(static=True)
    oversample: int = eqx.field(static=True)

    def __init__(self, ksize=3, oversample=4, orders=[1, 2], basis_length=4):
        self.ksize = int(3)
        self.oversample = int(oversample)
        self.orders = orders

        true_ksize = self.ksize * oversample
        padding = oversample * (self.ksize // 2)

        conv = jtu.tree_map(
            lambda x: np.zeros_like(x),
            eqx.nn.Conv2d(
                in_channels=1,
                out_channels=true_ksize**2,
                kernel_size=true_ksize,
                stride=self.oversample,
                padding=(padding, padding),
                use_bias=False,
                key=jr.PRNGKey(0),
            ),
        )

        zero_vec = np.zeros(true_ksize**2)
        conv_kernels = conv.weight
        for i in range(len(conv_kernels)):
            single_kern = zero_vec.at[i].set(1.0).reshape(true_ksize, true_ksize)
            conv_kernels = conv_kernels.at[i].set(single_kern)
        ncoeffs = len(conv_kernels) * basis_length * len(orders)

        self.conv = eqx.tree_at(lambda x: x.weight, conv, conv_kernels)
        self.coeffs = 1e-8 * jr.normal(jr.PRNGKey(0), (ncoeffs,))

    def apply(self, psf, flux, exposure, oversample):
        image = psf.data * flux
        downsampled_image = dlu.downsample(image, 4, mean=False)
        downsampled_ramp = model_ramp(downsampled_image, exposure.ngroups)
        ramp = model_ramp(image, exposure.ngroups)
        ramp = downsampled_ramp + vmap(self.calculate_bleeding)(ramp)
        return Ramp(ramp, psf.pixel_scale)

    def calculate_bleeding(self, image):
        basis = build_basis(image, norm=1.0, powers=self.orders)
        full_kernels = vmap(lambda im: self.conv(im[None]))(basis)
        # basis_length = np.prod(np.array(full_kernels.shape[:2]))
        true_kernels = full_kernels.reshape(len(self.coeffs), *full_kernels.shape[2:])
        return dlu.eval_basis(true_kernels, self.coeffs)

    def model():
        raise NotImplementedError


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


class NonLinCNN(dl.detectors.BaseDetector):
    conv: None
    amplitude: float
    steps = 20
    filter_norm: dict = eqx.field(static=True)

    # def __init__(self, layers, amplitude=1):
    def __init__(
        self,
        layers=None,
        widths=None,
        amplitude=1,
        key=jr.PRNGKey(0),
        powers=[1, 2],
        zero_bias=True,
    ):

        if layers is None:
            subkeys = jr.split(key, 5)
            use_bias = True

            in_size = 8 * len(powers)

            # in_size = 2 * len(basis)
            # print(in_size)
            # widths = [32, 16, 8, 4]
            # widths = [16, 8, 4, 4]
            # widths = [8, 8, 4, 4]
            # widths = [2, 2, 1, 1]

            if widths is None:
                widths = [1, 1, 1, 1]

            if len(widths) != 4:
                raise ValueError("Widths must be of length 4")

            layers = [
                eqx.nn.Conv2d(
                    in_channels=in_size,
                    out_channels=widths[0],
                    use_bias=use_bias,
                    padding=1,
                    kernel_size=3,
                    key=subkeys[0],
                ),
                eqx.nn.Conv2d(
                    in_channels=widths[0],
                    out_channels=widths[1],
                    use_bias=use_bias,
                    dilation=(2, 2),
                    padding=2,
                    kernel_size=3,
                    key=subkeys[1],
                ),
                eqx.nn.Conv2d(
                    in_channels=widths[1],
                    out_channels=widths[2],
                    use_bias=use_bias,
                    padding=1,
                    kernel_size=3,
                    key=subkeys[2],
                ),
                eqx.nn.Conv2d(
                    in_channels=widths[2],
                    out_channels=widths[3],
                    use_bias=use_bias,
                    dilation=(2, 2),
                    padding=2,
                    kernel_size=3,
                    key=subkeys[3],
                ),
                eqx.nn.Conv2d(
                    in_channels=widths[3],
                    out_channels=1,
                    use_bias=use_bias,
                    padding=1,
                    kernel_size=3,
                    key=subkeys[4],
                ),
            ]
            print(f"Field of Regard: {calc_rfield(layers)}")

        # def zero_bias(layer):
        if zero_bias:
            # zero_bias_fn = lambda layer: eqx.tree_at(
            #     lambda x: x.bias, layer, np.zeros_like(layer.bias)
            # )
            layers = [
                eqx.tree_at(lambda x: x.bias, layer, np.zeros_like(layer.bias)) for layer in layers
            ]

        from amigo.core import NNWrapper

        self.conv = NNWrapper(layers)
        self.amplitude = np.array(amplitude, float)
        self.filter_norm = {
            "F380M": 6.5e4,
            "F430M": 7.5e4,
            "F480M": 9e4,
        }

    def apply(self, psf, flux, exposure, oversample):
        ramp = self.bleeding_model(psf.data, exposure.filter)[0]
        ramp = self.sample_ramp(ramp, flux, exposure.filter, exposure.ngroups)
        dsample_fn = lambda x: dlu.downsample(x, oversample, mean=False)
        return Ramp(vmap(dsample_fn)(ramp), psf.pixel_scale)

    def bleeding_model(self, psf, filter):

        photons = psf * self.filter_norm[filter]
        charge, bleed = np.zeros(psf.shape), np.zeros(psf.shape)

        ramp, bleed_ramp = [], []
        for i in range(self.steps):

            # Build basis
            basis = np.concatenate(
                [
                    build_basis(charge),
                    build_basis(photons),
                    # build_basis(bleeding),
                ],
                0,
            )

            # Scale bleeding by amplitude
            bleeding = self.amplitude * np.squeeze(self.conv(basis))

            # TODO: We COULD make the bleeding a polynomial function of the current step
            # Since the bleeding is positive and negative, we need to use the `safe_pow`
            # function used in the basis creation

            # TODO: The incoming photons could also be a function of the current step
            # This might be an easier way to do pixel non-linearities

            charge += photons + bleeding
            # charge += photons
            bleed += bleeding

            ramp.append(charge)
            bleed_ramp.append(bleed)

        return np.array(ramp), np.array(bleed_ramp)

    def sample_ramp(self, ramp, flux, filter, ngroups):

        photons_in = self.filter_norm[filter] * self.steps

        # Pre-pend a zeroth group to allow interpolation for low values
        ramp = np.concatenate([np.zeros((1, *ramp.shape[1:])), ramp], axis=0)

        # Generate the coordinates up the ramp
        flux_coords = photons_in * np.arange(self.steps + 1) / self.steps
        group_coords = flux * (np.arange(ngroups) + 1) / ngroups

        # Check for fluxes exceeding the total flux
        _ = jax.lax.cond(
            group_coords[-1] > flux_coords[-1],
            lambda _: jax.debug.print("Warning: Group fluxes exceed the total flux"),
            lambda _: None,
            None,
        )

        # Interpolate the ramp
        ramp_fn = vmap(lambda fp: np.interp(group_coords, flux_coords, fp), 1, 1)
        ramp_out = ramp_fn(np.array(ramp).reshape(len(ramp), -1))
        return ramp_out.reshape(ngroups, *ramp[0].shape)

    def model(self, *args):
        raise NotImplementedError
