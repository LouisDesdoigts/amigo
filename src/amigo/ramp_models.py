import jax
import jax.numpy as np
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import dLux as dl
import dLux.utils as dlu
from jax import vmap
import interpax as ipx

# from .detector_layers import model_ramp, Ramp


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


class PolyNonLin(dl.detectors.BaseDetector):
    coeffs: jax.Array
    norm: float
    conv: None
    ksize: int = eqx.field(static=True)
    orders: list = eqx.field(static=True)
    oversample: int = eqx.field(static=True)

    def __init__(
        self,
        ksize=3,
        oversample=4,
        orders=[1, 2],
        # basis_length=4,
        coeffs_dict=True,
        seed=0,
        norm=1.0,
    ):
        self.ksize = int(ksize)
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

        basis_length = len(build_image_basis(np.zeros((10, 10))))
        coeffs_per_order = basis_length * len(conv_kernels)
        ncoeffs = basis_length * len(conv_kernels) * len(orders)

        self.conv = eqx.tree_at(lambda x: x.weight, conv, conv_kernels)
        self.coeffs = 1e-8 * jr.normal(jr.PRNGKey(0), (ncoeffs,))
        self.norm = norm

        if coeffs_dict:
            coeffs = {}
            for order in orders:
                coeffs[str(order)] = 1e-8 * jr.normal(jr.PRNGKey(seed), (coeffs_per_order,))
                seed += 1
            self.coeffs = coeffs
        else:
            self.coeffs = 1e-8 * jr.normal(jr.PRNGKey(seed), (ncoeffs,))

    def apply(self, psf, flux, exposure, oversample):
        image = psf.data * flux
        downsampled_image = dlu.downsample(image, 4, mean=False)
        downsampled_ramp = model_ramp(downsampled_image, exposure.ngroups)
        ramp = model_ramp(image * self.norm, exposure.ngroups)
        ramp = downsampled_ramp + vmap(self.calculate_bleeding)(ramp)
        return Ramp(ramp, psf.pixel_scale)

    def calculate_bleeding(self, image):
        basis = build_basis(image, powers=self.orders)
        full_kernels = vmap(lambda im: self.conv(im[None]))(basis)

        if isinstance(self.coeffs, dict):
            coeffs = np.concatenate(jtu.tree_leaves(self.coeffs))
        else:
            coeffs = self.coeffs
        true_kernels = full_kernels.reshape(len(coeffs), *full_kernels.shape[2:])
        return dlu.eval_basis(true_kernels, coeffs)
        # true_kernels = full_kernels.reshape(len(self.coeffs), *full_kernels.shape[2:])
        # return dlu.eval_basis(true_kernels, self.coeffs)

    def model():
        raise NotImplementedError


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


class NonLinCNN(dl.detectors.BaseDetector):
    conv: None
    amplitude: float
    steps: int = eqx.field(static=True)
    max_flux: dict = eqx.field(static=True)
    basis_norm: float = eqx.field(static=True)
    powers: list = eqx.field(static=True)

    def __init__(
        self,
        model,
        layers=None,
        widths=None,
        amplitude=1e-1,
        key=jr.PRNGKey(0),
        powers=[1, 2],
        zero_bias=True,
        max_pixel_depth=30e3,
        steps=25,
    ):

        # Calibrate filters - model an idealised psf
        basis_norm = 1.0
        max_flux = {}
        psfs = {}
        for filter in model.filters.keys():
            # Get wavelengths and weights
            wavels, weights = model.filters[filter]
            weights = weights / weights.sum()
            psf = model.optics.propagate(wavels, np.zeros(2), weights, return_psf=True)
            psf = model.detector.apply(psf).data
            psfs[filter] = psf

            # Now we rasterize over the oversample to find the peak flux
            max_depth = 0.0
            oversample = model.optics.oversample
            for i in range(oversample):
                for j in range(oversample):
                    roll = (i - (oversample - 1) // 2, j - (oversample - 1) // 2)
                    rolled = np.roll(psf, roll, axis=(0, 1))
                    downsampled = dlu.downsample(rolled, oversample, mean=False)
                    max_depth = np.maximum(max_depth, downsampled.max())

            # Calculate required values
            total_charge = (psf * max_pixel_depth / (max_depth * psf.sum())).sum()
            max_charge = total_charge * psf
            if max_charge.max() > basis_norm:
                basis_norm = max_charge.max()
            max_flux[filter] = total_charge

        self.basis_norm = float(basis_norm)
        self.max_flux = max_flux
        self.steps = int(steps)
        self.amplitude = np.array(amplitude, float)
        self.powers = powers

        if layers is None:
            subkeys = jr.split(key, 5)
            use_bias = True

            basis = build_basis(np.ones((320, 320)), powers=self.powers)
            in_size = 2 * len(basis)

            if widths is None:
                widths = [8, 8, 4, 4]

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

        if zero_bias:
            layers = [
                eqx.tree_at(lambda x: x.bias, layer, np.zeros_like(layer.bias)) for layer in layers
            ]

        # Import here to avoid circular imports
        from amigo.core import NNWrapper

        self.conv = NNWrapper(layers)

    def __getattr__(self, key):
        if hasattr(self.conv, key):
            return getattr(self.conv, key)
        raise AttributeError(f"NonLinCNN has no attribute {key}")

    def apply(self, psf, flux, exposure, oversample):
        ramp = self.bleeding_model(psf.data, exposure.filter)[0]
        ramp = self.sample_ramp(ramp, flux, exposure.filter, exposure.ngroups)
        dsample_fn = lambda x: dlu.downsample(x, oversample, mean=False)
        return Ramp(vmap(dsample_fn)(ramp), psf.pixel_scale)

    def bleeding_model(self, psf, filter):

        # photons = psf * self.filter_norm[filter] / self.steps
        photons = psf * self.max_flux[filter] / self.steps
        charge, bleed = np.zeros(psf.shape), np.zeros(psf.shape)

        ramp, bleed_ramp = [], []
        for i in range(self.steps):

            # Build basis
            basis = np.concatenate(
                [
                    build_basis(charge, norm=self.basis_norm, powers=self.powers),
                    build_basis(photons, norm=self.basis_norm, powers=self.powers),
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
        # Pre-pend a zeroth group to allow interpolation for low values
        ramp = np.concatenate([np.zeros((1, *ramp.shape[1:])), ramp], axis=0)

        # Generate the coordinates up the ramp
        flux_coords = self.max_flux[filter] * np.arange(self.steps + 1) / self.steps
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


class PolyConvInterp(eqx.Module):
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

    def eval_ramp_interp(self, psf, flux, ngroups):
        # Get the group flux coordinates and regular ramp
        groups = flux * (np.arange(ngroups) + 1) / ngroups
        ramp = groups[:, None, None] * dlu.downsample(psf, 4, mean=False)[None, ...]

        # Predict the polynomial coefficients
        norm_fn = vmap(lambda arr: arr / np.max(np.abs(arr)))
        full_bleed_ramp = self.conv(norm_fn(build_image_basis(psf)))

        #
        peak_flux = 2e6
        knots = np.linspace(0, 1, len(full_bleed_ramp))
        xs = (np.arange(ngroups) + 1) / ngroups
        sample_points = flux * xs / peak_flux

        # Interp fn
        def interp_fn(bleed):
            return ipx.interp1d(sample_points, knots, bleed, method="cubic2", extrap=True)

        bleed_ramp = vmap(vmap(interp_fn, 1, 1), 2, 2)(full_bleed_ramp)
        bleed_ramp *= 5e-3 * peak_flux / len(full_bleed_ramp)

        # Calculate the polynomial
        return ramp + bleed_ramp

    def eval_ramp(self, psf, flux, ngroups):
        # Get the input coordinates for the polynomial - arbitrary norm
        sample_ratio = flux / 2e6

        # Get the group flux coordinates and regular ramp
        groups = flux * (np.arange(ngroups) + 1) / ngroups
        ramp = groups[:, None, None] * dlu.downsample(psf, 4, mean=False)[None, ...]

        # Predict the polynomial coefficients
        norm_fn = vmap(lambda arr: arr / np.max(np.abs(arr)))
        conv_coeffs = self.conv(norm_fn(build_image_basis(psf)))
        # print(conv_coeffs.shape)

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

        # Regular polynomial, coeffs not raised to a power
        eval_points = sample_points[:, None] ** pows[None, :]
        bleed_ramp = np.sum(coeffs[None, ...] * eval_points[..., None, None], axis=1)

        # Calculate the polynomial
        return ramp + bleed_ramp


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

        # Regular polynomial, coeffs not raised to a power
        eval_points = sample_points[:, None] ** pows[None, :]
        bleed_ramp = np.sum(coeffs[None, ...] * eval_points[..., None, None], axis=1)

        # Calculate the polynomial
        return ramp + bleed_ramp
