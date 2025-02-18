import jax
import jax.numpy as np
import equinox as eqx
import jax.random as jr
import dLux.utils as dlu
from amigo.latent_ode_models import WrapperHolder, build_wrapper
from amigo.misc import interp_ramp
import zodiax as zdx
from amigo.detector_models import quadratic_SRF, broadcast_subpixel


class IllumEncoder(eqx.Module):
    layers: list

    def __init__(self, key, features=4, width=8, use_bias=False):

        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(
                in_channels=1,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[0],
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[1],
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[2],
            ),
            eqx.nn.AvgPool2d(kernel_size=3, stride=3, padding=1),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=features,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[3],
            ),
        ]

    def __call__(self, x):
        if x.ndim == 2:
            x = x[None, ...]
        for layer in self.layers:
            x = layer(x)
        return x


class ChargeEncoder(eqx.Module):
    layers: list

    def __init__(self, key, features=4, width=8, use_bias=False):

        keys = jax.random.split(key, 4)
        self.layers = [
            eqx.nn.Conv2d(
                in_channels=1,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[0],
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[1],
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[2],
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=features,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[3],
            ),
        ]

    def __call__(self, x):
        if x.ndim == 2:
            x = x[None, ...]
        for layer in self.layers:
            x = layer(x)
        return x


class DiffusionDecoder(eqx.Module):
    layers: list

    def __init__(self, key, features=8, width=8, use_bias=False):

        keys = jax.random.split(key, 3)
        self.layers = [
            # Regular conv to extract features
            eqx.nn.Conv2d(
                in_channels=features,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[0],
            ),
            jax.nn.relu,
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=width,
                kernel_size=3,
                padding=1,
                use_bias=use_bias,
                key=keys[1],
            ),
            jax.nn.relu,
            # Transpose conv to extract diffusion kernels
            eqx.nn.ConvTranspose2d(
                in_channels=width,
                out_channels=1,
                kernel_size=3,
                padding=0,
                stride=3,
                use_bias=use_bias,
                key=keys[2],
            ),
        ]

    def __call__(self, x):
        if x.ndim == 2:
            x = x[None, ...]

        for layer in self.layers:
            x = layer(x)

        return np.squeeze(x)


def update_layers(model, leaf, value):

    new_layers = []
    for layer in model.layers:
        if hasattr(layer, leaf):
            layer = eqx.tree_at(
                lambda model: getattr(model, leaf), layer, getattr(layer, leaf) * value
            )
        new_layers.append(layer)
    return eqx.tree_at(lambda model: model.layers, model, new_layers)


class ConvRNN(eqx.Module):
    illum_encoder: IllumEncoder
    charge_encoder: ChargeEncoder
    rnn_cell: DiffusionDecoder
    time_steps: int = eqx.field(static=True)

    def __init__(self, key, time_steps=10, use_bias=False, bias_value=1.0, weight_value=1.0):
        keys = jr.split(key, 3)

        illum_encoder = IllumEncoder(keys[0], features=2, width=4, use_bias=use_bias)
        charge_encoder = ChargeEncoder(keys[1], features=6, width=12, use_bias=use_bias)
        rnn_cell = DiffusionDecoder(keys[2], features=8, width=16, use_bias=use_bias)

        illum_encoder = update_layers(illum_encoder, "weight", weight_value)
        charge_encoder = update_layers(charge_encoder, "weight", weight_value)
        rnn_cell = update_layers(rnn_cell, "weight", weight_value)

        if use_bias:
            illum_encoder = update_layers(illum_encoder, "bias", bias_value)
            charge_encoder = update_layers(charge_encoder, "bias", bias_value)
            rnn_cell = update_layers(rnn_cell, "bias", bias_value)

        self.illum_encoder = illum_encoder
        self.charge_encoder = charge_encoder
        self.rnn_cell = rnn_cell
        self.time_steps = time_steps

    def __call__(self, bias, illuminance, gain_model, bleed=False):
        # Normalise by the time-steps
        illuminance /= self.time_steps

        # Extract the features from the illuminance and charge
        if bleed:
            illum_features = self.illum_encoder(illuminance)

        # Evolve the charge (include the relative bias)
        # Normalise the bias - we only care about the _relative_ pixel bias for bleeding
        # We take the median in order to avoid badpixels biasing the result
        charge = np.zeros_like(bias - np.median(bias))
        charges, diffusions = [charge], []
        for _ in range(self.time_steps):

            if bleed:
                # Calculate the total diffusion per pixel
                charge_features = self.charge_encoder(charge)
                diffusion = self.rnn_cell(np.concatenate([illum_features, charge_features]))
            else:
                diffusion = np.zeros_like(illuminance)

            # Electrons should have shape (240, 240)
            electrons = illuminance + diffusion

            # Add the new charge
            charge += dlu.downsample(gain_model(bias + charge) * electrons, 3, mean=False)
            charges.append(charge)
            diffusions.append(diffusion)

        return np.array(charges), np.array(diffusions)


class PixelNonLinearity(zdx.Base):
    FF: jax.Array
    SRF: jax.Array
    non_linearity: jax.Array

    def __init__(self, FF=np.ones((80, 80)), poly_order=2, SRF=0.1):
        self.FF = np.array(FF, float)
        self.SRF = np.array(SRF, float)
        self.non_linearity = np.zeros((poly_order, 80, 80))

    def __call__(self, charge):
        """Inputs an (80, 80) charge distribution, return a (240, 240) gain map"""
        # Get the non-linear, per-pixel gain
        coeffs = np.concatenate([self.non_linearity, self.FF[None, ...]], axis=0)
        gain = np.polyval(coeffs, charge)

        # Broadcast the SRF
        srf = quadratic_SRF(self.SRF, 3)
        return broadcast_subpixel(gain, srf)


class RNNRamp(WrapperHolder):
    gain_model: PixelNonLinearity
    bleed: bool
    norm: int

    def __init__(self, conv_rnn, gain_model, norm=2**15, bleed=True):
        values, structure = build_wrapper(conv_rnn)
        self.values = values
        self.structure = structure
        self.norm = norm
        self.gain_model = gain_model
        self.bleed = bleed

    def __getattr__(self, key):
        if hasattr(self.gain_model, key):
            return getattr(self.gain_model, key)
        raise AttributeError(f"RNNRamp has no attribute {key}")

    def evolve_ramp(self, illuminance, ngroups, z_point, sensitivity, badpix):
        # Get the charge (bias) and set badpixels to zero to stop stuff blowing up
        bias = z_point - (dlu.downsample(illuminance, 3, mean=False) / ngroups)
        bias = np.where(badpix, 0.0, bias)
        # bias -= 5e3 # subtract off the 'super bias'

        # Normalise the Illuminance and charge
        illuminance = illuminance / self.norm
        bias = bias / self.norm

        # Evolve the charge using the ConvRNN
        conv_rnn = self.build
        charge_ramp, bleed = conv_rnn(bias, illuminance, self.gain_model, self.bleed)

        # Interpolate the ramps to the number of groups
        return self.norm * interp_ramp(charge_ramp, ngroups), bleed
