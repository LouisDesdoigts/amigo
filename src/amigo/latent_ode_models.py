import jax
import equinox as eqx
import zodiax as zdx
import jax.numpy as np
import jax.tree as jtu
import jax.random as jr
import diffrax as dfx
import interpax as ipx
import dLux.utils as dlu
from jax.lax import dynamic_slice as lax_slice
from amigo.ramp_models import calc_rfield
from amigo.ode_models import ODEFunc
from amigo.ramp_models import Ramp, model_ramp
from jax import vmap


@eqx.filter_jit
def interp_ramp(ramp, ngroups, method="cubic2", extrap=True):
    # Assumes that the ramp time samples are from 0 to 1.
    ts = np.linspace(0, 1, len(ramp))
    groups = (np.arange(ngroups) + 1) / ngroups

    # Build the vectorised interpolator
    interpolator = eqx.filter_vmap(
        lambda f: ipx.interp1d(groups, ts, f, method=method, extrap=extrap),
        in_axes=1,
        out_axes=1,
    )

    # Get the group sample points and interpolate the ramp
    ramp_vec = ramp.reshape(len(ramp), -1)
    return interpolator(ramp_vec).reshape(ngroups, *ramp.shape[1:])


def build_wrapper(eqx_model, filter_fn=eqx.is_array):
    arr_mask = jtu.map(lambda leaf: filter_fn(leaf), eqx_model)
    dyn, static = eqx.partition(eqx_model, arr_mask)
    leaves, tree_def = jtu.flatten(dyn)
    values = np.concatenate([val.flatten() for val in leaves])
    return values, EquinoxWrapper(static, leaves, tree_def)


class EquinoxWrapper(zdx.Base):
    static: eqx.Module
    shapes: list
    sizes: list
    starts: list
    tree_def: None

    def __init__(self, static, leaves, tree_def):
        self.static = static
        self.tree_def = tree_def
        self.shapes = [v.shape for v in leaves]
        self.sizes = [int(v.size) for v in leaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]

    def inject(self, values):
        leaves = [
            lax_slice(values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return eqx.combine(jtu.unflatten(self.tree_def, leaves), self.static)


class WrapperHolder(zdx.Base):
    values: np.ndarray
    structure: EquinoxWrapper

    @property
    def build(self):
        return self.structure.inject(self.values)

    def __getattr__(self, name):
        if hasattr(self.structure, name):
            return getattr(self.structure, name)
        raise AttributeError(f"Attribute {name} not found in {self.__class__.__name__}")


class BaseNeuralNetwork(eqx.Module):
    layers: list

    def __init__(self, layers):
        self.layers = layers

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


def build_encoding_layers(
    in_channels=1,
    width=1,
    depth=1,
    n_latent=1,
    pools=[0, 1],
    key=jr.PRNGKey(0),
    pooling="avg",
    use_bias=False,
):
    """
    Assumes a 4x4 oversampled image.
    """

    # Check the pooling indexes
    if len(pools) != 2:
        raise ValueError("Must have 2 pooling layers")

    # Get the pooling layer
    if pooling == "avg":
        pooling_layer = eqx.nn.AvgPool2d(kernel_size=2, stride=(2, 2))
    elif pooling == "max":
        pooling_layer = eqx.nn.MaxPool2d(kernel_size=2, stride=(2, 2))
    else:
        raise ValueError("Pooling must be 'avg' or 'max'")

    # Get the widths
    if depth == 1:
        widths = np.array([in_channels, width, n_latent])
    else:
        widths = np.linspace(width, n_latent, depth + 1).astype(int)
        widths = np.concatenate([np.array([in_channels]), widths])

    # Build the layers
    layers = []
    keys = jr.split(key, (depth + 1,))
    for i in range(depth + 1):
        layers.append(
            eqx.nn.Conv2d(
                in_channels=int(widths[i]),
                out_channels=int(widths[i + 1]),
                kernel_size=3,
                padding=(1, 1),
                use_bias=use_bias,
                key=keys[i],
            )
        )

        # Add the pooling layer
        if i in pools:
            layers.append(pooling_layer)

        # Add the activation layer
        if i < depth:
            layers.append(jax.nn.relu)
    return layers


def build_decoding_layers(
    n_latent=1,
    # width=1,
    depth=1,
    out_channels=1,
    up_samples=[0, 1],
    key=jr.PRNGKey(0),
    use_bias=False,
):
    """
    Assumes a 4x4 oversampled image.
    """

    # Check the pooling indexes
    if len(up_samples) != 2:
        raise ValueError("Must have 2 up-sampling layers")

    # Check the up-sampling indexes
    if ((depth) < np.array(up_samples)).any():
        raise ValueError("Upsample indexes must be <= depth")

    widths = np.linspace(n_latent, out_channels, depth + 2).astype(int)

    # Build the layers
    layers = []
    keys = jr.split(key, (depth + 1,))
    for i in range(depth + 1):

        if i in up_samples:
            kwargs = dict(output_padding=(-1, -1), stride=(2, 2))
        else:
            kwargs = dict(padding=(1, 1))

        layers.append(
            eqx.nn.ConvTranspose2d(
                in_channels=int(widths[i]),
                out_channels=int(widths[i + 1]),
                kernel_size=3,
                use_bias=use_bias,
                key=keys[i],
                **kwargs,
            )
        )

        # Add the activation layer
        if i < depth:
            layers.append(jax.nn.relu)
    return layers


class ChargeODE(WrapperHolder, ODEFunc):

    def __init__(self, n_latent=16, width=4, depth=1, key=jr.PRNGKey(0)):
        in_size = n_latent + 2  # +2 for q, t
        mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=1,
            width_size=width * in_size,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
            # use_bias=False,
            use_final_bias=False,
        )
        values, structure = build_wrapper(mlp)
        self.values = values
        self.structure = structure

    def derivative(self, t, q, args):
        illuminance, z = args
        x = np.concatenate([z, np.array([q, t])])
        return illuminance + np.squeeze(self.build(x))

    def solve_fn(self, illuminance, z, ts, dt=0.1):
        return dfx.diffeqsolve(
            y0=0.0,
            t0=0.0,
            t1=1.0,
            dt0=dt,
            args=(illuminance, z),
            solver=dfx.Tsit5(),
            saveat=dfx.SaveAt(ts=ts),
            adjoint=dfx.DirectAdjoint(),
            terms=dfx.ODETerm(self.derivative),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
            max_steps=2048,
            throw=True,
        ).ys


class LatentDecoderMLP(WrapperHolder):

    def __init__(self, n_latent=16, width=3, depth=1, key=jr.PRNGKey(0)):
        # decoder = eqx.nn.Linear(
        #     in_features=n_latent,
        #     out_features=1,
        #     use_bias=False,
        #     key=key,
        # )
        # decoder = eqx.nn.MLP(
        #     in_size=n_latent,
        #     out_size=1,
        #     width_size=width * (n_latent),
        #     depth=depth,
        #     activation=jax.nn.relu,
        #     key=key,
        #     use_bias=False,
        #     use_final_bias=False,
        # )
        decoder = eqx.nn.MLP(
            in_size=n_latent,
            out_size=1,
            width_size=width * (n_latent),
            depth=depth,
            activation=jax.nn.relu,
            key=key,
            use_bias=False,
            use_final_bias=False,
        )
        values, structure = build_wrapper(decoder)
        self.values = values
        self.structure = structure

    def decode(self, z):
        return self.build(z)

    def decode_path(self, zs):
        return vmap(self.decode)(zs)


class LatentEncoder(WrapperHolder):

    def __init__(self, width=8, depth=2, n_latent=8, pools=[0, 1], key=jr.PRNGKey(0)):
        CNN = BaseNeuralNetwork(
            build_encoding_layers(
                in_channels=1,
                width=width,
                depth=depth,
                n_latent=n_latent,
                pools=pools,
                key=key,
                use_bias=False,
            )
        )
        values, structure = build_wrapper(CNN)
        self.values = values
        self.structure = structure

    @property
    def FoR(self):
        return calc_rfield(self.build.layers)

    def encode(self, image):
        return self.build(image[None, ...])

    def print_shapes(self):
        x = np.ones((1, 320, 320))
        print("->", x.shape)

        for layer in self.build.layers:
            print(str(type(layer))[8:-2])
            x = layer(x)
            print("->", x.shape)


class LatentDecoder(WrapperHolder):

    def __init__(self, n_latent=16, depth=2, up_samples=[1, 2], out_channels=1, key=jr.PRNGKey(0)):
        CNN = BaseNeuralNetwork(
            build_decoding_layers(
                n_latent=n_latent,
                depth=depth,
                out_channels=out_channels,
                up_samples=up_samples,
                key=key,
            )
        )
        values, structure = build_wrapper(CNN)
        self.values = values
        self.structure = structure

    def decode(self, latent_image):
        return self.build(latent_image)

    def print_shapes(self):
        x = np.ones((self.static.layers[0].in_channels, 80, 80))
        print("->", x.shape)

        for layer in self.build.layers:
            print(str(type(layer))[8:-2])
            x = layer(x)
            print("->", x.shape)


class LatentODE(WrapperHolder, ODEFunc):

    def __init__(self, n_latent=16, width=3, depth=1, key=jr.PRNGKey(0)):
        mlp = eqx.nn.MLP(
            in_size=n_latent + 1,
            out_size=n_latent,
            width_size=width * (n_latent + 1),
            depth=depth,
            activation=jax.nn.relu,
            key=key,
            use_bias=False,
            use_final_bias=False,
        )
        values, structure = build_wrapper(mlp)
        self.values = values
        self.structure = structure

    def derivative(self, t, z, args):
        return self.build(np.concatenate([z, np.array([t])]))

    def solve_fn(self, z, ts, dt=0.1):
        return dfx.diffeqsolve(
            y0=z,
            t0=0.0,
            t1=1.0,
            dt0=dt,
            args=(),
            solver=dfx.Tsit5(),
            saveat=dfx.SaveAt(ts=ts),
            adjoint=dfx.DirectAdjoint(),
            terms=dfx.ODETerm(self.derivative),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
            max_steps=8192,
            throw=True,
        ).ys


class LatentODERamp(zdx.Base):
    encoder: LatentEncoder
    decoder: LatentDecoder
    latent_ode: ODEFunc
    norm: int = eqx.field(static=True)

    def __init__(self, encoder, latent_ode, decoder=None, norm=2**15):
        self.encoder = encoder
        self.decoder = decoder
        self.latent_ode = latent_ode
        self.norm = int(norm)

    def __getattr__(self, name):
        if hasattr(self.encoder, name):
            return getattr(self.encoder, name)
        if hasattr(self.latent_ode, name):
            return getattr(self.latent_ode, name)
        raise AttributeError(f"LatentODERamp.ODE has no attribute {name}")

    def predict_ramp(self, illuminance, ngroups, return_paths=False):
        # Normalise the illuminance
        illuminance /= self.norm

        # Encode the image
        z = self.encoder.encode(illuminance)
        z_vec = z.reshape(len(z), -1).T

        # Solve the ODE
        ts = np.linspace(0, 1, 10)
        latent_path_vec = vmap(lambda z0: self.latent_ode.solve_fn(z0, ts))(z_vec)
        latent_paths = latent_path_vec.T.reshape(-1, len(ts), 80, 80)

        # # (Diffusion) Decode the latent paths
        # ramp_non_linearity_vec = vmap(self.decoder.decode_path)(latent_paths)
        # ramp_non_linearity = ramp_non_linearity_vec.T.reshape(ngroups, 80, 80)
        # ramp = model_ramp(pixel_illuminance, ngroups) + ramp_non_linearity

        # (Gain) Decode the latent paths
        # NOTE: We probably want to do this over a fix set of 10 time sample, so
        # we gain identical performance for all ngroups
        # We vmap the second axis, the _time_ axis
        # gain, diffusion = 1 + vmap(self.decoder.decode, 1, 1)(latent_paths)
        gain = 1 + np.squeeze(vmap(self.decoder.decode, 1)(latent_paths))

        # Apply gain term
        full_ramp = interp_ramp(gain, ngroups) * model_ramp(illuminance, ngroups)
        ramp = vmap(lambda x: dlu.downsample(x, 4, mean=False))(full_ramp)

        # Return the ramp
        if return_paths:
            return self.norm * ramp, latent_paths
        return self.norm * ramp

    def predict_slopes(self, illuminance, ngroups):
        ramp = self.predict_ramp(illuminance, ngroups)
        return np.diff(ramp, axis=0)

    def apply(self, psf, exposure, return_paths=False):
        # out = self.predict_ramp(psf.data * flux, exposure.ngroups, return_paths=return_paths)
        out = self.predict_ramp(psf.data, exposure.ngroups, return_paths=return_paths)
        if return_paths:
            ramp, latent_paths = out
            return Ramp(ramp, psf.pixel_scale), latent_paths
        return Ramp(out, psf.pixel_scale)
