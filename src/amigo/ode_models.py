import abc
import jax
import equinox as eqx
import zodiax as zdx
import diffrax as dfx
import jax.numpy as np
import jax.tree as jtu
import jax.random as jr
import dLux.utils as dlu
from .ramp_models import Ramp
from .misc import calc_laplacian
from jax import grad, vmap


class Func(eqx.Module):
    """Base class for unified API"""


class ODEFunc(Func):

    @abc.abstractmethod
    def derivative(self, t, y, args):
        raise NotImplementedError


class Polynomial(Func):
    powers: tuple = eqx.field(static=True)
    coeffs: np.ndarray

    def __init__(self, start=1, order=3):
        self.powers = tuple(range(start, order + 1))
        self.coeffs = np.zeros(len(self.powers))

    def evaluate(self, x):
        return np.sum(self.coeffs * x ** np.array(self.powers))

    def evaluate_arr(self, arr):
        return vmap(self.evaluate)(arr.flatten()).reshape(arr.shape)

    @property
    def n(self):
        return len(self.coeffs)


class ODESolver(eqx.Module):
    ODE: ODEFunc

    def __init__(self, ODE):
        self.ODE = ODE

    def __getattr__(self, name):
        if hasattr(self.ODE, name):
            return getattr(self.ODE, name)
        raise AttributeError(f"ODESolver.ODE has no attribute {name}")

    def solve_fn(self, y0, args, ts, dt=0.01):
        return dfx.diffeqsolve(
            y0=y0,
            t0=0.0,
            t1=1.0,
            dt0=dt,
            args=args,
            solver=dfx.Tsit5(),
            saveat=dfx.SaveAt(ts=ts),
            adjoint=dfx.DirectAdjoint(),
            terms=dfx.ODETerm(self.ODE.derivative),
            stepsize_controller=dfx.PIDController(rtol=1e-3, atol=1e-3),
            max_steps=4096,
            throw=True,
        )


class ODERamp(ODESolver):
    oversample: int = eqx.field(static=True)
    norm: int = eqx.field(static=True)

    def __init__(self, ODE, oversample=1, norm=2**15):
        super().__init__(ODE)
        self.oversample = oversample
        self.norm = int(norm)

    def initial_downsample(self, illuminance):
        if self.oversample == 4:
            return illuminance
        if self.oversample == 2:
            return dlu.downsample(illuminance, 2, mean=False)
        if self.oversample == 1:
            return dlu.downsample(illuminance, 4, mean=False)
        raise ValueError("oversample must be 1, 2, or 4")

    def final_downsample(self, ramp):
        dsample_fn = vmap(lambda x, n: dlu.downsample(x, n, mean=False), (0, None))
        if self.oversample == 4:
            return dsample_fn(ramp, 4)
        if self.oversample == 2:
            return dsample_fn(ramp, 2)
        if self.oversample == 1:
            return ramp
        raise ValueError("oversample must be 1, 2, or 4")

    def solve(self, illuminance, ngroups):
        ts = (np.arange(0, ngroups) + 1) / ngroups
        y0 = np.zeros_like(illuminance)
        args = illuminance
        return self.solve_fn(y0, args, ts)

    def predict_ramp(self, illuminance, ngroups):
        illuminance /= self.norm
        illuminance = self.initial_downsample(illuminance)
        ramp = self.solve(illuminance, ngroups).ys
        return self.norm * self.final_downsample(ramp)

    def predict_slopes(self, illuminance, ngroups):
        ramp = self.predict_ramp(illuminance, ngroups)
        return np.diff(ramp, axis=0)

    def apply(self, psf, flux, exposure, oversample):
        ramp = self.predict_ramp(psf.data * flux, exposure.ngroups)
        return Ramp(ramp, psf.pixel_scale)


class BaseNeuralODE(zdx.Base, ODEFunc):
    network: None

    def __init__(self, network):
        self.network = network

    def __getattr__(self, name):
        if hasattr(self.network, name):
            return getattr(self.network, name)
        super().__getattr__(name)

    @abc.abstractmethod
    def eval_network(self, t, charge, illuminance):
        raise NotImplementedError

    def derivative(self, t, charge, illuminance):
        # Linear response to incoming photons
        dqdt = illuminance

        # Nonlinear pixel value, predicted via a CNN
        dqdt += self.eval_network(t, charge, illuminance)
        return dqdt


class MiniCNN(eqx.Module):
    values: list
    tree_def: None
    shapes: list = eqx.field(static=True)
    sizes: list = eqx.field(static=True)
    starts: list = eqx.field(static=True)

    def __init__(self, in_channels, width, out_channels, seed):
        keys = jr.split(jr.PRNGKey(seed), 2)
        layers = [
            eqx.nn.Conv2d(
                in_channels=in_channels,
                out_channels=width,
                kernel_size=3,
                padding=(1, 1),
                key=keys[0],
                use_bias=False,
            ),
            eqx.nn.Conv2d(
                in_channels=width,
                out_channels=out_channels,
                kernel_size=3,
                padding=(1, 1),
                key=keys[1],
                use_bias=False,
            ),
        ]

        leaves, tree_def = jtu.flatten(layers)

        self.values = np.concatenate([val.flatten() for val in leaves])
        self.shapes = [v.shape for v in leaves]
        self.sizes = [int(v.size) for v in leaves]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
        self.tree_def = tree_def

    @property
    def layers(self):
        leaves = [
            jax.lax.dynamic_slice(self.values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return jtu.unflatten(self.tree_def, leaves)

    def __call__(self, x):
        layers = self.layers
        if x.ndim == 2:
            x = x[None, ...]
        for layer in layers[:-1]:
            x = jax.nn.relu(layer(x))
        out = np.squeeze(layers[-1](x))
        return out

    @property
    def size(self):
        return np.sum(np.array(jtu.leaves(jtu.map(lambda x: x.size, self.layers))))


class NeuralODE(BaseNeuralODE):

    def __init__(self, width=16):
        self.network = MiniCNN(2, width, 1, 0)

    def eval_network(self, t, charge, illuminance):
        x = np.array([t * illuminance, calc_laplacian(charge)])
        return self.network(x)


class PolynomialNeuralODE(NeuralODE):
    poly_fn: Polynomial

    def __init__(self, start=1, order=5, width=16):
        self.poly_fn = Polynomial(start, order)
        self.network = MiniCNN(2, width, self.poly_fn.n, 0)

    def __getattr__(self, name):
        if hasattr(self.poly_fn, name):
            return getattr(self.poly_fn, name)
        super().__getattr__(name)

    def eval_poly(self, t, coeffs):
        return eqx.at(lambda x: x.coeffs, self.poly_fn, coeffs).evaluate(t)

    def qt(self, t, charge, illuminance):
        coeffs_vec = self.eval_network(t, charge, illuminance).reshape(self.n, -1)
        eval_fn = vmap(self.eval_poly, (None, 0))
        return eval_fn(t, coeffs_vec.T).reshape(charge.shape)

    def dqdt(self, t, charge, illuminance):
        coeffs_vec = self.eval_network(t, charge, illuminance).reshape(self.n, -1)
        grad_fn = vmap(grad(self.eval_poly), (None, 0))
        return grad_fn(t, coeffs_vec.T).reshape(charge.shape)

    def derivative(self, t, charge, illuminance):
        # Linear response to incoming photons
        dqdt = illuminance

        # Nonlinear pixel value, predicted via a CNN
        # dqdt += self.dqdt(t, charge, illuminance)
        dqdt += self.qt(t, charge, illuminance)
        return dqdt


# Various things to try here:
# - Use qt rather than dqdt
# - Feed in illuminance * t, charge
# - Polynomial as function of t * illuminance, rather than just t
# - Different network architectures
# - Dont predict polynomial coefficients, predict dqdt directly
