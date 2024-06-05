import jax
import jax.numpy as np
import jax.tree_util as jtu
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
from jax.lax import dynamic_slice as lax_slice
from .misc import planck
from .optical_layers import AMIOptics
from .detector_layers import SUB80Ramp


class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):

        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values


class AmigoModel(BaseModeller):
    Teffs: dict
    filters: dict
    optics: AMIOptics
    detector: SUB80Ramp
    vis_model: None

    def __init__(self, params, optics, detector, Teffs, filters, vis_model=None):
        self.params = params
        self.Teffs = Teffs
        self.filters = filters
        self.optics = optics
        self.detector = detector
        self.vis_model = vis_model

    def model(self, exposure, **kwargs):
        return self.model_exposure(exposure, **kwargs)

    def model_psf(self, pos, wavels, weights):

        wfs = self.optics.propagate(wavels, dlu.arcsec2rad(pos), weights, return_wf=True)

        psfs = wfs.psf
        if self.vis_model is not None:
            psf = self.vis_model(psfs)
        else:
            psf = psfs.sum(0)
        return dl.PSF(psf, wfs.pixel_scale.mean(0))

    def model_detector(self, psf, to_BFE=False):

        for key, layer in self.detector.layers.items():
            if key == "EDM" and to_BFE:
                return psf.data
            psf = layer.apply(psf)
        return psf.data

    def model_exposure(self, exposure, to_BFE=False, slopes=False):
        # Get exposure key
        key = exposure.key

        # Get wavelengths and weights
        wavels, filt_weights = self.filters[exposure.filter]
        weights = filt_weights * planck(wavels, self.Teffs[exposure.star])
        weights = weights / weights.sum()

        position = self.positions[key]
        flux = 10 ** self.fluxes[key]
        aberrations = self.aberrations[key]
        one_on_fs = self.one_on_fs[key]
        # dark_current = self.dark_current
        opd = exposure.opd

        optics = self.optics.set(["pupil.coefficients", "pupil.opd"], [aberrations, opd])

        if "coherence" in self.params.keys():
            coherence = self.coherence[key]
            optics = optics.set("holes.reflectivity", coherence)

        detector = self.detector.set(
            ["EDM.ngroups", "EDM.flux", "EDM.filter", "one_on_fs"],
            [exposure.ngroups, flux, exposure.filter, one_on_fs],
        )  # , dark_current])

        self = self.set(["optics", "detector"], [optics, detector])
        psf = self.model_psf(position, wavels, weights)
        ramp = self.model_detector(psf, to_BFE=to_BFE)

        if to_BFE:
            return ramp

        if slopes:
            return np.diff(ramp, axis=0)

        return ramp

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        if hasattr(self.detector, key):
            return getattr(self.detector, key)
        if hasattr(self.vis_model, key):
            return getattr(self.vis_model, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be
    modelled.

    """

    data: jax.Array
    variance: jax.Array
    zero_point: jax.Array
    support: jax.Array = eqx.field(static=True)
    opd: jax.Array = eqx.field(static=True)
    nints: int = eqx.field(static=True)
    ngroups: int = eqx.field(static=True)
    nslopes: int = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    star: str = eqx.field(static=True)
    key: str = eqx.field(static=True)

    def __init__(self, file, data, variance, support, opd, key_fn):

        self.data = data
        self.variance = variance
        self.support = support
        self.opd = opd
        self.key = key_fn(file)
        self.nints = file[0].header["NINTS"]
        self.ngroups = file[0].header["NGROUPS"]
        self.nslopes = file[0].header["NGROUPS"] - 1
        self.filter = file[0].header["FILTER"]
        self.star = file[0].header["TARGPROP"]
        self.zero_point = np.asarray(file["ZPOINT"].data, float)

    def print_summary(self):
        print(
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"Filter {self.filter}\n"
            f"nints {self.nints}\n"
            f"ngroups {self.ngroups}\n"
        )

    def to_vec(self, image):
        return image[..., *self.support].T

    def from_vec(self, vec, fill=np.nan):
        return (fill * np.ones((80, 80))).at[*self.support].set(vec)


class ExposureFit(Exposure):
    position: jax.Array
    aberrations: jax.Array
    flux: jax.Array  # Log now
    one_on_fs: jax.Array
    coherence: jax.Array

    def __init__(self, exposure, position, flux, FDA, one_on_fs, coherence):

        self.data = exposure.data
        self.variance = exposure.variance
        self.support = exposure.support
        self.opd = exposure.opd
        self.key = exposure.key
        self.nints = exposure.nints
        self.ngroups = exposure.ngroups
        self.nslopes = exposure.nslopes
        self.filter = exposure.filter
        self.star = exposure.star
        self.zero_point = exposure.zero_point
        self.aberrations = FDA
        self.position = position
        self.flux = flux
        self.one_on_fs = one_on_fs
        self.coherence = coherence


class NNWrapper(zdx.Base):
    values: list
    shapes: list = eqx.field(static=True)
    sizes: list = eqx.field(static=True)
    starts: list = eqx.field(static=True)
    tree_def: None = eqx.field(static=True)

    def __init__(self, network):
        values, tree_def = jtu.tree_flatten(network)

        self.values = np.concatenate([val.flatten() for val in values])
        self.shapes = [v.shape for v in values]
        self.sizes = [v.size for v in values]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
        self.tree_def = tree_def

    @property
    def _layers(self):
        leaves = [
            lax_slice(self.values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return jtu.tree_unflatten(self.tree_def, leaves)

    def __call__(self, x):
        layers = self._layers
        for layer in layers[:-1]:
            x = jax.nn.relu(layer(x))
        return layers[-1](x)


class ModelParams(BaseModeller):

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def __getattr__(self, key):
        # print("In get attr")
        if key in self.keys:
            return self.params[key]
        for k, val in self.params.items():
            # print(k)
            if hasattr(val, key):
                return getattr(val, key)
        # return self.get(key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys]))

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(self.keys, self.values)


def _is_tree(x):
    """
    Here we check if the leaf is a leaf, or a tree. If it is a tree, we tree_map the
    operation around the leaves of that tree. We use the eqx.is_array_like to check if
    the leaf is a tree, but this could also be done with
    `isinstance(leaf, (list, dict, tuple, eqx.Module))`. The differences between these
    two methods needs to be investigated.
    """
    # return isinstance(x, (list, dict, tuple, eqx.Module))
    return not eqx.is_array_like(x)


class ModelHistory(ModelParams):
    """
    Tracks the history of a set of parameters in a model via tuples.

    Adds a series of convenience functions to interface with it.

    This could have issues with leaves not being jax.Arrays, so at some point it should be
    explicitly enforced that only array_likes are tracked.
    """

    def __init__(self, model, tracked):

        history = {}
        for param in tracked:
            leaf = model.get(param)
            if _is_tree(leaf):
                history[param] = jtu.tree_map(lambda sub_leaf: [sub_leaf], leaf)
            else:
                history[param] = [leaf]

        self.params = history

    def append(self, model):
        history = self.params
        for param, leaf_history in history.items():
            if hasattr(model, param):
                new_leaf = getattr(model, param)
            else:
                new_leaf = model.get(param)

            # Tree-like case
            if _is_tree(new_leaf):
                append_fn = lambda history, value: history + [value]
                leaf_fn = lambda leaf: isinstance(leaf, list)
                new_leaf_history = jtu.tree_map(append_fn, leaf_history, new_leaf, is_leaf=leaf_fn)
                history[param] = new_leaf_history

            # Non-tree case
            else:
                history[param] = leaf_history + [new_leaf]
        return self.set("params", history)
