import jax
import jax.numpy as np
import jax.tree_util as jtu
import equinox as eqx
import zodiax as zdx
from jax.lax import dynamic_slice as lax_slice
from .optics import AMIOptics
from .detectors import LinearDetectorModel, ReadModel
from .ramps import SimpleRamp
from .modelling import model_exposure
from .files import get_Teffs, get_filters


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
    visibilities: None
    detector: None
    ramp: None
    read: None

    def __init__(
        self,
        files,
        # exposures,
        params,
        optics=None,
        ramp=None,
        detector=None,
        read=None,
        visibilities=None,
    ):

        if optics is None:
            optics = AMIOptics()
        if detector is None:
            detector = LinearDetectorModel()
        if ramp is None:
            ramp = SimpleRamp()
        if read is None:
            read = ReadModel()

        self.Teffs = get_Teffs(files)
        self.filters = get_filters(files)
        self.optics = optics
        self.detector = detector
        self.ramp = ramp
        self.read = read
        self.visibilities = visibilities
        self.params = params

    def model(self, exposure, **kwargs):
        return model_exposure(self, exposure, **kwargs)

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
        if hasattr(self.ramp, key):
            return getattr(self.ramp, key)
        if hasattr(self.read, key):
            return getattr(self.read, key)
        if hasattr(self.visibilities, key):
            return getattr(self.visibilities, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be
    modelled.

    """

    data: jax.Array
    variance: jax.Array
    zero_point: jax.Array
    support: jax.Array
    opd: jax.Array
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

    def __init__(self, exposure, position, flux, aberrations, one_on_fs, coherence):

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
        self.aberrations = aberrations
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
        if key in self.keys:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
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

    def __mul__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(self.keys, self.values)


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
            if not eqx.is_array_like(leaf):
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
            if not eqx.is_array_like(new_leaf):
                append_fn = lambda history, value: history + [value]
                leaf_fn = lambda leaf: isinstance(leaf, list)
                new_leaf_history = jtu.tree_map(append_fn, leaf_history, new_leaf, is_leaf=leaf_fn)
                history[param] = new_leaf_history

            # Non-tree case
            else:
                history[param] = leaf_history + [new_leaf]
        return self.set("params", history)
