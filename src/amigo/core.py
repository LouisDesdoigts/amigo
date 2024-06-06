import jax
import jax.numpy as np
import jax.tree_util as jtu
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
from jax.lax import dynamic_slice as lax_slice
from .optics import AMIOptics
from .detectors import LinearDetectorModel, ReadModel
from .detectors import SimpleRamp
from .modelling import planck
from amigo.detectors import model_ramp
from xara.core import determine_origin
import pkg_resources as pkg
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


def find_position(psf, pixel_scale=0.065524085):
    origin = np.array(determine_origin(psf, verbose=False))
    origin -= (np.array(psf.shape) - 1) / 2
    origin += np.array([0.5, 0.5])
    position = origin * pixel_scale * np.array([1, -1])
    return position


def initialise_params(exposures, optics, pre_calc_FDA=False, amp_order=1):
    positions = {}
    fluxes = {}
    aberrations = {}
    one_on_fs = {}
    aberrations = {}
    for exp in exposures:

        im = exp.data[0]
        psf = np.where(np.isnan(im), 0.0, im)
        flux = np.log10(1.05 * exp.ngroups * np.nansum(exp.data[0]))
        position = find_position(psf, optics.psf_pixel_scale)
        n_fda = optics.pupil.coefficients.shape[1]

        if pre_calc_FDA:
            file_path = pkg.resource_filename(__name__, "data/FDA_coeffs.npy")
            coeffs = np.load(file_path)[:, :n_fda]
        else:
            coeffs = np.zeros((7, n_fda))

        positions[exp.key] = position
        aberrations[exp.key] = coeffs
        fluxes[exp.key] = flux
        one_on_fs[exp.key] = np.zeros((exp.ngroups, 80, amp_order + 1))

    return {
        "positions": positions,
        "fluxes": fluxes,
        "aberrations": aberrations,
        "one_on_fs": one_on_fs,
    }


class AmigoModel(BaseModeller):
    Teffs: dict
    filters: dict
    optics: AMIOptics
    vis_model: None
    linear_detector: None
    non_linear_detector: None
    read_detector: None

    def __init__(
        self,
        files,
        exposures,
        optics=None,
        non_linear_detector=None,
        linear_detector=None,
        read_detector=None,
        vis_model=None,
    ):

        if optics is None:
            optics = AMIOptics()
        if linear_detector is None:
            linear_detector = LinearDetectorModel()
        if non_linear_detector is None:
            non_linear_detector = SimpleRamp()
        if read_detector is None:
            read_detector = ReadModel()

        params = initialise_params(exposures, optics)
        self.params = params
        self.Teffs = get_Teffs(files)
        self.filters = get_filters(files)
        self.optics = optics
        self.linear_detector = linear_detector
        self.non_linear_detector = non_linear_detector
        self.read_detector = read_detector
        self.vis_model = vis_model

    def model(self, exposure, **kwargs):
        return self.model_exposure(exposure, **kwargs)

    def model_exposure(self, exposure, to_BFE=False, slopes=False):
        # Get wavelengths and weights
        wavels, filt_weights = self.filters[exposure.filter]
        weights = filt_weights * planck(wavels, self.Teffs[exposure.star])
        weights = weights / weights.sum()

        optics = self.optics.set(
            ["pupil.coefficients", "pupil.opd"], [self.aberrations[exposure.key], exposure.opd]
        )

        if "coherence" in self.params.keys():
            coherence = self.coherence[exposure.key]
            optics = optics.set("holes.reflectivity", coherence)

        # Model the optics
        pos = dlu.arcsec2rad(self.positions[exposure.key])
        wfs = optics.propagate(wavels, pos, weights, return_wf=True)

        psfs = wfs.psf
        if self.vis_model is not None:
            psf = self.vis_model(psfs)
        else:
            psf = psfs.sum(0)

        # PSF is still unitary here
        psf = self.linear_detector.apply(dl.PSF(psf, wfs.pixel_scale.mean(0)))

        # Get the hyper-parameters for the non-linear model
        flux = 10 ** self.fluxes[exposure.key]
        oversample = optics.oversample

        # Return the BFE and required meta-data
        if to_BFE:
            return psf, flux, oversample

        # Non linear model always goes from unit psf, flux, oversample to an 80x80 ramp
        if self.non_linear_detector is not None:
            ramp = self.non_linear_detector.apply(psf, flux, exposure, oversample)
        else:
            psf_data = dlu.downsample(psf.data * flux, oversample, mean=False)
            ramp = psf.set("data", model_ramp(psf_data, exposure.ngroups))

        # Model the read effects
        one_on_fs = self.one_on_fs[exposure.key]
        ramp = self.read_detector.set("one_on_fs", one_on_fs).apply(ramp)

        # Return the slopes if required
        if slopes:
            return np.diff(ramp.data, axis=0)

        # Return the ramp
        return ramp.data

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        if hasattr(self.linear_detector, key):
            return getattr(self.linear_detector, key)
        if hasattr(self.non_linear_detector, key):
            return getattr(self.non_linear_detector, key)
        if hasattr(self.read_detector, key):
            return getattr(self.read_detector, key)
        if hasattr(self.vis_model, key):
            return getattr(self.vis_model, key)
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
