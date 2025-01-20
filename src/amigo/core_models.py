import jax
import equinox as eqx
import zodiax as zdx
import jax.numpy as np
import jax.tree_util as jtu
from jax.lax import dynamic_slice as lax_slice

# from .misc import find_position
# from .model_fits import BinaryFit, SplineVisFit

# from .optical_models import AMIOptics
# from .vis_models import SplineVis
# from .detector_models import LinearDetectorModel
# from .ramp_models import SimpleRamp
# from .read_models import ReadModel
# from .files import initialise_params
# from .search_Teffs import get_Teffs
# from .misc import calc_throughput
# from .model_fits import SplineVisFit, BinaryFit


# class Exposure(zdx.Base):
#     """
#     A class to hold all the data relevant to a single exposure, allowing it to be
#     modelled.

#     """

#     slopes: jax.Array
#     variance: jax.Array
#     ramp: jax.Array
#     ramp_variance: jax.Array
#     support: jax.Array
#     badpix: jax.Array
#     nints: int = eqx.field(static=True)
#     filter: str = eqx.field(static=True)
#     star: str = eqx.field(static=True)
#     filename: str = eqx.field(static=True)
#     program: str = eqx.field(static=True)
#     observation: str = eqx.field(static=True)
#     act_id: str = eqx.field(static=True)
#     visit: str = eqx.field(static=True)
#     dither: str = eqx.field(static=True)
#     calibrator: bool = eqx.field(static=True)
#     fit: object = eqx.field(static=True)

#     def __init__(self, file, fit):
#         self.slopes = np.array(file["SLOPE"].data, float)
#         self.variance = np.array(file["SLOPE_ERR"].data, float) ** 2
#         self.badpix = np.array(file["BADPIX"].data, bool)
#         self.support = np.where(~np.array(file["BADPIX"].data, bool))
#         self.ramp = np.asarray(file["RAMP"].data, float)
#         self.ramp_variance = np.asarray(file["RAMP_ERR"].data, float) ** 2
#         self.nints = file[0].header["NINTS"]
#         self.filter = file[0].header["FILTER"]
#         self.star = file[0].header["TARGPROP"]
#         self.observation = file[0].header["OBSERVTN"]
#         self.program = file[0].header["PROGRAM"]
#         self.act_id = file[0].header["ACT_ID"]
#         self.visit = file[0].header["VISITGRP"]
#         self.dither = file[0].header["EXPOSURE"]
#         self.calibrator = bool(file[0].header["IS_PSF"])
#         self.filename = "_".join(file[0].header["FILENAME"].split("_")[:4])
#         self.fit = fit

#     def print_summary(self):
#         print(
#             f"File {self.key}\n"
#             f"Star {self.star}\n"
#             f"Filter {self.filter}\n"
#             f"nints {self.nints}\n"
#             f"ngroups {len(self.slopes)+1}\n"
#         )

#     def initialise_params(self, optics, vis_model=None, amp_order=1):
#         params = {}

#         im = np.where(self.badpix, np.nan, self.slopes[0])
#         psf = np.where(np.isnan(im), 0.0, im)

#         # # Get pixel scale in arcseconds
#         # if hasattr(optics, "focal_length"):
#         #     pixel_scale = dlu.rad2arcsec(1e-6 * optics.psf_pixel_scale / optics.focal_length)
#         # else:
#         #     pixel_scale = optics.psf_pixel_scale
#         params["positions"] = (self.get_key("positions"), find_position(psf, optics.pixel_scale))

#         # Log flux
#         slope_flux = self.ngroups + (1 / self.ngroups)
#         params["fluxes"] = (
#             self.get_key("fluxes"),
#             np.log10(slope_flux * np.nansum(self.slopes[0])),
#         )

#         # Aberrations
#         params["aberrations"] = (
#             self.get_key("aberrations"),
#             np.zeros_like(optics.pupil_mask.abb_coeffs),
#         )

#         # Reflectivity
#         if self.fit.fit_reflectivity:
#             params["reflectivities"] = (
#                 self.get_key("reflectivities"),
#                 np.zeros_like(optics.pupil_mask.amp_coeffs),
#             )

#         # One on fs
#         if self.fit.fit_one_on_fs:
#             params["one_on_fs"] = (
#                 self.get_key("one_on_fs"),
#                 np.zeros((self.ngroups, 80, amp_order + 1)),
#             )

#         # Biases
#         if self.fit.fit_bias:
#             params["biases"] = (self.get_key("biases"), np.zeros((80, 80)))

#         # Visibilities
#         if isinstance(self.fit, SplineVisFit):
#             if vis_model is None:
#                 raise ValueError("vis_model must be provided for SplineVisFit")
#             n = vis_model.knot_inds.size
#             params["amplitudes"] = (self.get_key("amplitudes"), np.ones(n))
#             params["phases"] = (self.get_key("phases"), np.zeros(n))

#         # Binary parameters
#         if isinstance(self.fit, BinaryFit):
#             raise NotImplementedError("BinaryFit initialisation not yet implemented")
#             params["seperation"] = (self.get_key("seperation"), 0.15)
#             params["contrast"] = (self.get_key("contrast"), 2.0)
#             params["position_angle"] = (self.get_key("position_angle"), 0.0)

#         return params

#     # Simple method to give nice syntax for getting keys
#     def get_key(self, param):
#         return self.fit.get_key(self, param)

#     def map_param(self, param):
#         return self.fit.map_param(self, param)

#     @property
#     def ngroups(self):
#         return len(self.slopes) + 1

#     @property
#     def nslopes(self):
#         return len(self.slopes)

#     @property
#     def std(self):
#         return np.sqrt(self.variance)

#     @property
#     def key(self):
#         return "_".join([self.program, self.observation, self.act_id, self.visit, self.dither])

#     def to_vec(self, image):
#         return image[..., *self.support].T

#     def from_vec(self, vec, fill=np.nan):
#         return (fill * np.ones((80, 80))).at[*self.support].set(vec)


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


# def initialise_model(
#     files,
#     fit=SplineVisFit(),
#     optics=AMIOptics(),
#     detector=LinearDetectorModel(),
#     ramp=SimpleRamp(),
#     read=ReadModel(),
#     vis_model=None,
#     nwavels=9,
#     Teff_cache="files/Teff_cache",
# ):
#     exposures = [Exposure(file, fit) for file in files]

#     # filters = {}
#     # for filt in list(set([exp.filter for exp in exposures])):
#     #     filters[filt] = calc_throughput(filt, nwavels=nwavels)
#     # optics = optics.set("filters", filters)

#     if isinstance(fit, SplineVisFit):
#         if vis_model is None:
#             vis_model = SplineVis(optics)
#         else:
#             vis_model = vis_model
#     else:
#         vis_model = None

#     if isinstance(fit, BinaryFit):
#         params = initialise_params(exposures, optics, binary_fit=True)
#     else:
#         params = initialise_params(exposures, optics, vis_model=vis_model)

#     # Add Teffs to params so we can fit it
#     params["Teffs"] = get_Teffs(files, Teff_cache=Teff_cache)
#     model = AmigoModel(params, optics, ramp, detector, read, vis_model)

#     return model, exposures


class AmigoModel(BaseModeller):
    optics: None
    vis_model: None
    detector: None
    # ramp: None
    read: None

    def __init__(self, params, optics, detector, read, vis_model=None):
        self.params = params
        self.optics = optics
        self.detector = detector
        # self.ramp = ramp
        self.read = read
        self.vis_model = vis_model

    # def model(self, exposure, **kwargs):
    #     return exposure.fit(self, exposure, **kwargs)

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
        # if hasattr(self.ramp, key):
        #     return getattr(self.ramp, key)
        if hasattr(self.read, key):
            return getattr(self.read, key)
        if hasattr(self.vis_model, key):
            return getattr(self.vis_model, key)
        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class NNWrapper(zdx.Base):
    values: list
    tree_def: None
    shapes: list
    sizes: list
    starts: list

    def __init__(self, network):
        values, tree_def = jtu.tree_flatten(network)

        self.values = np.concatenate([val.flatten() for val in values])
        self.shapes = [v.shape for v in values]
        self.sizes = [int(v.size) for v in values]
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
