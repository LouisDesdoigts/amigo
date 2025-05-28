import jax
import zodiax as zdx
import equinox as eqx
import jax.numpy as np
import jax.tree as jtu
from jax.lax import dynamic_slice as lax_slice
from jax.flatten_util import ravel_pytree


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
    optics: None
    vis_model: None
    detector: None
    ramp_model: None
    read: None

    def __init__(self, exposures, optics, detector, ramp_model, read, state=None, vis_model=None):
        if state is not None:
            optics = optics.set("transmission", state["transmission"])
            detector = detector.set("jitter", state["jitter"])
            ramp_model = ramp_model.set(
                ["FF", "SRF", "nn_weights"], [state["FF"], state["SRF"], state["nn_weights"]]
            )
            read = read.set(
                ["dark_current", "non_linearity"],
                [state["dark_current"], state["non_linearity"]],
            )

        params = {}
        for exp in exposures:
            if vis_model is not None:
                param_dict = exp.initialise_params(optics, vis_model=vis_model)
            else:
                param_dict = exp.initialise_params(optics)
            for param, (key, value) in param_dict.items():
                if param not in params.keys():
                    params[param] = {}
                params[param][key] = value

        if state is not None:
            params["defocus"] = state["defocus"]

            abb = {}
            for key in params["aberrations"].keys():
                prog, filt = key.split("_")
                abb[key] = state["aberrations"][filt]

            params["aberrations"] = abb  # jtu.map(lambda x: abb, params["aberrations"])

        # This seems to fix some recompile issues
        def fn(x):
            if isinstance(x, jax.Array):
                if "i" in x.dtype.str:
                    return x
                return np.array(x, dtype=float)
            return x

        self.params = jtu.map(lambda x: fn(x), params)
        self.optics = jtu.map(lambda x: fn(x), optics)
        self.detector = jtu.map(lambda x: fn(x), detector)
        self.ramp_model = jtu.map(lambda x: fn(x), ramp_model)
        self.read = jtu.map(lambda x: fn(x), read)
        self.vis_model = jtu.map(lambda x: fn(x), vis_model)

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        if hasattr(self.vis_model, key):
            return getattr(self.vis_model, key)
        if hasattr(self.ramp_model, key):
            return getattr(self.ramp_model, key)
        if hasattr(self.detector, key):
            return getattr(self.detector, key)
        if hasattr(self.read, key):
            return getattr(self.read, key)

        raise AttributeError(f"{self.__class__.__name__} has no attribute " f"{key}.")


class ModelParams(BaseModeller):

    def __getitem__(self, key):
        return self.params[key]

    def __getattr__(self, key):

        # Make the object act like a real dictionary
        if hasattr(self.params, key):
            return getattr(self.params, key)

        if key in self.params.keys():
            return self.params[key]

        for sub_key, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)

        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    # Remove?
    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys()]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys()]))

    def __add__(self, values):
        matched = self.replace(values)
        return jtu.map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def __mul__(self, values):
        matched = self.replace(values)
        return jtu.map(lambda x, y: x * y, self, matched)

    def __imul__(self, values):
        return self.__mul__(values)

    def map(self, fn):
        return jtu.map(lambda x: fn(x), self)

    def ravel(self, return_unvael=False):
        """Returns the flattened parameters"""
        X, unravel_fn = ravel_pytree(self)
        if return_unvael:
            return X, unravel_fn
        return X

    @property
    def X(self):
        """Returns the flattened parameters"""
        return self.ravel()

    # Re-name this donate, and it counterpart accept, receive?
    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(list(self.keys()), list(self.values()))

    def partition(self, params):
        """params can be a model params object or a list of keys"""
        if isinstance(params, ModelParams):
            params = list(params.params.keys())
        return (
            ModelParams({param: self[param] for param in params}),
            ModelParams({param: self[param] for param in self.keys() if param not in params}),
        )

    def combine(self, params2):
        return ModelParams({**self.params, **params2.params})

    def jacfwd(self, fn, n_batch=1):
        return self.jac(fn, n_batch=n_batch, type="fwd")

    def jacrev(self, fn, n_batch=1):
        return self.jac(fn, n_batch=n_batch, type="rev")

    def jac(self, fn, n_batch=1, type="fwd"):
        # X, unravel_fn = ravel_pytree(self)
        X, unravel_fn = self.ravel(return_unvael=True)
        Xs = np.array_split(X, n_batch)
        rebuild = lambda X_batch, index: X.at[index : index + len(X_batch)].set(X_batch)
        lens = np.cumsum(np.array([len(x) for x in Xs]))[:-1]
        starts = np.concatenate([np.array([0]), lens])

        def batch_fn(x, index):
            model_params = unravel_fn(rebuild(x, index))
            return eqx.filter_jit(fn)(model_params)

        if type == "fwd":
            batched_jac_fn = eqx.filter_jacfwd(batch_fn)

        elif type == "rev":
            batched_jac_fn = eqx.filter_jacrev(batch_fn)

        return np.concatenate([batched_jac_fn(x, index) for x, index in zip(Xs, starts)], axis=-1)


import numpy as onp


class ParamHistory(ModelParams):

    def __init__(self, model_params):
        self.params = jtu.map(lambda x: [onp.array(x)], model_params.params)
        # self.params = jtu.map(lambda x: [x], model_params.params)

    def append(self, model_params):
        # Wrap the leaves in a list to ensure the same tree structure as self.params
        updates_list = jtu.map(lambda x: [onp.array(x)], model_params.params)
        # updates_list = jtu.map(lambda x: [x], model_params.params)

        # We want to append the two dictionaries so we make the tree
        # map make it recognise lists as leaves
        is_leaf = lambda leaf: isinstance(leaf, list)

        # Append the new values to the history dictionary
        return self.set(
            "params",
            jtu.map(lambda a, b: a + b, self.params, updates_list, is_leaf=is_leaf),
        )


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


# class WrapperHolder(zdx.Base):
class NNWrapper(zdx.Base):
    nn_weights: np.ndarray
    structure: EquinoxWrapper

    def __init__(self, eqx_model):
        values, structure = build_wrapper(eqx_model)
        self.nn_weights = values
        self.structure = structure

    def __call__(self, *args, **kwargs):
        return self.build(*args, **kwargs)

    @property
    def build(self):
        return self.structure.inject(self.nn_weights)

    def __getattr__(self, name):
        if hasattr(self.structure, name):
            return getattr(self.structure, name)
        raise AttributeError(f"Attribute {name} not found in {self.__class__.__name__}")
