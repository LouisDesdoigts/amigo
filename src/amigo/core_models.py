import jax
import zodiax as zdx
import jax.tree as jtu


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

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys()]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys()]))

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

    def map(self, fn):
        return jtu.map(lambda x: fn(x), self)

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


class ParamHistory(ModelParams):

    def __init__(self, model_params):
        self.params = jtu.map(lambda x: [x], model_params.params)

    def append(self, model_params):
        # Wrap the leaves in a list to ensure the same tree structure as self.params
        updates_list = jtu.map(lambda x: [x], model_params.params)

        # We want to append the two dictionaries so we make the tree
        # map make it recognise lists as leaves
        is_leaf = lambda leaf: isinstance(leaf, list)

        # Append the new values to the history dictionary
        return self.set(
            "params",
            jtu.map(lambda a, b: a + b, self.params, updates_list, is_leaf=is_leaf),
        )


# class ModelHistory(ModelParams):
#     """
#     Tracks the history of a set of parameters in a model via tuples.

#     Adds a series of convenience functions to interface with it.

#     This could have issues with leaves not being jax.Arrays, so at some point it should be
#     explicitly enforced that only array_likes are tracked.
#     """

#     def __init__(self, model, tracked):

#         history = {}
#         for param in tracked:
#             leaf = model.get(param)
#             if not eqx.is_array_like(leaf):
#                 history[param] = jtu.map(lambda sub_leaf: [sub_leaf], leaf)
#             else:
#                 history[param] = [leaf]

#         self.params = history

#     def append(self, model):
#         history = self.params
#         for param, leaf_history in history.items():
#             if hasattr(model, param):
#                 new_leaf = getattr(model, param)
#             else:
#                 new_leaf = model.get(param)

#             # Tree-like case
#             if not eqx.is_array_like(new_leaf):
#                 append_fn = lambda history, value: history + [value]
#                 leaf_fn = lambda leaf: isinstance(leaf, list)
#                 new_leaf_history = jtu.map(append_fn, leaf_history, new_leaf, is_leaf=leaf_fn)
#                 history[param] = new_leaf_history

#             # Non-tree case
#             else:
#                 history[param] = leaf_history + [new_leaf]
#         return self.set("params", history)


# class ParamHistory(ModelParams):

#     def __init__(self, model_params):
#         self.params = jtu.map(lambda x: [], model_params)

#     def append(self, model_params):

#         history = self.params
#         for param, leaf_history in history.items():
#             if hasattr(model, param):
#                 new_leaf = getattr(model, param)
#             else:
#                 new_leaf = model.get(param)

#             # Tree-like case
#             if not eqx.is_array_like(new_leaf):
#                 append_fn = lambda history, value: history + [value]
#                 leaf_fn = lambda leaf: isinstance(leaf, list)
#                 new_leaf_history = jtu.map(append_fn, leaf_history, new_leaf, is_leaf=leaf_fn)
#                 history[param] = new_leaf_history

#             # Non-tree case
#             else:
#                 history[param] = leaf_history + [new_leaf]
#         return self.set("params", history)


# class NNWrapper(zdx.Base):
#     values: list
#     tree_def: None
#     shapes: list
#     sizes: list
#     starts: list

#     def __init__(self, network):
#         values, tree_def = jtu.flatten(network)

#         self.values = np.concatenate([val.flatten() for val in values])
#         self.shapes = [v.shape for v in values]
#         self.sizes = [int(v.size) for v in values]
#         self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
#         self.tree_def = tree_def

#     @property
#     def _layers(self):
#         leaves = [
#             lax_slice(self.values, (start,), (size,)).reshape(shape)
#             for start, size, shape in zip(self.starts, self.sizes, self.shapes)
#         ]
#         return jtu.unflatten(self.tree_def, leaves)

#     def __call__(self, x):
#         layers = self._layers
#         for layer in layers[:-1]:
#             x = jax.nn.relu(layer(x))
#         return layers[-1](x)
