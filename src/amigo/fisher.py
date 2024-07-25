import jax
import zodiax as zdx
import equinox as eqx
import jax.numpy as np
from jax import jit, grad, jvp, linearize, lax, vmap
import jax.tree_util as jtu
from .modelling import variance_model
from .stats import posterior
import os

# import tqdm appropriately
from IPython import get_ipython
if get_ipython() is not None:
    # Running in Jupyter Notebook
    from tqdm.notebook import tqdm
else:
    # Running in a script or other non-Jupyter environment
    from tqdm import tqdm


def fisher_fn(model, exposure, params, new_diag=False):
    return FIM(model, params, posterior, exposure, new_diag=new_diag)


def self_fisher_fn(model, exposure, params, read_noise=10, true_read_noise=False, new_diag=False):
    slopes, variance = variance_model(
        model, exposure, true_read_noise=true_read_noise, read_noise=read_noise
    )
    exposure = exposure.set(["slopes", "variance"], [slopes, variance])
    return fisher_fn(model, exposure, params, new_diag=new_diag)


def calc_fisher(
    model,
    exposure,
    param,
    file_path,
    recalculate=False,
    save=True,
    overwrite=False,
    new_diag=False,
):
    # Check that the param exists - caught later
    try:
        leaf = model.get(exposure.map_param(param))
        if not isinstance(leaf, np.ndarray):
            raise ValueError(f"Leaf at path '{param}' is not an array")
        N = leaf.size
    # except KeyError:
    #     print(
    #         f"KeyError: Unable to calculate fisher matrix for {param}",
    #         # f"key: {exposure.get_key(param)} not found in model {param}",
    #         # f"keys: {model.get(param).keys()}"
    #     )
    #     return None
    except ValueError:
        # Param doesn't exist, return None
        return None

    # Check for cached fisher mats
    exists = os.path.exists(file_path)

    # Check if we need to recalculate
    if exists and not recalculate:
        fisher = np.load(file_path)
        if fisher.shape[0] != N:

            # Overwrite shape miss-matches
            if overwrite:
                fisher = self_fisher_fn(model, exposure, [param], new_diag=new_diag)
                if save:
                    np.save(file_path, fisher)
            else:
                raise ValueError(f"Shape mismatch for {param}")

    # Calculate and save
    else:
        fisher = self_fisher_fn(model, exposure, [param], new_diag=new_diag)
        if save:
            np.save(file_path, fisher)
    return fisher


def calc_fishers(
    model,
    exposures,
    parameters,
    param_map_fn=None,
    recalculate=False,
    overwrite=False,
    save=True,
    new_diag=False,
    cache="files/fishers",
):

    if not os.path.exists(cache):
        os.makedirs(cache)

    # Iterate over exposures
    fisher_exposures = {}
    for exp in tqdm(exposures):

        # Iterate over params
        fisher_params = {}
        looper = tqdm(range(0, len(parameters)), leave=False, desc="")
        for idx in looper:
            param = parameters[idx]
            looper.set_description(param)

            # Ensure the path to save to exists
            save_path = f"{cache}/{exp.filename}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Path to the file
            file_path = os.path.join(save_path, f"{param}.npy")

            # Get path correct for parameters
            # param_path = key_mapper(model, exp, param)
            param_path = exp.map_param(param)

            # Allows for custom mapping of parameters
            if param_map_fn is not None:
                param_path = param_map_fn(model, exp, param)

            # Calculate fisher for each exposure
            fisher = calc_fisher(
                model, exp, param_path, file_path, recalculate, save, overwrite, new_diag
            )

            # Store the fisher
            if fisher is not None:
                fisher_params[param] = fisher
            else:
                print(f"Could not calculate fisher for {param_path} - {exp.key}")

        fisher_exposures[exp.key] = fisher_params

    return fisher_exposures


def hessian_diag(fn, x):
    """Source: https://github.com/google/jax/issues/924"""
    eye = np.eye(len(x))
    return np.array(
        [jvp(lambda x: jvp(fn, (x,), (eye[i],))[1], (x,), (eye[i],))[1] for i in range(len(x))]
    )


def hessian(f, x, fast=False):
    if fast:
        # print("Running Vmapped")
        # I think this basically just returns np.eye?
        basis = np.eye(x.size).reshape(-1, *x.shape)

        _, hvp = linearize(grad(f), x)
        hvp = jit(hvp)

        # Compile on first input
        # TODO: I Think this needs to be re-worked so that we call the vmapped jit fn,
        # not the function, then the vmapped version. ie
        # hvp = vmap(jit(hvp))
        # first = hvp(np.array([(basis[0])]))
        first = np.array([hvp(basis[0])])  # Add empty dim for concatenation

        # Vmap others
        others = vmap(hvp)(basis[1:])

        # Recombine
        return np.stack(np.concatenate([first, others], axis=0)).reshape(x.shape + x.shape)
    else:
        # print("Running non-vmapped")
        _, hvp = linearize(grad(f), x)
        # Jit the sub-function here since it is called many times
        # TODO: Test effect on speed
        hvp = jit(hvp)
        # basis = np.eye(np.prod(np.array(x.shape))).reshape(-1, *x.shape)
        basis = np.eye(x.size).reshape(-1, *x.shape)
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


"""
Some code adapted from here: 
https://github.com/google/jax/issues/3801#issuecomment-662131006

More resources:
https://github.com/google/jax/discussions/8456

I believe this efficient hessian diagonal methods only works _correctly_ if the output
hessian is _naturally_ diagonal, else the results are spurious.
"""


def hvp(f, x, v):
    return jvp(grad(f), (x,), (v,))[1]


# TODO: Update this to take the matrix mapper class
def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    shape_dict={},
    save_ram=True,
    vmapped=False,
    diag=False,
    new_diag=False,
    **loglike_kwargs,
):
    # Build X vec
    pytree = zdx.tree.set_array(pytree, parameters)

    if len(parameters) == 1:
        parameters = [parameters]

    leaves = [pytree.get(p) for p in parameters]
    shapes = [leaf.shape for leaf in leaves]
    lengths = [leaf.size for leaf in leaves]
    N = np.array(lengths).sum()
    X = np.zeros(N)

    # Build function to calculate FIM and calculate
    def loglike_fn_vec(X):
        parametric_pytree = _perturb(X, pytree, parameters, shapes, lengths)
        return loglike_fn(parametric_pytree, *loglike_args, **loglike_kwargs)

    if diag:
        diag = hvp(loglike_fn_vec, X, np.ones_like(X))
        return np.eye(diag.shape[0]) * diag[:, None]

    if new_diag:
        return hessian_diag(loglike_fn_vec, X)

    if save_ram:
        return hessian(loglike_fn_vec, X)

    if vmapped:
        return hessian(loglike_fn_vec, X, fast=True)

    return jax.hessian(loglike_fn_vec)(X)


def _perturb(X, pytree, parameters, shapes, lengths):
    n, xs = 0, []
    if isinstance(parameters, str):
        parameters = [parameters]
    indexes = range(len(parameters))

    for i, param, shape, length in zip(indexes, parameters, shapes, lengths):
        if length == 1:
            xs.append(X[i + n])
        else:
            xs.append(lax.dynamic_slice(X, (i + n,), (length,)).reshape(shape))
            n += length - 1

    return pytree.add(parameters, xs)


def recombine(matrices):
    lengths = np.array(jtu.tree_map(lambda x: len(x), matrices))
    mats = np.zeros((lengths.sum(), lengths.sum()))

    idx = 0
    for mat, length in zip(matrices, lengths):
        mats = mats.at[idx : idx + length, idx : idx + length].set(mat)
        idx += length

    return mats


def fix_diag(mat, thresh=1e-16, replace=1.0):
    """
    Some parameters have no effect on the PSF, for example if some pixels are nans. This
    leads to zero gradients on the diagonal of the fisher matrix corresponding to
    those values. If any diagonal entries are zero the inversion return a nan matrix.

    We can fix this by setting the diagonals to one. This should have no effect on the
    result as all correlation terms will also be zero, and the gradients of those
    parameters will also be zero, so they will have no effect
    """
    # Get the indices
    lin_inds = np.arange(len(mat))
    inds = np.array([lin_inds, lin_inds])

    # Fix the diagonal
    diag = np.diag(mat)
    fixed_diag = diag.at[np.where(np.abs(diag) <= thresh)].set(replace)

    # Return the fixed matrix
    return mat.at[*inds].set(fixed_diag)


def create_block_diagonal(size, block_size):
    # Initialize an empty matrix of zeros
    matrix = np.zeros((size, size))

    # Iterate over the matrix in steps of block_size
    for i in range(0, size, block_size):
        # Determine the end of the current block
        end = min(i + block_size, size)

        # Set the elements in the current block to 1
        matrix = matrix.at[i:end, i:end].set(1)

    return matrix


# class MatrixMapper(eqx.Module):
class MatrixMapper(zdx.Base):
    """Class to map matrices to and across pytree leaves."""

    params: list[str] = eqx.field(static=True)
    step_type: str = eqx.field(static=True)
    fisher_matrix: jax.Array
    step_matrix: jax.Array

    def __init__(self, params, fisher_matrix, step_type):
        self.params = params
        self.fisher_matrix = fisher_matrix

        if step_type not in ["matrix", "vector"]:
            raise ValueError("Step type must be 'matrix' or 'vec'")
        self.step_type = step_type

        if self.step_type == "matrix":
            self.step_matrix = -np.linalg.inv(self.fisher_matrix)
        else:
            self.step_matrix = -1.0 / np.diag(self.fisher_matrix)

    def update(self, model, vec):
        idx = 0
        for param in self.params:
            n = model.get(param).size
            leaf = vec[idx : idx + n].reshape(model.get(param).shape)
            model = model.set(param, leaf)
            idx += n
        return model

    def to_vec(self, model):
        return np.concatenate([model.get(p).flatten() for p in self.params])

    def apply(self, model):
        if self.step_type == "matrix":
            return self.update(model, self.step_matrix @ self.to_vec(model))
        elif self.step_type == "vector":
            return self.update(model, self.step_matrix * self.to_vec(model))
        else:
            raise ValueError("Step type must be 'matrix' or 'vector'")

    def get_cross_terms(self):
        raise NotImplementedError("Method not implemented")

    def get_diagonal_terms(self):
        raise NotImplementedError("Method not implemented")
