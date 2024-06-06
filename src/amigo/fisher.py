import jax
import zodiax as zdx
import equinox as eqx
import jax.numpy as np
from jax import jit, grad, jvp, linearize, lax, vmap
import jax.tree_util as jtu
import dLux.utils as dlu
from amigo.modelling import variance_model
from amigo.stats import posterior
import os


def hessian(f, x, fast=False):
    if fast:
        # print("Running Vmapped")
        # I think this basically just returns np.eye?
        basis = np.eye(x.size).reshape(-1, *x.shape)

        _, hvp = linearize(grad(f), x)
        hvp = jit(hvp)

        # Compile on first input
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


def fisher_fn(model, exposure, params):
    return FIM(model, params, posterior, exposure)


def self_fisher_fn(model, exposure, params, read_noise=10, true_read_noise=False):
    psf, variance = variance_model(
        model, exposure, true_read_noise=true_read_noise, read_noise=read_noise
    )
    exposure = exposure.set(["data", "variance"], [psf, variance])
    return fisher_fn(model, exposure, params)


def calc_fisher(model, exposure, param, save_path, file_name, recalculate=False):

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # Check that the param exists - caught later
    try:
        leaf = model.get(param)
        if not isinstance(leaf, np.ndarray):
            raise ValueError(f"Leaf at path '{param}' is not an array")
        N = leaf.size
    except ValueError:
        return None

    # Check for cached fisher mats
    file_path = os.path.join(save_path, file_name)
    exists = os.path.exists(file_path)

    # Check if we need to recalculate
    if exists and not recalculate:
        fisher = np.load(file_path)
        if fisher.shape[0] != N:
            raise ValueError(f"Shape mismatch for {param}")
    else:
        fisher = self_fisher_fn(model, exposure, [param])
        np.save(file_path, fisher)
    return fisher


def calc_fishers(
    model,
    exposures,
    parameters,
    local=False,
    recalculate=False,
    cache="files/fishers",
):

    if not os.path.exists(cache):
        os.makedirs(cache)

    fishers = {}

    # Iterate over params
    for param in parameters:

        # Iterate over exposures
        for exp in exposures:

            # Get file and parameter path
            save_path = f"{cache}/{exp.key}/"
            file_name = f"{param}.npy"

            # Get path correct for 'local' parameters
            if local:
                param_path = f"{param}.{exp.key}"
            else:
                param_path = param

            # Calculate fisher for each exposure
            fisher = calc_fisher(model, exp, param_path, save_path, file_name, recalculate)

            # Store the fisher
            if fisher is not None:
                fishers[param_path] = fisher
            else:
                print(f"Could not calculate Fisher for {param_path}")

    return fishers


def load_fisher(fisher_path, value):
    try:
        fisher = np.load(fisher_path)
    except FileNotFoundError:
        # fisher = - np.eye(value.size)
        fisher = np.zeros((value.size, value.size))

        # Ensure sizes match
        if len(fisher) != value.size:
            raise ValueError(f"Shape mismatch for {fisher_path}")

    return fisher


def fisher_to_lr(fisher, value, order=1):
    # Take negative inverse of diagonal (first order)
    if order == 1:

        # Take the negative inverse, ensure zeros are ones
        lr = dlu.nandiv(-1, np.diag(fisher), 1).reshape(value.shape)
    else:
        raise NotImplementedError(
            "Second order fishers not yet implemented, Code needs to be ported "
            "from the MatrixMapper class"
        )

    return lr


def calc_lrs(param_model, exposures, fisher_cache="files/fishers", order=1):

    # Just work with a raw dict here for simplicity
    fisher_dict = jtu.tree_map(lambda x: np.zeros((x.size, x.size)), param_model.params)

    # Loop over parameters and get fishers
    for param_key, value in param_model.params.items():
        # print(param_key)

        # Loop over exposures and accumulate fishers
        for exp in exposures:
            fisher_path = os.path.join(fisher_cache, exp.key, param_key) + ".npy"

            # Dict case
            if isinstance(value, dict):
                fisher = load_fisher(fisher_path, value[exp.key])
                fisher_dict[param_key][exp.key] += fisher

            # Array case
            else:
                fisher = load_fisher(fisher_path, value)
                fisher_dict[param_key] += fisher

    lr_fn = lambda fisher, value: fisher_to_lr(fisher, value, order=order)
    lr_params = jtu.tree_map(lr_fn, fisher_dict, param_model.params)
    return param_model.set("params", lr_params)
