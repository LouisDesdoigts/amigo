import os
import zodiax as zdx
import jax.numpy as np
from jax import jit, grad, linearize, lax
from .misc import tqdm
import jax


def calc_fisher(
    model,
    exposure,
    param,
    file_path,
    fisher_fn,
    recalculate=False,
    overwrite=False,
):
    param_path = exposure.map_param(param)

    # Check that the param exists - caught later
    try:
        leaf = model.get(param_path)
        if not isinstance(leaf, np.ndarray):
            print(f"{exposure.key} - Leaf at path '{param_path}' is not an array.")
            return None
        N = leaf.size
    except ValueError:
        print(f"{exposure.key} - Invalid path {param_path}, no leaf found")
        return None

    # Check for cached fisher mats
    if os.path.exists(file_path):
        fisher = np.load(file_path)

        # Always recalculate nan values
        if np.isnan(fisher).any():
            recalculate = True

        # Check shape matches expectation
        if fisher.shape[0] != N:

            # If overwrite, set recalculate to True
            if overwrite:
                recalculate = True

            # Else raise an error
            else:
                raise ValueError(
                    f"Saved fisher has a shape miss-match for {exposure.key}, {param_path}"
                )

    # File doesn't exists, need to recalculate
    else:
        recalculate = True

    # Finally calculate fisher matrix if needed
    if recalculate:
        # fisher = FIM(model, [param], loss_fn, exposure)
        fisher = fisher_fn(model, exposure, [param])

    # Check for nans
    if np.isnan(fisher).any():
        raise ValueError(f"Fisher matrix has nan value for exposure {exposure.key}, {param_path}")

    return fisher


def calc_fishers(
    model,
    exposures,
    parameters,
    fisher_fn,
    recalculate=False,
    overwrite=False,
    save=True,
    verbose=True,
    cache="files/fishers",
):

    # Ensure the cache directory exists
    if not os.path.exists(cache):
        os.makedirs(cache)

    # Set up tqdm looper if verbose
    if verbose:
        looper = tqdm(exposures, desc="")
    else:
        looper = exposures

    # Loop over exposures
    fishers = {}
    for exp in looper:

        # Iterate over params
        for param in parameters:

            # Update the looper if verbose
            if verbose:
                looper.set_description(param)

            # Ensure the path to save to exists
            save_path = f"{cache}/{exp.filename}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Get the path to the file
            file_path = os.path.join(save_path, f"{param}.npy")

            # Get path correct for parameters
            param_path = exp.map_param(param)

            # Calculate fisher for each exposure
            fisher = calc_fisher(
                model, exp, param_path, file_path, fisher_fn, recalculate, overwrite
            )

            # Cache the fisher matrix
            if save:
                np.save(file_path, fisher)

            # Put the fisher matrix into the dictionary
            fishers[f"{exp.key}.{param}"] = fisher

    return fishers


"""
Some code adapted from here: 
https://github.com/google/jax/issues/3801#issuecomment-662131006

More resources:
https://github.com/google/jax/discussions/8456

I believe this efficient hessian diagonal methods only works _correctly_ if the output
hessian is _naturally_ diagonal, else the results are spurious.
"""


def hessian(f, x):
    # Jit the sub-function here since it is called many times
    _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)

    # Build and stack
    basis = np.eye(x.size).reshape(-1, *x.shape)
    return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)


def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
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

    # return hessian(loglike_fn_vec, X)
    return jit(jax.hessian(loglike_fn_vec))(X)


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
