import os
import zodiax as zdx
import jax.numpy as np
from jax import jit, grad, linearize, lax
from .stats import posterior, variance_model


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
    except ValueError:
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
    # for exp in tqdm(exposures):
    for exp in exposures:

        # Iterate over params
        fisher_params = {}
        # looper = tqdm(range(0, len(parameters)), leave=False, desc="")
        for idx in range(0, len(parameters)):
            param = parameters[idx]
            # looper.set_description(param)

            # Ensure the path to save to exists
            save_path = f"{cache}/{exp.filename}/"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Path to the file
            file_path = os.path.join(save_path, f"{param}.npy")

            # Get path correct for parameters
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

    return hessian(loglike_fn_vec, X)


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
