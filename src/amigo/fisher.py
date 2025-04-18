import os
import zodiax as zdx
import jax.numpy as np
from jax import jit, grad, linearize, lax, vmap
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

        try:
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

        # Some bug causes non-arrays to be saved. Overwrite them in this case
        except ValueError:
            recalculate = True

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


def hessian(f, x, has_aux=False, batch_size=1):
    # Jit the sub-function here since it is called many times
    if has_aux:
        _, hvp, aux = linearize(grad(f, has_aux=has_aux), x, has_aux=has_aux)
    else:
        _, hvp = linearize(grad(f), x)
    hvp = jit(hvp)

    # Build the basis
    basis = np.eye(x.size).reshape(-1, *x.shape)

    if batch_size == 1:
        return np.stack([hvp(e) for e in basis]).reshape(x.shape + x.shape)

    hvp = vmap(hvp)

    # Break it into batches
    n_batch = np.maximum(1, len(basis) // batch_size)
    basis = np.array_split(basis, n_batch)
    return np.concatenate([hvp(batch) for batch in basis])


def FIM(
    pytree,
    parameters,
    loglike_fn,
    *loglike_args,
    has_aux=False,
    reduce_ram=False,
    batch_size=1,
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

    # Note reduce ram is removed until has_aux is implemented
    if reduce_ram:
        return hessian(loglike_fn_vec, X, has_aux=has_aux, batch_size=batch_size)
    else:
        if has_aux:
            fim, aux = jit(jax.hessian(loglike_fn_vec, has_aux=has_aux))(X)
        else:
            fim = jit(jax.hessian(loglike_fn_vec, has_aux=has_aux))(X)
    return fim


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
