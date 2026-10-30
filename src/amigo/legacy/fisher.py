import os
import jax.numpy as np
from ..misc import tqdm


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
