import zodiax as zdx
import jax.numpy as np
import equinox as eqx
import jax
import optax
import jax.tree_util as jtu
import time
from datetime import timedelta
from typing import Any
from jax import Array
from .core import AmigoHistory

# import tqdm appropriately
from IPython import get_ipython
if get_ipython() is not None:
    # Running in Jupyter Notebook
    from tqdm.notebook import tqdm
else:
    # Running in a script or other non-Jupyter environment
    from tqdm import tqdm

def get_optimiser(pytree, parameters, optimisers):
    # Pre-wrap single inputs into a list since optimisers have a length of 2
    if not isinstance(optimisers, list):
        optimisers = [optimisers]

    # parameters have to default be wrapped in a list to match optimiser
    if isinstance(parameters, str):
        parameters = [parameters]

    # Construct groups and get param_spec
    groups = [str(i) for i in range(len(optimisers))]
    param_spec = jtu.tree_map(lambda _: "null", pytree)

    # Does this need to be tree_mapped with an 'isinstance' to only map optimisers
    # to array likes ?
    param_spec = param_spec.set(parameters, groups)

    # Generate optimiser dictionary and Assign the null group
    opt_dict = dict([(groups[i], optimisers[i]) for i in range(len(groups))])
    opt_dict["null"] = optax.sgd(0.0)

    # Get optimiser object and filtered optimiser
    optim = optax.multi_transform(opt_dict, param_spec)

    # Here we build a 'none tree' that has None's at all the leaves except those we
    # are optimising. This ensures that the leaf shape and dtype matches those returned
    # from the gradient tree. This is important as without this, the PyTreeDef of the
    # 'opt_state' will change when passed through a 'step function', forcing a
    # recompile of the JIT'd function.
    none_tree = jtu.tree_map(lambda _: None, pytree)
    opt_tree = none_tree.set(parameters, pytree.get(parameters))
    opt_state = optim.init(eqx.filter(opt_tree, eqx.is_inexact_array))

    # Return
    return (optim, opt_state)


# Array
def _to_array(leaf: Any):
    if not isinstance(leaf, Array):
        try:
            return np.asarray(leaf, dtype=float)
        except TypeError:
            # TODO: Try recursive tree map here?
            return leaf
    else:
        return leaf


# def set_array(pytree: Base(), parameters: Params) -> Base():
def set_array(pytree, parameters):
    """
    Converts all leaves specified by parameters in the pytree to arrays to
    ensure they have a .shape property for static dimensionality and size
    checks. This allows for 'dynamicly generated' array shapes from the path
    based `parameters` input. This is used for dynamically generating the
    latent X parameter that we need to generate in order to calculate the
    hessian.

    Parameters
    ----------
    pytree : Base()
        The pytree to be converted.
    parameters : Params
        The leaves to be converted to arrays.

    Returns
    -------
    pytree : Base()
        The pytree with the specified leaves converted to arrays.
    """
    new_leaves = jtu.tree_map(_to_array, pytree.get(parameters))
    return pytree.set(parameters, new_leaves)


def optimise(
    model,
    args,
    loss_fn,
    epochs,
    optimisers,
    grad_fn=lambda model, grads, args, optimisers: grads,
    norm_fn=lambda model, args: model,
    update_fn=lambda updates, args: updates,
    print_grads=False,
    verbose=True,
    return_state=False,
    nan_method="none",
    no_history=[],  # List of parameters to NOT track histry for (ie NN weights)
    args_updates=[],
    args_fn=lambda model, args: args,
):
    """nan_method: str, either 'debug' or 'zero', 'none'"""
    params = list(optimisers.keys())
    opts = list(optimisers.values())

    model = set_array(model, params)
    optim, opt_state = get_optimiser(model, params, opts)
    val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)

    if print_grads:
        loss, grads = val_grad_fn(model, args)
        for param in params:
            print(f"{param}: {grads.get(param)}")

    if nan_method == "debug":

        def nan_check(grads):
            bool_tree = jax.tree_map(lambda x: np.isnan(x).any(), grads)
            vals = np.array(jax.tree_util.tree_flatten(bool_tree)[0])
            eqx.debug.breakpoint_if(vals.sum() > 0)
            return grads

    elif nan_method == "zero":

        def nan_check(grads):
            return jax.tree_map(lambda x: np.where(np.isnan(x), 0.0, x), grads)

    elif nan_method == "none":

        def nan_check(grads):
            return grads

    else:
        raise ValueError(f"nan_method must be 'debug', 'zero' or 'none', got {nan_method}")

    # Define faster step function - uses args from inside fn
    @eqx.filter_jit
    # @eqx.debug.assert_max_traces(max_traces=1)  # Probably not needed anymore
    def step_fn(model, opt_state, args):
        # This disappears when compiled, so use it as a compile check
        print("Step fn compiling...")

        # Calculate the loss and gradient
        loss, grads = val_grad_fn(model, args)

        grads = grad_fn(model, grads, args, optimisers)
        grads = nan_check(grads)

        # Apply the update
        updates, opt_state = optim.update(grads, opt_state, model)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model, args)
        return model, loss, opt_state

    # Create model history
    tracked = [param for param in params if param not in no_history]
    model_history = AmigoHistory(model, tracked)

    # Compile call
    t0 = time.time()
    model, loss, opt_state = step_fn(model, opt_state, args)
    elapsed_time = time.time() - t0
    formatted_time = str(timedelta(seconds=int(elapsed_time)))

    print(f"Compile Time: {formatted_time}")
    print(f"Initial Loss: {loss:,.2f}")

    # Append to model histroy
    model_history = model_history.append(model)

    looper = range(1, epochs)
    if verbose:
        looper = tqdm(looper, desc=f"Loss: {loss:,.2f}, Change: {0.}", initial=1, total=epochs)

    losses = [loss]
    for i in looper:
        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch, exiting loop")
            if return_state:
                return model, losses, model_history, opt_state
            return model, losses, model_history

        # if i in args_updates:
        #     args = args_fn(model, args)
        model, args = args_fn(model, args)

        model, _loss, opt_state = step_fn(model, opt_state, args)

        if verbose:
            delta_loss = _loss - loss
            looper.set_description(f"Loss: {loss:,.2f}, Change: {delta_loss:,.2f}")

        loss = _loss
        losses.append(loss)
        # model_history = append_fn(tracked, model_history, model)
        model_history = model_history.append(model)

    # Final execution time
    elapsed_time = time.time() - t0
    formatted_time = str(timedelta(seconds=int(elapsed_time)))

    print(f"Full Time: {formatted_time}")
    print(f"Final Loss: {loss:,.2f}")

    if return_state:
        return model, losses, model_history, opt_state
    return model, losses, model_history


#
