from tqdm.notebook import tqdm
import zodiax as zdx
import jax.numpy as np
import equinox as eqx
import jax
import optax
import jax.tree_util as jtu


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
    # t0 array likes ?
    param_spec = param_spec.set(parameters, groups)
    # for param, group in zip(parameters, groups):
    #     sub_tree = pytree.get(param)
    #     sub_param_spec = param_spec.get(param)
    #     if not isinstance(sub_tree, Array):
    #         param_spec_leaf = jtu.tree_map(lambda _: group, sub_param_spec)
    #     else:
    #         param_spec_leaf = group
    #     param_spec = param_spec.set(param, param_spec_leaf)
    #     # param_spec = param_spec.set(param, group)
    # # print(param_spec)

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
    # print(none_tree)
    # for param in parameter:
    #     sub_tree = pytree.get(param)
    #     if not isinstance(sub_tree, Array):

    #     none_tree = none_tree.set(param, pytree.get(param))
    opt_tree = none_tree.set(parameters, pytree.get(parameters))
    # print(opt_tree)
    opt_state = optim.init(eqx.filter(opt_tree, eqx.is_inexact_array))
    # opt_state = optim.init(opt_tree)

    # Return
    return (optim, opt_state)


from typing import Any
from jax import Array


# Array
def _to_array(leaf: Any):
    if not isinstance(leaf, Array):

        # Try except here allows for leaves to be _pytrees_ and not just arrays.
        # This should probably recursively tree_map the _to_array function in this case.
        # For now, they already all are arrays so this is fine.
        try:
            return np.asarray(leaf, dtype=float)
        except TypeError:
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
    grad_fn=lambda grads, args, optimisers: grads,
    norm_fn=lambda model, args: model,
    update_fn=lambda updates, args: updates,
    print_grads=False,
    verbose=True,
    return_state=False,
    nan_method="none",
):
    """nan_method: str, either 'debug' or 'zero', 'none'"""
    params = list(optimisers.keys())
    opts = list(optimisers.values())

    # model = zdx.set_array(model, params)
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
        raise ValueError(
            f"nan_method must be 'debug', 'zero' or 'none', got {nan_method}"
        )

    # Define faster step function - uses args from inside fn
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)  # Probably not needed anymore
    def step_fn(model, opt_state, args):
        # This disappears when compiled, so use it as a compile check
        print("Step fn compiling...")

        # Calculate the loss and gradient
        loss, grads = val_grad_fn(model, args)

        grads = grad_fn(grads, args, optimisers)
        grads = nan_check(grads)

        # Apply the update
        updates, opt_state = optim.update(grads, opt_state, model)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model, args)
        return model, loss, opt_state

    # Get the params from each model
    from copy import deepcopy
    from CNN import ConvBFE

    params = {}
    for param in optimisers.keys():
        leaf = deepcopy(model.get(param))  # Mother fucker is mutable
        # Store dict as list and append along entries
        if isinstance(leaf, dict):
            for p in leaf.keys():
                leaf[p] = [leaf[p]]
            params[param] = leaf

        # Same with CNN model
        elif isinstance(leaf, ConvBFE):
            weights = []
            biases = []
            for layer in leaf.layers:
                if isinstance(layer, eqx.nn.Conv):
                    weights.append(layer.weight)
                    biases.append(layer.bias)

            params["BFE.weights"] = weights
            params["BFE.biases"] = biases

        else:
            # Else is array, store as list
            params[param] = [leaf]

    def configure_params(model, params):
        print(params)
        for key, value in params.items():
            # If entry is list, we must append along the entries
            if isinstance(value, dict):
                for p in value.keys():
                    value[p].append(model.get(key)[p])
            # If entry is
            elif isinstance(value, ConvBFE):
                weights = []
                biases = []
                for layer in model.layers:
                    if isinstance(layer, eqx.nn.Conv):
                        weights.append(layer.weight)
                        biases.append(layer.bias)
                params["BFE.weights"].append(weights)
                params["BFE.biases"].append(biases)
            else:
                # Else is array, append to list
                params[key].append(model.get(key))

        return params

    import time
    from datetime import timedelta

    # Compile call
    t0 = time.time()
    model, loss, opt_state = step_fn(model, opt_state, args)
    params = configure_params(model, params)

    # Calculate elapsed time
    elapsed_time = time.time() - t0
    formatted_time = str(timedelta(seconds=int(elapsed_time)))

    print(f"Compile Time: {formatted_time}")
    print(f"Initial Loss: {loss:,.2f}")

    # TODO: Add a delta loss parameter too?
    looper = range(1, epochs)
    if verbose:
        looper = tqdm(looper, desc=f"Loss: {loss:,.2f}")

    losses = [loss]
    for i in looper:
        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch, exiting loop")
            if return_state:
                return model, losses, params, opt_state
            return model, losses, params

        model, loss, opt_state = step_fn(model, opt_state, args)

        losses.append(loss)
        params = configure_params(model, params)

        if verbose:
            looper.set_description(f"Loss: {loss:,.2f}")

    print(f"Final Loss: {loss:,.2f}")
    # TODO: Add a "Full execution time" print statement

    if return_state:
        return model, losses, params, opt_state
    return model, losses, params
