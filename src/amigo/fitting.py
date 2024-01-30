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
    param_spec = param_spec.set(parameters, groups)

    # Generate optimiser dictionary and Assign the null group
    opt_dict = dict([(groups[i], optimisers[i]) for i in range(len(groups))])
    opt_dict["null"] = optax.adam(0.0)

    # Get optimiser object and filtered optimiser
    optim = optax.multi_transform(opt_dict, param_spec)

    # Here we build a 'none tree' that has None's at all the leaves except those we
    # are optimising. This ensures that the leaf shape and dtype matches those returned
    # from the gradient tree. This is important as without this, the PyTreeDef of the
    # 'opt_state' will change when passed through a 'step function', forcing a
    # recompile of the JIT'd function.
    none_tree = jtu.tree_map(lambda _: None, pytree)
    opt_state = optim.init(none_tree.set(parameters, pytree.get(parameters)))

    # Return
    return (optim, opt_state)


def optimise(
    model,
    args,
    loss_fn,
    epochs,
    config,
    grad_fn=lambda grads, args, config: grads,
    norm_fn=lambda model, args: model,
    update_fn=lambda updates, args: updates,
    print_grads=False,
    verbose=True,
    return_state=False,
    nan_method="none",
):
    """nan_method: str, either 'debug' or 'zero', 'none'"""
    params = list(config.keys())
    optimisers = list(config.values())

    model = zdx.set_array(model, params)
    # optim, opt_state = zdx.get_optimiser(model, params, optimisers)
    optim, opt_state = get_optimiser(model, params, optimisers)
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

        grads = grad_fn(grads, args, config)
        grads = nan_check(grads)

        # Apply the update
        updates, opt_state = optim.update(grads, opt_state, model)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model, args)
        return model, loss, opt_state

    # Get the params from each model
    from copy import deepcopy

    params = {}
    for param in config.keys():
        leaf = deepcopy(model.get(param))  # Mother fucker is mutable
        # Store dict as list and append along entries
        if isinstance(leaf, dict):
            for p in leaf.keys():
                leaf[p] = [leaf[p]]
            params[param] = leaf

        else:
            # Else is array, store as list
            params[param] = [leaf]

    def configure_params(model, params):
        for key, value in params.items():
            # If entry is list, we must append along the entries
            if isinstance(value, dict):
                for p in value.keys():
                    value[p].append(model.get(key)[p])

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
        # looper = tqdm(looper, desc=f"Loss: {loss:.2f}")
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
            # looper.set_description("Loss: %.2f" % (loss))
            looper.set_description(f"Loss: {loss:,.2f}")

    print(f"Final Loss: {loss:,.2f}")
    # TODO: Add a "Full execution time" print statement

    if return_state:
        return model, losses, params, opt_state
    return model, losses, params
