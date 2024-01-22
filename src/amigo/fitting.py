"""This probably ends up as a zodiax function"""

from tqdm.notebook import tqdm
import zodiax as zdx
import jax.numpy as np


# def optimise(
#     model,
#     args,
#     loss_grad_fn,
#     epochs,
#     config,
#     # grad_fn=lambda grads, config, epoch: grads,
#     norm_fn=lambda model: model,
#     print_grads=False,
#     verbose=True,
# ):
#     params = list(config.keys())
#     optimisers = list(config.values())

#     model = zdx.set_array(model, params)
#     optim, opt_state = zdx.get_optimiser(model, params, optimisers)

#     loss, grads = loss_grad_fn(model, args)
#     if print_grads:
#         for param in params:
#             print(f"{param}: {grads.get(param)}")

#     # Define faster step function
#     @zdx.filter_jit
#     def step_fn(model, opt_state):  # , epoch):
#         # calculate the loss and gradient
#         loss, grads = loss_grad_fn(model, args)

#         # # Apply any processing to the gradients
#         # grads = grad_fn(grads, config, epoch)

#         # apply the update
#         updates, opt_state = optim.update(grads, opt_state)
#         model = zdx.apply_updates(model, updates)

#         # Apply normalisation
#         model = norm_fn(model)

#         return model, loss, opt_state

#     # Compile
#     _ = step_fn(model, opt_state)

#     if verbose:
#         looper = tqdm(range(epochs), desc="Loss %.2f" % (loss))
#     else:
#         looper = range(epochs)

#     # Get the params from each model
#     params = {}
#     for param in config.keys():
#         params[param] = [model.get(param)]

#     losses = []
#     for i in looper:
#         # model, loss, opt_state = step_fn(model, opt_state, i)
#         model, loss, opt_state = step_fn(model, opt_state)
#         # # calculate the loss and gradient
#         # loss, grads = loss_fn(model, args)

#         if np.isnan(loss):
#             print(f"Loss is NaN on {i} th epoch, exiting loop")
#             return model, losses, params

#         # save results
#         losses.append(loss)

#         for param in config.keys():
#             params[param].append(model.get(param))

#         if verbose:
#             looper.set_description("Loss %.2f" % (loss))

#     return model, losses, params


from tqdm.notebook import tqdm
import zodiax as zdx
import jax.numpy as np
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


# def optimise(
#     model,
#     args,
#     loss_fn,
#     epochs,
#     config,
#     # grad_fn=lambda grads, config, epoch: grads,
#     norm_fn=lambda model: model,
#     print_grads=False,
#     verbose=True,
# ):
#     params = list(config.keys())
#     optimisers = list(config.values())

#     model = zdx.set_array(model, params)
#     optim, opt_state = zdx.get_optimiser(model, params, optimisers)
#     val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)

#     if print_grads:
#         loss, grads = val_grad_fn(model, args)
#         for param in params:
#             print(f"{param}: {grads.get(param)}")

#     # Define faster step function - uses args from inside fn
#     @zdx.filter_jit
#     def step_fn(model, opt_state, args):
#         # This disappears when compiled, so use it as a compile check
#         print("Step fn: Python version running (compiling)")

#         # calculate the loss and gradient
#         loss, grads = val_grad_fn(model, args)

#         # apply the update
#         updates, opt_state = optim.update(grads, opt_state, model)
#         model = zdx.apply_updates(model, updates)

#         # Apply normalisation
#         model = norm_fn(model)
#         return model, loss, opt_state

#     # Get the params from each model
#     from copy import deepcopy

#     params = {}
#     for param in config.keys():
#         leaf = deepcopy(model.get(param))  # Mother fucker is mutable
#         # Store dict as dist and append along entries
#         if isinstance(leaf, dict):
#             for p in leaf.keys():
#                 leaf[p] = [leaf[p]]
#             params[param] = leaf

#         else:
#             # Else is array, store as list
#             params[param] = [leaf]

#     def configure_params(model, params):
#         for key, value in params.items():
#             # If entry is list, we must append along the entries
#             if isinstance(value, dict):
#                 for p in value.keys():
#                     value[p].append(model.get(key)[p])

#             else:
#                 # Else is array, append to list
#                 params[key].append(model.get(key))

#         return params

#     # from jax import tree_map
#     # shape = lambda x: x.shape
#     # dtype = lambda x: x.dtype

#     # fs = zdx.tree.boolean_filter(model, params)
#     # m = eqx.filter(model, fs)
#     # print(tree_map(shape, m))
#     # print(tree_map(dtype, m))

#     # Compile call used as first epoch to circumvent odd recompiles
#     model, loss, opt_state = step_fn(model, opt_state, args)
#     params = configure_params(model, params)

#     looper = range(1, epochs)
#     if verbose:
#         looper = tqdm(looper, desc=f"Loss: {loss:.2f}")

#     losses = [loss]
#     for i in looper:
#         if np.isnan(loss):
#             print(f"Loss is NaN on {i} th epoch, exiting loop")
#             return model, losses, params

#         model, loss, opt_state = step_fn(model, opt_state, args)

#         losses.append(loss)
#         params = configure_params(model, params)

#         if verbose:
#             looper.set_description("Loss: %.2f" % (loss))

#     return model, losses, params

import equinox as eqx


def optimise(
    model,
    args,
    loss_fn,
    epochs,
    config,
    # grad_fn=lambda grads, config, epoch: grads,
    norm_fn=lambda model: model,
    print_grads=False,
    verbose=True,
):
    params = list(config.keys())
    optimisers = list(config.values())

    model = zdx.set_array(model, params)
    # optim, opt_state = zdx.get_optimiser(model, params, optimisers)
    optim, opt_state = get_optimiser(model, params, optimisers)
    flat1, tree_def1 = jtu.tree_flatten(opt_state)
    val_grad_fn = zdx.filter_value_and_grad(params)(loss_fn)

    if print_grads:
        loss, grads = val_grad_fn(model, args)
        for param in params:
            print(f"{param}: {grads.get(param)}")

    # Define faster step function - uses args from inside fn
    @eqx.filter_jit
    @eqx.debug.assert_max_traces(max_traces=1)
    def step_fn(model, opt_state, args):
        # This disappears when compiled, so use it as a compile check
        print("Step fn: Python version running (compiling)")

        # calculate the loss and gradient
        loss, grads = val_grad_fn(model, args)

        # apply the update
        updates, opt_state = optim.update(grads, opt_state, model)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model)
        return model, loss, opt_state

    # Get the params from each model
    from copy import deepcopy

    params = {}
    for param in config.keys():
        leaf = deepcopy(model.get(param))  # Mother fucker is mutable
        # Store dict as dist and append along entries
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

    # Compile call used as first epoch to circumvent odd recompiles
    model, loss, opt_state = step_fn(model, opt_state, args)
    params = configure_params(model, params)

    looper = range(1, epochs)
    if verbose:
        looper = tqdm(looper, desc=f"Loss: {loss:.2f}")

    losses = [loss]
    for i in looper:
        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch, exiting loop")
            return model, losses, params

        model, loss, opt_state = step_fn(model, opt_state, args)

        losses.append(loss)
        params = configure_params(model, params)

        if verbose:
            looper.set_description("Loss: %.2f" % (loss))

    return model, losses, params
