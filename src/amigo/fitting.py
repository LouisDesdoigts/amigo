"""This probably ends up as a zodiax function"""

from tqdm.notebook import tqdm
import zodiax as zdx
import jax.numpy as np


def optimise(
    model,
    args,
    loss_grad_fn,
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
    optim, opt_state = zdx.get_optimiser(model, params, optimisers)

    loss, grads = loss_grad_fn(model, args)
    if print_grads:
        for param in params:
            print(f"{param}: {grads.get(param)}")

    # Define faster step function
    @zdx.filter_jit
    def step_fn(model, opt_state):  # , epoch):
        # calculate the loss and gradient
        loss, grads = loss_grad_fn(model, args)

        # # Apply any processing to the gradients
        # grads = grad_fn(grads, config, epoch)

        # apply the update
        updates, opt_state = optim.update(grads, opt_state)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model)

        return model, loss, opt_state

    # Compile
    _ = step_fn(model, opt_state)

    if verbose:
        looper = tqdm(range(epochs), desc="Loss %.2f" % (loss))
    else:
        looper = range(epochs)

    # Get the params from each model
    params = {}
    for param in config.keys():
        params[param] = [model.get(param)]

    losses = []
    for i in looper:
        # model, loss, opt_state = step_fn(model, opt_state, i)
        model, loss, opt_state = step_fn(model, opt_state)
        # # calculate the loss and gradient
        # loss, grads = loss_fn(model, args)

        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch, exiting loop")
            return model, losses, params

        # save results
        losses.append(loss)

        for param in config.keys():
            params[param].append(model.get(param))

        if verbose:
            looper.set_description("Loss %.2f" % (loss))

    return model, losses, params
