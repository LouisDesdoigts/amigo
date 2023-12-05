"""This probably ends up as a zodiax function"""

from tqdm.notebook import tqdm
import zodiax as zdx
import jax.numpy as np


def optimise(
    model,
    args,
    loss_fn,
    epochs,
    config,
    grad_fn=lambda grads, config, epoch: grads,
    norm_fn=lambda model: model,
    verbose=True,
    print_grads=False,
):
    params = list(config.keys())
    optimisers = list(config.values())

    model = zdx.set_array(model, params)
    optim, opt_state = zdx.get_optimiser(model, params, optimisers)

    loss, grads = loss_fn(model, args)
    if print_grads:
        for param in params:
            print(f"{param}: {grads.get(param)}")
    losses, models_out = [], [model]

    # Define faster step function
    @zdx.filter_jit
    def step_fn(model, opt_state):  # , epoch):
        # calculate the loss and gradient
        loss, grads = loss_fn(model, args)

        # # Apply any processing to the gradients
        # grads = grad_fn(grads, config, epoch)

        # apply the update
        updates, opt_state = optim.update(grads, opt_state)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model)

        return model, loss, opt_state

    # Compile
    # _ = step_fn(model, opt_state, 0)
    _ = step_fn(model, opt_state)

    if verbose:
        looper = tqdm(range(epochs), desc="Loss %.2f" % (loss))
    else:
        looper = range(epochs)

    for i in looper:
        # model, loss, opt_state = step_fn(model, opt_state, i)
        model, loss, opt_state = step_fn(model, opt_state)
        # # calculate the loss and gradient
        # loss, grads = loss_fn(model, args)

        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch, exiting loop")
            return losses, models_out

        # # Apply any processing to the gradients
        # grads = grad_fn(grads, config, i)

        # # apply the update
        # updates, opt_state = optim.update(grads, opt_state)
        # model = zdx.apply_updates(model, updates)

        # # Apply normalisation
        # model = norm_fn(model)

        # save results
        models_out.append(model)
        losses.append(loss)

        if verbose:
            looper.set_description("Loss %.2f" % (loss))

    return losses, models_out
