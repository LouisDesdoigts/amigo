import zodiax as zdx
import jax.numpy as np
import jax.random as jr
from jax import vmap
import equinox as eqx
import jax
import optax
import jax.tree_util as jtu
import time
from datetime import timedelta
from amigo.core import ModelParams, ModelHistory
from .stats import loss_fn

# import tqdm appropriately
from IPython import get_ipython

if get_ipython() is not None:
    # Running in Jupyter Notebook
    from tqdm.notebook import tqdm
else:
    # Running in a script or other non-Jupyter environment
    from tqdm import tqdm


def debug_nan_check(grads):
    bool_tree = jax.tree_map(lambda x: np.isnan(x).any(), grads)
    vals = np.array(jax.tree_util.tree_flatten(bool_tree)[0])
    eqx.debug.breakpoint_if(vals.sum() > 0)
    return grads


def zero_nan_check(grads):
    return jax.tree_map(lambda x: np.where(np.isnan(x), 0.0, x), grads)


def get_optimiser(pytree, optimisers, parameters=None):

    # Get the parameters and opt_dict
    if parameters is not None:
        optimisers = dict([(p, optimisers[p]) for p in parameters])
    else:
        parameters = list(optimisers.keys())

    model_params = ModelParams(dict([(p, pytree.get(p)) for p in parameters]))
    param_spec = ModelParams(dict([(param, param) for param in parameters]))
    optim = optax.multi_transform(optimisers, param_spec)

    # Build the optimised object - the 'model_params' object
    state = optim.init(model_params)
    return model_params, optim, state


# def set_array(pytree: Base(), parameters: Params) -> Base():
def set_array(pytree, parameters):
    # WARNING: Presently statically set to 64bit
    # Enforce everything to be a float (of the same precision)
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=np.float64), floats)
    return eqx.combine(floats, other)


def optimise(
    model,
    args,
    epochs,
    optimisers,
    batch_size,
    grad_fn=lambda model, grads: grads,
    norm_fn=lambda model_params, args: model_params,
    args_fn=lambda model, args: (model, args),
    print_grads=False,
    no_history=[],
    batch_params=[],
):
    opt_params = list(optimisers.keys())

    # Get the parameter classes and the optimisers
    model = set_array(model, opt_params)
    reg_params = [p for p in opt_params if p not in batch_params]
    batch_params = [p for p in opt_params if p in batch_params]

    # Get the model, optimiser and state
    reg_model, reg_optim, reg_state = get_optimiser(model, optimisers, reg_params)
    batch_model, batch_optim, batch_state = get_optimiser(model, optimisers, batch_params)

    @eqx.filter_jit
    def model_update_fn(optim, model, grads, model_params, state):
        print("Compiling update function")
        updates, state = optim.update(grads, state, model_params)
        model_params = zdx.apply_updates(model_params, updates)
        model_params = norm_fn(model_params, args)
        model = model_params.inject(model)
        return model, model_params, state

    # Binds optimisers to update functions
    update_batch = lambda *args: model_update_fn(batch_optim, *args)
    update_reg = lambda *args: model_update_fn(reg_optim, *args)
    val_grad_fn = zdx.filter_value_and_grad(opt_params)(loss_fn)

    def grad_batch_fn(model, batch):
        print("Grad Batch fn compiling...")
        loss, grads = val_grad_fn(model, batch)
        grads = grad_fn(model, grads)

        # Optionally print the grads
        if print_grads:
            jax.debug.print("{x}", x=jtu.tree_leaves(grads))
        return loss, grads

    # Create model history
    reg_history = ModelHistory(model, [p for p in reg_params if p not in no_history])
    batch_history = ModelHistory(model, [p for p in batch_params if p not in no_history])

    # Get batches
    # TODO: Randomise the order of the exposures before batching
    exposures = args["exposures"]
    batches = [exposures[i : i + batch_size] for i in range(0, len(exposures), batch_size)]

    # Bind batches to grad_fn - This _shouldn't_ be necessary but its the easiest way
    # to get guarantees about inputs to the compiled function
    batch_fns = [eqx.filter_jit(lambda model: grad_batch_fn(model, batch)) for batch in batches]

    # Get a random batch order
    keys = jr.split(args["key"], epochs)
    rand_batch_inds = vmap(lambda key: jr.permutation(key, len(batches)))(keys)

    # Epoch loop
    losses = []
    looper = tqdm(range(0, epochs))
    epoch_loss = 0.0
    t0 = time.time()
    for idx in looper:
        model, args = args_fn(model, args)

        # Loop over batches
        reg_grads = jax.tree_map(lambda x: np.zeros_like(x), reg_model)
        batch_inds = rand_batch_inds[idx]
        batch_losses = np.zeros(len(batches))
        for i in batch_inds:

            # Calculate loss and grads
            _loss, grads = batch_fns[i](model)

            # Update losses and grads
            batch_losses = batch_losses.at[i].set(_loss)
            batch_grads = batch_model.from_model(grads)
            reg_grads += reg_grads.from_model(grads)

            # Update the batch params and accumulate grads
            model, batch_model, batch_state = update_batch(
                model, batch_grads, batch_model, batch_state
            )
            batch_history = batch_history.append(batch_model)  # could be JIT'd with work

        # Update the reg params
        model, reg_model, reg_state = update_reg(model, reg_grads, reg_model, reg_state)
        reg_history = reg_history.append(reg_model)  # could be JIT'd with work

        # Check for NaNs
        if np.isnan(_loss):
            print(f"Loss is NaN on {i} th epoch, exiting loop")
            return model, losses, (reg_history, batch_history), (reg_state, batch_state)

        # Update the looper
        batch_loss = np.array(batch_losses).sum()
        looper.set_description(f"Loss: {epoch_loss:,.2f}, Change: {batch_loss - epoch_loss:,.2f}")
        losses.append(batch_losses)
        epoch_loss = batch_loss

        if idx == 0:
            print(f"Compile Time: {str(timedelta(seconds=int(time.time() - t0)))}")
            print(f"Initial Loss: {epoch_loss:,.2f}")

    # Final execution time
    elapsed_time = time.time() - t0
    formatted_time = str(timedelta(seconds=int(elapsed_time)))

    print(f"Full Time: {formatted_time}")
    print(f"Final Loss: {epoch_loss:,.2f}")

    return model, losses, (reg_history, batch_history), (reg_state, batch_state)
