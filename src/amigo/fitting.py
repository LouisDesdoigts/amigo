import jax
import optax
import time
import zodiax as zdx
import jax.numpy as np
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import dLux.utils as dlu
from jax import vmap
from .core import ModelParams, ModelHistory
from .stats import batch_loss_fn
from datetime import timedelta

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


def calc_lrs(model, exposures, fishers, params=None, order=1):

    # Get the parameters from the fishers
    if params is None:
        params = []
        for exp_key, fisher_dict in fishers.items():
            for param in fisher_dict.keys():
                params.append(param)
        params = list(set(params))

    # Build a filter, we need to handle parameters that are stored in dicts
    # TODO: Add this to model?
    bool_model = jtu.tree_map(lambda _: False, model)
    for param in params:
        leaf = model.get(param)
        if isinstance(leaf, dict):
            true_leaf = jtu.tree_map(lambda x: True, leaf)
        else:
            true_leaf = True
        bool_model = bool_model.set(param, true_leaf)

    # Make an empty fisher model
    grad_model = eqx.filter(model, bool_model)
    fisher_model = jtu.tree_map(lambda x: np.zeros((x.size, x.size)), grad_model)

    # Loop over exposures
    # for exp_key, fisher_dict in fishers.items():
    for exp in exposures:

        # Loop over parameters
        # for param in fisher_dict.keys():
        for param in fishers[exp.key]:
            leaf = model.get(param)

            # Handle dict case
            if isinstance(leaf, dict):
                leaf_param = f"{param}.{exp.key}"
                fisher_model = fisher_model.add(leaf_param, fishers[exp.key][param])
            else:
                fisher_model = fisher_model.add(param, fishers[exp.key][param])

    # Convert fisher to lr model
    inv_fn = lambda fmat, leaf: dlu.nandiv(-1, np.diag(fmat), 1).reshape(leaf.shape)
    lr_model = jtu.tree_map(inv_fn, fisher_model, model)
    return lr_model


# def set_array(pytree: Base(), parameters: Params) -> Base():
def set_array(pytree, parameters):
    # WARNING: Presently statically set to 64bit
    # Enforce everything to be a float (of the same precision)
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=np.float64), floats)
    return eqx.combine(floats, other)


def optimise(
    model,
    exposures,
    optimisers,
    fishers=None,
    key=jr.PRNGKey(0),
    epochs=10,
    batch_size=1,
    args={},
    grad_fn=lambda model, grads, args, key: (grads, key),
    norm_fn=lambda model, model_params, args, key: (model_params, key),
    args_fn=lambda model, args, key: (model, args, key),
    print_grads=False,
    no_history=[],
    batch_params=[],
):

    # Define an update function to improve step speed
    @eqx.filter_jit
    def model_update_fn(optim, model, grads, model_params, state, args, key):
        print("Compiling update function")
        # NOTE: We apply the normalisation after calculating the state, so we should
        # re-update the state to reflect this
        updates, state = optim.update(grads, state, model_params)
        model_params = zdx.apply_updates(model_params, updates)
        model_params, key = norm_fn(model, model_params, args, key)
        model = model_params.inject(model)
        return model, model_params, state, key

    # Get params and assert array
    opt_params = list(optimisers.keys())
    model = set_array(model, opt_params)

    # Get the LR normalisation from the fisher matrices
    lr_model = calc_lrs(model, exposures, fishers, params=opt_params)

    # Get the parameter classes and the optimisers
    reg_params = [p for p in opt_params if p not in batch_params]
    batch_params = [p for p in batch_params if p in opt_params]

    # Deal with batch param inputs that aren't in the optimisers
    batch_params = [p for p in opt_params if p in batch_params]

    # Get the model, optimiser and state
    reg_model, reg_optim, reg_state = get_optimiser(model, optimisers, reg_params)
    batch_model, batch_optim, batch_state = get_optimiser(model, optimisers, batch_params)

    # Binds optimisers to update functions
    update_batch = lambda *args: model_update_fn(batch_optim, *args)
    update_reg = lambda *args: model_update_fn(reg_optim, *args)

    # Apply gradient to loss function
    val_grad_fn = zdx.filter_value_and_grad(opt_params)(batch_loss_fn)

    @eqx.filter_jit
    def batched_loss_fn(model, batch, args, key):
        print("Grad Batch fn compiling...")
        loss, grads = val_grad_fn(model, batch)

        # Apply the lr normalisation
        grads = jtu.tree_map(lambda x, y: x * y, grads, lr_model)

        # Apply user normalisation
        grads, key = grad_fn(model, grads, args, key)

        # Optionally print the final gradients
        if print_grads:
            jax.debug.print("{x}", x=jtu.tree_leaves(grads))
        return loss, grads

    # Create model history
    reg_history = ModelHistory(model, [p for p in reg_params if p not in no_history])
    batch_history = ModelHistory(model, [p for p in batch_params if p not in no_history])

    # Randomise exposures and get batches
    (key, exp_key) = jr.split(key, 2)
    exposures = [exposures[i] for i in jr.permutation(exp_key, len(exposures))]
    batches = [exposures[i : i + batch_size] for i in range(0, len(exposures), batch_size)]

    # Get a random batch order
    key, batch_key = jr.split(key, 2)
    batch_keys = jr.split(batch_key, epochs)
    rand_batch_inds = vmap(lambda key: jr.permutation(key, len(batches)))(batch_keys)

    # Epoch loop
    losses = []
    looper = tqdm(range(0, epochs))
    epoch_loss = 0.0
    t0 = time.time()
    for idx in looper:
        model, args, key = args_fn(model, args, key, idx)

        # Loop over batches
        reg_grads = jax.tree_map(lambda x: np.zeros_like(x), reg_model)
        batch_inds = rand_batch_inds[idx]
        batch_losses = np.zeros(len(batches))
        for i in batch_inds:
            _loss, grads = batched_loss_fn(model, batches[i], args, key)

            # Update losses and grads
            batch_losses = batch_losses.at[i].set(_loss)
            batch_grads = batch_model.from_model(grads)
            reg_grads += reg_grads.from_model(grads)

            # Update the batch params and accumulate grads
            model, batch_model, batch_state, key = update_batch(
                model, batch_grads, batch_model, batch_state, args, key
            )
            batch_history = batch_history.append(batch_model)  # could be JIT'd with work

            # Check for NaNs
            if np.isnan(_loss):
                print(f"Loss is NaN on {idx} th epoch, exiting loop")
                return model, losses, (reg_history, batch_history), (reg_state, batch_state)

        # Update the reg params
        model, reg_model, reg_state, key = update_reg(
            model, reg_grads, reg_model, reg_state, args, key
        )
        reg_history = reg_history.append(reg_model)  # could be JIT'd with work

        # Update the looper
        batch_loss = np.array(batch_losses).mean()
        looper.set_description(f"Loss: {epoch_loss:,.2f}, Change: {batch_loss - epoch_loss:,.2f}")
        losses.append(batch_losses)
        epoch_loss = batch_loss

        # Print helpful things
        if idx == 0:
            compile_time = int(time.time() - t0)
            print(f"Compile Time: {str(timedelta(seconds=compile_time))}")
            print(f"Initial Loss: {epoch_loss:,.2f}")
            t1 = time.time()
        if idx == 1:
            epoch_time = time.time() - t1
            est_runtime = compile_time + epoch_time * (epochs - 1)
            print("Est time per epoch: ", str(timedelta(seconds=int(epoch_time))))
            print("Est run remaining: ", str(timedelta(seconds=int(est_runtime))))

    # Final execution time
    elapsed_time = time.time() - t0
    formatted_time = str(timedelta(seconds=int(elapsed_time)))

    print(f"Full Time: {formatted_time}")
    print(f"Final Loss: {epoch_loss:,.2f}")

    return model, losses, (reg_history, batch_history), (reg_state, batch_state)
