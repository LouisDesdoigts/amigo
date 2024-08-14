import jax
import optax
import time
import zodiax as zdx
import jax.numpy as np
import jax.random as jr
import jax.tree_util as jtu
import equinox as eqx
import dLux.utils as dlu
from jax import vmap, config
from datetime import timedelta
from .core_models import ModelParams, ModelHistory
from .stats import reg_loss_fn
from zodiax.experimental import serialise

# import tqdm appropriately
from IPython import get_ipython

if get_ipython() is not None:
    # Running in Jupyter Notebook
    from tqdm.notebook import tqdm
else:
    # Running in a script or other non-Jupyter environment
    from tqdm import tqdm


def scheduler(lr, start, *args):
    shed_dict = {start: 1e100}
    for start, mul in args:
        shed_dict[start] = mul
    return optax.piecewise_constant_schedule(lr / 1e100, shed_dict)


base_sgd = lambda vals: optax.sgd(vals, nesterov=True, momentum=0.6)
base_adam = lambda vals: optax.adam(vals)

sgd = lambda lr, start, *schedule: base_sgd(scheduler(lr, start, *schedule))
adam = lambda lr, start, *schedule: base_adam(scheduler(lr, start, *schedule))


def debug_nan_check(grads):
    bool_tree = jax.tree_map(lambda x: np.isnan(x).any(), grads)
    vals = np.array(jax.tree_util.tree_flatten(bool_tree)[0])
    eqx.debug.breakpoint_if(vals.sum() > 0)
    return grads


def zero_nan_check(grads):
    return jax.tree_map(lambda x: np.where(np.isnan(x), 0.0, x), grads)


def set_array(pytree, parameters):
    dtype = np.float64 if config.x64_enabled else np.float32
    floats, other = eqx.partition(pytree, eqx.is_inexact_array_like)
    floats = jtu.tree_map(lambda x: np.array(x, dtype=dtype), floats)
    return eqx.combine(floats, other)


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
    # Flag and deal with large arrays
    grad_model = eqx.filter(model, bool_model)
    is_large = jtu.tree_map(lambda x: x.size > 1e4, grad_model)
    bool_model = jtu.tree_map(lambda x, y: x and not y, bool_model, is_large)
    grad_model = eqx.filter(model, bool_model)
    fisher_model = jtu.tree_map(lambda x: np.zeros((x.size, x.size)), grad_model)
    large_grad_model = eqx.filter(model, is_large)
    large_lr_model = jtu.tree_map(lambda x: np.ones(x.shape), large_grad_model)

    # Loop over exposures
    for exp in exposures:

        # Loop over parameters
        for param in params:

            # Check if the parameter is in the fisher
            if param not in fishers[exp.key].keys():
                continue

            param_path = exp.map_param(param)
            fisher_model = fisher_model.add(param_path, fishers[exp.key][param])

    # Convert fisher to lr model
    inv_fn = lambda fmat, leaf: dlu.nandiv(-1, np.diag(fmat), 1).reshape(leaf.shape)
    lr_model = jtu.tree_map(inv_fn, fisher_model, model)
    lr_model = eqx.combine(lr_model, large_lr_model)
    return lr_model


# def reg_loss_fn(model, exposure, args):
#     return -np.array(posterior(model, exposure, per_pix=True)).sum()


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
    args_fn=lambda model, args, key, epoch: (model, args, key),
    loss_fn=None,  # Must have input signature (model, exposure, args)
    print_grads=False,
    no_history=[],
    batch_params=[],
    save_every=None,
    save_path="",
    save_ext="",
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
    if loss_fn is None:
        loss_fn = reg_loss_fn
    batch_loss_fn = lambda model, batch, args: np.array(
        [loss_fn(model, exp, args) for exp in batch]
    ).sum()
    val_grad_fn = zdx.filter_value_and_grad(opt_params)(batch_loss_fn)

    @eqx.filter_jit
    def batched_loss_fn(model, batch, args, key):
        print("Grad Batch fn compiling...")
        loss, grads = val_grad_fn(model, batch, args)

        # Apply the lr normalisation
        grads = jtu.tree_map(lambda x, y: x * y, grads, lr_model)

        # Apply user normalisation
        grads, key = grad_fn(model, grads, args, key)

        # Optionally print the final gradients
        if print_grads:
            for param in opt_params:
                print(param)
                jax.debug.print("{x}", x=jtu.tree_leaves(grads.get(param)))
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
            batch_history = batch_history.append(batch_model)

            # Check for NaNs
            if np.isnan(_loss):
                print(f"Loss is NaN on {idx} th epoch, exiting loop")
                return (
                    model,
                    losses,
                    (reg_history, batch_history),
                    (reg_state, batch_state),
                )

        # Update the reg params
        model, reg_model, reg_state, key = update_reg(
            model, reg_grads, reg_model, reg_state, args, key
        )
        reg_history = reg_history.append(reg_model)  # could be JIT'd with work

        # Update the looper
        loss = np.array(batch_losses).mean()
        if idx == 0:
            looper.set_description(f"Loss: {loss:,.2f}")
            prev_loss = loss  # This line is here to make the linter happy
        else:
            looper.set_description(f"Loss: {loss:,.2f}, \u0394: {loss - prev_loss:,.2f}")
        prev_loss = loss

        losses.append(batch_losses)

        # Save progress along the way
        if save_every is not None and ((idx + 1) % save_every) == 0:
            if save_ext != "":
                save_ext = f"_{save_ext}"
            np.save(save_path + f"losses_{idx+1}{save_ext}.npy", losses)
            serialise(save_path + f"reg_history_{idx+1}{save_ext}.zdx", reg_history)
            serialise(save_path + f"batch_history_{idx+1}{save_ext}.zdx", batch_history)

        # Print helpful things
        if idx == 0:
            compile_time = int(time.time() - t0)
            print(f"Compile Time: {str(timedelta(seconds=compile_time))}")
            print(f"Initial Loss: {loss:,.2f}")
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
    print(f"Final Loss: {loss:,.2f}")

    batch_model = jtu.tree_map(lambda x: np.array(x[-batch_size:]).mean(axis=0), batch_model)
    final_state = reg_model.set("params", {**reg_model.params, **batch_model.params})
    history = reg_history.set("params", {**reg_history.params, **batch_history.params})
    return model, losses, final_state, history
