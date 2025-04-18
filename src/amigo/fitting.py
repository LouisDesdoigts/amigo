from amigo.fisher import FIM
import equinox as eqx
import zodiax as zdx
import time
from datetime import timedelta
import jax.tree as jtu
from .core_models import ModelParams, ParamHistory
from .fisher import calc_fishers
from .misc import tqdm
from .stats import covariance_model
import optax
import jax
import jax.numpy as np
from jax import config
import jax.random as jr
import dLux.utils as dlu


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


def get_optimiser(model_params, optimisers):
    param_spec = ModelParams({param: param for param in list(model_params.keys())})
    optim = optax.multi_transform(optimisers, param_spec)
    return optim, optim.init(model_params)


def get_random_batch_order(batches, epochs, args):
    """Exposures is a dictionary of lists"""
    # Generate a random order for the batches to be fed
    key, subkey = jr.split(args["key"], 2)
    args["key"] = subkey

    # This is the number of batches we have
    inds = np.tile(np.arange(len(batches)), (epochs, 1))

    # Permutate the oder of each individual batch
    batch_inds = jr.permutation(subkey, inds, axis=1, independent=True)
    return batch_inds, args


# Returns the batched and gradded loss function
def get_val_grad_fn(loss_fn):

    # Injects the model parameters, and takes the loss over the batch, and applied
    # the gradient transformation
    def _val_grad_fn(model_params, model, exposures, args):
        model = model_params.inject(model)
        keys = [exp.key for exp in exposures]
        losses, aux = zip(*[loss_fn(model, exp, args) for exp in exposures])
        return np.array(losses).sum(), dict(zip(keys, aux))

    return eqx.filter_value_and_grad(_val_grad_fn, has_aux=True)


# Returns the jitted the loss function
def get_norm_loss_fn(val_grad_fn, grad_fn=None):

    @eqx.filter_jit
    def _norm_loss_fn(model_params, lr_model, model, batch, args):
        print("Compiling Loss function...")
        # loss, grads = val_grad_fn(model_params, model, batch, args)
        (loss, aux), grads = val_grad_fn(model_params, model, batch, args)

        # Apply the lr normalisation
        grads = jtu.map(lambda x, y: x * y, grads, lr_model)

        # Apply user normalisation if exists
        if grad_fn is not None:
            grads, args = grad_fn(model, grads, args)
        return loss, grads, args, aux

    # return eqx.debug.assert_max_traces(_norm_loss_fn, max_traces=3)
    return _norm_loss_fn


# Returns the jitted the update function
def get_update_fn(optim, norm_fn=None):

    # Inner function simply binds the optim object from self
    @eqx.filter_jit
    def update_fn(param_grads, model_params, opt_state, args):
        print("Compiling update function...")
        # NOTE: We apply the normalisation after calculating the state, so we should
        # re-update the opt_state to reflect this. Im not sure how this would affect
        # the opt_state object though
        updates, opt_state = optim.update(param_grads, opt_state, model_params)
        model_params = zdx.apply_updates(model_params, updates)

        if norm_fn is not None:
            model_params, args = norm_fn(model_params, args)
        return model_params, opt_state, args

    return update_fn


def populate_lr_model(fishers, exposures, model_params):

    # Build the lr model structure
    params_dict = jtu.map(lambda x: np.zeros((x.size, x.size)), model_params).params

    # Loop over exposures
    for exp in exposures:

        # Loop over parameters
        for param in model_params.keys():

            # Check if the fishers have values for this exposure
            key = f"{exp.key}.{param}"
            if key not in fishers.keys():
                continue

            # Add the Fisher matrices
            if isinstance(params_dict[param], dict):
                params_dict[param][exp.get_key(param)] += fishers[key]
            else:
                params_dict[param] += fishers[key]

    fisher_params = model_params.set("params", params_dict)

    # Convert fisher to lr model
    def inv_fn(fmat, leaf):
        return dlu.nandiv(-1, np.diag(fmat), fill=1).reshape(leaf.shape)

    return jtu.map(inv_fn, fisher_params, model_params)


def batch_exposures(exposures, n_batch=None, batch_size=None, key=None):
    # Both have a value
    if n_batch is not None and batch_size is not None:
        raise ValueError

    if key is None:
        key = "batch_"
    else:
        key += "_"

    # Both None
    if n_batch is None and batch_size is None:
        batch_size = len(exposures)

    # We have batch size, need n_batch
    if n_batch is None:
        n_batch = np.ceil(len(exposures) / batch_size).astype(int)

    # We have n_batch, need batch size
    if batch_size is None:
        batch_size = np.ceil(len(exposures) / n_batch).astype(int)
        n_batch = np.minimum(n_batch, np.ceil(len(exposures) / batch_size).astype(int))

    # Populate the batches
    batches = {}
    for i in range(n_batch):
        start = i * batch_size
        end = np.minimum(start + batch_size, len(exposures))
        batches[f"{key}{i}"] = exposures[start:end]
    return batches


class Result(zdx.Base):
    losses: dict
    model: zdx.Base
    state: ModelParams
    lr_model: ModelParams
    history: ParamHistory
    aux: dict
    meta_data: dict

    def __init__(self, losses, model, aux, state, history, lr_model, meta_data=None):
        self.losses = losses
        self.model = model
        self.state = state
        self.history = history
        self.lr_model = lr_model
        self.meta_data = meta_data
        self.aux = aux


class Trainer(zdx.Base):
    fishers: dict
    loss_fn: callable
    args_fn: callable
    grad_fn: callable
    norm_fn: callable
    looper_fn: callable
    aux_fn: callable
    cache: str

    def __init__(
        self,
        loss_fn,
        args_fn=None,
        grad_fn=None,
        norm_fn=None,
        looper_fn=None,
        aux_fn=None,
        cache="cache",
    ):
        """
        loss_fn(model, exposure, args): -> loss
        args_fn(model, args, key, epoch): -> (mode, args, key)
        grad_fn(model, grads, args, key): -> (grads, key)
        norm_fn(model_params, args, key): -> model_params, key
        looper_fn(looper, loss_dict): -> ()
        """
        self.loss_fn = loss_fn
        self.args_fn = args_fn
        self.grad_fn = grad_fn
        self.norm_fn = norm_fn
        self.looper_fn = looper_fn
        self.aux_fn = aux_fn
        self.fishers = None
        self.cache = cache

    def default_looper(self, looper, loss_dict):
        loss = np.array([v[-1] for v in loss_dict.values()]).mean(0)
        looper.set_description(f"Loss: {loss:.2f}")

    def update_fishers(
        self,
        model,
        exposures,
        parameters=[],
        recalculate=False,
        overwrite=False,
        verbose=True,
        save=True,
        args=None,
        reduce_ram=False,
        batch_size=1,
    ):
        # RANDOM TODO: Models params should be renamed ParamsDict everywhere and always
        # Use the dark current data to get a measure of the read noise

        def fisher_fn(model, exposure, param):
            slopes, cov = covariance_model(model, exposure)
            exposure = exposure.set(["slopes", "cov"], [slopes, cov])

            # NOTE: FIM doesnt return aux, so even though has_aux=True, it wont be returned
            if args is not None:
                fmat = FIM(
                    model,
                    param,
                    self.loss_fn,
                    exposure,
                    args,
                    reduce_ram=reduce_ram,
                    has_aux=True,
                    batch_size=batch_size,
                )
            else:
                fmat = FIM(
                    model,
                    param,
                    self.loss_fn,
                    exposure,
                    reduce_ram=reduce_ram,
                    has_aux=True,
                    batch_size=batch_size,
                )

            return -fmat

        # Calculate the fishers
        fishers = calc_fishers(
            model,
            exposures,
            parameters,
            fisher_fn=fisher_fn,
            overwrite=overwrite,
            recalculate=recalculate,
            verbose=verbose,
            save=save,
            cache=f"{self.cache}/fishers",
        )

        # The second dictionary (new fishers) will overwrite the existing fishers
        if self.fishers is not None:
            return self.set("fishers", {**self.fishers, **fishers})
        return self.set("fishers", fishers)

    def finalise(
        self, t0, model, loss_dict, aux, model_params, history, lr_model, epochs, success
    ):
        """Prints stats and returns the result object"""
        # Final execution time
        elapsed_time = time.time() - t0
        formatted_time = str(timedelta(seconds=int(elapsed_time)))
        print(f"Full Time: {formatted_time}")

        # Get the final loss
        final_loss = np.array([losses[-1] for losses in loss_dict.values()]).mean()
        print(f"Final Loss: {final_loss:,.2f}")

        # Return
        return Result(
            losses=loss_dict,
            model=model_params.inject(model),
            aux=aux,
            state=model_params,
            history=history,
            lr_model=lr_model,
            meta_data={
                "elapsed_time": formatted_time,
                "epochs": epochs,
                "successful": success,
            },
        )

    def unwrap_batches(self, batches):
        # Format the batches and exposures
        if isinstance(batches, list):
            exposures = batches
            batches = {0: exposures}
        else:
            exposures = []
            for batch_key, batch in batches.items():
                exposures += batch
        return batches, exposures

    def initial_print(self, loss_dict):
        # Get the initial loss
        initial_loss = np.array([losses[-1] for losses in loss_dict.values()]).mean()
        print(f"\nInitial_loss Loss: {initial_loss:,.2f}")

    def second_print(self, t1, epochs):
        estimated_time = epochs * (time.time() - t1)
        formatted_time = str(timedelta(seconds=int(estimated_time)))
        print(f"Estimated run time: {formatted_time}")

    def check_args_key(self, args):
        # Ensure args key exists and is the right type
        if "key" in args.keys():
            if not isinstance(args["key"], (type(jr.key(0)), type(jr.PRNGKey(0)))):
                raise ValueError("the 'key' entry of 'args' must be a jax prng key.")
        if "key" not in args.keys():
            args["key"] = jr.key(0)
        return args

    def train(
        self,
        model,
        optimisers,
        epochs,
        batches: dict,
        args={},
    ):
        # Ensure args key exists and is the right type
        args = self.check_args_key(args)

        # Get the batches and raw exposures
        batches, exposures = self.unwrap_batches(batches)

        # Get the model parameters
        model_params = ModelParams({p: model.get(p) for p in optimisers.keys()})
        lrs = populate_lr_model(self.fishers, exposures, model_params)

        # Get the optax optimiser and history bits
        optim, state = get_optimiser(model_params, optimisers)
        history = ParamHistory(model_params)

        # Get the loss and update functions
        val_grad_fn = get_val_grad_fn(self.loss_fn)
        loss_fn = get_norm_loss_fn(val_grad_fn, self.grad_fn)
        update_fn = get_update_fn(optim, self.norm_fn)

        # Looping things
        t0 = time.time()
        looper = tqdm(range(0, epochs))
        loss_dict = dict([(key, []) for key in batches.keys()])
        aux_dict = {}

        # Loop
        for epoch in looper:
            if epoch == 1:
                t1 = time.time()

            if self.args_fn is not None:
                model, args = self.args_fn(model, args, epoch)

            # Create an empty gradient model to append gradients to
            grads = model_params.map(lambda x: x * 0.0)

            # Loop over randomised batch order
            for batch_key, batch in batches.items():
                # Calculate the loss and gradients
                loss, new_grads, args, aux = loss_fn(model_params, lrs, model, batch, args)
                grads += new_grads

                # Append the mean batch loss to the loss dictionary and update aux dict
                loss_dict[batch_key].append(loss / len(batch))

                #
                if self.aux_fn is not None:
                    aux_dict = self.aux_fn(aux_dict, aux)

                # Check for NaNs and exit if so
                if np.isnan(loss):
                    print(f"Loss is NaN on epoch {epoch}, exiting fit")
                    return self.finalise(
                        t0, model, loss_dict, aux, model_params, history, lrs, epochs, False
                    )

            # Update the regular parameters and append to history
            model_params, state, args = update_fn(grads, model_params, state, args)
            history = history.append(model_params)

            # Update the looper
            loop_fn = self.default_looper if self.looper_fn is None else self.looper_fn
            loop_fn(looper, loss_dict)

            # Print helpful things run time
            if epoch == 0:
                self.initial_print(loss_dict)
            if epoch == 1:
                self.second_print(t1, epochs)

        # Print the runtime stats and return Result object
        return self.finalise(t0, model, loss_dict, aux, model_params, history, lrs, epochs, True)
