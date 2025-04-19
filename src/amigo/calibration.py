import jax.numpy as np
import jax.random as jr
from jax import vmap


def mv_zscore(x, mu, cov):
    """Multivariate z-score, return identical gradients to normal log-likelihood"""
    return -0.5 * np.dot(x - mu, np.dot(np.linalg.inv(cov), x - mu))


def log_likelihood(slope, exposure, return_im=False):
    # Get the model, data, and variances
    slope_vec = exposure.to_vec(slope)
    data_vec = exposure.to_vec(exposure.slopes)
    cov_vec = exposure.to_vec(exposure.cov)

    # Calculate per-pixel likelihood
    loglike_vec = vmap(mv_zscore)(slope_vec, data_vec, cov_vec)

    # Return image or vector
    if return_im:
        # NOTE: Adds nans to the empty spots
        return exposure.from_vec(loglike_vec)
    return loglike_vec


def prior(model, bleeds, l2_norm=0.0, bleed_norm=0.0):
    l2_reg = -l2_norm * np.mean(model.ramp.values**2)
    bleed_reg = -bleed_norm * (np.sum(bleeds, axis=1) ** 2).sum()
    return l2_reg, bleed_reg


def posterior(model, exposure, l2_norm=0.0, bleed_norm=0.0):
    slopes, bleed = exposure(model, return_bleed=True)
    loglike = log_likelihood(slopes, exposure)
    l2_reg, bleed_reg = prior(model, bleed, l2_norm=l2_norm, bleed_norm=bleed_norm)
    return loglike + l2_reg + bleed_reg, (loglike, l2_reg, bleed_reg)


def loss_fn(model, exposure, args={"l2": 0.0, "bleed": 0.0}):
    loss, (loglike, l2_reg, bleed_reg) = posterior(
        model, exposure, l2_norm=args["l2"]  # , bleed_norm=args["bleed"]
    )
    return -np.nanmean(loss), (-np.nanmean(loglike), l2_reg, bleed_reg)


def args_fn(model, args, epoch):
    args["l2"] = args["l2_schedule"][epoch]
    # args["bleed"] = args["bleed_schedule"][epoch]
    return model, args


def cosine_warmup(t, n_max, min_lr):
    # Make the cosine curve
    x = t * np.pi / n_max
    half_cos = 0.5 * (1 + (np.cos(x + np.pi)))

    # Shift and scale by the min lr
    amp = 1 - min_lr
    half_cos = half_cos * amp + min_lr

    # Set all values > n to 1.
    return np.where(t > n_max, 1.0, half_cos)


def temp_decay(T0, k, t):
    return T0 * np.exp(-k * t)


def get_warmup(args):
    return cosine_warmup(args["t"], args["n_max"], args["min_lr"])


def get_temperature(args):
    return temp_decay(args["T0"], args["k"], args["t"])


def grads_fn(model, grads, args):
    # Get the parameters
    grad_params = grads.params

    # Get the key and update args with new key
    key, subkey = jr.split(args["key"], 2)
    args["key"] = subkey

    # Adds a temperature to the NN gradients
    values = grad_params["ramp.values"]
    rand_vals = get_temperature(args) * jr.normal(key, values.shape)
    values += rand_vals

    # Add the learning rate warm-up (we also warm up the temperature here)
    values *= get_warmup(args)

    # Increment the t parameter
    args["t"] += 1.0

    # Update with the new values
    grad_params["ramp.values"] = values
    grads = grads.set("params", grad_params)
    return grads, args


import zodiax as zdx
import jax.tree as jtu
import numpy as onp
import time
from datetime import timedelta
from amigo.core_models import ModelParams, ParamHistory
from amigo.misc import tqdm
from amigo.fitting import (
    get_optimiser,
    get_val_grad_fn,
    get_norm_loss_fn,
    get_update_fn,
    get_random_batch_order,
    populate_lr_model,
    Trainer,
)


def aux_fn(batch_key, aux_dict, aux):
    # Aux should have exposure keys, with values (loglike, l2_reg, bleed_reg)
    for exp_key, val in aux.items():
        aux_key = (batch_key, exp_key)
        aux_dict["loglike"][aux_key].append(onp.array(val[0]))
        aux_dict["l2_reg"][aux_key].append(onp.array(-val[1]))
        aux_dict["bleed_reg"][aux_key].append(onp.array(-val[2]))
    return aux_dict


def looper_fn(loss_dict, aux_dict):

    cal_losses, flat_losses, val_losses = {}, {}, {}
    for key, value in aux_dict["loglike"].items():
        batch_key, exp_key = key
        if "cal" in batch_key:
            cal_losses[key] = value
        if "flat" in batch_key:
            flat_losses[key] = value
        if "val" in batch_key:
            val_losses[key] = value

    print_str = ""
    if len(cal_losses) > 0:
        print_str += "Cal: "

        losses = np.array(list(cal_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    if len(val_losses) > 0:
        print_str += " | Val: "

        losses = np.array(list(val_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    if len(flat_losses) > 0:
        print_str += " | Flat: "
        losses = np.array(list(flat_losses.values())).mean(0)
        print_str += f"{losses[-1]:.2f}"
        if len(losses) > 1:
            print_str += f" \u0394 {np.diff(losses)[-1]:.2f}"

    l2 = np.array(jtu.leaves(aux_dict["l2_reg"]))
    if len(l2) > 0:
        print_str += f" | L2: {l2[-1]:.2f}"
        if len(l2) > 1:
            print_str += f" \u0394 {np.diff(l2)[-1]:.2f}"

    # bleed = np.array(jtu.leaves(aux_dict["bleed_reg"]))
    # if len(bleed) > 0:
    #     print_str += f" | Bleed: {bleed[-1]:.2f}"
    #     if len(bleed) > 1:
    #         print_str += f" \u0394 {np.diff(bleed)[-1]:.2f}"

    return print_str


class BatchedTrainer(Trainer):

    def train(
        self,
        model,
        optimisers,
        epochs,
        batches: dict,
        batched_params: list = None,
        key=jr.PRNGKey(0),
        args={},
    ):
        # If no batch params, just call the parent class version
        if batched_params is None:
            return super().train(model, optimisers, epochs, batches)

        # Get the batches and raw exposures
        batches, exposures = self.unwrap_batches(batches)

        # Get the model parameters
        model_params = ModelParams({p: model.get(p) for p in optimisers.keys()})
        batch_params, reg_params = model_params.partition(batched_params)
        lrs = populate_lr_model(self.fishers, exposures, model_params)

        # Get the optax optimiser bits
        reg_optim, reg_state = get_optimiser(reg_params, optimisers)
        batch_optim, batch_state = get_optimiser(batch_params, optimisers)

        # Make the history objects
        reg_history = ParamHistory(reg_params)
        batch_history = ParamHistory(batch_params)

        # Get the loss and update functions
        val_grad_fn = get_val_grad_fn(self.loss_fn)
        loss_fn = get_norm_loss_fn(val_grad_fn, self.grad_fn)
        reg_update_fn = get_update_fn(reg_optim, self.norm_fn)
        batch_update_fn = get_update_fn(batch_optim, self.norm_fn)

        # Randomise batch inputs
        batch_keys = list(batches.keys())
        batch_inds, key = get_random_batch_order(batches, epochs, key)

        # Make the loss dictionary
        loss_dict = dict([(key, []) for key in batches.keys()])

        # Epoch loop
        t0 = time.time()
        looper = tqdm(range(0, epochs))
        for epoch in looper:
            if epoch == 1:
                t1 = time.time()

            model, args, key = self.args_fn(model, args, key, epoch)

            # Create an empty gradient model to append gradients to
            reg_grads = reg_params.map(lambda x: x * 0.0)

            # Loop over randomised batch order
            for i in batch_inds[epoch]:

                # Get the batch key and batch
                batch_key = batch_keys[i]
                batch = batches[batch_key]

                # Calculate the loss and gradients
                loss, grads, key = loss_fn(model_params, lrs, model, batch, args, key)

                # Append the mean batch loss to the loss dictionary
                loss_dict[batch_key].append(loss / len(batch))

                # Split the gradients into regular and batched, accumulate gradients
                batch_grads, new_grads = grads.partition(batch_params)
                reg_grads += new_grads

                # Update the batched parameters
                batch_params, batch_state, key = batch_update_fn(
                    batch_grads, batch_params, batch_state, args, key
                )

                # Append to history and update the model parameters
                batch_history = batch_history.append(batch_params)
                model_params = reg_params.combine(batch_params)

                # Check for NaNs and exit if so
                if np.isnan(loss):
                    print(f"Loss is NaN on epoch {epoch}, exiting fit")
                    history = reg_history.combine(batch_history)
                    return self.finalise(
                        t0, model, loss_dict, model_params, history, lrs, epochs, False
                    )

            # Update the regular parameters and append to history
            reg_params, reg_state, key = reg_update_fn(reg_grads, reg_params, reg_state, args, key)
            reg_history = reg_history.append(reg_params)

            # Paste together the batch and regular params
            model_params = reg_params.combine(batch_params)

            # Update the looper
            loop_fn = self.default_looper if self.looper_fn is None else self.looper_fn
            loop_fn(looper, loss_dict)

            # Print estimated run time
            if epoch == 0:
                # Get the final loss
                initial_loss = np.array([losses[-1] for losses in loss_dict.values()]).mean()
                print(f"\nInitial_loss Loss: {initial_loss:,.2f}")

            if epoch == 1:
                estimated_time = epochs * (time.time() - t1)
                formatted_time = str(timedelta(seconds=int(estimated_time)))
                print(f"Estimated run time: {formatted_time}")

        # Print the runtime stats and return Result object
        history = reg_history.combine(batch_history)
        return self.finalise(t0, model, loss_dict, model_params, history, lrs, epochs, True)


class Result(zdx.Base):
    losses: dict
    model: zdx.Base
    state: ModelParams
    lr_model: ModelParams
    history: ParamHistory
    aux: dict
    meta_data: dict
    best_batch: None
    best_state: None

    def __init__(
        self,
        losses,
        model,
        aux,
        state,
        history,
        lr_model,
        meta_data=None,
        best_batch=None,
        best_state=None,
    ):
        self.losses = losses
        self.model = model
        self.state = state
        self.history = history
        self.lr_model = lr_model
        self.meta_data = meta_data
        self.aux = aux
        self.best_batch = best_batch
        self.best_state = best_state


class ValBatchedTrainer(BatchedTrainer):

    def unwrap_batches(self, batches, validators):
        # Format the batches and exposures
        exposures = []
        for batch_key, batch in batches.items():
            exposures += batch
        for val_key, batch in validators.items():
            exposures += batch
        return {**batches, **validators}, exposures

    def finalise(
        self,
        t0,
        model,
        loss_dict,
        aux,
        model_params,
        history,
        lr_model,
        epochs,
        success,
        best_batch,
        best_state,
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
            best_batch=best_batch,
            best_state=best_state,
        )

    def train(
        self,
        model,
        optimisers,
        epochs,
        batches: dict,
        batched_params: list,
        validators: dict,
        validator_params: list,
        args={},
    ):
        # Ensure args key exists and is the right type
        args = self.check_args_key(args)

        # Get the batches and raw exposures
        batches, exposures = self.unwrap_batches(batches, validators)

        # Get the model parameters
        model_params = ModelParams({p: model.get(p) for p in optimisers.keys()})
        batch_params, reg_params = model_params.partition(batched_params)

        # Get the learning rate normalisation
        reg_lrs = populate_lr_model(self.fishers, exposures, reg_params)
        batch_lrs = jtu.map(lambda x: np.ones_like(x), batch_params)
        lrs = model_params.set("params", {**reg_lrs.params, **batch_lrs.params})

        # Get the optax optimiser bits
        reg_optim, reg_state = get_optimiser(reg_params, optimisers)
        batch_optim, batch_state = get_optimiser(batch_params, optimisers)

        # Make the history objects
        reg_history = ParamHistory(reg_params)
        batch_history = ParamHistory(batch_params)

        # Get the loss and update functions
        val_grad_fn = get_val_grad_fn(self.loss_fn)
        loss_fn = get_norm_loss_fn(val_grad_fn, self.grad_fn)
        reg_update_fn = get_update_fn(reg_optim, self.norm_fn)
        batch_update_fn = get_update_fn(batch_optim, self.norm_fn)

        # Randomise batch inputs
        batch_inds, args = get_random_batch_order(batches, epochs, args)

        # Make the loss dictionary
        batch_keys = list(batches.keys())
        loss_dict = {key: [] for key in batch_keys}

        aux_dict = {
            "loglike": {},
            "l2_reg": {},
            "bleed_reg": {},
        }
        for batch_key, exposures in batches.items():
            for exp in exposures:
                aux_dict["loglike"][(batch_key, exp.key)] = []
                aux_dict["l2_reg"][(batch_key, exp.key)] = []
                aux_dict["bleed_reg"][(batch_key, exp.key)] = []

        loop_fn = self.default_looper if self.looper_fn is None else self.looper_fn

        aux = {}
        best_val = 1e100
        best_batch = batch_params
        best_state = model_params

        # Epoch loop
        t0 = time.time()
        looper = tqdm(range(0, epochs))
        for epoch in looper:
            if epoch == 1:
                t1 = time.time()

            if self.args_fn is not None:
                model, args = self.args_fn(model, args, epoch)

            # Create an empty gradient model to append gradients to
            reg_grads = reg_params.map(lambda x: x * 0.0)
            batch_history = ParamHistory(batch_params)

            # Loop over randomised calibration batch
            for i in batch_inds[epoch]:

                # Get the batch key and batch
                batch_key = batch_keys[i]
                batch = batches[batch_key]
                loss, grads, args, aux = loss_fn(model_params, lrs, model, batch, args)

                # Append the mean batch loss to the loss dictionary
                loss_dict[batch_key].append(onp.array(loss) / len(batch))

                #
                if self.aux_fn is not None:
                    aux_dict = self.aux_fn(batch_key, aux_dict, aux)

                # Nuke the non-validator grads since we take gradients wrt all parameters
                if "val" in batch_key:
                    grad_params = grads.params
                    for param, value in grad_params.items():
                        if param not in validator_params:
                            if isinstance(value, dict):
                                grad_params[param] = jtu.map(lambda x: x * 0, value)
                            else:
                                grad_params[param] = value * 0
                    grads = grads.set("params", grad_params)

                # Split the gradients into regular and batched, accumulate gradients
                batch_grads, new_grads = grads.partition(batch_params)
                reg_grads += new_grads

                # Update the batched parameters if calibrator or flat
                if "cal" in batch_key or "flat" in batch_key:

                    # Update the batched parameters
                    batch_params, batch_state, args = batch_update_fn(
                        batch_grads, batch_params, batch_state, args
                    )

                    # Append to history and update the model parameters
                    batch_history = batch_history.append(batch_params)
                    model_params = reg_params.combine(batch_params)

                # Check for NaNs and exit if so
                if np.isnan(loss):
                    print(f"Loss is NaN on epoch {epoch}, exiting fit")
                    # history = reg_history.combine(batch_history)
                    history = reg_history
                    return self.finalise(
                        t0,
                        model,
                        loss_dict,
                        aux_dict,
                        model_params,
                        history,
                        lrs,
                        epochs,
                        True,
                        best_batch,
                        best_state,
                    )

            # Check if this is the best validation loss
            leaf_fn = lambda x: isinstance(x, list)
            loglikes = jtu.map(lambda x: x[-1], aux_dict["loglike"], is_leaf=leaf_fn)
            val = np.array([val for key, val in loglikes.items() if "val" in key[0]]).mean()

            if val < best_val:
                best_val = val
                best_batch = batch_history
                best_state = model_params

            # Update the regular parameters and append to history
            reg_params, reg_state, args = reg_update_fn(reg_grads, reg_params, reg_state, args)
            reg_history = reg_history.append(reg_params)

            # Paste together the batch and regular params
            model_params = reg_params.combine(batch_params)

            # Update the looper
            looper.set_description(loop_fn(loss_dict, aux_dict))

            # Print estimated run time
            if epoch == 0:
                compile_time = str(timedelta(seconds=int(time.time() - t0)))
                print(f"Compile time: {compile_time}")

                initial_loss = np.array([losses[-1] for losses in loss_dict.values()]).mean()
                print(f"\nInitial_loss Loss: {initial_loss:,.2f}")

            if epoch == 1:
                estimated_time = epochs * (time.time() - t1)
                formatted_time = str(timedelta(seconds=int(estimated_time)))
                print(f"Estimated run time: {formatted_time}")

        # Print the runtime stats and return Result object
        # history = reg_history.combine(batch_history)
        history = reg_history

        return self.finalise(
            t1,
            model,
            loss_dict,
            aux_dict,
            model_params,
            history,
            lrs,
            epochs,
            True,
            best_batch,
            best_state,
        )
