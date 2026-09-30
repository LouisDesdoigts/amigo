import jax.numpy as np
import jax.random as jr
from jax import vmap
import jax.tree as jtu
import numpy as onp
import time
import os
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from tqdm.auto import tqdm
from .core_models import ModelParams, ParamHistory
from .misc import BIG
from .fitting import (
    get_optimiser,
    get_val_grad_fn,
    get_norm_loss_fn,
    get_update_fn,
    get_random_batch_order,
    populate_lr_model,
    device_put_pytree,
    assign_batches_to_devices,
    Trainer,
    Result,
)



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


# def prior(model, bleeds, l2_norm=0.0, bleed_norm=0.0):
#     l2_reg = -l2_norm * np.mean(model.ramp.values**2)
#     bleed_reg = -bleed_norm * (np.sum(bleeds, axis=1) ** 2).sum()
#     return l2_reg, bleed_reg


# def posterior(model, exposure, l2_norm=0.0, bleed_norm=0.0):
#     slopes, bleed = exposure(model, return_bleed=True)
#     loglike = log_likelihood(slopes, exposure)
#     l2_reg, bleed_reg = prior(model, bleed, l2_norm=l2_norm, bleed_norm=bleed_norm)
#     return loglike + l2_reg + bleed_reg, (loglike, l2_reg, bleed_reg)


# def loss_fn(model, exposure, args={"l2": 0.0, "bleed": 0.0}):
#     loss, (loglike, l2_reg, bleed_reg) = posterior(
#         model, exposure, l2_norm=args["l2"]  # , bleed_norm=args["bleed"]
#     )
#     return -np.nanmean(loss), (-np.nanmean(loglike), l2_reg, bleed_reg)


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


def aux_fn(batch_key, aux_dict, aux):
    # Aux should have exposure keys, with values (loglike, l2_reg, bleed_reg)
    for exp_key, val in aux.items():
        aux_key = (batch_key, exp_key)
        aux_dict["loglike"][aux_key].append(onp.array(val[0]))
        aux_dict["l2_reg"][aux_key].append(onp.array(-val[1]))
        # aux_dict["bleed_reg"][aux_key].append(onp.array(-val[2]))
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


class ValBatchedTrainer(Trainer):

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
        batch_updates_per_epoch=None,
    ):
        """Prints stats and returns the result object"""
        # Final execution time
        elapsed_time = time.time() - t0
        formatted_time = str(timedelta(seconds=int(elapsed_time)))
        print(f"Full Time: {formatted_time}")

        # Get the final loss
        final_loss = np.array([losses[-1] for losses in loss_dict.values()]).mean()
        print(f"Final Loss: {final_loss:,.2f}")

        meta_data = {
            "elapsed_time": formatted_time,
            "epochs": epochs,
            "successful": success,
        }
        # Tells retrain_fns.py how many batch_history entries make up one epoch's
        # worth of batched-param (e.g. nn_weights) updates -- 1 under the
        # multi_device path (one accumulated-gradient update/epoch), or the
        # number of cal/flat batches under the sequential path (one update per
        # batch). Without this, `history["nn_weights"][-n_batch:].mean(0)` would
        # silently average across multiple epochs instead of within the final one
        # when running multi-GPU. See the multigpu-validator-required memory note.
        if batch_updates_per_epoch is not None:
            meta_data["batch_updates_per_epoch"] = batch_updates_per_epoch

        # Return
        return Result(
            losses=loss_dict,
            model=model_params.inject(model),
            aux=aux,
            state=model_params,
            history=history,
            lr_model=lr_model,
            meta_data=meta_data,
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
        summarise_kwargs={},
    ):
        # Ensure args key exists and is the right type
        args = self.check_args_key(args)

        # Get the batches and raw exposures
        batches, exposures = self.unwrap_batches(batches, validators)

        # Get the model parameters
        model_params = ModelParams({p: model.get(p) for p in optimisers.keys()})
        batch_params, reg_params = model_params.partition(batched_params)

        multi_device = self.devices is not None and len(self.devices) > 1
        if len(batched_params) > 0:
            cadence = "once per epoch (multi-GPU)" if multi_device else "every batch"
            print(
                f"Batched params {batched_params} update {cadence} "
                f"({len(batches)} batches/epoch) -- any `start`/schedule step "
                f"passed to their optimiser is in units of batches, not epochs "
                f"under the sequential (single-device) path."
            )

        # Get the learning rate normalisation
        reg_lrs = populate_lr_model(self.fishers, exposures, reg_params)
        batch_lrs = jtu.map(lambda x: np.ones_like(x), batch_params)
        lrs = model_params.set("params", {**reg_lrs.params, **batch_lrs.params})

        # Get the optax optimiser bits
        #
        # NOTE: `reg_optim`'s update_fn is called once per EPOCH (below), so an
        # `sgd(lr, start)`/`adam(lr, start)` schedule for a regular parameter
        # activates at epoch `start`. `batch_optim`'s update_fn is instead
        # called once per BATCH (there are `len(batches)` batches per epoch),
        # so `start` for any parameter in `batched_params` is in units of
        # *batches*, not epochs -- e.g. `start=5` fires after 5 batches, which
        # is a fraction of one epoch, not epoch 5. If a batched parameter ever
        # needs a nonzero warm-up start, multiply it by the number of batches
        # per epoch when constructing its optimiser.
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

        # Multi-device setup. The sequential path below updates batch_params
        # (e.g. nn_weights) immediately after every single cal/flat batch --
        # online-SGD-style, inherently sequential, since batch N's forward pass
        # depends on batch N-1's update. That can't be split across devices as-is.
        # Under multi_device, every batch (cal/flat/val alike) is instead
        # evaluated against ONE frozen model_params snapshot per epoch and all
        # gradients (reg AND batch) are accumulated and applied once at epoch
        # end -- exactly like the plain Trainer. This makes batches embarrassingly
        # parallel across devices, at the cost of nn_weights moving from ~n_batch
        # small updates/epoch to 1 larger update/epoch. See the
        # multigpu-validator-required memory note for why that tradeoff was
        # chosen deliberately.
        if multi_device:
            device_map, device_load = assign_batches_to_devices(batches, self.devices)
            print(f"Splitting {len(batches)} batches across {len(self.devices)} devices by estimated cost:")
            for d in self.devices:
                keys_here = [k for k, dd in device_map.items() if dd == d]
                print(f"  {d}: {keys_here} (total cost {device_load[d]:,.0f})")

            model_by_device = {d: device_put_pytree(model, d) for d in self.devices}
            lrs_by_device = {d: device_put_pytree(lrs, d) for d in self.devices}
            batches_by_device = {
                key: device_put_pytree(batch, device_map[key]) for key, batch in batches.items()
            }
            pool_by_device = {d: ThreadPoolExecutor(max_workers=1) for d in self.devices}

            # One-time rebalance after the first (post-compile) epoch, using
            # measured wall time instead of the a-priori cost estimate -- see
            # Trainer.train()'s identical mechanism in fitting.py for why.
            rebalance_epoch = 1
            measured_cost = {}

            def _timed_loss_fn(*a):
                t_start = time.time()
                r = loss_fn(*a)
                return r, time.time() - t_start

        # Randomise batch inputs (only consumed by the sequential path below --
        # multi_device evaluates every batch every epoch regardless of order)
        batch_inds, args = get_random_batch_order(batches, epochs, args)

        # Make the loss dictionary
        batch_keys = list(batches.keys())
        loss_dict = {key: [] for key in batch_keys}
        # How many batch_history entries make up one epoch (see finalise()'s
        # batch_updates_per_epoch docstring above).
        batch_updates_per_epoch = (
            1 if multi_device
            else sum(1 for k in batch_keys if "cal" in k or "flat" in k)
        )

        aux_dict = {
            "loglike": {},
            "l2_reg": {},
            # "bleed_reg": {},
        }
        for batch_key, exposures in batches.items():
            for exp in exposures:
                aux_dict["loglike"][(batch_key, exp.key)] = []
                aux_dict["l2_reg"][(batch_key, exp.key)] = []
                # aux_dict["bleed_reg"][(batch_key, exp.key)] = []

        loop_fn = self.default_looper if self.looper_fn is None else self.looper_fn

        aux = {}
        best_val = BIG
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
            _batch_history = ParamHistory(batch_params)

            if multi_device:
                batch_grads = batch_params.map(lambda x: x * 0.0)
                nan_hit = False

                # Replicate this epoch's frozen snapshot to every device with a batch.
                params_by_device = {
                    d: device_put_pytree(model_params, d) for d in set(device_map.values())
                }

                # self.grad_fn (retrain_fns.grads_fn) mutates args["t"] and args["key"]
                # once per batch call, threaded sequentially in the sequential path
                # below. To get the same result regardless of dispatch order,
                # precompute what each batch's args would have been at its position in
                # that sequential order, same as Trainer's multi-device path -- except
                # here the per-epoch call count (n_b, every batch: cal/flat/val alike)
                # isn't necessarily args["n_batch"] itself (that's set by the caller to
                # the cal+flat count only), so the final advance uses n_b/n_batch, not
                # a bare 1.0.
                n_b = len(batches)
                epoch_keys = jr.split(args["key"], n_b + 1)
                dispatched = []
                for i, (batch_key, batch) in enumerate(batches.items()):
                    d = device_map[batch_key]
                    batch_args = dict(args)
                    batch_args["t"] = args["t"] + i / args["n_batch"]
                    batch_args["key"] = epoch_keys[i + 1]
                    batch_args = device_put_pytree(batch_args, d)
                    future = pool_by_device[d].submit(
                        _timed_loss_fn,
                        params_by_device[d], lrs_by_device[d], model_by_device[d],
                        batches_by_device[batch_key], batch_args,
                    )
                    dispatched.append((batch_key, d, future))
                dispatched = [(k, d, f.result()) for k, d, f in dispatched]
                if epoch == rebalance_epoch:
                    for k, d, (_r, dt) in dispatched:
                        measured_cost[k] = dt
                dispatched = [(k, d, r) for k, d, (r, _dt) in dispatched]

                # One-time rebalance using this epoch's measured times.
                if epoch == rebalance_epoch:
                    new_map, new_load = assign_batches_to_devices(
                        {k: [] for k in measured_cost}, self.devices,
                        _cost_override=measured_cost,
                    )
                    moved = {k: (device_map[k], new_map[k]) for k in measured_cost if new_map[k] != device_map[k]}
                    if moved:
                        print(f"Rebalancing {len(moved)}/{len(measured_cost)} batches after epoch {epoch} "
                              f"using measured time instead of the cost estimate:")
                        for d in self.devices:
                            keys_here = [k for k, dd in new_map.items() if dd == d]
                            total_t = sum(measured_cost[k] for k in keys_here)
                            print(f"  {d}: {keys_here} (measured {total_t:.2f}s)")
                        for k, (_old, new_d) in moved.items():
                            batches_by_device[k] = device_put_pytree(batches[k], new_d)
                        device_map = new_map
                    else:
                        print("Rebalancing after epoch 1: measured split already matches the cost-estimate split.")

                for batch_key, d, (loss, new_grads, _returned_args, aux) in dispatched:
                    if np.isnan(loss):
                        nan_hit = True

                    # loss/new_grads were never explicitly moved off their batch's
                    # assigned device -- see Trainer's identical comment in
                    # fitting.py for why this matters once results get combined.
                    loss = device_put_pytree(loss, self.devices[0])
                    loss_dict[batch_key].append(onp.array(loss) / len(batches[batch_key]))

                    if self.aux_fn is not None:
                        aux_dict = self.aux_fn(batch_key, aux_dict, aux)

                    new_grads = device_put_pytree(new_grads, self.devices[0])

                    # Nuke the non-validator grads since we take gradients wrt all
                    # parameters -- identical to the sequential path below.
                    if "val" in batch_key:
                        grad_params = new_grads.params
                        for param, value in grad_params.items():
                            if param not in validator_params:
                                if isinstance(value, dict):
                                    grad_params[param] = jtu.map(lambda x: x * 0, value)
                                else:
                                    grad_params[param] = value * 0
                        new_grads = new_grads.set("params", grad_params)

                    batch_grads_i, reg_grads_i = new_grads.partition(batch_params)
                    reg_grads += reg_grads_i
                    batch_grads += batch_grads_i

                args["t"] += n_b / args["n_batch"]
                args["key"] = epoch_keys[0]

                if nan_hit:
                    print(f"Loss is NaN on epoch {epoch}, exiting fit")
                    history = reg_history.combine(batch_history)
                    return self.finalise(
                        t0, model, loss_dict, aux_dict, model_params, history, lrs,
                        epochs, True, best_batch, best_state,
                        batch_updates_per_epoch=batch_updates_per_epoch,
                    )

                # One update per epoch for both param groups (see the multi_device
                # comment above for why nn_weights moves from ~n_batch updates/epoch
                # to 1).
                batch_params, batch_state, args = batch_update_fn(
                    batch_grads, batch_params, batch_state, args
                )
                batch_history = batch_history.append(batch_params, max_len=self.batch_history_max_len)

                # Matches the sequential path's convention: by the time the "best
                # validation loss" check below runs, model_params reflects this
                # epoch's fresh batch_params combined with the *old* (not yet
                # updated this epoch) reg_params -- reg_params only updates once,
                # further down, after that check.
                model_params = reg_params.combine(batch_params)

            else:
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

                        # Append to history and update the model parameters. _batch_history
                        # is reset every epoch (bounded naturally, at most n_batch entries
                        # at a time) so doesn't need capping; batch_history accumulates for
                        # the whole run and is exactly what OOM-killed a long run -- see
                        # batch_history_max_len's docstring on Trainer.__init__.
                        _batch_history = _batch_history.append(batch_params)
                        batch_history = batch_history.append(batch_params, max_len=self.batch_history_max_len)

                        model_params = reg_params.combine(batch_params)

                    # Check for NaNs and exit if so
                    if np.isnan(loss):
                        print(f"Loss is NaN on epoch {epoch}, exiting fit")
                        history = reg_history.combine(batch_history)
                        # history = reg_history
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
                            batch_updates_per_epoch=batch_updates_per_epoch,
                        )

            # Check if this is the best validation loss
            leaf_fn = lambda x: isinstance(x, list)
            loglikes = jtu.map(lambda x: x[-1], aux_dict["loglike"], is_leaf=leaf_fn)
            val = np.array([val for key, val in loglikes.items() if "val" in key[0]]).mean()

            if val < best_val:
                best_val = val
                # Average each batched param (e.g. nn_weights) over the winning epoch's
                # own batches, matching final_state's convention (which averages
                # `history[-n_batch:]`) instead of just keeping whatever batch_params
                # happened to be after the last batch processed this epoch.
                # _batch_history is reset to ParamHistory(batch_params) at the top of
                # the epoch, which seeds itself with one pre-epoch snapshot before any
                # append() below -- so this epoch's actual len(batches) updates are its
                # last len(batches) entries, same slice final_state uses. Wrapped in
                # try/except: this runs unprotected inside the training loop (unlike
                # retrain_fns.py's best_state save, which already has one), so a bug
                # here shouldn't be able to crash a long run -- fall back to the plain
                # snapshot and keep training.
                try:
                    n_b = len(batches)
                    is_leaf = lambda leaf: isinstance(leaf, list)
                    avg_params = jtu.map(
                        lambda leaf: onp.array(leaf[-n_b:]).mean(0), _batch_history.params, is_leaf=is_leaf
                    )
                    best_batch = batch_params.set("params", avg_params)
                except Exception as e:
                    print(f"best_batch epoch-averaging failed, falling back to the plain snapshot: {e}")
                    best_batch = batch_params
                best_state = model_params

            # Update the regular parameters and append to history
            reg_params, reg_state, args = reg_update_fn(reg_grads, reg_params, reg_state, args)
            # Same per-epoch device sync + O(epochs^2) list growth as plain Trainer
            # (see its history_stride docstring), so stride it the same way. No
            # hi-res exemption needed here: unlike plain Trainer, nn_weights lives in
            # batch_params/batch_history under this trainer (appended every batch,
            # already full resolution, untouched by this), not in reg_params -- so
            # reg_history holds nothing that retrain_fns.py needs at full resolution.
            if epoch % self.history_stride == 0:
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

            if epoch in self.intermediate_prints:
                history = reg_history.combine(batch_history)
                intermediate_result = self.finalise(t1, model, loss_dict, aux_dict, model_params, history, lrs, epochs, True, best_batch, best_state, batch_updates_per_epoch=batch_updates_per_epoch)
                intermediate_save_dir = os.path.join(self.save_path, f"epoch_{epoch:06d}") if self.save_path is not None else None
                if intermediate_save_dir is not None:
                    os.mkdir(intermediate_save_dir)
                self.summarise_fn(intermediate_result, intermediate_save_dir, **summarise_kwargs)


        # Print the runtime stats and return Result object
        history = reg_history.combine(batch_history)
        # history = reg_history

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
            batch_updates_per_epoch=batch_updates_per_epoch,
        )
