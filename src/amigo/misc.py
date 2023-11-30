import jax
import zodiax as zdx
import jax.numpy as np
import jax.scipy as jsp
import dLux as dl
import tqdm.notebook as tqdm

def fit_image(
    model,
    data,
    err,
    loss_fn,
    grad_fn,
    norm_fn,
    epochs,
    config,
    loss_scale=1e-4,
    verbose=True,
    print_grads=False,
):
    params = list(config.keys())
    optimisers = [config[param]["optim"] for param in params]

    model = zdx.set_array(model, params)
    optim, opt_state = zdx.get_optimiser(model, params, optimisers)

    if verbose:
        print("Compiling...")
    loss, grads = loss_fn(model, data, err)
    if print_grads:
        for param in params:
            print(f"{param}: {grads.get(param)}")
    losses, models_out = [], [model]

    if verbose:
        looper = tqdm(range(epochs), desc="Loss %.2f" % (loss * loss_scale))
    else:
        looper = range(epochs)

    for i in looper:
        # calculate the loss and gradient
        new_loss, grads = loss_fn(model, data, err)

        if new_loss > loss:
            print(
                f"Loss increased from {loss * loss_scale:.2f} to "
                f"{new_loss * loss_scale:.2f} on {i} th epoch"
            )
        loss = new_loss
        if np.isnan(loss):
            print(f"Loss is NaN on {i} th epoch")
            return losses, models_out

        # Apply any processing to the gradients
        grads = grad_fn(grads, config, i)

        # apply the update
        updates, opt_state = optim.update(grads, opt_state)
        model = zdx.apply_updates(model, updates)

        # Apply normalisation
        model = norm_fn(model)

        # save results
        models_out.append(model)
        losses.append(loss)

        if verbose:
            looper.set_description("Loss %.2f" % (loss * loss_scale))

    return losses, models_out


from typing import List, Any


class FresnelOptics(dl.CartesianOpticalSystem):
    """
    fl = pixel_scale_m / pixel_scale_rad -> NIRISS pixel scales are 18um  and
    0.0656 arcsec respectively, so fl ~= 56.6m
    """

    defocus: jax.Array  # metres, is this actually um??

    def __init__(self, *args, **kwargs):
        self.defocus = np.array(0.0)
        super().__init__(*args, **kwargs)

    def propagate_mono(
        self: dl.optical_systems.OpticalSystem,
        wavelength: jax.Array,
        offset: jax.Array = np.zeros(2),
        return_wf: bool = False,
    ) -> jax.Array:
        """
        Propagates a monochromatic point source through the optical layers.

        Parameters
        ----------
        wavelength : float, metres
            The wavelength of the wavefront to propagate through the optical layers.
        offset : Array, radians = np.zeros(2)
            The (x, y) offset from the optical axis of the source.
        return_wf: bool = False
            Should the Wavefront object be returned instead of the psf Array?

        Returns
        -------
        object : Array, Wavefront
            if `return_wf` is False, returns the psf Array.
            if `return_wf` is True, returns the Wavefront object.
        """
        # Unintuitive syntax here, this is saying call the _parent class_ of
        # CartesianOpticalSystem, ie LayeredOpticalSystem, which is what we want.
        wf = super(dl.optical_systems.CartesianOpticalSystem, self).propagate_mono(
            wavelength, offset, return_wf=True
        )

        # Propagate
        true_pixel_scale = self.psf_pixel_scale / self.oversample
        pixel_scale = 1e-6 * true_pixel_scale
        psf_npixels = self.psf_npixels * self.oversample

        wf = wf.propagate_fresnel(
            psf_npixels,
            pixel_scale,
            self.focal_length,
            focal_shift=self.defocus,
        )

        # Return PSF or Wavefront
        if return_wf:
            return wf
        return wf.psf


# def gettr(im, support):
#     return im[support[0], support[1]]


# def like_fn(model, data, sigma, sup):
#     return jsp.stats.norm.pdf(
#         gettr(model.model(), sup), loc=gettr(data, sup), scale=gettr(sigma, sup)
#     )


# def loglike_fn(model, data, sigma, sup):
#     return jsp.stats.norm.logpdf(
#         gettr(model.model(), sup), loc=gettr(data, sup), scale=gettr(sigma, sup)
#     )
#     # return jsp.stats.norm.logpdf(model.model(), loc=data, scale=sigma)


def get_likelihoods(psf, data, err):
    return (
        jsp.stats.norm.pdf(psf, loc=data, scale=err),
        -jsp.stats.norm.logpdf(psf, loc=data, scale=err),
    )
