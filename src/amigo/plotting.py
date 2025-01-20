import jax.numpy as np
import jax.scipy as jsp
import dLux.utils as dlu
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from .stats import posterior

inferno = colormaps["inferno"]
seismic = colormaps["seismic"]


def plot_losses(losses, start, stop=-1):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Full Loss")
    plt.plot(losses)

    if start >= len(losses):
        start = 0
    last_losses = losses[start:stop]
    n = len(last_losses)
    plt.subplot(1, 2, 2)
    plt.title(f"Final {n} Losses")
    plt.plot(np.arange(start, start + n), last_losses)

    plt.tight_layout()
    plt.show()


def summarise_fit(
    model,
    exposure,
    residuals=False,
    histograms=False,
    flat_field=False,
    up_the_ramp=False,
    up_the_ramp_norm=False,
    full_bias=False,
    aberrations=False,
    pow=0.5,
    loglike_fn=None,
):

    inferno = colormaps["inferno"]
    seismic = colormaps["seismic"]

    # slopes = model_fn(model, exposure)
    slopes = exposure(model)
    data = exposure.slopes

    residual = data - slopes

    #
    if loglike_fn is not None:
        posterior_im = posterior(model, exposure, return_im=True)
    else:
        posterior_im = posterior(model, exposure, return_im=True)
        # posterior_im = exposure.from_vec(posterior_vec)

    # loglike_im = exposure.log_likelihood(slopes, return_im=True)

    nan_mask = np.where(np.isnan(posterior_im))
    slopes = slopes.at[:, *nan_mask].set(np.nan)
    data = data.at[:, *nan_mask].set(np.nan)

    final_loss = np.nansum(-posterior_im) / np.prod(np.array(data.shape[-2:]))

    norm_res_slope = residual / (exposure.variance**0.5)
    norm_res_slope = norm_res_slope.at[:, *nan_mask].set(np.nan)

    norm_res_vec = exposure.to_vec(norm_res_slope)
    norm_res_vec = norm_res_vec[~np.isnan(norm_res_vec)]
    norm_res_vec = norm_res_vec[~np.isinf(norm_res_vec)]

    x = np.nanmax(np.abs(norm_res_vec))
    xs = np.linspace(-x, x, 200)
    ys = jsp.stats.norm.pdf(xs)

    effective_data = data.sum(0)
    effective_psf = slopes.sum(0)
    vmax = np.maximum(np.nanmax(np.abs(effective_data)), np.nanmax(np.abs(effective_psf)))
    vmin = np.minimum(np.nanmin(np.abs(effective_data)), np.nanmin(np.abs(effective_psf)))

    inferno.set_bad("k", 0.5)
    seismic.set_bad("k", 0.5)

    skip = False
    if np.isnan(vmin) or np.isnan(vmax):
        skip = True

    if not skip:

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 3, 1)
        plt.title(f"Pixel neg log posterior: {final_loss:,.1f}")
        plt.imshow(-posterior_im, cmap=inferno)
        plt.colorbar()

        v = np.nanmax(np.abs(norm_res_slope.mean(0)))
        plt.subplot(1, 3, 2)
        plt.title("Mean noise normalised slope residual")
        plt.imshow(norm_res_slope.mean(0), vmin=-v, vmax=v, cmap=seismic)
        plt.colorbar()

        ax = plt.subplot(1, 3, 3)
        ax.set_title(f"Noise normalised residual sigma: {norm_res_vec.std():.3}")
        ax.hist(norm_res_vec.flatten(), bins=50, density=True)

        ax2 = ax.twinx()
        ax2.plot(xs, ys, c="k")
        ax2.set_ylim(0)

        plt.tight_layout()
        plt.show()

        if residuals:
            norm = colors.PowerNorm(gamma=pow, vmin=-vmin, vmax=vmax)

            plt.figure(figsize=(15, 4))
            plt.subplot(1, 3, 1)
            plt.title(f"Data $^{{{str(pow)}}}$")
            plt.imshow(effective_data, cmap=inferno, norm=norm)
            plt.colorbar()

            plt.subplot(1, 3, 2)
            plt.title(f"Effective PSF $^{{{str(pow)}}}$")
            plt.imshow(effective_psf, cmap=inferno, norm=norm)
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.title(f"Pixel neg log posterior: {final_loss:,.1f}")
            plt.imshow(-posterior_im, cmap=inferno)
            plt.colorbar()

            plt.tight_layout()
            plt.show()

        if histograms:

            plt.figure(figsize=(15, 4))
            ax = plt.subplot(1, 3, 1)
            ax.set_title(f"Noise normalised residual sigma: {norm_res_vec.std():.3}")
            ax.hist(norm_res_vec.flatten(), bins=50, density=True)

            ax2 = ax.twinx()
            ax2.plot(xs, ys, c="k")
            ax2.set_ylim(0)

            ax = plt.subplot(1, 3, 2)
            ax.set_title(f"Noise normalised residual sigma: {norm_res_vec.std():.3}")
            ax.hist(norm_res_vec.flatten(), bins=50)[0]
            ax.semilogy()

            # ax2 = ax.twinx()
            # ax2.plot(xs, bins.max() * ys, c="k")
            # ax2.semilogy()

            v = np.nanmax(np.abs(norm_res_slope.mean(0)))
            plt.subplot(1, 3, 3)
            plt.title("Mean noise normalised slope residual")
            plt.imshow(norm_res_slope.mean(0), vmin=-v, vmax=v, cmap=seismic)
            plt.colorbar()

            plt.tight_layout()
            plt.show()

    if flat_field:
        plt.figure(figsize=(15, 4))

        plt.subplot(1, 3, 1)
        plt.title("Mean Pixel Response Function")
        v = np.max(np.abs(model.detector.sensitivity.SRF - 1))
        plt.imshow(model.detector.sensitivity.SRF, vmin=1 - v, vmax=1 + v, cmap=seismic)
        plt.colorbar()

        FF = dlu.resize(model.detector.sensitivity.FF, 80)
        nan_mask = np.where(np.isnan(data.mean(0)))
        FF = FF.at[nan_mask].set(np.nan)
        v = np.nanmax(np.abs(FF - 1))

        plt.subplot(1, 3, 2)
        plt.title("Flat Field")
        plt.imshow(FF, vmin=1 - v, vmax=1 + v, cmap=seismic)
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.title("Flat Field Histogram")
        plt.hist(FF.flatten(), bins=100)
        # plt.xlim(0, 2)
        plt.show()

    if full_bias:
        coeffs = model.one_on_fs[exposure.get_key("one_on_fs")]
        nan_mask = 1 + (np.nan * np.isnan(data.sum(0)))
        bias = nan_mask * model.biases[exposure.get_key("bias")]

        plt.figure(figsize=(15, 4))
        plt.subplot(1, 2, 1)
        plt.title("Pixel Bias")
        plt.imshow(bias, cmap=inferno)
        plt.colorbar()

        plt.subplot(2, 4, (3, 4))
        plt.title("1/f Gradient")
        plt.imshow(coeffs[..., 0])
        plt.colorbar()
        plt.xlabel("x-pixel")
        plt.ylabel("Group")

        plt.subplot(2, 4, (7, 8))
        plt.title("1/f Bias")
        plt.imshow(coeffs[..., 1])
        plt.colorbar()
        plt.xlabel("x-pixel")
        plt.ylabel("Group")

        plt.tight_layout()
        plt.show()

    if aberrations:
        # Get the AMI mask and applied mask
        pupil_mask = model.optics.pupil_mask
        pupil_mask = pupil_mask.set(
            "abb_coeffs", model.aberrations[exposure.get_key("aberrations")]
        )
        if hasattr(model, "reflectivity"):
            pupil_mask = pupil_mask.set(
                "amp_coeffs", model.reflectivity[exposure.get_key("reflectivity")]
            )

        optics = model.optics
        amp = pupil_mask.calc_transmission()
        mask = pupil_mask.calc_mask(optics.wf_npixels, optics.diameter)
        abb = pupil_mask.calc_aberrations()
        # mask, amp, abb = pupil_mask.calculate(optics.wf_npixels, optics.diameter)
        amp_in = np.where(mask < 1.0, np.nan, mask * amp)
        abb_in = np.where(mask < 1.0, np.nan, 1e9 * abb)

        # # Get the applied opds in nm and flip to match the mask
        # static_opd = np.flipud(exposure.opd) * 1e9
        # total_opd = np.where(mask < 1.0, np.nan, abb_in)

        plt.figure(figsize=(15, 4))

        v = np.nanmax(amp_in - 1)
        plt.subplot(1, 3, 1)
        plt.title("Applied mask")
        plt.imshow(amp_in, cmap=inferno)
        plt.colorbar(label="Transmission")

        v = np.nanmax(np.abs(abb_in))
        plt.subplot(1, 3, 2)
        plt.title("Applied OPD")
        plt.imshow(abb_in, cmap=seismic, vmin=-v, vmax=v)
        plt.colorbar(label="OPD (nm)")

        # v = np.nanmax(np.abs(total_opd))
        # plt.subplot(1, 3, 3)
        # plt.title("Total OPD")
        # plt.imshow(total_opd, cmap=seismic, vmin=-v, vmax=v)
        # plt.colorbar(label="OPD (nm)")

        plt.tight_layout()
        plt.show()

    if up_the_ramp:
        ncols = 4
        nrows = exposure.nslopes // ncols
        if exposure.nslopes % ncols > 0:
            nrows += 1

        plt.figure(figsize=(5 * ncols, 4 * nrows))
        plt.suptitle("Up The Ramp Residuals")

        for i in range(exposure.nslopes):
            # plt.subplot(4, 4, i + 1)
            plt.subplot(nrows, ncols, i + 1)
            v = np.nanmax(np.abs(residual[i]))
            plt.imshow(residual[i], cmap=seismic, vmin=-v, vmax=v)
            plt.colorbar()
        plt.show()

    if up_the_ramp_norm:
        ncols = 4
        nrows = exposure.nslopes // ncols
        if exposure.nslopes % ncols > 0:
            nrows += 1

        plt.figure(figsize=(5 * ncols, 4 * nrows))
        plt.suptitle("Normalised Up The Ramp Residuals")

        for i in range(exposure.nslopes):
            # plt.subplot(4, 4, i + 1)
            plt.subplot(nrows, ncols, i + 1)
            v = np.nanmax(np.abs(norm_res_slope[i]))
            plt.imshow(norm_res_slope[i], cmap=seismic, vmin=-v, vmax=v)
            plt.colorbar()
        plt.show()


def plot(history, exposures=None, key_fn=None, ignore=[], start=0, end=-1):

    params = list(history.params.keys())
    params_in = [param for param in params if param not in ignore]

    # Plot in groups of two
    for i in np.arange(0, len(params_in), 2):
        plt.figure(figsize=(16, 5))
        ax = plt.subplot(1, 2, 1)

        param = params_in[i]
        leaf = history.params[param]
        _plot_ax(leaf, ax, param, exposures, key_fn, start=start, end=end)

        ax = plt.subplot(1, 2, 2)
        if i + 1 == len(params_in):
            plt.tight_layout()
            plt.show()
            break

        param = params_in[i + 1]
        leaf = history.params[param]
        _plot_ax(leaf, ax, param, exposures, key_fn, start=start, end=end)

        plt.tight_layout()
        plt.show()


def _format_leaf(leaf, per_exp=False, keys=None):
    """
    Takes in a tuple, or a dictionary of a tuples of parameters and returns a 2D
    array of values for plotting.
    """

    if isinstance(leaf, list):
        # I think we can return an array here, first axis in then always the
        # history. We also need to deal with potential dimensionality (such as
        # mirror aberrations) so we reshape the remaining axes into a single axis
        return np.array(leaf).reshape(len(leaf), -1)

    # leaf should always be a dictionary now, but that is presently not enforced
    if keys is None:
        keys = list(leaf.keys())
    values = [_format_leaf(leaf[key]) for key in keys]

    if per_exp:
        return values
    return np.concatenate(values, axis=-1)


def _get_styles(n):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    linestyles = ["-", "--", "-.", ":"]

    color_list = [colors[i % len(colors)] for i in range(n)]
    linestyle_list = [linestyles[i // len(colors)] for i in range(n)]

    return color_list, linestyle_list


def _plot_ax(leaf, ax, param, exposures=None, key_fn=lambda x: x.key, start=0, end=-1):

    if exposures is not None:
        keys = [exp.key for exp in exposures]
        labels = [key_fn(exp) for exp in exposures]
    else:
        keys = None

    if isinstance(leaf, dict):
        values = _format_leaf(leaf, per_exp=True, keys=keys)

        if keys is None:
            keys = list(leaf.keys())
            labels = keys

        colors, linestyles = _get_styles(len(keys))

        for val, c, ls, label in zip(values, colors, linestyles, labels):
            kwargs = {"c": c, "ls": ls}
            _plot_param(ax, val, param, start=start, end=end, **kwargs)
            ax.plot([], label=label, **kwargs)

        plt.legend()

    else:
        arr = _format_leaf(leaf)
        _plot_param(ax, arr, param, start=start, end=end)


def _plot_param(ax, arr, param, start=0, end=-1, **kwargs):
    """This is the ugly gross function that is necessary"""
    arr = arr[start:end]
    epochs = np.arange(len(arr))
    ax.set(xlabel="Epochs", title=param)

    match param:
        case "positions":
            norm_pos = arr - arr[0]
            # rs = np.hypot(norm_pos[:, 0], norm_pos[:, 1])
            # ax.plot(epochs, rs, **kwargs)
            ax.plot(epochs, norm_pos, **kwargs)
            ax.set(ylabel="$\Delta$ Position (arcsec)")

        case "fluxes":
            norm_flux = arr - arr[0]
            # norm_flux = 100 * (1 - arr / arr[0])
            ax.plot(epochs, norm_flux, **kwargs)
            ax.set(ylabel="$\Delta$ Flux (log)")

        case "aberrations":
            norm_ab = arr - arr[0]
            ax.plot(epochs, norm_ab, alpha=0.4, **kwargs)
            ax.set(ylabel="$\Delta$ Aberrations (nm)")

        case "one_on_fs":
            norm_oneonf = arr - arr[0]
            ax.plot(epochs, norm_oneonf, alpha=0.25, **kwargs)
            ax.set(ylabel="$\Delta$ one_on_fs")

        case "SRF":
            srf = arr - arr[0]
            ax.plot(epochs, srf, **kwargs)
            ax.set(ylabel="$\Delta$ SRF")

        case "holes":
            arr *= 1e3
            norm_holes = arr - arr[0]
            ax.plot(epochs, norm_holes, **kwargs)
            ax.set(ylabel="$\Delta$ Pupil Mask Holes (mm)")

        case "f2f":
            arr *= 1e2
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Pupil Mask f2f (cm)")

        case "biases":
            norm_bias = arr - arr[0]
            ax.plot(epochs, norm_bias, alpha=0.25, **kwargs)
            ax.set(ylabel="$\Delta$ Bias")

        case "rotation":
            # arr = dlu.rad2deg(arr)
            norm_rot = arr
            ax.plot(epochs, norm_rot, **kwargs)
            ax.set(ylabel="Rotation (deg)")

        case "compression":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Compression")

        case "shear":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Shear")

        case "translation":
            norm_arr = arr - arr[0]
            norm_arr *= 1e2
            ax.plot(epochs, norm_arr, **kwargs)
            ax.set(ylabel="$\Delta$ translation (cm)")

        case "holes":
            norm_arr = arr - arr[0]
            norm_arr *= 1e2
            ax.plot(epochs, norm_arr, **kwargs)
            ax.set(ylabel="$\Delta$ Hole position (cm)")

        case "dark_current":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Dark Current")

        case "defocus":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Defocus")

        case "jitter.r":
            ax.plot(epochs, 1e3 * arr, **kwargs)
            ax.set(ylabel="Jitter Magnitude (mas)")

        case "amplitudes":
            norm_amplitudes = arr
            ax.plot(epochs, norm_amplitudes, **kwargs)
            ax.set(ylabel="Visibility Amplitude")

        case "phases":
            norm_phases = dlu.rad2deg(arr)
            ax.plot(epochs, norm_phases, **kwargs)
            ax.set(ylabel="Visibility Phase (deg)")

        case "separations":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Binary Separation (arcsec)")

        case "position_angles":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Position Angle (deg)")

        case "contrasts":
            ax.plot(epochs, arr, **kwargs)
            ax.set(ylabel="Log Contrast")

        case "EDM.conv.values":
            norm_weights = arr - arr[0]
            ax.plot(epochs, norm_weights, **kwargs)
            ax.set(ylabel="$\Delta$ Convolutional Values")

        case "EDM.amplitude":
            norm_weights = arr - arr[0]
            ax.plot(epochs, norm_weights, **kwargs)
            ax.set(ylabel="$\Delta$ Convolutional Output Amplitude")

        case _:
            print(f"No formatting function for {param}")
            ax.plot(epochs, arr - arr[0], **kwargs)
            ax.set(ylabel="$\Delta$ values")
