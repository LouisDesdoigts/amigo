import jax.numpy as np
import jax.scipy as jsp
import dLux.utils as dlu
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# from .lib import get_likelihoods, pairwise_vectors, osamp_freqs
from misc import get_likelihoods
from interferometry import pairwise_vectors, osamp_freqs

from matplotlib import colormaps, colors
inferno = colormaps["inferno"]
seismic = colormaps["seismic"]


def compare_mask(cplx, masks, pad):
    ampl = np.abs(cplx)
    mask = masks.sum(0) > 0

    c = ampl.shape[0] // 2
    s = 30 * pad
    cut = slice(c - s, c + s, 1)
    ampl = ampl[cut, cut]
    mask = mask[cut, cut]

    inner = ampl * mask
    outer = ampl * np.abs(mask - 1)

    logged = np.where(np.log10(ampl) == -np.inf, np.nan, np.log10(ampl))
    vmin, vmax = np.nanmin(logged), np.nanmax(logged)

    # plt.figure(figsize=(20, 8))
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(np.log10(inner), vmin=vmin, vmax=vmax)
    plt.colorbar()

    plt.subplot(1, 2, 2)
    plt.imshow(np.log10(outer), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.show()

    inner_mask_min = np.nanmin(np.log10(inner[np.where(mask)]))
    masked_outer = np.where(mask, -np.inf, np.log10(outer))
    nand_outer = np.where(masked_outer == -np.inf, np.nan, masked_outer)

    outer_mask_max = np.nanmax(nand_outer)
    inner_mask_min = np.nanmin(np.log10(inner[np.where(mask)]))
    print(f"Inner min: {inner_mask_min}")
    print(f"Outer max: {outer_mask_max}")
    print(f"Inner - outer: {inner_mask_min - outer_mask_max}")


def plot_params(losses, params_out, format_fn, k=10, l=-1):
    # nparams = len(params_out.keys())
    # nplots = (nparams + 1) // 2

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Full Loss")
    plt.plot(losses)

    if k >= len(losses):
        k = 0
    last_losses = losses[k:l]
    n = len(last_losses)
    plt.subplot(1, 2, 2)
    plt.title(f"Final {n} Losses")
    plt.plot(np.arange(k, k + n), last_losses)

    plt.tight_layout()
    plt.show()

    params = list(params_out.keys())
    for i in np.arange(0, len(params), 2):
        plt.figure(figsize=(16, 5))
        plt.subplot(1, 2, 1)
        plt.title(params[i])
        plt.plot(format_fn(params_out, params[i]))

        plt.subplot(1, 2, 2)
        if i + 1 == len(params):
            plt.tight_layout()
            plt.show()
            break
        plt.title(params[i + 1])
        plt.plot(format_fn(params_out, params[i + 1]))

        plt.tight_layout()
        plt.show()


def plot_radial_residual(model, psf, im, support_mask, rmin=0, rmax=3, n=12):
    # Plot flux and residual with radius
    radii = np.linspace(rmin, rmax, n)
    coords = dlu.pixel_coords(
        model.psf_npixels, model.psf_pixel_scale * model.psf_npixels
    )
    coords += (model.position * np.array([-1, 1]))[:, None, None]
    rs = np.hypot(*coords)

    im_radial_mean, psf_radial_mean, radial_support = [], [], []
    for r in range(len(radii) - 1):
        inner = rs <= radii[r]
        outer = rs <= radii[r + 1]
        annulus = np.logical_xor(inner, outer)

        nan_im = im.at[np.where(~annulus)].set(np.nan)
        nan_psf = psf.at[np.where(~annulus | ~support_mask)].set(np.nan)

        im_radial_mean.append(np.nanmean(nan_im))
        psf_radial_mean.append(np.nanmean(nan_psf))
        radial_support.append(np.sum(~np.isnan(nan_im)))

    im_radial_mean = np.array(im_radial_mean)
    psf_radial_mean = np.array(psf_radial_mean)
    radial_support = np.array(radial_support)

    hex_fringe = dlu.rad2arcsec(4.8e-6 / 0.82)  # Is this right??
    alpha = (radial_support / radial_support.max()) ** 0.5

    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.scatter(radii[1:], im_radial_mean, alpha=alpha, label="Data")
    plt.scatter(radii[1:], psf_radial_mean, alpha=alpha, label="Model")
    plt.axhline(0, c="k", ls="--")
    plt.title("Mean Flux vs Radius")
    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Mean Flux")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.title("Mean Residual vs Radius")
    plt.scatter(radii[1:], im_radial_mean - psf_radial_mean, alpha=alpha, label="Data")
    # plt.scatter(radii[1:], im_radial_mean / psf_radial_mean, alpha=alpha, label="Data")
    plt.axhline(0, c="k", ls="--")
    plt.xlabel("Radius (arcsec)")
    plt.ylabel("Mean Residual")

    plt.tight_layout()
    plt.show()


def visualise_residual(psf, data, log_likeli, k=0.5, pow=0.25):
    # from lib import get_likelihoods

    res = data - psf
    # likeli, neg_loglikeli = get_likelihoods(psf, im, err)

    vmax = np.maximum(np.nanmax(np.abs(data)), np.nanmax(np.abs(psf)))
    vmin = np.minimum(np.nanmin(np.abs(data)), np.nanmin(np.abs(psf)))
    norm = colors.PowerNorm(gamma=pow, vmin=-vmin, vmax=vmax)

    inferno.set_bad("k", k)
    seismic.set_bad("k", k)

    plt.figure(figsize=(15, 8))
    plt.subplot(2, 3, 1)
    plt.title(r"Data $^{}$".format(pow))
    plt.imshow(data, cmap=inferno, norm=norm)
    plt.colorbar()

    plt.subplot(2, 3, 2)
    plt.title(f"Model $^{pow}$")
    plt.imshow(psf, cmap=inferno, norm=norm)
    plt.colorbar()

    v = np.nanmax(np.abs(res))
    plt.subplot(2, 3, 3)
    plt.title("Residual")
    plt.imshow(res, cmap=seismic, vmin=-v, vmax=v)
    plt.colorbar()

    # plt.subplot(2, 3, 4)
    # plt.title("Pixel likelihood")
    # plt.imshow(likeli, cmap=inferno)
    # plt.colorbar()

    # v = 1.0
    # plt.subplot(2, 3, 4)
    # plt.title("Fractional Residual")
    # plt.imshow(res / im, vmin=-v, vmax=v, cmap=seismic)
    # plt.colorbar()

    plt.subplot(2, 3, 4)
    plt.title("Pixel neg log likelihood")
    plt.imshow(-log_likeli, cmap=inferno)
    plt.colorbar()

    # norm_res = (data - psf) / err
    norm_res = (data - psf) / 1  # Not sure how to get 'err' equiv right now
    x = np.nanmax(np.abs(norm_res))
    xs = np.linspace(-x, x, 200)
    ys = jsp.stats.norm.pdf(xs)

    ax = plt.subplot(2, 3, 5)
    ax.set_title("Noise normalised residual hist")
    ax.hist(norm_res.flatten(), bins=50, density=True)

    ax2 = ax.twinx()
    ax2.plot(xs, ys, c="k")
    ax2.set_ylim(0)

    v = np.nanmax(np.abs(norm_res))
    plt.subplot(2, 3, 6)
    plt.title("Noise normalised Residual")
    plt.imshow(norm_res, vmin=-v, vmax=v, cmap=seismic)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# def plot_extra(model):
#     # Get the AMI mask and applied mask
#     applied_mask = model.pupil_mask.gen_AMI(model.wf_npixels, model.diameter)

#     # Get the applied opds in nm and flip to match the mask
#     mirror_opd = np.flipud(model.pupil.opd + model.basis_opd) * 1e9
#     mirror_opd = mirror_opd.at[np.where(~(applied_mask > 1e-6))].set(np.nan)

#     plt.figure(figsize=(15, 8))

#     v = np.nanmax(np.abs(mirror_opd))
#     plt.subplot(2, 3, 1)
#     plt.title("Applied Mask")
#     plt.imshow(mirror_opd, cmap=seismic, vmin=-v, vmax=v)
#     plt.colorbar()

#     plt.subplot(2, 3, 2)
#     plt.title(f"Spectral Weights")
#     plt.plot(model.wavelengths, model.weights, marker="x")

#     plt.subplot(2, 3, 3)
#     # plt.title(r"$\sqrt{\text{IPC Kernel}}$")
#     # plt.imshow(model.IPC.kernel**0.5, cmap=inferno)
#     # plt.colorbar()

#     plt.tight_layout()
#     plt.show()


# def show_likelihoods(model, file, show_res=True, n_mask=1, order=1, k=0.5):
#     im = np.array(file[1].data).astype(float)
#     err = np.array(file[2].data).astype(float)
#     support, support_mask = get_nan_support(file, n_mask=n_mask, order=order)

#     like_px = like_fn(model, im, err, support)  # pixel likelihood 1d
#     loglike_px = -loglike_fn(model, im, err, support)  # pixel likelihood, 1d

#     like_im = np.ones_like(im) * np.nan
#     like_im = like_im.at[support[0], support[1]].set(like_px)
#     loglike_im = like_im.at[support[0], support[1]].set(loglike_px)

#     return like_im, loglike_im, support_mask

# return like_im, loglike_im

# if show_res:
#     psf = model.model().at[~support_mask].set(np.nan)
#     data = im.at[~support_mask].set(np.nan)
#     res = data - psf
#     n = 3
# else:
#     n = 2

# inferno.set_bad("k", k)
# plt.figure(figsize=(n * 5, 4))
# plt.subplot(1, n, 1)
# plt.title("Pixel likelihood")
# plt.imshow(like_im, cmap=inferno)
# plt.colorbar()

# plt.subplot(1, n, 2)
# plt.title("Pixel neg log likelihood")
# plt.imshow(loglike_im, cmap=inferno)
# plt.colorbar()

# if show_res:
#     seismic.set_bad("k", k)
#     v = np.nanmax(np.abs(res))
#     plt.subplot(1, n, 3)
#     plt.title("Residual")
#     plt.imshow(res, cmap=seismic, vmin=-v, vmax=v)
#     plt.colorbar()

# plt.tight_layout()
# plt.show()


def plot_image(fits_file, idx=0):
    def getter(file, k):
        if file[k].data.ndim == 2:
            return file[k].data
        else:
            return file[k].data[idx]

    plt.figure(figsize=(15, 4))

    plt.suptitle(fits_file[1].data.shape)

    plt.subplot(1, 3, 1)
    plt.title("SCI")
    plt.imshow(getter(fits_file, 1))
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("ERR")
    plt.imshow(getter(fits_file, 2))
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("DQ")
    plt.imshow(getter(fits_file, 3))
    plt.colorbar()

    plt.tight_layout()
    plt.show()


def compare(arr1, arr2, cut=0, normres=True, titles=["arr1", "arr2"], k=0.5):
    arr1 = np.array(arr1).astype(float)
    arr2 = np.array(arr2).astype(float)

    if arr1.ndim != 2 or arr2.ndim != 2:
        raise ValueError("Arrays must be 2D")

    if arr1.shape != arr2.shape:
        raise ValueError("Arrays must have same shape")

    c = arr1.shape[0] // 2
    s = c - cut
    cut = slice(c - s, c + s, 1)

    res = arr1 - arr2

    plt.figure(figsize=(15, 4))

    plt.subplot(1, 3, 1)
    plt.title(titles[0])
    plt.imshow(arr1[cut, cut])
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title(titles[1])
    plt.imshow(arr2[cut, cut])
    plt.colorbar()

    v = np.nanmax(np.abs(res[cut, cut]))
    seismic.set_bad("k", k)
    plt.subplot(1, 3, 3)
    plt.title("Residual")
    plt.imshow(res[cut, cut], cmap=seismic, vmin=-v, vmax=v)
    plt.colorbar()

    plt.tight_layout()
    plt.show()


# def plot_vis(holes, model, fim):
def plot_vis(holes, model, fim=None):
    if fim is not None:
        N = len(model.amplitudes)
        # Dont marginalise over pos and flux for now
        small_fim = fim[-2 * N :, -2 * N :]
        stds = np.abs(np.diag(-np.linalg.inv(small_fim))) ** 0.5
        ampl_sig = stds[-2 * N : -N]
        phase_sig = stds[-N:]

    hbls = pairwise_vectors(holes)
    bls_r = np.array(np.hypot(hbls[:, 0], hbls[:, 1]))
    # bls_r = np.concatenate([np.zeros(1), bls_r])  # Add DC term

    plt.figure(figsize=(14, 5))
    plt.suptitle("Visibilities")

    plt.subplot(1, 2, 1)
    if fim is not None:
        plt.title(f"Amplitudes, mean sigma: {100*ampl_sig.mean():.3f}%")
        plt.errorbar(bls_r, model.amplitudes, yerr=ampl_sig, fmt="o", capsize=5)
    else:
        plt.title("Amplitudes")
        plt.scatter(bls_r, model.amplitudes)
    plt.axhline(1, c="k", ls="--")
    plt.ylabel("Amplitudes")
    plt.xlabel("Baseline (m)")

    plt.subplot(1, 2, 2)
    phases = dlu.rad2deg(model.phases)
    if fim is not None:
        plt.title(f"Phases, mean sigma: {100*phase_sig.mean():.3f}%")
        plt.errorbar(bls_r, phases, yerr=phase_sig, fmt="o", capsize=5)
    else:
        plt.title("Phases")
        plt.scatter(bls_r, phases)
    plt.axhline(0, c="k", ls="--")
    plt.ylabel("Phases")
    plt.xlabel("Baseline (m)")

    plt.tight_layout()
    plt.show()


def show_splodges(model, s=65, pupil_phases=False, k=0.5):
    splodges = model.source.splodges

    c = splodges.shape[-1] // 2
    cut = slice(c - s, c + s, 1)

    inferno.set_bad("k", k)
    seismic.set_bad("k", k)

    # Get the coordaintes
    dx = dlu.arcsec2rad(model.psf_pixel_scale) / model.oversample
    wl = model.wavelengths.mean()

    shifted_coords = osamp_freqs(splodges.shape[-1], dx, 1) * wl / 2
    rmin, rmax = shifted_coords[cut].min(), shifted_coords[cut].max()
    extent = [rmin, rmax, rmin, rmax]

    if pupil_phases:
        n = 3
    else:
        n = 2
    plt.figure(figsize=(n * 5, 4))

    plt.subplot(1, n, 1)
    plt.title("Applied Amplitudes")
    plt.imshow(np.abs(splodges).mean(0)[cut, cut], extent=extent)
    plt.colorbar()
    plt.xlabel("meters")
    plt.ylabel("meters")

    plt.subplot(1, n, 2)
    plt.title("Applied Phases")
    plt.imshow(np.angle(splodges).mean(0)[cut, cut], extent=extent, cmap=seismic)
    plt.colorbar(label="radians")
    plt.xlabel("meters")
    plt.ylabel("meters")

    if pupil_phases:
        seismic.set_bad("k", k)
        if hasattr(model.pupil_mask, "transformed"):
            transmisson = np.flipud(model.pupil_mask.transformed.transmission)
        else:
            transmisson = np.flipud(model.pupil_mask.transmission)
        opd = (model.basis_opd * 1e9).at[np.where(~(transmisson > 1e-6))].set(np.nan)
        v = np.nanmax(np.abs(opd))
        plt.subplot(1, n, 3)
        plt.title("Pupil Phases")
        plt.imshow(opd, extent=extent, vmin=-v, vmax=v, cmap=seismic)
        plt.colorbar(label="nm")
        plt.xlabel("meters")
        plt.ylabel("meters")

    plt.tight_layout()
    plt.show()
