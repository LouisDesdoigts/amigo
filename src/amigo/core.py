import jax
import jax.numpy as np
import jax.scipy as jsp
import jax.tree_util as jtu
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
from jax.lax import dynamic_slice as lax_slice
from .misc import planck
from .optical_layers import AMIOptics
from .detector_layers import SUB80Ramp
from .stats import posterior


class BaseModeller(zdx.Base):
    params: dict

    def __init__(self, params):
        self.params = params

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def __getitem__(self, key):

        values = {}
        for param, item in self.params.items():
            if isinstance(item, dict) and key in item.keys():
                values[param] = item[key]

        return values


class AmigoModel(BaseModeller):
    Teffs: dict
    filters: dict
    optics: AMIOptics
    detector: SUB80Ramp
    vis_model: None

    def __init__(self, params, optics, detector, Teffs, filters, vis_model=None):
        self.params = params
        self.Teffs = Teffs
        self.filters = filters
        self.optics = optics
        self.detector = detector
        self.vis_model = vis_model

    def model(self, exposure, **kwargs):
        return self.model_exposure(exposure, **kwargs)

    def model_psf(self, pos, wavels, weights):

        wfs = self.optics.propagate(wavels, dlu.arcsec2rad(pos), weights, return_wf=True)

        psfs = wfs.psf
        if self.vis_model is not None:
            psf = self.vis_model(psfs)
        else:
            psf = psfs.sum(0)
        return dl.PSF(psf, wfs.pixel_scale.mean(0))

    def model_detector(self, psf, to_BFE=False):

        for key, layer in self.detector.layers.items():
            if key == "EDM" and to_BFE:
                return psf.data
            psf = layer.apply(psf)
        return psf.data

    def model_exposure(self, exposure, to_BFE=False, slopes=False):
        # Get exposure key
        key = exposure.key

        # Get wavelengths and weights
        wavels, filt_weights = self.filters[exposure.filter]
        weights = filt_weights * planck(wavels, self.Teffs[exposure.star])
        weights = weights / weights.sum()

        position = self.positions[key]
        flux = 10 ** self.fluxes[key]
        aberrations = self.aberrations[key]
        one_on_fs = self.one_on_fs[key]
        # dark_current = self.dark_current
        opd = exposure.opd

        optics = self.optics.set(["pupil.coefficients", "pupil.opd"], [aberrations, opd])

        if "coherence" in self.params.keys():
            coherence = self.coherence[key]
            optics = optics.set("holes.reflectivity", coherence)

        detector = self.detector.set(
            ["EDM.ngroups", "EDM.flux", "EDM.filter", "one_on_fs"],
            [exposure.ngroups, flux, exposure.filter, one_on_fs],
        )  # , dark_current])

        self = self.set(["optics", "detector"], [optics, detector])
        psf = self.model_psf(position, wavels, weights)
        ramp = self.model_detector(psf, to_BFE=to_BFE)

        if to_BFE:
            return ramp

        if slopes:
            return np.diff(ramp, axis=0)

        return ramp

    def __getattr__(self, key):
        if key in self.params:
            return self.params[key]
        for k, val in self.params.items():
            if hasattr(val, key):
                return getattr(val, key)
        if hasattr(self.optics, key):
            return getattr(self.optics, key)
        if hasattr(self.detector, key):
            return getattr(self.detector, key)
        if hasattr(self.vis_model, key):
            return getattr(self.vis_model, key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be
    modelled.

    """

    data: jax.Array
    variance: jax.Array
    zero_point: jax.Array
    support: jax.Array = eqx.field(static=True)
    opd: jax.Array = eqx.field(static=True)
    nints: int = eqx.field(static=True)
    ngroups: int = eqx.field(static=True)
    nslopes: int = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    star: str = eqx.field(static=True)
    key: str = eqx.field(static=True)

    def __init__(self, file, data, variance, support, opd, key_fn):

        self.data = data
        self.variance = variance
        self.support = support
        self.opd = opd
        self.key = key_fn(file)
        self.nints = file[0].header["NINTS"]
        self.ngroups = file[0].header["NGROUPS"]
        self.nslopes = file[0].header["NGROUPS"] - 1
        self.filter = file[0].header["FILTER"]
        self.star = file[0].header["TARGPROP"]
        self.zero_point = np.asarray(file["ZPOINT"].data, float)

    def print_summary(self):
        print(
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"Filter {self.filter}\n"
            f"nints {self.nints}\n"
            f"ngroups {self.ngroups}\n"
        )

    def to_vec(self, image):
        return image[..., *self.support].T

    def from_vec(self, vec, fill=np.nan):
        return (fill * np.ones((80, 80))).at[*self.support].set(vec)

    def summarise_fit(
        self,
        model,
        residuals=False,
        histograms=False,
        flat_field=False,
        up_the_ramp=False,
        up_the_ramp_norm=False,
        full_bias=False,
        aberrations=False,
        pow=0.5,
    ):

        inferno = colormaps["inferno"]
        seismic = colormaps["seismic"]

        # slopes = model_fn(model, self)
        slopes = model.model(self, slopes=True)
        data = self.data

        residual = data - slopes
        # loglike_im = self.loglike_im(slope)

        posterior_im = posterior(model, self, return_im=True)

        # loglike_im = self.log_likelihood(slopes, return_im=True)

        nan_mask = np.where(np.isnan(posterior_im))
        slopes = slopes.at[:, *nan_mask].set(np.nan)
        data = data.at[:, *nan_mask].set(np.nan)

        final_loss = np.nansum(-posterior_im) / np.prod(np.array(data.shape[-2:]))

        norm_res_slope = residual / (self.variance**0.5)
        norm_res_slope = norm_res_slope.at[:, *nan_mask].set(np.nan)

        norm_res_vec = self.to_vec(norm_res_slope)
        norm_res_vec = norm_res_vec[~np.isnan(norm_res_vec)]
        norm_res_vec = norm_res_vec[~np.isinf(norm_res_vec)]

        x = np.nanmax(np.abs(norm_res_vec))
        xs = np.linspace(-x, x, 200)
        ys = jsp.stats.norm.pdf(xs)

        effective_data = data.sum(0)
        effective_psf = slopes.sum(0)
        vmax = np.maximum(np.nanmax(np.abs(effective_data)), np.nanmax(np.abs(effective_psf)))
        vmin = np.minimum(np.nanmin(np.abs(effective_data)), np.nanmin(np.abs(effective_psf)))

        skip = False
        if np.isnan(vmin) or np.isnan(vmax):
            skip = True

        if not skip:
            if residuals:
                norm = colors.PowerNorm(gamma=0.5, vmin=-vmin, vmax=vmax)
                inferno.set_bad("k", 0.5)
                seismic.set_bad("k", 0.5)

                plt.figure(figsize=(15, 4))
                plt.subplot(1, 3, 1)
                plt.title(f"Data $^{str(pow)}$")
                plt.imshow(effective_data, cmap=inferno, norm=norm)
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.title(f"Effective PSF $^{str(pow)}$")
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

            plt.subplot(1, 3, 2)
            plt.title("Flat Field")
            plt.imshow(FF, vmin=0, vmax=2, cmap=seismic)
            plt.colorbar()

            plt.subplot(1, 3, 3)
            plt.title("Flat Field Histogram")
            plt.hist(FF.flatten(), bins=100)
            # plt.xlim(0, 2)
            plt.show()

        if up_the_ramp:
            ncols = 4
            nrows = self.nslopes // ncols
            if self.nslopes % ncols > 0:
                nrows += 1

            plt.figure(figsize=(5 * ncols, 4 * nrows))
            plt.suptitle("Up The Ramp Residuals")

            for i in range(self.nslopes):
                # plt.subplot(4, 4, i + 1)
                plt.subplot(nrows, ncols, i + 1)
                v = np.nanmax(np.abs(residual[i]))
                plt.imshow(residual[i], cmap=seismic, vmin=-v, vmax=v)
                plt.colorbar()
            plt.show()

        if up_the_ramp_norm:
            ncols = 4
            nrows = self.nslopes // ncols
            if self.nslopes % ncols > 0:
                nrows += 1

            plt.figure(figsize=(5 * ncols, 4 * nrows))
            plt.suptitle("Normalised Up The Ramp Residuals")

            for i in range(self.nslopes):
                # plt.subplot(4, 4, i + 1)
                plt.subplot(nrows, ncols, i + 1)
                v = np.nanmax(np.abs(norm_res_slope[i]))
                plt.imshow(norm_res_slope[i], cmap=seismic, vmin=-v, vmax=v)
                plt.colorbar()
            plt.show()

        if full_bias:
            coeffs = model.one_on_fs[self.key]
            nan_mask = 1 + (np.nan * np.isnan(data.sum(0)))
            bias = nan_mask * model.biases[self.key]

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
            optics = model.optics.set("coefficients", model.aberrations[self.key])
            applied_mask = optics.pupil_mask.gen_AMI(optics.wf_npixels, optics.diameter)

            # Get the applied opds in nm and flip to match the mask
            static_opd = np.flipud(self.opd) * 1e9
            added_opd = np.flipud(optics.basis_opd) * 1e9
            static_opd = static_opd.at[np.where(~(applied_mask > 1e-6))].set(np.nan)
            added_opd = added_opd.at[np.where(~(applied_mask > 1e-6))].set(np.nan)
            mirror_opd = static_opd + added_opd

            plt.figure(figsize=(15, 4))

            v = np.nanmax(np.abs(static_opd))
            plt.subplot(1, 3, 1)
            plt.title("Static OPD")
            plt.imshow(static_opd, cmap=seismic, vmin=-v, vmax=v)
            plt.colorbar()

            v = np.nanmax(np.abs(added_opd))
            plt.subplot(1, 3, 2)
            plt.title("Added OPD")
            plt.imshow(added_opd, cmap=seismic, vmin=-v, vmax=v)
            plt.colorbar()

            v = np.nanmax(np.abs(mirror_opd))
            plt.subplot(1, 3, 3)
            plt.title("Total OPD")
            plt.imshow(mirror_opd, cmap=seismic, vmin=-v, vmax=v)
            plt.colorbar()

            plt.tight_layout()
            plt.show()


class ExposureFit(Exposure):
    position: jax.Array
    aberrations: jax.Array
    flux: jax.Array  # Log now
    one_on_fs: jax.Array
    coherence: jax.Array

    def __init__(self, exposure, position, flux, FDA, one_on_fs, coherence):

        self.data = exposure.data
        self.variance = exposure.variance
        self.support = exposure.support
        self.opd = exposure.opd
        self.key = exposure.key
        self.nints = exposure.nints
        self.ngroups = exposure.ngroups
        self.nslopes = exposure.nslopes
        self.filter = exposure.filter
        self.star = exposure.star
        self.zero_point = exposure.zero_point
        self.aberrations = FDA
        self.position = position
        self.flux = flux
        self.one_on_fs = one_on_fs
        self.coherence = coherence


class NNWrapper(zdx.Base):
    values: list
    shapes: list = eqx.field(static=True)
    sizes: list = eqx.field(static=True)
    starts: list = eqx.field(static=True)
    tree_def: None = eqx.field(static=True)

    def __init__(self, network):
        values, tree_def = jtu.tree_flatten(network)

        self.values = np.concatenate([val.flatten() for val in values])
        self.shapes = [v.shape for v in values]
        self.sizes = [v.size for v in values]
        self.starts = [int(i) for i in np.cumsum(np.array([0] + self.sizes))]
        self.tree_def = tree_def

    @property
    def _layers(self):
        leaves = [
            lax_slice(self.values, (start,), (size,)).reshape(shape)
            for start, size, shape in zip(self.starts, self.sizes, self.shapes)
        ]
        return jtu.tree_unflatten(self.tree_def, leaves)

    def __call__(self, x):
        layers = self._layers
        for layer in layers[:-1]:
            x = jax.nn.relu(layer(x))
        return layers[-1](x)


class ModelParams(BaseModeller):

    @property
    def keys(self):
        return list(self.params.keys())

    @property
    def values(self):
        return list(self.params.values())

    def __getattr__(self, key):
        # print("In get attr")
        if key in self.keys:
            return self.params[key]
        for k, val in self.params.items():
            # print(k)
            if hasattr(val, key):
                return getattr(val, key)
        # return self.get(key)
        raise AttributeError(
            f"Attribute {key} not found in params of {self.__class__.__name__} object"
        )

    def replace(self, values):
        # Takes in a super-set class and updates this class with input values
        return self.set("params", dict([(param, getattr(values, param)) for param in self.keys]))

    def from_model(self, values):
        return self.set("params", dict([(param, values.get(param)) for param in self.keys]))

    def __add__(self, values):
        matched = self.replace(values)
        return jax.tree_map(lambda x, y: x + y, self, matched)

    def __iadd__(self, values):
        return self.__add__(values)

    def inject(self, other):
        # Injects the values of this class into another class
        return other.set(self.keys, self.values)


def _is_tree(x):
    """
    Here we check if the leaf is a leaf, or a tree. If it is a tree, we tree_map the
    operation around the leaves of that tree. We use the eqx.is_array_like to check if
    the leaf is a tree, but this could also be done with
    `isinstance(leaf, (list, dict, tuple, eqx.Module))`. The differences between these
    two methods needs to be investigated.
    """
    # return isinstance(x, (list, dict, tuple, eqx.Module))
    return not eqx.is_array_like(x)


class ModelHistory(ModelParams):
    """
    Tracks the history of a set of parameters in a model via tuples.

    Adds a series of convenience functions to interface with it.

    This could have issues with leaves not being jax.Arrays, so at some point it should be
    explicitly enforced that only array_likes are tracked.
    """

    def __init__(self, model, tracked):

        history = {}
        for param in tracked:
            leaf = model.get(param)
            if _is_tree(leaf):
                history[param] = jtu.tree_map(lambda sub_leaf: [sub_leaf], leaf)
            else:
                history[param] = [leaf]

        self.params = history

    def append(self, model):
        history = self.params
        for param, leaf_history in history.items():
            if hasattr(model, param):
                new_leaf = getattr(model, param)
            else:
                new_leaf = model.get(param)

            # Tree-like case
            if _is_tree(new_leaf):
                append_fn = lambda history, value: history + [value]
                leaf_fn = lambda leaf: isinstance(leaf, list)
                new_leaf_history = jtu.tree_map(append_fn, leaf_history, new_leaf, is_leaf=leaf_fn)
                history[param] = new_leaf_history

            # Non-tree case
            else:
                history[param] = leaf_history + [new_leaf]
        return self.set("params", history)


class AmigoHistory(ModelHistory):
    """
    Adds plotting and convenience functions

    TODO: Get ith
    TODO: Get exposure vals
    TODO: Get exposure fit?
    """

    def plot(self, exposures=None, key_fn=None, ignore=[], start=0, end=-1):

        params = list(self.params.keys())
        params_in = [param for param in params if param not in ignore]

        # Plot in groups of two
        for i in np.arange(0, len(params_in), 2):
            plt.figure(figsize=(16, 5))
            ax = plt.subplot(1, 2, 1)

            param = params_in[i]
            leaf = self.params[param]
            self._plot_ax(leaf, ax, param, exposures, key_fn, start=start, end=end)

            ax = plt.subplot(1, 2, 2)
            if i + 1 == len(params_in):
                plt.tight_layout()
                plt.show()
                break

            param = params_in[i + 1]
            leaf = self.params[param]
            self._plot_ax(leaf, ax, param, exposures, key_fn, start=start, end=end)

            plt.tight_layout()
            plt.show()

    def _format_leaf(self, leaf, per_exp=False, keys=None):
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
        values = [self._format_leaf(leaf[key]) for key in keys]

        if per_exp:
            return values
        return np.concatenate(values, axis=-1)

    def _plot_ax(self, leaf, ax, param, exposures=None, key_fn=lambda x: x.key, start=0, end=-1):

        if exposures is not None:
            keys = [exp.key for exp in exposures]
            labels = [key_fn(exp) for exp in exposures]
        else:
            keys = None

        if isinstance(leaf, dict):
            values = self._format_leaf(leaf, per_exp=True, keys=keys)

            if keys is None:
                keys = list(leaf.keys())
                labels = keys

            colors, linestyles = self._get_styles(len(keys))

            for val, c, ls, label in zip(values, colors, linestyles, labels):
                kwargs = {"c": c, "ls": ls}
                self._plot_param(ax, val, param, start=start, end=end, **kwargs)
                ax.plot([], label=label, **kwargs)

            plt.legend()

        else:
            arr = self._format_leaf(leaf)
            self._plot_param(ax, arr, param, start=start, end=end)

    def _get_styles(self, n):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyles = ["-", "--", "-.", ":"]

        color_list = [colors[i % len(colors)] for i in range(n)]
        linestyle_list = [linestyles[i // len(colors)] for i in range(n)]

        return color_list, linestyle_list

    def _plot_param(self, ax, arr, param, start=0, end=-1, **kwargs):
        """This is the ugly gross function that is necessary"""
        # print(start, end)
        arr = arr[start:end]
        epochs = np.arange(len(arr))
        ax.set(xlabel="Epochs", title=param)  # , xlim=(start, epochs[end]))

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

            case "BFE":
                pass
                # norm_bfe_lin = arr - arr[0]
                # ax.plot(epochs, norm_bfe_lin, alpha=0.2, **kwargs)
                # ax.set(ylabel="$\Delta$ Linear Coefficients")

            case "BFE.linear":
                norm_bfe_lin = arr - arr[0]
                ax.plot(epochs, norm_bfe_lin, alpha=0.2, **kwargs)
                ax.set(ylabel="$\Delta$ Linear Coefficients")

            case "BFE.quadratic":
                norm_bfe_quad = arr - arr[0]
                ax.plot(epochs, norm_bfe_quad, alpha=0.1, **kwargs)
                ax.set(ylabel="$\Delta$ BFE Quadratic Coefficients")

            case "BFE.coeffs":
                norm_bfe = arr - arr[0]
                ax.plot(epochs, norm_bfe, alpha=0.5, **kwargs)
                ax.set(ylabel="$\Delta$ Linear Coefficients")

            case "SRF":
                srf = arr - arr[0]
                ax.plot(epochs, srf, **kwargs)
                ax.set(ylabel="SRF")

            case "pupil_mask.holes":
                arr *= 1e3
                norm_holes = arr - arr[0]
                ax.plot(epochs, norm_holes, **kwargs)
                ax.set(ylabel="$\Delta$ Pupil Mask Holes (mm)")

            case "pupil_mask.f2f":
                arr *= 1e2
                ax.plot(epochs, arr, **kwargs)
                ax.set(ylabel="Pupil Mask f2f (cm)")

            case "biases":
                norm_bias = arr - arr[0]
                ax.plot(epochs, norm_bias, alpha=0.25, **kwargs)
                ax.set(ylabel="$\Delta$ Bias")

            case "rotation":
                arr = dlu.rad2deg(arr)
                norm_rot = arr
                ax.plot(epochs, norm_rot, **kwargs)
                ax.set(ylabel="Rotation (deg)")

            case "amplitudes":
                norm_amplitudes = arr
                ax.plot(epochs, norm_amplitudes, **kwargs)
                ax.set(ylabel="Visibility Amplitude")

            case "phases":
                arr = dlu.rad2deg(arr)
                norm_phases = arr
                ax.plot(epochs, norm_phases, **kwargs)
                ax.set(ylabel="Visibility Phase (deg)")

            case _:
                print(f"No formatting function for {param}")
                ax.plot(epochs, arr, **kwargs)
