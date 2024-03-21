import equinox as eqx
import dLuxWebbpsf as dlw
from jax import Array
import zodiax as zdx
import jax.numpy as np
import jax.tree_util as jtu
from jax import vmap
import dLux.utils as dlu
import dLux as dl
from jax.scipy.stats import multivariate_normal as mvn
from .detector_layers import Rotate, ApplySensitivities
from .optical_layers import DynamicAMI
from .files import prep_data, get_wss_ops
import pkg_resources as pkg
import matplotlib.pyplot as plt
from matplotlib import colormaps, colors
import jax.scipy as jsp


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be 
    modelled.

    """
    data: Array
    covariance: Array
    bias: Array
    bias_var: Array
    support: Array = eqx.field(static=True)
    pipeline_bias: Array = eqx.field(static=True)
    opd: Array = eqx.field(static=True)
    nints: int = eqx.field(static=True)
    ngroups: int = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    star: str = eqx.field(static=True)
    key: str = eqx.field(static=True)

    def __init__(self, file, add_read_noise=False, opd=None, key_fn=None):

        if key_fn is None:
            key_fn = lambda file: "_".join(file[0].header["FILENAME"].split("_")[:3])

        if opd is None:
            opd = get_wss_ops([file])[0]

        if add_read_noise:
            file_path = pkg.resource_filename(__name__, "data/SUB80_readnoise.npy")
            read_noise = np.load(file_path)
        else:
            read_noise = None
        data, covariance, support = prep_data(file, read_noise=read_noise)
        # bias = np.asarray(file['BIAS'].data, float)
        # data += bias

        self.nints = file[0].header["NINTS"]
        self.ngroups = file[0].header["NGROUPS"]
        self.filter = file[0].header["FILTER"]
        self.star = file[0].header["TARGPROP"]
        self.data = data
        self.covariance = covariance
        self.bias = np.zeros(data.shape[-2:]) # Dont add bias to data, fit it relative
        self.bias_var = np.asarray(file["BIAS_VAR"].data, float)
        self.pipeline_bias = np.asarray(file["BIAS"].data, float)
        self.support = np.array(support)
        self.key = key_fn(file)
        self.opd = opd
    
    def print_summary(self):
        print(
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"Filter {self.filter}\n"
            f"nints {self.nints}\n"
            f"ngroups {self.ngroups}\n"
        )

    # TODO: Probs dont need this anymore
    @property
    def nims(self):
        return self.nints * self.ngroups

    # TODO: This should return the leading dimension as the pixels, using np.swap_axes
    # to make the leading dimension the pixels, and vmap loss along first dimension
    def to_vec(self, image):
        return image[..., *self.support]

    def loglike_vec(self, ramp):
        # Error is _standard error of the mean_, so we dont need to multiply by nints
        ramp_vec = self.to_vec(ramp)
        data_vec = self.to_vec(self.data)
        cov_vec = self.to_vec(self.covariance)
        return vmap(mvn.logpdf, (-1, -1, -1))(ramp_vec, data_vec, cov_vec)

    def loglike_im(self, ramp):
        loglike_vec = self.loglike_vec(ramp)
        return (np.nan * np.ones_like(ramp[0])).at[*self.support].set(loglike_vec)

    def summarise_fit(
        self,
        model,
        model_fn,
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

        # for exp in exposures:
        # self.print_summary()

        ramp = model_fn(model, self)
        data = self.data

        residual = data - ramp
        loglike_im = self.loglike_im(ramp)
        
        nan_mask = np.where(np.isnan(loglike_im))
        ramp = ramp.at[:, *nan_mask].set(np.nan)
        data = data.at[:, *nan_mask].set(np.nan)

        final_loss = np.nansum(-loglike_im) / np.prod(np.array(data.shape[-2:]))

        err = vmap(vmap(np.diag, -1, -1), -1, -1)(self.covariance) ** 0.5
        norm_res_ramp = residual / err
        norm_res_ramp = norm_res_ramp.at[:, *nan_mask].set(np.nan)

        norm_res_vec = self.to_vec(norm_res_ramp)
        norm_res_vec = norm_res_vec[~np.isnan(norm_res_vec)]
        norm_res_vec = norm_res_vec[~np.isinf(norm_res_vec)]

        x = np.nanmax(np.abs(norm_res_vec))
        xs = np.linspace(-x, x, 200)
        ys = jsp.stats.norm.pdf(xs)

        vmax = np.maximum(np.nanmax(np.abs(data)), np.nanmax(np.abs(ramp)))
        vmin = np.minimum(np.nanmin(np.abs(data)), np.nanmin(np.abs(ramp)))

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
                plt.title(r"Mean Data Ramp $^{}$".format(pow))
                plt.imshow(data.mean(0), cmap=inferno, norm=norm)
                plt.colorbar()

                plt.subplot(1, 3, 2)
                plt.title(f"Mean Model Ramp $^{pow}$")
                plt.imshow(ramp.mean(0), cmap=inferno, norm=norm)
                plt.colorbar()

                plt.subplot(1, 3, 3)
                plt.title(f"Pixel neg log likelihood: {final_loss:,.1f}")
                plt.imshow(-loglike_im, cmap=inferno)
                plt.colorbar()

                plt.tight_layout()
                plt.show()

            if histograms:

                plt.figure(figsize=(15, 4))
                ax = plt.subplot(1, 3, 1)
                ax.set_title(
                    f"Noise normalised residual sigma: {norm_res_vec.std():.3}"
                )
                ax.hist(norm_res_vec.flatten(), bins=50, density=True)

                ax2 = ax.twinx()
                ax2.plot(xs, ys, c="k")
                ax2.set_ylim(0)

                ax = plt.subplot(1, 3, 2)
                ax.set_title(
                    f"Noise normalised residual sigma: {norm_res_vec.std():.3}"
                )
                bins = ax.hist(norm_res_vec.flatten(), bins=50)[0]
                ax.semilogy()

                # ax2 = ax.twinx()
                # ax2.plot(xs, bins.max() * ys, c="k")
                # ax2.semilogy()

                v = np.nanmax(np.abs(norm_res_ramp.mean(0)))
                plt.subplot(1, 3, 3)
                plt.title("Mean noise normalised ramp residual")
                plt.imshow(norm_res_ramp.mean(0), vmin=-v, vmax=v, cmap=seismic)
                plt.colorbar()

                plt.tight_layout()
                plt.show()

        if flat_field:
            plt.figure(figsize=(15, 4))

            plt.subplot(1, 3, 1)
            plt.title("Mean Pixel Response Function")
            v = np.max(np.abs(model.detector.sensitivity.SRF - 1))
            plt.imshow(
                model.detector.sensitivity.SRF, vmin=1 - v, vmax=1 + v, cmap=seismic
            )
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
            plt.figure(figsize=(20, 16))
            plt.suptitle("Up The Ramp Residuals")

            for i in range(self.ngroups):
                plt.subplot(4, 4, i + 1)
                v = np.nanmax(np.abs(residual[i]))
                plt.imshow(residual[i], cmap=seismic, vmin=-v, vmax=v)
                plt.colorbar()
            plt.show()

        if up_the_ramp_norm:
            plt.figure(figsize=(20, 16))
            plt.suptitle("Up The Ramp Residuals")

            for i in range(self.ngroups):
                plt.subplot(4, 4, i + 1)
                v = np.nanmax(np.abs(norm_res_ramp[i]))
                plt.imshow(norm_res_ramp[i], cmap=seismic, vmin=-v, vmax=v)
                plt.colorbar()
            plt.show()

        if full_bias:
            coeffs = model.OneOnFs[self.key]
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



class AMIOptics(dl.optical_systems.AngularOpticalSystem):
    def __init__(
        self,
        radial_orders=4,
        pupil_mask=None,
        opd=None,
        normalise=True,
        psf_npixels=80,
        oversample=4,
        pixel_scale = 0.065524085,
        diameter = 6.603464,
        wf_npixels = 1024,
    ):
        self.wf_npixels = wf_npixels
        self.diameter = diameter
        self.psf_npixels = psf_npixels
        self.oversample = oversample
        self.psf_pixel_scale = pixel_scale

        # Get the primary mirror transmission
        file_path = pkg.resource_filename(__name__, 'data/primary.npy')
        transmission = np.load(file_path)
        # Create the primary
        primary = dlw.JWSTAberratedPrimary(
            transmission,
            opd=np.zeros_like(transmission),
            radial_orders=np.arange(radial_orders),
            AMI=True,
        )

        # Load the values into the primary
        n_fda = primary.basis.shape[1]
        file_path = pkg.resource_filename(__name__, 'data/FDA_coeffs.npy')
        primary = primary.set("coefficients", np.load(file_path)[:, :n_fda])

        if opd is None:
            opd = np.zeros_like(transmission)
        primary = primary.set("opd", opd)
        primary = primary.multiply("basis", 1e-9)  # Normalise to nm
        # primary = primary.multiply("basis", 1e-6)  # Normalise to um
        # primary = primary.multiply("basis", 1e-3)  # Normalise to mm

        if pupil_mask is None:
            pupil_mask = DynamicAMI(f2f=0.80, normalise=normalise)

        # Set the layers
        self.layers = dlu.list2dictionary(
            [
                ("pupil", primary),
                ("InvertY", dl.Flip(0)),
                ("pupil_mask", pupil_mask),
            ],
            ordered=True,
        )



class SUB80Ramp(dl.detectors.LayeredDetector):
    def __init__(
        self,
        angle=-0.56126717,
        oversample=4,
        SRF=None,
        FF=None,
        downsample=False,
        npixels_in=80,

    ):
        # Load the FF
        if FF is None:
            file_path = pkg.resource_filename(__name__, "data/SUB80_flatfield.npy")
            FF = np.load(file_path)
            if npixels_in != 80:
                pad = (npixels_in - 80) // 2
                FF = np.pad(FF, pad, constant_values=1)
        
        if SRF is None:
            SRF = np.ones((oversample, oversample))

        layers = [
            ("rotate", Rotate(angle)),
            ("sensitivity", ApplySensitivities(FF, SRF)),
        ]

        if downsample:
            layers.append(("downsample", dl.Downsample(oversample)))

        self.layers = dlu.list2dictionary(layers, ordered=True)




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


class ModelHistory(BaseModeller):
    """
    Tracks the history of a set of parameters in a model via tuples.

    Adds a series of convenience functions to interface with it.

    This could have issues with leaves not being Arrays, so at some point it should be
    explicitly enforced that only array_likes are tracked.
    """

    def __init__(self, model, tracked):

        history = {}
        for param in tracked:
            leaf = model.get(param)
            if _is_tree(leaf):
                history[param] = jtu.tree_map(lambda sub_leaf: (sub_leaf,), leaf)
            else:
                history[param] = (leaf,)

        self.params = history

    def append(self, model):

        history = self.params
        for param, leaf_history in history.items():
            new_leaf = model.get(param)

            # Tree-like case
            if _is_tree(new_leaf):
                append_fn = lambda history, value: history + (value,)
                leaf_fn = lambda leaf: isinstance(leaf, tuple)
                new_leaf_history = jtu.tree_map(
                    append_fn, leaf_history, new_leaf, is_leaf=leaf_fn
                )
                history[param] = new_leaf_history

            # Non-tree case
            else:
                history[param] = leaf_history + (new_leaf,)
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
            self._plot_ax(leaf, ax, param, exposures, key_fn, start, end)

            ax = plt.subplot(1, 2, 2)
            if i + 1 == len(params_in):
                plt.tight_layout()
                plt.show()
                break

            param = params_in[i + 1]
            leaf = self.params[param]
            self._plot_ax(leaf, ax, param, exposures, key_fn)

            plt.tight_layout()
            plt.show()

    def _format_leaf(self, leaf, per_exp=False, keys=None):
        """
        Takes in a tuple, or a dictionary of a tuples of parameters and returns a 2D
        array of values for plotting.
        """

        if isinstance(leaf, tuple):
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
                kwargs = {'c':c, 'ls':ls}
                self._plot_param(ax, val, param, kwargs=kwargs)
                ax.plot([], label=label, **kwargs)

            plt.legend()

        else:
            arr = self._format_leaf(leaf)
            self._plot_param(ax, arr, param)


    def _get_styles(self, n):
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        linestyles = ["-", "--", "-.", ":"]

        color_list = [colors[i % len(colors)] for i in range(n)]
        linestyle_list = [linestyles[i // len(colors)] for i in range(n)]

        return color_list, linestyle_list

    def _plot_param(self, ax, arr, param, kwargs={}):
        """This is the ugly gross function that is necessary"""
        ax.set(xlabel="Epochs", title=param)
        epochs = np.arange(len(arr))

        match param:
            case "positions":
                ax.plot(epochs, arr - arr[0], **kwargs)
                ax.set(ylabel="$\Delta$ Position (arcsec)")

            case "fluxes":
                ax.plot(epochs, 100 * (1 - arr / arr[0]), **kwargs)
                ax.set(ylabel="$\Delta$ Flux (%)")

            case "aberrations":
                ax.plot(epochs, arr - arr[0], alpha=0.4, **kwargs)
                ax.set(ylabel="$\Delta$ Aberrations (nm)")

            case "OneOnFs":
                ax.plot(epochs, arr - arr[0], alpha=0.25, **kwargs)
                ax.set(ylabel="$\Delta$ OneOnFs")

            case "BFE.linear":
                ax.plot(epochs, arr - arr[0], alpha=0.2, **kwargs)
                ax.set(ylabel="$\Delta$ Linear Coefficients")

            case "BFE.quadratic":
                ax.plot(epochs, arr - arr[0], alpha=0.1, **kwargs)
                ax.set(ylabel="$\Delta$ BFE Quadratic Coefficients")

            case "SRF":
                ax.plot(epochs, arr, **kwargs)
                ax.set(ylabel="SRF")

            case "pupil_mask.holes":
                arr *= 1e3
                ax.plot(epochs, arr - arr[0], **kwargs)
                ax.set(ylabel="$\Delta$ Pupil Mask Holes (mm)")

            case "pupil_mask.f2f":
                arr *= 1e2
                ax.plot(epochs, arr, **kwargs)
                ax.set(ylabel="Pupil Mask f2f (cm)")

            case "bias":
                ax.plot(epochs, arr - arr[0], alpha=0.25, **kwargs)
                ax.set(ylabel="$\Delta$ Bias")

            case "rotation":
                ax.plot(epochs, dlu.rad2deg(arr), **kwargs)
                ax.set(ylabel="Rotation (deg)")

            case "amplitudes":
                ax.plot(epochs, arr, **kwargs)
                ax.set(ylabel="Visibility Amplitude")

            case "phases":
                ax.plot(epochs, dlu.rad2deg(arr), **kwargs)
                ax.set(ylabel="Visibility Phase (deg)")

            case _:
                print(f"No formatting function for {param}")
                ax.plot(epochs, arr, **kwargs)