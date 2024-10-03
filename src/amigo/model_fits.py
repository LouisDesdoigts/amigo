import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from abc import abstractmethod
from jax import lax, vmap
from .ramp_models import model_ramp
from .misc import planck


class ModelFit(zdx.Base):

    @abstractmethod
    def __call__(self, model, exposure):
        pass

    def get_key(self, exposure, param):

        # TODO: Update to switch statement
        if param in ["amplitudes", "phases"]:
            return "_".join([exposure.star, exposure.filter])

        if param == "dispersion":
            return exposure.filter

        if param == "fluxes":
            return "_".join([exposure.star, exposure.filter])

        if param == "aberrations":
            return "_".join([exposure.program, exposure.filter])

        if param == "reflectivity":
            return "_".join([exposure.program, exposure.filter])

        if param == "positions":
            return exposure.key

        if param == "one_on_fs":
            return exposure.key

        if param == "biases":
            return exposure.key

        if param == "Teffs":
            return exposure.star

        raise ValueError(f"Parameter {param} has no key")

    def map_param(self, exposure, param):
        """
        The `key` argument will return only the _key_ extension of the parameter path,
        which is required for object initialisation.
        """

        # TODO: Update to switch statement
        if param in ["amplitudes", "phases"]:
            return f"{param}.{exposure.get_key(param)}"

        if param in ["shifts", "contrasts"]:
            return f"{param}.{exposure.get_key(param)}"

        if param == "fluxes":
            return f"{param}.{exposure.get_key(param)}"

        if param == "aberrations":
            return f"{param}.{exposure.get_key(param)}"

        if param == "reflectivity":
            return f"{param}.{exposure.get_key(param)}"

        if param == "positions":
            return f"{param}.{exposure.get_key(param)}"

        if param == "one_on_fs":
            return f"{param}.{exposure.get_key(param)}"

        if param == "biases":
            return f"{param}.{exposure.get_key(param)}"

        if param == "Teffs":
            return f"{param}.{exposure.get_key(param)}"

        # Else its global
        return param

    def get_spectra(self, model, exposure):
        wavels, filt_weights = model.filters[exposure.filter]
        weights = filt_weights * planck(wavels, model.Teffs[exposure.star])
        return wavels, weights / weights.sum()

    def update_optics(self, model, exposure):
        optics = model.optics
        if "aberrations" in model.params.keys():
            coefficients = model.aberrations[self.get_key(exposure, "aberrations")]

            # Stop gradient for science targets
            if not exposure.calibrator:
                coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil_mask.abb_coeffs", coefficients)

        if "reflectivity" in model.params.keys():
            coefficients = model.reflectivity[self.get_key(exposure, "reflectivity")]

            # Stop gradient for science targets
            if not exposure.calibrator:
                coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil_mask.amp_coeffs", coefficients)

        return optics

    def model_wfs(self, model, exposure):
        wavels, weights = self.get_spectra(model, exposure)
        optics = self.update_optics(model, exposure)

        pos = dlu.arcsec2rad(model.positions[exposure.key])
        wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)

        # Convert Cartesian to Angular wf
        if wfs.units == "Cartesian":
            wfs = wfs.multiply("pixel_scale", 1 / optics.focal_length)
            wfs = wfs.set(["plane", "units"], ["Focal", "Angular"])
        return wfs

    def model_psf(self, model, exposure):
        wfs = self.model_wfs(model, exposure)

        return dl.PSF(wfs.psf.sum(0), wfs.pixel_scale.mean(0))

    def model_detector(self, psf, model, exposure):
        return eqx.filter_jit(model.detector.apply)(psf)

    def model_ramp(self, psf, model, exposure, to_BFE=False):

        # Get the hyper-parameters for the non-linear model
        flux = 10 ** model.fluxes[self.get_key(exposure, "fluxes")]
        oversample = model.optics.oversample
        bias = model.biases[self.get_key(exposure, "biases")]

        # Need to get the _true_ bias to feed to NN
        zpoint = exposure.ramp[0]  # Includes first group of photons
        dsamp_psf = dlu.downsample(psf.data * flux, oversample, mean=False)
        est_bias = bias + zpoint - model_ramp(dsamp_psf, exposure.ngroups)[0]
        # Note this bis estimate will be poor when the psf is in-accurate

        # Return the BFE and required meta-data
        if to_BFE:
            return psf.data, flux, oversample

        # Non linear model always goes from unit psf, flux, oversample to an 80x80 ramp
        # NOTE: Should be able to remove this if statement and just use SimpleRamp
        if model.ramp is not None:
            # from amigo.ramp_models import PolyBias

            # if isinstance(model.ramp, PolyBias):
            #     ramp = eqx.filter_jit(model.ramp.apply)(
            # psf, flux, est_bias, exposure, oversample
            # )
            # else:
            # ramp = eqx.filter_jit(model.ramp.apply)(psf, flux, exposure, oversample)
            ramp = eqx.filter_jit(model.ramp.apply(psf, flux, exposure, oversample))
        else:
            psf_data = dlu.downsample(psf.data * flux, oversample, mean=False)
            ramp = psf.set("data", model_ramp(psf_data, exposure.ngroups))

        # Re-add the bias to the ramp
        ramp = ramp.set("data", ramp.data + est_bias)

        return ramp

    def model_read(self, ramp, model, exposure):  # , slopes=False):
        # Model the read effects
        if "one_on_fs" in model.params.keys():
            model = model.set("read.one_on_fs", model.one_on_fs[exposure.key])

        # Zero point is added in the ramp modelling
        # # Add the zero point to the ramp
        # zpoint = exposure.ramp[:1]
        # zpoint = np.where(np.isnan(zpoint), 0, zpoint)
        # slopes = np.diff(ramp.data, axis=0)
        # true_ramp = np.concatenate([zpoint, zpoint + np.cumsum(slopes, axis=0)])
        # ramp = ramp.set("data", true_ramp)

        # Apply the read effects
        ramp = model.read.apply(ramp)

        return np.diff(ramp.data, axis=0)


class PointFit(ModelFit):

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        psf = self.model_detector(psf, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)


class SplineVisFit(PointFit):
    joint_fit: bool = eqx.field(static=True)

    def __init__(self, joint_fit=True):
        self.joint_fit = bool(joint_fit)

    def get_key(self, exposure, param):

        # Return the per exposure key if not joint fitting
        if not self.joint_fit:
            if param in ["amplitudes", "phases"]:
                return exposure.key

        return super().get_key(exposure, param)

    def model_vis(self, wfs, model, exposure):

        # Get the visibilities
        amps = model.amplitudes[self.get_key(exposure, "amplitudes")]
        phases = model.phases[self.get_key(exposure, "phases")]
        wavels = model.optics.filters[exposure.filter][0]
        psfs = model.vis_model.model_vis(wfs.psf, wavels, amps, phases, exposure.filter)

        return dl.PSF(psfs.sum(0), wfs.pixel_scale.mean(0))

    def model_psf(self, model, exposure):
        wfs = self.model_wfs(model, exposure)
        psf = self.model_vis(wfs, model, exposure)
        return psf


# class SplineVisFit(PointFit):
#     # uv_pad: int = eqx.field(static=True)
#     # crop_size: int = eqx.field(static=True)
#     joint_fit: bool = eqx.field(static=True)
#     # per_wavelength: bool = eqx.field(static=True)

#     # def __init__(self, uv_pad=2, crop_size=160, joint_fit=True, per_wavelength=True):
#     def __init__(self, joint_fit=True):
#         # self.uv_pad = int(uv_pad)
#         # self.crop_size = int(crop_size)
#         self.joint_fit = bool(joint_fit)
#         # self.per_wavelength = bool(per_wavelength)

#     def get_key(self, exposure, param):

#         # Return the per exposure key if not joint fitting
#         if not self.joint_fit:
#             if param in ["amplitudes", "phases"]:
#                 return exposure.key

#         return super().get_key(exposure, param)

#     def model_vis(self, wfs, model, exposure):
#         vis_pts = build_vis_pts(
#             model.amplitudes[self.get_key(exposure, "amplitudes")],
#             model.phases[self.get_key(exposure, "phases")],
#             model.vis_model.knots[0].shape,
#         )
#         # vis_pts = self.get_vis_pts(model, exposure)
#         psf = model.vis_model.apply_vis(wfs.psf, vis_pts, exposure.filter)
#         return dl.PSF(psf, wfs.pixel_scale.mean(0))

#     def model_psf(self, model, exposure):
#         wfs = self.model_wfs(model, exposure)
#         psf = self.model_vis(wfs, model, exposure)
#         return psf


class BinaryFit(ModelFit):

    # Maybe overwrite this to get the binary spectra
    def get_spectra(self, model, exposure):
        return super().get_spectra(model, exposure)

    def model_wfs(self, model, exposure):
        wavels, weights = self.get_spectra(model, exposure)

        # Update the weights for each binary component
        contrast = 10 ** model.contrasts[self.get_key(exposure, "contrasts")]
        flux_weights = np.array([contrast * 1, 1]) / (1 + contrast)
        weights = flux_weights[:, None] * weights[None, :]

        # Get the binary positions
        position = dlu.arcsec2rad(model.positions[exposure.key])
        pos_angle = dlu.deg2rad(model.position_angles[exposure.star])
        r = model.separations[exposure.star] / 2
        sep_vec = np.array([r * np.sin(pos_angle), r * np.cos(pos_angle)])
        positions = np.array([position + sep_vec, position - sep_vec])
        positions = vmap(dlu.arcsec2rad)(positions)

        # Model the optics - unit weights to apply each flux
        optics = self.update_optics(model, exposure)
        prop_fn = lambda pos: optics.propagate(wavels, pos, return_wf=True)
        wfs = eqx.filter_jit(eqx.filter_vmap(prop_fn))(positions)

        # Return the correctly weighted wfs - needs sqrt because its amplitude not psf
        return wfs * np.sqrt(weights)[..., None, None]

    def __call__(self, model, exposure):
        wfs = self.model_wfs(model, exposure)
        psf = dl.PSF(wfs.psf.sum(0, 1), wfs.pixel_scale.mean((0, 1)))
        psf = self.model_detector(wfs, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)
