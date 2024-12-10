import jax
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from jax import lax
from .ramp_models import model_ramp
from .misc import find_position, planck

# from .core_models import Exposure


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be
    modelled.

    """

    slopes: jax.Array
    variance: jax.Array
    ramp: jax.Array
    ramp_variance: jax.Array
    support: jax.Array
    badpix: jax.Array
    nints: int = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    star: str = eqx.field(static=True)
    filename: str = eqx.field(static=True)
    program: str = eqx.field(static=True)
    observation: str = eqx.field(static=True)
    act_id: str = eqx.field(static=True)
    visit: str = eqx.field(static=True)
    dither: str = eqx.field(static=True)
    calibrator: bool = eqx.field(static=True)
    # fit: object = eqx.field(static=True)

    def __init__(self, file):  # , fit):
        self.slopes = np.array(file["SLOPE"].data, float)
        self.variance = np.array(file["SLOPE_ERR"].data, float) ** 2
        self.badpix = np.array(file["BADPIX"].data, bool)
        self.support = np.where(~np.array(file["BADPIX"].data, bool))
        self.ramp = np.asarray(file["RAMP"].data, float)
        self.ramp_variance = np.asarray(file["RAMP_ERR"].data, float) ** 2
        self.nints = file[0].header["NINTS"]
        self.filter = file[0].header["FILTER"]
        self.star = file[0].header["TARGPROP"]
        self.observation = file[0].header["OBSERVTN"]
        self.program = file[0].header["PROGRAM"]
        self.act_id = file[0].header["ACT_ID"]
        self.visit = file[0].header["VISITGRP"]
        self.dither = file[0].header["EXPOSURE"]
        self.calibrator = bool(file[0].header["IS_PSF"])
        self.filename = "_".join(file[0].header["FILENAME"].split("_")[:4])
        # self.fit = fit

    def print_summary(self):
        print(
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"Filter {self.filter}\n"
            f"nints {self.nints}\n"
            f"ngroups {len(self.slopes)+1}\n"
        )

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = {}

        im = np.where(self.badpix, np.nan, self.slopes[0])
        psf = np.where(np.isnan(im), 0.0, im)

        # positions
        params["positions"] = (
            self.get_key("positions"),
            find_position(psf, optics.psf_pixel_scale),
        )

        # Log flux
        slope_flux = self.ngroups + (1 / self.ngroups)
        params["fluxes"] = (
            self.get_key("fluxes"),
            np.log10(slope_flux * np.nansum(self.slopes[0])),
        )

        # Aberrations
        params["aberrations"] = (
            self.get_key("aberrations"),
            np.zeros_like(optics.pupil_mask.abb_coeffs),
        )

        # Reflectivity
        if self.fit_reflectivity:
            params["reflectivities"] = (
                self.get_key("reflectivities"),
                np.zeros_like(optics.pupil_mask.amp_coeffs),
            )

        # One on fs
        if self.fit_one_on_fs:
            params["one_on_fs"] = (
                self.get_key("one_on_fs"),
                np.zeros((self.ngroups, 80, one_on_fs_order + 1)),
            )

        # Biases
        if self.fit_bias:
            params["biases"] = (self.get_key("biases"), np.zeros((80, 80)))

        # Visibilities
        if isinstance(self, SplineVisFit):
            if vis_model is None:
                raise ValueError("vis_model must be provided for SplineVisFit")
            n = vis_model.knot_inds.size
            params["amplitudes"] = (self.get_key("amplitudes"), np.ones(n))
            params["phases"] = (self.get_key("phases"), np.zeros(n))

        # Binary parameters
        if isinstance(self, BinaryFit):
            raise NotImplementedError("BinaryFit initialisation not yet implemented")
            params["seperation"] = (self.get_key("seperation"), 0.15)
            params["contrast"] = (self.get_key("contrast"), 2.0)
            params["position_angle"] = (self.get_key("position_angle"), 0.0)

        return params

    # # Simple method to give nice syntax for getting keys
    # def get_key(self, param):
    #     return self.fit.get_key(self, param)

    # def map_param(self, param):
    #     return self.fit.map_param(self, param)

    @property
    def ngroups(self):
        return len(self.slopes) + 1

    @property
    def nslopes(self):
        return len(self.slopes)

    @property
    def std(self):
        return np.sqrt(self.variance)

    @property
    def key(self):
        return "_".join([self.program, self.observation, self.act_id, self.visit, self.dither])

    def to_vec(self, image):
        return image[..., *self.support].T

    def from_vec(self, vec, fill=np.nan):
        return (fill * np.ones((80, 80))).at[*self.support].set(vec)


class ModelFit(Exposure):
    fit_one_on_fs: bool = eqx.field(static=True)
    fit_reflectivity: bool = eqx.field(static=True)
    fit_bias: bool = eqx.field(static=True)

    def __init__(
        self,
        *args,
        fit_reflectivity=False,
        fit_one_on_fs=False,
        fit_bias=False,
        **kwargs,
    ):
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = fit_reflectivity
        self.fit_bias = fit_bias
        super().__init__(*args, **kwargs)

    # def get_key(self, exposure, param):
    def get_key(self, param):

        # Unique to each exposure
        if param in [
            "positions",
            "one_on_fs",
            "contrasts",
            "separations",
            "position_angles",
        ]:
            return self.key

        if param in ["amplitudes", "phases"]:
            return "_".join([self.star, self.filter])

        if param in ["aberrations", "reflectivity"]:
            # return "_".join([self.program, self.filter])
            return self.program

        if param == "fluxes":
            return "_".join([self.star, self.filter])

        if param == "biases":
            return self.program

        if param == "Teffs":
            return self.star

        raise ValueError(f"Parameter {param} has no key")

    # def map_param(self, exposure, param):
    def map_param(self, param):
        """
        The `key` argument will return only the _key_ extension of the parameter path,
        which is required for object initialisation.
        """

        # Map the appropriate parameter to the correct key
        if param in [
            "amplitudes",
            "phases",
            "fluxes",
            "aberrations",
            "reflectivity",
            "positions",
            "one_on_fs",
            "biases",
            "Teffs",
            "contrasts",
            "separations",
            "position_angles",
        ]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return param

    # def get_spectra(self, model, exposure):
    def get_spectra(self, model):
        wavels, filt_weights = model.filters[self.filter]
        weights = filt_weights * planck(wavels, model.Teffs[self.star])
        return wavels, weights / weights.sum()

    # def update_optics(self, model, exposure):
    def update_optics(self, model):
        optics = model.optics
        if "aberrations" in model.params.keys():
            coefficients = model.aberrations[self.get_key("aberrations")]

            # Stop gradient for science targets
            if not self.calibrator:
                coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil_mask.abb_coeffs", coefficients)

        if self.fit_reflectivity:
            coefficients = model.reflectivity[self.get_key("reflectivity")]

            # Stop gradient for science targets
            if not self.calibrator:
                coefficients = lax.stop_gradient(coefficients)
            optics = optics.set("pupil_mask.amp_coeffs", coefficients)

        return optics

    # def model_wfs(self, model, exposure):
    def model_wfs(self, model):
        wavels, weights = self.get_spectra(model)
        optics = self.update_optics(model)

        pos = dlu.arcsec2rad(model.positions[self.key])
        wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)

        # Convert Cartesian to Angular wf
        if wfs.units == "Cartesian":
            wfs = wfs.multiply("pixel_scale", 1 / optics.focal_length)
            wfs = wfs.set(["plane", "units"], ["Focal", "Angular"])
        return wfs

    # def model_psf(self, model, exposure):
    def model_psf(self, model):
        wfs = self.model_wfs(model)
        return dl.PSF(wfs.psf.sum(0), wfs.pixel_scale.mean(0))

    # def model_detector(self, psf, model, exposure):
    def model_detector(self, psf, model):
        return eqx.filter_jit(model.detector.apply)(psf)

    # def model_ramp(self, psf, model, exposure, to_BFE=False, return_paths=False):
    # def model_ramp(self, psf, model, to_BFE=False, return_paths=False):
    def model_ramp(self, psf, model, return_paths=False):

        # Apply the flux scaling to get the illuminance function
        psf = psf.multiply("data", 10 ** model.fluxes[self.get_key("fluxes")])

        # Non linear model always goes from unit psf, flux, oversample to an 80x80 ramp
        if model.ramp is not None:
            ramp, latent_paths = model.ramp.apply(psf, self, return_paths=True)
        else:
            psf_data = dlu.downsample(psf.data, model.optics.oversample, mean=False)
            ramp = psf.set("data", model_ramp(psf_data, self.ngroups))
            latent_paths = None

        # Re-add the bias to the ramp
        est_bias = self.ramp[0] - ramp.data[0]
        if self.fit_bias:
            est_bias += model.biases[self.get_key("biases")]
        ramp = ramp.set("data", ramp.data + est_bias)

        if return_paths:
            return ramp, latent_paths
        return ramp

    # def model_read(self, ramp, model, exposure):  # , slopes=False):
    def model_read(self, ramp, model):
        # Model the read effects
        if self.fit_one_on_fs:
            # if "one_on_fs" in model.params.keys():
            # model = model.set("one_on_fs", self.map_param(
            # "one_one_fs", model.one_on_fs[exposure.key]
            # ))
            model = model.set("read.one_one_fs", model.one_on_fs[self.get_key("one_one_fs")])

        # Apply the read effects
        return model.read.apply(ramp)

    def simulate(self, model, return_paths=False):
        psf = self.model_psf(model)
        psf = self.model_detector(psf, model)
        if return_paths:
            ramp, latent_path = self.model_ramp(psf, model, return_paths=return_paths)
            return self.model_read(ramp, model), latent_path
        else:
            ramp = self.model_ramp(psf, model)
            return self.model_read(ramp, model)

    # def __call__(self, model, exposure, return_paths=False, return_ramp=False):
    def __call__(self, model, return_paths=False, return_ramp=False):
        ramp, latent_path = self.simulate(model, return_paths=True)

        if return_ramp:
            if return_paths:
                return ramp.data, latent_path
            return ramp.data

        if return_paths:
            return np.diff(ramp.data, axis=0), latent_path
        return np.diff(self.simulate(model).data, axis=0)


class PointFit(ModelFit):
    pass
    # def __call__(self, model, exposure, return_paths=False, return_ramp=False):
    #     psf = self.model_psf(model, exposure)
    #     psf = self.model_detector(psf, model, exposure)
    #     if return_paths:
    #         ramp, latent_path = self.model_ramp(psf, model, exposure, return_paths=return_paths)
    #         ramp = self.model_read(ramp, model, exposure)
    #         return np.diff(ramp.data, axis=0), latent_path

    #     ramp = self.model_ramp(psf, model, exposure)
    #     ramp = self.model_read(ramp, model, exposure)
    #     if return_ramp:
    #         return ramp.data
    #     return np.diff(ramp.data, axis=0)


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

    # def model_vis(self, wfs, model, exposure):

    #     # Get the visibilities
    #     amps = model.amplitudes[self.get_key(exposure, "amplitudes")]
    #     phases = model.phases[self.get_key(exposure, "phases")]
    #     wavels = model.optics.filters[exposure.filter][0]
    #     psfs = model.vis_model.model_vis(wfs.psf, wavels, amps, phases, exposure.filter)

    #     return dl.PSF(psfs.sum(0), wfs.pixel_scale.mean(0))
    def model_vis(self, wfs, model, exposure):
        # NOTE: Returns a psf

        # Get the visibilities
        amps = model.amplitudes[self.get_key(exposure, "amplitudes")]
        phases = model.phases[self.get_key(exposure, "phases")]
        return model.vis_model.model_vis(wfs, amps, phases)
        # wavels = model.optics.filters[exposure.filter][0]
        # psfs = model.vis_model.model_vis(wfs, amps, phases)

        # return dl.PSF(psfs.sum(0), wfs.pixel_scale.mean(0))

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
        position = dlu.arcsec2rad(model.positions[self.get_key(exposure, "positions")])
        pos_angle = dlu.deg2rad(model.position_angles[self.get_key(exposure, "position_angles")])
        r = dlu.arcsec2rad(model.separations[self.get_key(exposure, "separations")] / 2)
        sep_vec = np.array([r * np.sin(pos_angle), r * np.cos(pos_angle)])
        positions = np.array([position + sep_vec, position - sep_vec])
        # positions = vmap(dlu.arcsec2rad)(positions)

        # Model the optics - unit weights to apply each flux
        optics = self.update_optics(model, exposure)
        prop_fn = lambda pos: optics.propagate(wavels, pos, return_wf=True)
        wfs = eqx.filter_jit(eqx.filter_vmap(prop_fn))(positions)

        # Return the correctly weighted wfs - needs sqrt because its amplitude not psf
        return wfs * np.sqrt(weights)[..., None, None]

    def model_psf(self, model, exposure):
        wfs = self.model_wfs(model, exposure)
        return dl.PSF(wfs.psf.sum((0, 1)), wfs.pixel_scale.mean((0, 1)))

    # def __call__(self, model, exposure):
    #     # wfs = self.model_wfs(model, exposure)
    #     # psf = dl.PSF(wfs.psf.sum((0, 1)), wfs.pixel_scale.mean((0, 1)))
    #     psf = self.model_psf(model, exposure)
    #     psf = self.model_detector(psf, model, exposure)
    #     ramp = self.model_ramp(psf, model, exposure)
    #     return self.model_read(ramp, model, exposure)

    # def __call__(self, model, exposure, return_paths=False, return_ramp=False):
    #     psf = self.model_psf(model, exposure)
    #     psf = self.model_detector(psf, model, exposure)
    #     if return_paths:
    #         ramp, latent_path = self.model_ramp(psf, model, exposure, return_paths=return_paths)
    #         ramp = self.model_read(ramp, model, exposure)
    #         return np.diff(ramp.data, axis=0), latent_path

    #     ramp = self.model_ramp(psf, model, exposure)
    #     ramp = self.model_read(ramp, model, exposure)
    #     if return_ramp:
    #         return ramp.data
    #     return np.diff(ramp.data, axis=0)
