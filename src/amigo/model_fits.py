import jax
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from jax import lax, vmap
from .misc import find_position, planck, gen_surface
from .ramp_models import Ramp
from .latent_ode_models import GainDiffusionRamp
from .optical_models import gen_powers

# from .core_models import Exposure


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be
    modelled.

    """

    slopes: jax.Array
    # variance: jax.Array
    cov: jax.Array
    ramp: jax.Array
    # ramp_variance: jax.Array
    ramp_cov: jax.Array
    support: jax.Array
    badpix: jax.Array
    slope_support: jax.Array
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
        # self.variance = np.array(file["SLOPE_ERR"].data, float) ** 2
        self.cov = np.array(file["SLOPE_ERR"].data, float)
        self.badpix = np.array(file["BADPIX"].data, bool)
        self.support = np.where(~np.array(file["BADPIX"].data, bool))
        self.ramp = np.asarray(file["RAMP"].data, float)
        # self.ramp_variance = np.asarray(file["RAMP_ERR"].data, float) ** 2
        self.ramp_cov = np.asarray(file["RAMP_ERR"].data, float)
        self.slope_support = np.asarray(file["SLOPE_SUP"].data, int)
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

        # Defocus
        params["defocus"] = self.get_key("defocus"), np.array(0.01)

        # Reflectivity
        if self.fit_reflectivity:
            params["reflectivity"] = (
                self.get_key("reflectivity"),
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
            params["separation"] = (self.get_key("separation"), 0.15)
            params["contrast"] = (self.get_key("contrast"), 2.0)
            params["position_angle"] = (self.get_key("position_angle"), 0.0)

        return params

    @property
    def ngroups(self):
        return len(self.slopes) + 1

    @property
    def nslopes(self):
        return len(self.slopes)

    @property
    def variance(self):
        variance = vmap(np.diag)(self.to_vec(self.cov))
        return vmap(self.from_vec, in_axes=(1), out_axes=(0))(variance)

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
    validator: bool = eqx.field(static=True)

    def __init__(
        self,
        *args,
        fit_reflectivity=False,
        fit_one_on_fs=False,
        fit_bias=False,
        validator=False,
        **kwargs,
    ):
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = fit_reflectivity
        self.fit_bias = fit_bias
        self.validator = bool(validator)
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
            return "_".join([self.program, self.filter])
            # return self.program

        if param == "fluxes":
            return "_".join([self.star, self.filter])

        if param == "biases":
            return self.program

        if param == "Teffs":
            return self.star

        if param == "defocus":
            return self.filter

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
            "defocus",
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

            # Nuke the piston gradient to prevent degeneracy
            fixed_piston = lax.stop_gradient(coefficients[0, 0])
            coefficients = coefficients.at[0, 0].set(fixed_piston)

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

        optics = optics.set("defocus", model.defocus[self.get_key("defocus")])

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
    def model_illuminance(self, psf, model):
        flux = 10 ** model.fluxes[self.get_key("fluxes")]
        psf = eqx.filter_jit(model.detector.apply)(psf)
        return psf.multiply("data", flux)

    def model_ramp(self, illuminance, model, return_bleed=False):

        # # Ensure latent_paths exists if we need it
        # if return_paths:
        #     latent_paths = 0.0

        # Special case of latent ODE models returning paths
        if isinstance(model.detector.ramp, GainDiffusionRamp):
            ramp, latent_paths = model.detector.evolve_ramp(illuminance.data, self.ngroups)
            # ramp, latent_paths = eqx.filter_jit(model.detector.evolve_ramp)(
            #     illuminance.data, self.ngroups
            # )
        else:
            # ramp = model.detector.evolve_ramp(illuminance.data, self.ngroups)
            # ramp = eqx.filter_jit(model.detector.evolve_ramp)(illuminance.data, self.ngroups)
            ramp, bleed = model.detector.evolve_ramp(
                illuminance.data, self.ngroups, self.ramp[0], self.badpix
            )

        # Return the ramp
        ramp = Ramp(ramp, illuminance.pixel_scale)
        if return_bleed:
            return ramp, bleed
        return ramp

    def model_read(self, ramp, model):
        # Re-add the bias to the ramp
        est_bias = self.ramp[0] - ramp.data[0]

        # Update one on fs if we are fitting for it
        if self.fit_one_on_fs:
            model = model.set("read.one_one_fs", model.one_on_fs[self.get_key("one_one_fs")])

        # Update bias value
        if self.fit_bias:
            est_bias += model.biases[self.get_key("biases")]
        model = model.set("pixel_bias.bias", est_bias)

        # Apply the read effects
        return eqx.filter_jit(model.read.apply)(ramp)

    def simulate(self, model, return_bleed=False):
        psf = self.model_psf(model)
        illuminance = self.model_illuminance(psf, model)
        if return_bleed:
            ramp, latent_path = self.model_ramp(illuminance, model, return_bleed=return_bleed)
            return self.model_read(ramp, model), latent_path
        else:
            ramp = self.model_ramp(illuminance, model)
            return self.model_read(ramp, model)

    def __call__(self, model, return_bleed=False, return_ramp=False):
        ramp, bleed = self.simulate(model, return_bleed=return_bleed)

        if return_ramp:
            if return_bleed:
                return ramp.data, bleed
            return ramp.data

        if return_bleed:
            return np.diff(ramp.data, axis=0), bleed
        return np.diff(self.simulate(model).data, axis=0)


class PointFit(ModelFit):
    pass


class FlatFit(ModelFit):
    polynomial_powers: np.ndarray

    def __init__(self, file, fit_one_on_fs=False):
        self.slopes = np.array(file["SLOPE"].data, float)
        # self.variance = np.array(file["SLOPE_ERR"].data, float) ** 2
        self.cov = np.array(file["SLOPE_ERR"].data, float)
        self.badpix = np.array(file["BADPIX"].data, bool)
        self.support = np.where(~np.array(file["BADPIX"].data, bool))
        self.ramp = np.asarray(file["RAMP"].data, float)
        # self.ramp_variance = np.asarray(file["RAMP_ERR"].data, float) ** 2
        self.ramp_cov = np.asarray(file["RAMP_ERR"].data, float)
        self.slope_support = np.asarray(file["SLOPE_SUP"].data, int)
        self.nints = file[0].header["NINTS"]
        self.filter = file[0].header["FILTER"]
        self.star = "NIS_LAMP"
        self.observation = "FLAT"
        self.program = "FLAT"
        self.act_id = file[0].header["ACT_ID"]
        self.visit = file[0].header["VISITGRP"]
        self.dither = file[0].header["EXPOSURE"]
        self.calibrator = False
        self.filename = f"FLAT_{self.filter}"
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = False
        self.fit_bias = False
        self.validator = False

        # Remove the 0, 0 power term since its invariant
        self.polynomial_powers = np.array(gen_powers(2))[:, 1:]

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

        # Log flux
        slope_flux = self.ngroups + (1 / self.ngroups)
        params["fluxes"] = (
            self.get_key("fluxes"),
            np.log10(slope_flux * np.nansum(self.slopes[0])),
        )

        # Polynomial fit coefficients
        params["flat_coeffs"] = (
            self.get_key("flat_coeffs"),
            np.zeros_like(self.polynomial_powers),
        )

        # One on fs
        if self.fit_one_on_fs:
            params["one_on_fs"] = (
                self.get_key("one_on_fs"),
                np.zeros((self.ngroups, 80, one_on_fs_order + 1)),
            )

        return params

    @property
    def key(self):
        return "_".join(["flat", self.filter, str(self.ngroups)])

    # def get_key(self, exposure, param):
    def get_key(self, param):

        # Unique to each exposure
        if param in [
            "fluxes",
            "one_on_fs",
            "flat_coeffs",
        ]:
            return self.key

        raise ValueError(f"Parameter {param} has no key")

    # def map_param(self, exposure, param):
    def map_param(self, param):
        """
        The `key` argument will return only the _key_ extension of the parameter path,
        which is required for object initialisation.
        """

        # Map the appropriate parameter to the correct key
        if param in [
            "fluxes",
            "one_on_fs",
            "flat_coeffs",
        ]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return param

    def model_illuminance(self, model):
        # Get the pixel scale (arcseconds)
        pixel_scale = model.optics.psf_pixel_scale / model.optics.oversample
        npix = model.optics.psf_npixels * model.optics.oversample
        coords = dlu.pixel_coords(npix, pixel_scale)

        # Get the illuminance
        coeffs = model.flat_coeffs[self.get_key("flat_coeffs")]
        flux = 10 ** model.fluxes[self.get_key("fluxes")]
        surface = 1.0 + gen_surface(coords, coeffs, self.polynomial_powers)
        illuminance = flux * (surface / surface.sum())

        # Make the object and return
        return dl.PSF(illuminance, dlu.arcsec2rad(pixel_scale))

    def simulate(self, model, return_bleed=False):
        illuminance = self.model_illuminance(model)
        if return_bleed:
            ramp, bleed = self.model_ramp(illuminance, model, return_bleed=return_bleed)
            return self.model_read(ramp, model), bleed
        else:
            ramp = self.model_ramp(illuminance, model)
            return self.model_read(ramp, model)


class SplineVisFit(PointFit):
    joint_fit: bool = eqx.field(static=True)

    def __init__(self, *args, joint_fit=True, **kwargs):
        self.joint_fit = bool(joint_fit)
        super().__init__(*args, **kwargs)

    def get_key(self, param):

        # Return the per exposure key if not joint fitting
        if not self.joint_fit:
            if param in ["amplitudes", "phases"]:
                return self.key

        return super().get_key(param)

    def model_vis(self, wfs, model):
        # NOTE: Returns a psf

        # Get the visibilities
        amps = model.amplitudes[self.get_key("amplitudes")]
        phases = model.phases[self.get_key("phases")]
        return model.vis_model.model_vis(wfs, amps, phases)

    def model_psf(self, model):
        wfs = self.model_wfs(model)
        psf = self.model_vis(wfs, model)
        return psf


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
