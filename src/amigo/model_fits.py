import jax
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from jax import lax, vmap

# import pkg_resources as pkg
from importlib import resources
from .misc import find_position, get_pos, gen_surface
from .ramp_models import Ramp
from .optical_models import gen_powers
from .stats import mv_zscore, loglike


class Exposure(zdx.Base):
    """
    A class to hold all the data relevant to a single exposure, allowing it to be
    modelled.

    """

    slopes: jax.Array
    cov: jax.Array
    ramp: jax.Array
    ramp_cov: jax.Array
    support: jax.Array
    badpix: jax.Array
    slope_support: jax.Array
    parang: jax.Array
    calibrator: bool = eqx.field(static=True)

    #
    nints: int = eqx.field(static=True)
    filter: str = eqx.field(static=True)
    star: str = eqx.field(static=True)
    filename: str = eqx.field(static=True)
    program: str = eqx.field(static=True)
    observation: str = eqx.field(static=True)
    act_id: str = eqx.field(static=True)
    visit: str = eqx.field(static=True)
    dither: str = eqx.field(static=True)
    POS: str = eqx.field(static=True)

    def __init__(self, file):
        self.slopes = np.array(file["SLOPE"].data, float)
        self.cov = np.array(file["SLOPE_COV"].data, float)
        self.support = np.where(~np.array(file["BADPIX"].data, bool))
        self.ramp = np.asarray(file["RAMP"].data, float)
        self.ramp_cov = np.asarray(file["RAMP_COV"].data, float)
        self.slope_support = np.asarray(file["SLOPE_SUP"].data, int)
        self.parang = np.array(file[0].header["ROLL_REF"], float)

        # Make sure we have all the baxpixels
        # static_badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy"))
        static_badpix = np.load(resources.files(__package__) / "data" / "badpix.npy")
        pipeline_badpix = np.array(file["BADPIX"].data, bool)
        self.badpix = static_badpix | pipeline_badpix

        #
        self.nints = file[0].header["NINTS"]
        self.filter = file[0].header["FILTER"]
        self.star = file[0].header["TARGPROP"]
        self.observation = file[0].header["OBSERVTN"]
        self.program = file[0].header["PROGRAM"]
        self.act_id = file[0].header["ACT_ID"]
        self.visit = file[0].header["VISITGRP"]
        self.dither = file[0].header["EXPOSURE"]
        self.POS = get_pos(file)
        self.calibrator = bool(file[0].header["IS_PSF"])
        self.filename = "_".join(file[0].header["FILENAME"].split("_")[:4])

    def add_badpix(self, badpix):
        badpix = self.badpix | badpix
        support = np.where(~badpix)
        return self.set(["badpix", "support"], [badpix, support])

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

        # Position
        pos = find_position(psf, optics.psf_pixel_scale)
        pos += np.array([-optics.psf_pixel_scale / 4, 0])  # apply small shift, seems to help

        # Log flux (1.6 is the ~ gain)
        log_flux = np.log10((80**2) * 1.61 * np.nanmean(im))

        # Initialise flat WF
        abb = np.zeros_like(optics.pupil_mask.abb_coeffs)

        # positions
        params["positions"] = (self.get_key("positions"), pos)
        params["fluxes"] = (self.get_key("fluxes"), log_flux)
        params["aberrations"] = (self.get_key("aberrations"), abb)
        params["spectra"] = (self.get_key("spectra"), np.array(0.0))
        params["defocus"] = (self.get_key("defocus"), np.array(0.01))

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
    use_cov: bool = eqx.field(static=True)

    def __init__(
        self,
        *args,
        fit_reflectivity=False,
        fit_one_on_fs=False,
        fit_bias=False,
        validator=False,
        use_cov=False,
        only_diag=False,
        **kwargs,
    ):
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = fit_reflectivity
        self.fit_bias = fit_bias
        self.validator = bool(validator)
        self.use_cov = bool(use_cov)
        super().__init__(*args, **kwargs)

        if not bool(use_cov):
            self.cov = self.cov * np.eye(len(self.slopes))[..., None, None]

        if only_diag:
            n = self.nslopes
            mask = np.eye(n, dtype=bool)
            mask |= np.eye(n, k=1, dtype=bool)
            mask |= np.eye(n, k=-1, dtype=bool)
            self.cov = self.cov * mask[..., None, None]

    def mv_zscore(self, model, return_im=False, return_slopes=False):
        slopes = self(model)

        # Get the model, data, and variances
        slope_vec = self.to_vec(slopes)
        data_vec = self.to_vec(self.slopes)
        cov_vec = self.to_vec(self.cov)

        # Calculate per-pixel z-scores
        z_vec = vmap(mv_zscore)(slope_vec, data_vec, cov_vec)

        # Return image or vector
        if return_im:
            # NOTE: Adds nans to the empty spots
            z_vec = self.from_vec(z_vec)
        if return_slopes:
            return z_vec, slopes
        return z_vec

    def loglike(self, model, return_im=False):
        slopes = self(model)

        # Get the model, data, and variances
        slope_vec = self.to_vec(slopes)
        data_vec = self.to_vec(self.slopes)
        cov_vec = self.to_vec(self.cov)

        # Calculate per-pixel z-scores
        z_vec = vmap(loglike)(slope_vec, data_vec, cov_vec)

        # Return image or vector
        if return_im:
            # NOTE: Adds nans to the empty spots
            return self.from_vec(z_vec)
        return z_vec

    def get_key(self, param):

        # Unique to each exposure
        if param in [
            "positions",
            "one_on_fs",
            "contrasts",
            "separations",
            "position_angles",
            "fluxes",
        ]:
            return self.key

        if param in ["amplitudes", "phases", "vis"]:
            return "_".join([self.star, self.filter])

        # if param in ["aberrations", "reflectivity"]:
        #     return "_".join([self.program, self.filter])
        if param == "aberrations":
            return "_".join([self.program, self.filter])

        # if param in ["reflectivity", "beam_coeffs", "defocus"]:
        #     return self.filter

        if param in ["reflectivity", "defocus"]:
            return self.filter

        # if param == "fluxes":
        #     return "_".join([self.star, self.filter])

        if param == "biases":
            return self.program

        if param == "Teffs":
            return self.star

        if param == "spectra":
            return "_".join([self.star, self.filter])

        # if param == "defocus":
        #     return self.filter

        raise ValueError(f"Parameter {param} has no key")

    def map_param(self, param):
        """
        The `key` argument will return only the _key_ extension of the parameter path,
        which is required for object initialisation.
        """

        # Map the appropriate parameter to the correct key
        if param in [
            "amplitudes",
            "phases",
            "vis",
            "fluxes",
            "aberrations",
            "reflectivity",
            "beam_coeffs",
            "positions",
            "one_on_fs",
            "biases",
            "Teffs",
            "spectra",
            "contrasts",
            "separations",
            "position_angles",
            "defocus",
        ]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return param

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

        if hasattr(model, "reflectivity"):
            coefficients = model.reflectivity[self.get_key("reflectivity")]
            optics = optics.set("pupil_mask.amp_coeffs", coefficients)

        # Set the defocus
        optics = optics.set("defocus", model.defocus[self.get_key("defocus")])

        return optics

    def get_spectra(self, model):
        wavels, filt_weights = model.filters[self.filter]
        xs = np.linspace(-1, 1, len(wavels), endpoint=True)
        spectra_slopes = 1 + model.get(self.map_param("spectra")) * xs
        weights = filt_weights * spectra_slopes
        weights = np.where(weights < 0, 0.0, weights)
        return wavels, weights / weights.sum()

    def model_wfs(self, model):
        pos = dlu.arcsec2rad(model.positions[self.key])
        wavels, weights = self.get_spectra(model)

        optics = self.update_optics(model)
        wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)

        return wfs

    def model_psf(self, model):
        wfs = self.model_wfs(model)
        return dl.PSF(wfs.psf.sum(0), wfs.pixel_scale)

    def model_illuminance(self, psf, model):
        flux = self.ngroups * 10 ** model.fluxes[self.get_key("fluxes")]
        psf = eqx.filter_jit(model.detector.apply)(psf)
        return psf.multiply("data", flux)

    def calc_bias(self, illuminance, model):
        # Get the charge (bias)
        illum_small = dlu.downsample(illuminance.data, 3, mean=False)

        # NOTE: This bias estimate is inadequate becuase it doesnt correctly account
        # for the non-linear component of the gain. This ultimately should be properly
        # calibrated, WITH the gain term using the ramp rather than slope data.
        #
        # TODO: USe quadratic formula to get correct non-linear inversion
        true_bias = model.read.gain * self.ramp[0]
        bias = true_bias - (illum_small / self.ngroups)

        # bias = self.ramp[0] - (illum_small / self.ngroups)
        # bias = model.read.gain * bias

        # Paste badpixels with median
        return np.where(self.badpix, np.median(bias), bias)

    def model_ramp(self, illuminance, model):
        bias = self.calc_bias(illuminance, model)

        # Evolve the illuminance
        ramp = model.ramp_model.evolve_illuminance(illuminance.data, bias, self.ngroups)
        return Ramp(ramp, illuminance.pixel_scale)

    def model_read(self, ramp, model):
        # Update one on fs if we are fitting for it
        if self.fit_one_on_fs:
            model = model.set("read.one_on_fs", model.one_on_fs[self.get_key("one_on_fs")])

        # Update bias value
        if self.fit_bias:
            model = model.set("pixel_bias.bias", model.biases[self.get_key("biases")])

        # Apply the read effects
        return model.read(ramp, return_psf=True)

    def nuke_pixel_grads(self, model):
        FF = lax.stop_gradient(model.FF)
        non_linearity = lax.stop_gradient(model.non_linearity)
        return model.set(["FF", "non_linearity"], [FF, non_linearity])

    def nuke_dark_grads(self, model):
        dark_current = lax.stop_gradient(model.dark_current)
        return model.set("dark_current", dark_current)

    def simulate(self, model, return_slopes=True):
        model = self.nuke_pixel_grads(model)
        model = self.nuke_dark_grads(model)
        psf = self.model_psf(model)
        illuminance = self.model_illuminance(psf, model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp

    def __call__(self, model, return_slopes=True):
        return self.simulate(model, return_slopes=return_slopes).data


class PointFit(ModelFit):
    pass


class SplineVisFit(PointFit):
    joint_fit: bool = eqx.field(static=True)

    def __init__(self, *args, joint_fit=False, **kwargs):
        self.joint_fit = bool(joint_fit)
        super().__init__(*args, **kwargs)

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = super().initialise_params(
            optics, vis_model=vis_model, one_on_fs_order=one_on_fs_order
        )
        if vis_model is None:
            raise ValueError("vis_model must be provided for SplineVisFit")
        n = vis_model.n_basis
        params["amplitudes"] = (self.get_key("amplitudes"), np.zeros(n))
        params["phases"] = (self.get_key("phases"), np.zeros(n))
        return params

    def get_key(self, param):

        # Return the per exposure key if not joint fitting
        if not self.joint_fit:
            if param in ["amplitudes", "phases", "vis"]:
                return self.key

        return super().get_key(param)

    def model_vis(self, wfs, model):
        amps = model.amplitudes[self.get_key("amplitudes")]
        phases = model.phases[self.get_key("phases")]
        return model.vis_model.model_vis(wfs, amps, phases, self.filter)

    def model_psf(self, model):
        wfs = self.model_wfs(model)
        psf = self.model_vis(wfs, model)
        return psf


class FlatFit(ModelFit):
    polynomial_powers: np.ndarray

    def __init__(self, file, fit_one_on_fs=False, **kwargs):
        file[0].header["IS_PSF"] = False

        super().__init__(file, **kwargs)
        self.star = "NIS_LAMP"
        self.observation = "FLAT"
        self.program = "FLAT"
        self.POS = "N/A"
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

        im = np.where(self.badpix, np.nan, self.slopes[0])
        # psf = np.where(np.isnan(im), 0.0, im)

        # Log flux
        params["fluxes"] = (
            self.get_key("fluxes"),
            np.log10((80**2) * 1.61 * np.nanmean(im)),
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
        # flux = 10 ** model.fluxes[self.get_key("fluxes")]
        flux = self.ngroups * 10 ** model.fluxes[self.get_key("fluxes")]
        surface = 1.0 + gen_surface(coords, coeffs, self.polynomial_powers)
        illuminance = flux * (surface / surface.sum())

        # Make the object and return
        return dl.PSF(illuminance, dlu.arcsec2rad(pixel_scale))

    def simulate(self, model, return_slopes=False):
        model = self.nuke_dark_grads(model)
        illuminance = self.model_illuminance(model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp


class DarkFit(ModelFit):

    def __init__(self, file, fit_one_on_fs=False, **kwargs):
        file[0].header["IS_PSF"] = False

        super().__init__(file, **kwargs)
        self.star = "NIS_DARK"
        self.observation = "DARK"
        self.program = "DARK"
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = False
        self.fit_bias = False
        self.validator = False

    def print_summary(self):
        print(
            f"File {self.key}\n"
            f"Star {self.star}\n"
            f"nints {self.nints}\n"
            f"ngroups {len(self.slopes)+1}\n"
        )

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = {}
        return params

    @property
    def key(self):
        return "_".join(["dark", str(self.ngroups)])

    # def get_key(self, param):
    #     if param in ["dark_A"]:
    #         return self.key
    #     return super().get_key(param)

    def model_illuminance(self, model):
        """
        There is no illuminance! Haha!
        """
        # Get the pixel scale (arcseconds)
        pixel_scale = model.optics.psf_pixel_scale / model.optics.oversample
        npix = model.optics.psf_npixels * model.optics.oversample

        # illuminance is just zeros
        illuminance = np.zeros((npix, npix))

        # Make the object and return
        return dl.PSF(illuminance, dlu.arcsec2rad(pixel_scale))

    def model_ramp(self, illuminance, model):
        bias = self.calc_bias(illuminance, model)

        # Evolve the illuminance
        # Don't need to bother with modelling charge bleeding here
        no_bleed = model.ramp_model.set("bleed", False)
        ramp = no_bleed.evolve_illuminance(illuminance.data, bias, self.ngroups)
        return Ramp(ramp, illuminance.pixel_scale)

    def simulate(self, model, return_slopes=False):
        model = self.nuke_pixel_grads(model)
        illuminance = self.model_illuminance(model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp

class BinaryFit(PointFit):

    sub_exps: dict
    unique_params: list

    def __init__(self, file, unique_params=None, calibrator=True):

        super().__init__(file)

        # OVERIDE self.calbrator
        self.calibrator = calibrator

        self.sub_exps = {
            "A": PointFit(file),
            "B": PointFit(file),
        }

        if unique_params is None:
            unique_params = [
                "spectra",
            ]
        self.unique_params = unique_params

    def initialise_params(self, optics, one_on_fs_order=1):
        params = super().initialise_params(optics, one_on_fs_order)
        params["pas"] = (self.get_key("pas"), np.array(0.0))  # degrees
        params["separations"] = (self.get_key("separations"), np.array(0.1))
        params["contrasts"] = (self.get_key("contrasts"), np.array(0.5))
        for param, (key, value) in params.items():
            if param in self.unique_params:
                params[param] = key, np.array(2 * [value])  # one for each source

        return params

    def get_key(self, param):
        if param in ["pas"]:
            return self.star
        if param in ["separations"]:
            return self.star
        if param in ["contrasts"]:
            return "_".join([self.star, self.filter])
        return super().get_key(param)

    def map_param(self, param):
        if param in ["pas", "separations", "contrasts"]:
            return f"{param}.{self.get_key(param)}"
        return super().map_param(param)

    def model_interferogram(self, model):

        mean_pos = model.positions[self.get_key("positions")]
        total_flux = 10 ** model.fluxes[self.get_key("fluxes")]

        pa = model.pas[self.get_key("pas")]  # in degrees, measured from N toward E
        
        separation = model.separations[self.get_key("separations")]
        contrast = model.contrasts[self.get_key("contrasts")]

        # Converting position angle from deg to radians
        # and offsetting by JWST Parallactic Angle (ROLL_REF)
        phi = dlu.deg2rad((90 + pa) - self.parang)      
    
        # separation vector d in radians
        # I believe it's negative cos for x co-ordinate because
        # of the YAxis Flip in the optical model. 
        # TODO Check this
        # d = separation * np.array([-np.cos(phi), np.sin(phi)])  # in radians
        d = separation * np.array([np.cos(phi), np.sin(phi)])  # in radians

        posA = mean_pos - d / 2  # brighter source
        posB = mean_pos + d / 2  # dimmer source

        logfluxA = np.log10(contrast * total_flux)
        logfluxB = np.log10((1 - contrast) * total_flux)

        modelA = model.set(self.map_param("positions"), posA).set(
            self.map_param("fluxes"), logfluxA
        )
        modelB = model.set(self.map_param("positions"), posB).set(
            self.map_param("fluxes"), logfluxB
        )

        # Stack the per-source parameters and vmap the A/B sources through the
        # model instead of looping (positions/fluxes/unique_params are the
        # only things that differ between sources; everything else in
        # `model` -- optics, detector, ramp_model, other exposures' params --
        # stays shared/un-vmapped, matching what the loop version did).
        pos_key = self.map_param("positions")
        flux_key = self.map_param("fluxes")
        positions_AB = np.stack([posA, posB])
        fluxes_AB = np.stack([logfluxA, logfluxB])

        unique_AB = {}
        for param in self.unique_params:
            pA, pB = model.get(self.map_param(param))
            unique_AB[param] = np.stack([pA, pB])

        def single_source(pos, flux, uniques):
            m = model.set(pos_key, pos).set(flux_key, flux)
            for param, val in uniques.items():
                m = m.set(self.map_param(param), val)
            psf = self.model_psf(m)
            return self.model_illuminance(psf, m).data, psf.pixel_scale

        illuminances, pixel_scales = vmap(single_source, in_axes=(0, 0, {k: 0 for k in unique_AB}))(
            positions_AB, fluxes_AB, unique_AB
        )

        illuminance = dl.PSF(illuminances.sum(axis=0), pixel_scales[0])

        return illuminance


    def simulate(self, model, return_slopes: bool = True):
        model = self.nuke_pixel_grads(model)
        model = self.nuke_dark_grads(model)
        illuminance = self.model_interferogram(model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)
        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp

