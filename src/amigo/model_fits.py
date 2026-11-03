import jax
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from jax import lax, vmap

# import pkg_resources as pkg
from importlib import resources
from .misc import find_position, gen_surface, spectral_weights
from .ramp_models import Ramp
from .optical_models import gen_powers
from .opd import apply_opds
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
    detector_to_sky_arcsec: tuple | None = eqx.field(default=None, kw_only=True, static=True)

    def __init__(self, file):
        # Display geometry only: retain the FITS tangent-plane Jacobian, including
        # parity. Do not substitute ROLL_REF (V3 orientation) for detector WCS.
        header = file[0].header
        self.detector_to_sky_arcsec = None
        if (str(header.get('CTYPE1', '')).startswith('RA---') and
                str(header.get('CTYPE2', '')).startswith('DEC--')):
            if any(header.get(f'CUNIT{i}', 'deg').lower() != 'deg' for i in (1, 2)):
                raise ValueError('Detector WCS must use degrees')
            if all(f'CD{i}_{j}' in header for i in (1, 2) for j in (1, 2)):
                self.detector_to_sky_arcsec = tuple(tuple(3600*float(header[f'CD{i}_{j}'])
                    for j in (1, 2)) for i in (1, 2))
            elif all(f'CDELT{i}' in header for i in (1, 2)):
                self.detector_to_sky_arcsec = tuple(tuple(3600*float(header[f'CDELT{i}']) *
                    float(header.get(f'PC{i}_{j}', float(i == j)))
                    for j in (1, 2)) for i in (1, 2))
        self.slopes = np.array(file["SLOPE"].data, float)
        self.cov = np.array(file["SLOPE_COV"].data, float)
        self.ramp = np.asarray(file["RAMP"].data, float)
        self.ramp_cov = np.asarray(file["RAMP_COV"].data, float)
        self.slope_support = np.asarray(file["SLOPE_SUP"].data, int)
        self.parang = np.array(file[0].header["ROLL_REF"], float)

        # Make sure we have all the baxpixels
        # static_badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy"))
        static_badpix = np.load(resources.files(__package__) / "data" / "badpix.npy")
        pipeline_badpix = np.array(file["BADPIX"].data, bool)
        self.badpix = static_badpix | pipeline_badpix
        self.support = np.where(~self.badpix)

        #
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

        # The exposure fitter uses the aperture's chosen coefficient alias.
        if hasattr(optics.pupil_mask, "aberrations"):
            abb = np.zeros_like(optics.pupil_mask.aberrations)
            params["aberrations"] = (self.get_key("aberrations"), abb)

        # positions
        params["positions"] = (self.get_key("positions"), pos)
        params["fluxes"] = (self.get_key("fluxes"), log_flux)
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
        use_cov=True,
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

    def mv_zscore(self, model, return_im=False):
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
            return self.from_vec(z_vec)
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
        """Apply the supplied optical state, retaining its derivatives.

        Fit parameter selection and wavefront gauge belong to the caller's
        coordinates. The exposure's science/calibrator role does not change
        the derivative of the same forward model.
        """
        optics = model.optics
        if optics.pupil_mask.opd_sharing is not None:
            coefficients = {**optics.opd_coefficients, **model.params.get("opds", {})}
            aperture = apply_opds(optics.pupil_mask, coefficients,
                                 optics.pupil_mask.opd_sharing, self)
            optics = optics.set(["pupil_mask", "opd_selected"], [aperture, True])
        elif "aberrations" in model.params.keys():
            coefficients = model.aberrations[self.get_key("aberrations")]
            optics = optics.set("pupil_mask.aberrations", coefficients)
        if "reference_opd" in model.params:
            opds = dict(optics.pupil_mask.opds or {})
            opds["reference"] = model.reference_opd[self.get_key("aberrations")]
            optics = optics.set("pupil_mask.opds", opds)

        if hasattr(model, "reflectivity"):
            coefficients = model.reflectivity[self.get_key("reflectivity")]
            optics = optics.set("pupil_mask.amp_coeffs", coefficients)

        # Set the defocus
        optics = optics.set("defocus", model.defocus[self.get_key("defocus")])

        return optics

    def get_spectra(self, model):
        wavels, filt_weights = model.filters[self.filter]
        slope = model.get(self.map_param("spectra"))
        return wavels, spectral_weights(filt_weights, slope)

    def model_wfs(self, model, return_wf=True):
        """Return optical fields, or integrated Intensity images when requested.

        Wavefronts are returned before subpixel intensity averaging. The fitting
        path requests intensity explicitly to avoid reconstructing optical phase.
        """
        pos = dlu.arcsec2rad(model.positions[self.key])
        wavels, weights = self.get_spectra(model)

        optics = self.update_optics(model)
        return eqx.filter_jit(optics.propagate)(
            wavels, pos, weights, return_wf=return_wf, return_intensity=not return_wf
        )

    def model_psf(self, model):
        intensity = self.model_wfs(model, return_wf=False)
        return intensity.to_psf()

    def model_illuminance(self, psf, model):
        flux = self.ngroups * 10 ** model.fluxes[self.get_key("fluxes")]
        psf = eqx.filter_jit(model.detector)(psf, return_psf=True)
        return psf.multiply("data", flux)

    def model_ramp(self, illuminance, model):
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
        bias = np.where(self.badpix, np.median(bias), bias)

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
        return eqx.filter_jit(model.read)(ramp, return_psf=True)

    def nuke_pixel_grads(self, model):
        FF = lax.stop_gradient(model.FF)
        non_linearity = lax.stop_gradient(model.non_linearity)
        return model.set(["FF", "non_linearity"], [FF, non_linearity])

    def simulate(self, model, return_slopes=True):
        model = self.nuke_pixel_grads(model)
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
    vis_key: str | None = eqx.field(static=True, default=None)

    def __init__(self, *args, joint_fit=False, vis_key=None, **kwargs):
        self.joint_fit = bool(joint_fit)
        self.vis_key = vis_key
        super().__init__(*args, **kwargs)

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = super().initialise_params(
            optics, vis_model=vis_model, one_on_fs_order=one_on_fs_order
        )
        if vis_model is None:
            raise ValueError("vis_model must be provided for SplineVisFit")
        n = vis_model.n_basis
        params["vis"] = (self.get_key("vis"), np.zeros(n))
        return params

    def get_key(self, param):

        if param == "vis" and self.vis_key is not None:
            return self.vis_key

        # Return the per exposure key if not joint fitting
        if not self.joint_fit:
            if param in ["amplitudes", "phases", "vis"]:
                return self.key

        return super().get_key(param)

    def model_vis(self, intensity, model):
        coefficients = model.vis[self.get_key("vis")]
        return model.vis_model.model_vis(intensity, coefficients, self.filter)

    def model_psf(self, model):
        intensity = self.model_wfs(model, return_wf=False)
        return self.model_vis(intensity, model)


class FlatFit(ModelFit):
    polynomial_powers: np.ndarray

    def __init__(self, file, fit_one_on_fs=False, **kwargs):
        file[0].header["IS_PSF"] = False

        super().__init__(file, **kwargs)
        self.star = "NIS_LAMP"
        self.observation = "FLAT"
        self.program = "FLAT"
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
        illuminance = self.model_illuminance(model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)

        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp


class BinaryFit(ModelFit):

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("BinaryFit initialisation not yet implemented")

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = super().initialise_params(
            optics, vis_model=vis_model, one_on_fs_order=one_on_fs_order
        )
        # Binary parameters
        raise NotImplementedError("BinaryFit initialisation not yet implemented")
        params["separation"] = (self.get_key("separation"), 0.15)
        params["contrast"] = (self.get_key("contrast"), 2.0)
        params["position_angle"] = (self.get_key("position_angle"), 0.0)
        return params

    # Maybe overwrite this to get the binary spectra
    def get_spectra(self, model, exposure):
        return super().get_spectra(model, exposure)

    def model_wfs(self, model, exposure, return_wf=True):
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

        optics = self.update_optics(model, exposure)
        prop_fn = lambda pos: optics.propagate(
            wavels, pos, return_wf=return_wf, return_intensity=not return_wf
        )
        result = eqx.filter_jit(eqx.filter_vmap(prop_fn))(positions)
        if return_wf:
            return result.multiply("phasor", np.sqrt(weights)[..., None, None])
        return result.multiply("data", weights[..., None, None])

    def model_psf(self, model, exposure):
        intensity = self.model_wfs(model, exposure, return_wf=False)
        return intensity.to_psf(axis=(0, 1))
