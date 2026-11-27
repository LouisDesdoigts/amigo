import warnings
from datetime import date

import jax
import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from jax import lax, vmap

# import pkg_resources as pkg
from importlib import resources
from .misc import find_position, get_pos, gen_surface, primary_offset, spectral_weights
from .ramp_models import Ramp
from .optical_models import gen_powers
from .opd import apply_opds
from .stats import mv_zscore, loglike, tridiagonal_covariance

# Opt-in: when True, point-source exposures (calibrators, binaries -- anything
# going through ModelFit.simulate or BinaryFit.simulate) stop blocking their
# likelihood gradient to FF/non_linearity (normally done via nuke_pixel_grads, so
# those only ever get real gradients from flats). Off by default -- zero effect on
# every existing caller. DarkFit's own nuke_pixel_grads call is untouched either
# way: darks have essentially no illumination, so this isn't the interesting case.
# Diagnostic for whether letting FF/non_linearity respond to PSF data actually
# improves the fit, or just lets them absorb optical-model residuals (e.g. the
# core speckle pattern) that should really be attributed to aberrations/nn_weights
# instead -- worth checking the resulting FF/non_linearity maps for PSF-shaped
# structure, not just whether the loss goes down.
ALLOW_CAL_PIXEL_GRADS = False

# The aperture's wavefront coefficients, each fitted under the name of the aperture alias it sets: the legacy
# per-hole Zernikes ("aberrations", radial_orders), or the three parts of opd.standard_opds (Fourier and Zernike
# shared, piston/tip/tilt per key). A parameter exists only for the aliases the aperture has.
WF_PARAMS = ("aberrations", "fourier_coeffs", "zernike_coeffs", "ptt_coeffs")


def observation_date(header):
    """The exposure's UTC start date, ISO "YYYY-MM-DD", from DATE-OBS (None without one)."""
    value = header.get("DATE-OBS")
    if value is None:
        return None
    return date.fromisoformat(str(value).strip()[:10]).isoformat()


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
    date: str | None = eqx.field(static=True)
    observation: str = eqx.field(static=True)
    act_id: str = eqx.field(static=True)
    visit: str = eqx.field(static=True)
    dither: str = eqx.field(static=True)
    POS: str = eqx.field(static=True)
    detector_to_sky_arcsec: tuple | None = eqx.field(default=None, kw_only=True, static=True)
    # The calibrator epoch whose Fourier wavefront set this exposure uses, when an instrument
    # record has several (amigo.files.assign_fourier_epochs); None until assigned.
    fourier_epoch: str | None = eqx.field(default=None, kw_only=True, static=True)
    # The pupil wheel's reported angle (PWCPOS, deg; None without one) and the primary dither
    # position (amigo.misc.primary_offset): both set where the exposure sits on the detector
    pwcpos: float | None = eqx.field(default=None, kw_only=True, static=True)
    primary_offset: str | None = eqx.field(default=None, kw_only=True, static=True)
    # Degrees added to the fitted rotation this exposure reads (rotation()), held fixed: e.g.
    # amigo-pipeline's -(PWCPOS - its calibrators' mean PWCPOS). 0 unless set.
    rotation_offset: float = eqx.field(default=0.0, kw_only=True, static=True)

    def __init__(self, file):
        # Display geometry only: retain the FITS tangent-plane Jacobian, including
        # parity. Do not substitute ROLL_REF (V3 orientation) for detector WCS.
        header = file[0].header
        self.detector_to_sky_arcsec = None
        if str(header.get("CTYPE1", "")).startswith("RA---") and str(
            header.get("CTYPE2", "")
        ).startswith("DEC--"):
            if any(header.get(f"CUNIT{i}", "deg").lower() != "deg" for i in (1, 2)):
                raise ValueError("Detector WCS must use degrees")
            if all(f"CD{i}_{j}" in header for i in (1, 2) for j in (1, 2)):
                self.detector_to_sky_arcsec = tuple(
                    tuple(3600 * float(header[f"CD{i}_{j}"]) for j in (1, 2)) for i in (1, 2)
                )
            elif all(f"CDELT{i}" in header for i in (1, 2)):
                self.detector_to_sky_arcsec = tuple(
                    tuple(
                        3600
                        * float(header[f"CDELT{i}"])
                        * float(header.get(f"PC{i}_{j}", float(i == j)))
                        for j in (1, 2)
                    )
                    for i in (1, 2)
                )
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
        self.date = observation_date(header)
        self.fourier_epoch = None
        self.act_id = file[0].header["ACT_ID"]
        self.visit = file[0].header["VISITGRP"]
        self.dither = file[0].header["EXPOSURE"]
        self.POS = get_pos(file)
        self.pwcpos = None if header.get("PWCPOS") is None else float(header["PWCPOS"])
        self.primary_offset = primary_offset(header)
        self.rotation_offset = 0.0
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

        # positions
        params["positions"] = (self.get_key("positions"), pos)
        params["fluxes"] = (self.get_key("fluxes"), log_flux)
        # Flat wavefront under this exposure's keys: the legacy per-hole Zernikes and, on
        # an aperture without an opd_sharing table, the standard_opds coefficients
        # (WF_PARAMS). With a table, the model initialises the shared OPD coefficients
        # per group instead (core_models, opd.init_opds).
        sharing = getattr(optics.pupil_mask, "opd_sharing", None)
        for param in WF_PARAMS if sharing is None else ("aberrations",):
            if hasattr(optics.pupil_mask, param):
                params[param] = (
                    self.get_key(param),
                    np.zeros_like(getattr(optics.pupil_mask, param)),
                )
        params["spectra"] = (self.get_key("spectra"), np.array(0.0))
        params["defocus"] = (self.get_key("defocus"), np.array(0.01))
        params["sigmas"] = (self.get_key("sigmas"), np.array(0.0214))  # detector blur, arcsec
        params["rotations"] = (self.get_key("rotations"), np.array(0.0))  # degrees

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

    def rotation(self, model):
        """This exposure's rotation (deg): the fitted one it reads, plus its fixed offset."""
        return model.rotations[self.get_key("rotations")] + self.rotation_offset

    def to_vec(self, image):
        return image[..., *self.support].T

    def from_vec(self, vec, fill=np.nan):
        return (fill * np.ones((80, 80))).at[*self.support].set(vec)


COVARIANCES = ("full", "tridiagonal", "diagonal")


def _covariance_option(covariance, use_cov, only_diag):
    """Resolve the deprecated use_cov/only_diag aliases to a covariance mode."""
    if use_cov is None and only_diag is None:
        return covariance
    if covariance != "full":
        raise ValueError("Pass covariance, not covariance with use_cov/only_diag")
    warnings.warn(
        "use_cov and only_diag are deprecated: use covariance='full', 'tridiagonal', 'diagonal'",
        FutureWarning,
        stacklevel=3,
    )
    if use_cov is not None and not use_cov:
        return "diagonal"
    return "tridiagonal" if only_diag else "full"


class ModelFit(Exposure):
    fit_one_on_fs: bool = eqx.field(static=True)
    fit_reflectivity: bool = eqx.field(static=True)
    fit_bias: bool = eqx.field(static=True)
    validator: bool = eqx.field(static=True)
    covariance: str = eqx.field(static=True)

    def __init__(
        self,
        *args,
        fit_reflectivity=False,
        fit_one_on_fs=False,
        fit_bias=False,
        validator=False,
        covariance="full",
        use_cov=None,
        only_diag=None,
        **kwargs,
    ):
        """
        covariance sets how each exposure's slope covariance is used: "full" uses
        the measured matrix, "tridiagonal" keeps its variances and replaces the rest
        with the read-noise correlation of neighbouring slopes (see
        tridiagonal_covariance), and "diagonal" treats every slope as independent.
        use_cov and only_diag are deprecated aliases for it.
        """
        covariance = _covariance_option(covariance, use_cov, only_diag)
        if covariance not in COVARIANCES:
            raise ValueError(f"covariance must be one of {COVARIANCES}, got {covariance!r}")
        self.fit_one_on_fs = fit_one_on_fs
        self.fit_reflectivity = fit_reflectivity
        self.fit_bias = fit_bias
        self.validator = bool(validator)
        self.covariance = covariance
        super().__init__(*args, **kwargs)

        if covariance == "diagonal":
            self.cov = self.cov * np.eye(len(self.slopes))[..., None, None]

        if covariance == "tridiagonal":
            self.cov = tridiagonal_covariance(self.cov)

    def mv_zscore(self, model, return_im=False, return_slopes=False):
        slopes = self(model)

        # Get the model, data, and variances
        slope_vec = self.to_vec(slopes)
        data_vec = self.to_vec(self.slopes)

        if self.covariance != "diagonal":
            cov_vec = self.to_vec(self.cov)
            # Calculate per-pixel z-scores
            z_vec = vmap(mv_zscore)(slope_vec, data_vec, cov_vec)
        else:
            # covariance="diagonal" already zeroes self.cov's off-diagonal entries in
            # __init__, so the "multivariate" z-score above is mathematically just
            # a per-group sum-of-squared-residuals-over-variance -- but it still
            # pays for a full (ngroups-1)x(ngroups-1) np.linalg.inv per pixel
            # (vmapped, so once per valid pixel per exposure) to get there, an
            # O(ngroups^3) cost for a result that only ever used the diagonal.
            # Skip straight to the diagonal case: identical numbers, O(ngroups)
            # per pixel instead.
            var_vec = vmap(np.diag)(self.to_vec(self.cov))
            z_vec = -np.sum((slope_vec - data_vec) ** 2 / var_vec, axis=-1)

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

        if param in ["aberrations", "ptt_coeffs"]:
            return "_".join([self.program, self.filter])

        if param == "fourier_coeffs":
            return "shared"

        if param == "zernike_coeffs":
            return self.filter

        if param in ["reflectivity", "defocus", "sigmas"]:
            return self.filter

        if param in ["biases", "rotations"]:
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
            *WF_PARAMS,
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
            "rotations",
            "sigmas",
        ]:
            return f"{param}.{self.get_key(param)}"

        # Else its global
        return param

    def update_optics(self, model):
        """Apply the supplied optical state.

        With an opd_sharing table, the grouped OPD coefficients are applied with
        their derivatives; which ones are fitted is the caller's choice. Without
        one, each wavefront parameter (WF_PARAMS) is read under this exposure's
        key, hole 0's piston is held, and a science target's wavefront carries no
        gradient, so it inherits the calibrators'.
        """
        optics = model.optics
        if getattr(optics.pupil_mask, "opd_sharing", None) is not None:
            coefficients = {**optics.opd_coefficients, **model.params.get("opds", {})}
            aperture = apply_opds(
                optics.pupil_mask, coefficients, optics.pupil_mask.opd_sharing, self
            )
            optics = optics.set(["pupil_mask", "opds_applied"], [aperture, True])
        else:
            for param in WF_PARAMS:
                if param not in model.params.keys():
                    continue
                coefficients = model.get(self.map_param(param))

                # Nuke the piston gradient to prevent degeneracy. Only for per-hole
                # bases, (n_holes, n_modes), where [0, 0] is hole 0's piston; a
                # pupil-wide basis, (n_modes,), has no global piston to fix.
                if coefficients.ndim == 2:
                    fixed_piston = lax.stop_gradient(coefficients[0, 0])
                    coefficients = coefficients.at[0, 0].set(fixed_piston)

                # Stop gradient for science targets
                if not self.calibrator:
                    coefficients = lax.stop_gradient(coefficients)
                optics = optics.set(f"pupil_mask.{param}", coefficients)
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
        """Optical fields (before the subpixel intensity averaging), or the spectral Intensity
        images with return_wf=False, which is what the PSF is built from."""
        # The detector's resampler turns the whole image about the array centre (model_illuminance),
        # but the mask turns the pattern about the star, not the star: so turn the star back by the
        # same angle first, which also keeps the rotation from pulling on the fitted position.
        angle = dlu.deg2rad(self.rotation(model))
        x, y = model.positions[self.key]
        c, s = np.cos(angle), np.sin(angle)
        pos = dlu.arcsec2rad(np.array([c * x + s * y, -s * x + c * y]))
        wavels, weights = self.get_spectra(model)

        optics = self.update_optics(model)
        return eqx.filter_jit(optics.propagate)(
            wavels, pos, weights, return_wf=return_wf, return_intensity=not return_wf
        )

    def model_psf(self, model):
        return self.model_wfs(model, return_wf=False).to_psf()

    def model_illuminance(self, psf, model):
        flux = self.ngroups * 10 ** model.fluxes[self.get_key("fluxes")]
        # This exposure's rotation, on top of the detector's (the NRM sits in the pupil wheel, which
        # does not come back to exactly the same angle between visits), and its filter's blur
        # (mostly charge diffusion, which depends on how deep the photons are absorbed)
        detector = model.detector.add("layers.resampler.rotation", self.rotation(model))
        detector = detector.set("layers.convolution.sigma", model.sigmas[self.get_key("sigmas")])
        psf = detector(psf, return_psf=True)
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
        if not ALLOW_CAL_PIXEL_GRADS:
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
    share_vis: bool = eqx.field(static=True)
    vis_key: str | None = eqx.field(static=True, default=None)

    def __init__(self, *args, share_vis=False, vis_key=None, joint_fit=None, **kwargs):
        """
        vis_key names the group of exposures that share one visibility vector, and
        takes precedence. Without it, share_vis=True shares by star and filter and
        False gives each exposure its own. joint_fit is a deprecated alias.
        """
        if joint_fit is not None:
            if share_vis:
                raise ValueError("Pass share_vis, not both share_vis and joint_fit")
            warnings.warn("joint_fit is deprecated: use share_vis", FutureWarning, stacklevel=2)
            share_vis = joint_fit
        self.share_vis = bool(share_vis)
        self.vis_key = vis_key
        super().__init__(*args, **kwargs)

    def initialise_params(self, optics, vis_model=None, one_on_fs_order=1):
        params = super().initialise_params(
            optics, vis_model=vis_model, one_on_fs_order=one_on_fs_order
        )
        if vis_model is None:
            raise ValueError("vis_model must be provided for SplineVisFit")
        n = vis_model.n_modes
        params["vis"] = (self.get_key("vis"), np.zeros(n))
        return params

    def get_key(self, param):

        if param == "vis" and self.vis_key is not None:
            return self.vis_key

        # Return the per exposure key if not joint fitting
        if not self.share_vis:
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

    def __init__(self, file, unique_params=None, calibrator=True, **kwargs):

        super().__init__(file, **kwargs)

        # OVERIDE self.calbrator
        self.calibrator = calibrator

        self.sub_exps = {
            "A": PointFit(file, **kwargs),
            "B": PointFit(file, **kwargs),
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

        # separation vector d (from A to B) in the frame of `positions`. That frame's x is the
        # detector column and its y is minus the row (the optics' InvertY), so x points to
        # PA = parang - 90 and y to parang + 180: position angles run CLOCKWISE in it, hence
        # the minus on y. Checked against the header WCS (PC matrix) and the fitted positions
        # of 48 exposures (data centroid = +x column, -y row), 2026-10-13. The earlier
        # (cos, sin) had the handedness wrong: it put B at 2 * parang - 180 - pa.
        d = separation * np.array([np.cos(phi), -np.sin(phi)])  # arcsec, like positions

        posA = mean_pos - d / 2  # brighter source (contrast is A's share of the flux)
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

        illuminances, pixel_scales = vmap(
            single_source, in_axes=(0, 0, {k: 0 for k in unique_AB})
        )(positions_AB, fluxes_AB, unique_AB)

        illuminance = dl.PSF(illuminances.sum(axis=0), pixel_scales[0])

        return illuminance

    def simulate(self, model, return_slopes: bool = True):
        if not ALLOW_CAL_PIXEL_GRADS:
            model = self.nuke_pixel_grads(model)
        model = self.nuke_dark_grads(model)
        illuminance = self.model_interferogram(model)
        ramp = self.model_ramp(illuminance, model)
        ramp = self.model_read(ramp, model)
        if return_slopes:
            return ramp.set("data", np.diff(ramp.data, axis=0))
        return ramp
