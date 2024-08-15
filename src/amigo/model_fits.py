import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from abc import abstractmethod
from jax import lax, vmap
from .ramp_models import model_ramp
from .misc import planck
from .vis_models import (
    build_vis_pts,
    get_mean_wavelength,
    get_uv_coords,
    sample_spline,
    to_uv,
    from_uv,
)


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
        key = self.get_key(exposure, "fluxes")
        flux = 10 ** model.fluxes[key]
        oversample = model.optics.oversample

        # Return the BFE and required meta-data
        if to_BFE:
            return psf.data, flux, oversample

        # Non linear model always goes from unit psf, flux, oversample to an 80x80 ramp
        # NOTE: Should be able to remove this if statement and just use SimpleRamp
        if model.ramp is not None:
            ramp = eqx.filter_jit(model.ramp.apply)(psf, flux, exposure, oversample)
        else:
            psf_data = dlu.downsample(psf.data * flux, oversample, mean=False)
            ramp = psf.set("data", model_ramp(psf_data, exposure.ngroups))

        return ramp

    def model_read(self, ramp, model, exposure):  # , slopes=False):
        # Model the read effects
        if "one_on_fs" in model.params.keys():
            model = model.set("read.one_on_fs", model.one_on_fs[exposure.key])

        # Add the zero point to the ramp
        zpoint = exposure.ramp[:1]
        zpoint = np.where(np.isnan(zpoint), 0, zpoint)
        slopes = np.diff(ramp.data, axis=0)
        true_ramp = np.concatenate([zpoint, zpoint + np.cumsum(slopes, axis=0)])
        ramp = ramp.set("data", true_ramp)

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
    uv_pad: int = eqx.field(static=True)
    crop_size: int = eqx.field(static=True)
    joint_fit: bool = eqx.field(static=True)
    per_wavelength: bool = eqx.field(static=True)

    def __init__(self, uv_pad=2, crop_size=160, joint_fit=True, per_wavelength=True):
        self.uv_pad = int(uv_pad)
        self.crop_size = int(crop_size)
        self.joint_fit = bool(joint_fit)
        self.per_wavelength = bool(per_wavelength)

    def get_key(self, exposure, param):

        # Return the per exposure key if not joint fitting
        if not self.joint_fit:
            if param in ["amplitudes", "phases"]:
                return exposure.key

        return super().get_key(exposure, param)

    def get_uv_coords(self, model, exposure):
        pscale = dlu.arcsec2rad(model.optics.psf_pixel_scale / model.optics.oversample)
        full_size = self.uv_pad * model.optics.psf_npixels * model.optics.oversample

        # Per wavelength, we need to calculate the UV coordinates for each wavelength
        if self.per_wavelength:
            wavels = model.filters[exposure.filter][0]
            coord_fn = lambda lam: get_uv_coords(lam, pscale, full_size, self.crop_size)
            return vmap(coord_fn)(wavels)

        # Otherwise, we can just use the weighted mean wavelength
        lam = get_mean_wavelength(*self.get_spectra(model, exposure))
        return get_uv_coords(lam, pscale, full_size, self.crop_size)

    def get_vis_pts(self, model, exposure):
        return build_vis_pts(
            model.amplitudes[self.get_key(exposure, "amplitudes")],
            model.phases[self.get_key(exposure, "phases")],
            model.visibilities.knots[0].shape,
        )

    def get_vis_map(self, model, exposure):
        # Get the inputs
        vis_pts = self.get_vis_pts(model, exposure)

        # If per_wavelength, this has an extra dimension
        uv_coords = np.array(self.get_uv_coords(model, exposure))
        knots = model.visibilities.knots

        # Interpolate the visibilities
        if self.per_wavelength:
            uv_coords = np.swapaxes(uv_coords, 0, 1)
            interp_fn = lambda im, coords: sample_spline(im, knots, coords)
            amp_map = vmap(interp_fn, (None, 0))(np.abs(vis_pts), uv_coords)
            phase_map = vmap(interp_fn, (None, 0))(np.angle(vis_pts), uv_coords)
        else:
            interp_fn = lambda im: sample_spline(im, knots, uv_coords)
            amp_map = interp_fn(np.abs(vis_pts))
            phase_map = interp_fn(np.angle(vis_pts))
        return np.maximum(amp_map, 0) * np.exp(1j * phase_map)

    def calc_vis_psf(self, wfs, model, exposure):
        psfs = wfs.psf
        npix_in = psfs.shape[-1]
        npix_pad = self.uv_pad * npix_in

        if self.per_wavelength:
            splodges = vmap(lambda psf: to_uv(dlu.resize(psf, npix_pad)))(psfs)
            resize_fn = lambda vis_map: dlu.resize(vis_map, npix_pad)
            vis_map = vmap(resize_fn)(self.get_vis_map(model, exposure))

            amps = np.abs(splodges)
            norm_amps = amps / np.max(amps, axis=(1, 2), keepdims=True)
            applied = np.where(norm_amps > 1e-3, splodges * vis_map, splodges)

            psf_fn = lambda cplx: np.abs(dlu.resize(from_uv(cplx), npix_in))
            return vmap(psf_fn)(applied).sum(0)

        else:
            psf = wfs.psf.sum(0)
            vis_map = dlu.resize(self.get_vis_map(model, exposure), npix_pad)
            splodges = to_uv(dlu.resize(psf, npix_pad))

            # The support threshold is some what arbitrary and determined experimentally
            # To be safe, in future we should normalise the splodge map first
            applied = np.where(np.abs(splodges) > 1e-3, splodges * vis_map, splodges)
            np.where(np.abs(splodges) > 1e-3, vis_map, np.nan)
            return dlu.resize(np.abs(from_uv(applied)), npix_in)

    def model_vis(self, wfs, model, exposure):
        psf = self.calc_vis_psf(wfs, model, exposure)
        return dl.PSF(psf, wfs.pixel_scale.mean(0))

    def model_psf(self, model, exposure):
        wfs = self.model_wfs(model, exposure)
        psf = self.model_vis(wfs, model, exposure)
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
