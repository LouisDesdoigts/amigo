import equinox as eqx
import zodiax as zdx
import dLux as dl
import dLux.utils as dlu
import jax.numpy as np
from abc import abstractmethod
from jax import jit, lax, vmap
from .modelling import planck
from .ramp_models import model_ramp
from .interferometry import apply_vis, to_uv, from_uv
from .vis_models import build_vis_pts, get_mean_wavelength, get_uv_coords, sample_spline


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

        # # TODO: Add mapping
        # if param == "dispersion":
        #     return f"{param}.{exposure.get_key(param)}"

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

        if "wfs_opd" in optics.layers.keys():
            optics = optics.set("wfs_opd.opd", exposure.opd)
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

        # Apply the read effects
        ramp = model.read.apply(ramp)

        return np.diff(ramp.data, axis=0)

        # # Return the slopes if required
        # if slopes:
        #     return np.diff(ramp.data, axis=0)

        # # Return the ramp
        # return ramp.data


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

    def __init__(self, uv_pad=2, crop_size=160, joint_fit=True):
        self.uv_pad = int(uv_pad)
        self.crop_size = int(crop_size)
        self.joint_fit = bool(joint_fit)

    def get_key(self, exposure, param):

        # Return the per exposure key if not joint fitting
        if not self.joint_fit:
            if param in ["amplitudes", "phases"]:
                return exposure.key

        return super().get_key(exposure, param)

    def get_uv_coords(self, model, exposure):
        lam = get_mean_wavelength(*self.get_spectra(model, exposure))
        pscale = dlu.arcsec2rad(model.optics.psf_pixel_scale / model.optics.oversample)
        full_size = self.uv_pad * model.optics.psf_npixels * model.optics.oversample
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
        uv_coords = self.get_uv_coords(model, exposure)
        knots = model.visibilities.knots

        # Interpolate the visibilities
        interp_fn = lambda im: sample_spline(im, knots, uv_coords)
        amp_map = np.maximum(interp_fn(np.abs(vis_pts)), 0)
        phase_map = interp_fn(np.angle(vis_pts))
        return amp_map * np.exp(1j * phase_map)

    def calc_vis_psf(self, psf, model, exposure):
        splodges = to_uv(dlu.resize(psf, self.uv_pad * len(psf)))
        vis_map = dlu.resize(self.get_vis_map(model, exposure), splodges.shape[0])

        # The support threshold is some what arbitrary and determined experimentally
        # To be safe, in future we should normalise the splodge map first
        applied = np.where(np.abs(splodges) > 1e-3, splodges * vis_map, splodges)
        return dlu.resize(np.abs(from_uv(applied)), psf.shape[0])

    def model_vis(self, psf, model, exposure):
        return psf.set("data", self.calc_vis_psf(psf.data, model, exposure))

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        psf = self.model_vis(psf, model, exposure)
        psf = self.model_detector(psf, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)


class FringeFit(PointFit):

    def model_wfs(self, model, exposure):
        wavels, weights = self.get_spectra(model, exposure)
        optics = self.update_optics(model, exposure)

        pos1 = dlu.arcsec2rad(model.positions[exposure.key])
        wfs1 = eqx.filter_jit(optics.propagate)(wavels, pos1, weights, return_wf=True)

        pos2 = pos1 + dlu.arcsec2rad(model.shifts[exposure.filter])
        wfs2 = eqx.filter_jit(optics.propagate)(wavels, pos2, weights, return_wf=True)

        contrast = 10 ** model.contrasts[exposure.filter]
        f1, f2 = np.array([1, contrast]) / (1 + contrast)
        combined_phasors = (f1 * wfs1.phasor) + (f2 * wfs2.phasor)
        wfs = wfs1.set(
            ["amplitude", "phase"], [np.abs(combined_phasors), np.angle(combined_phasors)]
        )

        # Convert Cartesian to Angular wf
        if wfs.units == "Cartesian":
            wfs = wfs.multiply("pixel_scale", 1 / optics.focal_length)
            wfs = wfs.set(["plane", "units"], ["Focal", "Angular"])
        return wfs

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        psf = self.model_detector(psf, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)


class VisFit(ModelFit):
    pad: int = eqx.field(static=True)

    def __init__(self, pad=2):
        self.pad = int(pad)

    def get_key(self, exposure, param):
        if param in ["shifts", "contrasts"]:
            return exposure.filter
        return super().get_key(exposure, param)

    def model_vis(self, wfs, model, exposure, cplx=False):
        # Get the bits we need
        basis = model.visibilities.basis[exposure.filter]
        weights = model.visibilities.weight[exposure.filter]
        inv_support = model.visibilities.inv_support[exposure.filter]

        # Get visibilities
        key = self.get_key(exposure, "amplitudes")
        vis = model.amplitudes[key] * np.exp(1j * model.phases[key])

        # Apply the visibilities
        psfs = wfs.psf
        vis_fn = vmap(lambda *args: apply_vis(vis, *args), 4 * (0,))
        pad_fn = vmap(lambda x: dlu.resize(x, self.pad * psfs.shape[-1]))
        cplx_psfs = jit(vis_fn)(pad_fn(psfs), basis, weights, inv_support)

        # Crop back to original size
        cplx_psfs = vmap(lambda x: dlu.resize(x, psfs.shape[-1]))(cplx_psfs)

        # Return complex or psf
        if cplx:
            return cplx_psfs
        return np.abs(cplx_psfs)
        # return dl.PSF(np.abs(cplx_psfs), wfs.pixel_scale.mean(0))

    def model_psf(self, model, exposure):
        wfs = self.model_wfs(model, exposure)
        psfs = self.model_vis(wfs, model, exposure)
        return dl.PSF(psfs.sum(0), wfs.pixel_scale.mean(0))

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        psf = self.model_detector(psf, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)


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


class DispersedFit(ModelFit):

    def model_wfs(self, model, exposure):
        wavels, weights = self.get_spectra(model, exposure)
        pos = dlu.arcsec2rad(model.positions[exposure.key])
        optics = self.update_optics(model, exposure)
        return eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)

    def model_psf(self, model, exposure):
        wavels, weights = self.get_spectra(model, exposure)
        optics = self.update_optics(model, exposure)

        # Model the optics
        pos = dlu.arcsec2rad(model.positions[exposure.key])

        # Dispersion contrast
        contrast = 10**model.contrast
        flux_weights = np.array([contrast * 1, 1]) / (1 + contrast)

        # Model the primary psf
        wfs = eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)
        primary_psfs = flux_weights[0] * wfs.psf

        # Model the dispersed psf
        wf_prop = lambda *args: optics.propagate_mono(*args, return_wf=True)
        prop_fn = lambda wav, disp: wf_prop(wav, pos + disp)

        # # This one does free-floating (x, y)
        # dispersion = dlu.arcsec2rad(model.dispersion[exposure.filter])

        # This one does furthest point (x, y)
        xmax, ymax = dlu.arcsec2rad(model.dispersion[exposure.filter])
        xs = np.linspace(-xmax, xmax, len(wavels))
        ys = np.linspace(-ymax, ymax, len(wavels))
        dispersion = np.array([xs, ys]).T

        # Apply it
        wfs = eqx.filter_jit(eqx.filter_vmap(prop_fn))(wavels, dispersion)
        wfs = wfs.multiply("amplitude", weights[:, None, None] ** 0.5)
        secondary_psfs = flux_weights[1] * wfs.psf
        psfs = primary_psfs + secondary_psfs

        return dl.PSF(psfs.sum(0), wfs.pixel_scale.mean(0))

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        psf = self.model_detector(psf, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)


from jax.nn import sigmoid


class PolarisedFit(ModelFit):

    def model_psf(self, model, exposure):
        wavels, weights = self.get_spectra(model, exposure)
        optics = self.update_optics(model, exposure)
        pos = dlu.arcsec2rad(model.positions[exposure.key])

        # Get the polarisation keys - is this just all of pupil_mask?
        polarisation_keys = [
            "abb_coeffs",
            "amp_coeffs",
            "holes",
            "f2f",
            "transformation",
        ]

        # Partition the optics - assumed the model is already partition-vectorised
        filter_spec = zdx.boolean_filter(optics, polarisation_keys)
        polarised, unpolarised = eqx.partition(optics, filter_spec)

        # Function to evaluate the polarised optics
        def fn(polar_optics, null_optics):
            optics = eqx.combine(polar_optics, null_optics)
            return eqx.filter_jit(optics.propagate)(wavels, pos, weights, return_wf=True)

        # Model the psf
        wfs = eqx.filter_jit(eqx.filter_vmap(fn, (0, None)))(polarised, unpolarised)

        # Get the polarisation contrast
        contrast = sigmoid(model.contrasts[exposure.filter])
        contrasts = np.array([contrast, 1 - contrast])[:, None, None, None]

        # Apply the contrast and return
        psfs = (contrasts * wfs.psf).sum(0)
        return dl.PSF(psfs, wfs.pixel_scale.mean((0, 1)))

    def __call__(self, model, exposure):
        psf = self.model_psf(model, exposure)
        psf = self.model_detector(psf, model, exposure)
        ramp = self.model_ramp(psf, model, exposure)
        return self.model_read(ramp, model, exposure)
