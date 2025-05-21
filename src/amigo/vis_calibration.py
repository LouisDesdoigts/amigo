import jax.numpy as np
import dLux.utils as dlu


def calibrate_phases(sci_phase, cal_phases, sci_cov, cal_cov):
    """Propagate covariance for element-wise phase difference Δφ = phase_1 - phase_2."""
    n = sci_phase.shape[0]

    J1 = np.eye(n)
    J2 = -np.eye(n)

    cov_dphi = J1 @ sci_cov @ J1.T + J2 @ cal_cov @ J2.T
    dphi = sci_phase - cal_phases
    return dphi, cov_dphi


def calibrate_vis(vis_outputs, filt, kernel=True):

    if kernel:
        k_type = "K_"
    else:
        k_type = ""
    # Calibrator values
    cal_key = f"cal_{filt}"
    cal_amp = vis_outputs[cal_key][f"{k_type}vis"]
    cal_phase = vis_outputs[cal_key][f"{k_type}phi"]
    cal_amp_cov = vis_outputs[cal_key][f"{k_type}vis_cov"]
    cal_phase_cov = vis_outputs[cal_key][f"{k_type}phi_cov"]

    # Science values
    sci_key = f"sci_{filt}"
    sci_amp = vis_outputs[sci_key][f"{k_type}vis"]
    sci_phase = vis_outputs[sci_key][f"{k_type}phi"]
    sci_amp_cov = vis_outputs[sci_key][f"{k_type}vis_cov"]
    sci_phase_cov = vis_outputs[sci_key][f"{k_type}phi_cov"]

    # Calibrate
    amp, amp_cov = calibrate_phases(sci_amp, cal_amp, sci_amp_cov, cal_amp_cov)
    phase, phase_cov = calibrate_phases(sci_phase, cal_phase, sci_phase_cov, cal_phase_cov)

    return {
        f"{k_type}vis": amp,
        f"{k_type}vis_cov": amp_cov,
        f"{k_type}phi": phase,
        f"{k_type}phi_cov": phase_cov,
    }


def average_vis_fits(fit_list):
    #
    amps = np.array([fit["amplitudes"] for fit in fit_list])
    phases = np.array([fit["phases"] for fit in fit_list])
    K_amps = np.array([fit["K_amp"] for fit in fit_list])
    K_phases = np.array([fit["K_phase"] for fit in fit_list])

    #
    amp_covs = np.array([fit["amp_cov"] for fit in fit_list])
    phase_covs = np.array([fit["phase_cov"] for fit in fit_list])
    K_amp_covs = np.array([fit["K_amp_cov"] for fit in fit_list])
    K_phase_covs = np.array([fit["K_phase_cov"] for fit in fit_list])

    #
    parangs = np.array([fit["parang"] for fit in fit_list])
    wavels = np.array([fit["wavel"] for fit in fit_list])

    # Am I weighting this mean appropriately by the errors?
    amps = np.array(amps).mean(0)
    phases = np.array(phases).mean(0)
    K_amps = np.array(K_amps).mean(0)
    K_phases = np.array(K_phases).mean(0)

    amp_covs = np.array(amp_covs).mean(0)
    K_amp_covs = np.array(K_amp_covs).mean(0)
    phase_covs = np.array(phase_covs).mean(0)
    K_phase_covs = np.array(K_phase_covs).mean(0)

    # Put it all together
    return {
        "vis": amps,
        "phi": phases,
        "K_vis": K_amps,
        "K_phi": K_phases,
        "vis_cov": amp_covs,
        "phi_cov": phase_covs,
        "K_vis_cov": K_amp_covs,
        "K_phi_cov": K_phase_covs,
        "wavels": wavels,
        "parangs": parangs,
    }


def get_mean_wavelength(wavels, filt_weights, spectra):
    """Get the spectrally weighted mean wavelength"""
    xs = np.linspace(-1, 1, len(wavels), endpoint=True)
    spectra_slopes = 1 + spectra * xs
    weights = filt_weights * spectra_slopes
    weights /= weights.sum()
    return ((wavels * weights).sum() / weights.sum()).mean()


def vis_jac_fn(model_params, args):
    optics, vis_model, filter = args

    # Populate the optics with any bits we want
    for key, value in model_params.items():
        if hasattr(optics, key):
            optics = optics.set(key, value)

    # Get the spectrally weighted wavelengths
    wavels, weights = optics.filters[filter]

    if "sectra" in model_params.keys():
        xs = np.linspace(-1, 1, len(wavels), endpoint=True)
        spectra_slopes = 1 + model_params.spectra * xs
        weights = weights * spectra_slopes
        weights /= weights.sum()

    # Apply flux if in there
    if "flux" in model_params.keys():
        weights *= 10**model_params.flux

    # Propagate the wavefront and project to the latent space
    if "positions" in model_params.keys():
        offset = dlu.arcsec2rad(model_params.positions)
        wfs = optics.propagate(wavels, offset, weights, return_wf=True)
    else:
        wfs = optics.propagate(wavels, weights=weights, return_wf=True)
    return vis_model.wfs_to_latent(wfs, filter)
