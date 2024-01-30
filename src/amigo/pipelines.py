import os
import logging

import numpy as np
from astropy.io import fits
import astropy

from tqdm.auto import tqdm
from matplotlib import pyplot as plt

from jwst.pipeline import Image2Pipeline

# step imports
from jwst.group_scale import group_scale_step
from jwst.dq_init import dq_init_step
from jwst.saturation import saturation_step
from jwst.ipc import ipc_step
from jwst.superbias import superbias_step
from jwst.refpix import refpix_step
from jwst.rscd import rscd_step
from jwst.firstframe import firstframe_step
from jwst.lastframe import lastframe_step
from jwst.linearity import linearity_step
from jwst.dark_current import dark_current_step
from jwst.reset import reset_step
from jwst.persistence import persistence_step
from jwst.jump import jump_step
from jwst.charge_migration import charge_migration_step
from jwst.ramp_fitting import ramp_fit_step
from jwst.gain_scale import gain_scale_step
from jwst.stpipe import Pipeline
from stdatamodels.jwst import datamodels

__all__ = ["Detector1Pipeline"]

# Define logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)


class Detector1Pipeline(Pipeline):

    def __init__(self, lin_ramp_only=True, from_lin_ramp=False, *args, **kwargs):
        super(Detector1Pipeline, self).__init__(*args, **kwargs)
        self.lin_ramp_only = lin_ramp_only
        self.from_lin_ramp = from_lin_ramp

    """
    Detector1Pipeline: Apply all calibration steps to raw JWST
    ramps to produce a 2-D slope product. Included steps are:
    group_scale, dq_init, saturation, ipc, superbias, refpix, rscd,
    lastframe, linearity, dark_current, persistence, jump detection,
    ramp_fit, and gain_scale.
    """

    class_alias = "calwebb_detector1"

    spec = """
        save_calibrated_ramp = boolean(default=False)
    """

    # Define aliases to steps
    step_defs = {
        "group_scale": group_scale_step.GroupScaleStep,
        "dq_init": dq_init_step.DQInitStep,
        "saturation": saturation_step.SaturationStep,
        "ipc": ipc_step.IPCStep,
        "superbias": superbias_step.SuperBiasStep,
        "refpix": refpix_step.RefPixStep,
        "rscd": rscd_step.RscdStep,
        "firstframe": firstframe_step.FirstFrameStep,
        "lastframe": lastframe_step.LastFrameStep,
        "linearity": linearity_step.LinearityStep,
        "dark_current": dark_current_step.DarkCurrentStep,
        "reset": reset_step.ResetStep,
        "persistence": persistence_step.PersistenceStep,
        "jump": jump_step.JumpStep,
        "charge_migration": charge_migration_step.ChargeMigrationStep,
        "ramp_fit": ramp_fit_step.RampFitStep,
        "gain_scale": gain_scale_step.GainScaleStep,
    }

    def process(self, input):
        """
        Performs the actual data processing.
        """

        log.info("Starting calwebb_detector1 ...")

        # open the input as a RampModel
        input = datamodels.RampModel(input)

        # propagate output_dir to steps that might need it
        self.dark_current.output_dir = self.output_dir
        self.ramp_fit.output_dir = self.output_dir

        instrument = input.meta.instrument.name

        # process Near-IR exposures
        log.debug("Processing a Near-IR exposure")

        if self.from_lin_ramp:

            pass

        else:
            input = self.group_scale(input)
            input = self.dq_init(input)
            input = self.saturation(input)
            # input = self.ipc(input)
            input = self.superbias(input)
            input = self.refpix(input)
            input = self.linearity(input)

            input = self.dark_current(input)

            # apply the charge_migration step
            # input = self.charge_migration(input)

            # apply the jump step
            input = self.jump(input)

            # save the corrected ramp data, if requested
            if self.save_calibrated_ramp:
                self.save_model(input, "ramp")

        if self.lin_ramp_only:
            # skip the rest of the steps
            return None

        # apply the ramp_fit step
        # This explicit test on self.ramp_fit.skip is a temporary workaround
        # to fix the problem that the ramp_fit step ordinarily returns two
        # objects, but when the step is skipped due to `skip = True`,
        # only the input is returned when the step is invoked.
        if self.ramp_fit.skip:
            input = self.ramp_fit(input)
            ints_model = None
        else:
            input, ints_model = self.ramp_fit(input)

        # apply the gain_scale step to the exposure-level product
        if input is not None:
            self.gain_scale.suffix = "gain_scale"
            input = self.gain_scale(input)
        else:
            log.info("NoneType returned from ramp_fit.  Gain Scale step skipped.")

        # apply the gain scale step to the multi-integration product,
        # if it exists, and then save it
        if ints_model is not None:
            self.gain_scale.suffix = "gain_scaleints"
            ints_model = self.gain_scale(ints_model)
            self.save_model(ints_model, "rateints")

        # setup output_file for saving
        self.setup_output(input)

        log.info("... ending calwebb_detector1")

        return input

    def setup_output(self, input):
        if input is None:
            return None
        # Determine the proper file name suffix to use later
        if input.meta.cal_step.ramp_fit == "COMPLETE":
            self.suffix = "rate"
        else:
            self.suffix = "ramp"


def generate_ramps(files: str | list, output_dir: str):
    """
    Performs the modified Pipeline1 processing.
    Takes a list of _uncal.fits file directories
    as input and outputs _ramp.fits files in a
    specified output directory.

    Parameters
    ----------
    files : list
        List of _uncal.fits file directories.
    output_dir : str
        Output directory for _ramp.fits files.
    """
    # Check whether the specified output directory exists

    # string manip to make sure we have the right format
    if output_dir[-1] != "/":
        output_dir += "/"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in tqdm(files):
        # stage 1
        pl1 = Detector1Pipeline()
        pl1.save_results = True
        pl1.output_dir = str(output_dir)

        pl1.save_calibrated_ramp = True  # save linearized ramp
        pl1.lin_ramp_only = True  # generate a linearized ramp
        pl1.from_lin_ramp = False  # start from uncal file

        # pl1.dark_current.skip = True
        pl1.charge_migration.skip = True
        pl1.ipc.skip = True

        pl1.run(str(file))  # run pipeline from uncal file


def generate_calgrps(
    input_dir: str | list,
    output_dir: str,
    filename_base: str,
    tframe: float = 0.07544,
):
    """
    Performs the modified Pipeline2 processing.
    Takes a list of _ramp.fits file directories
    as input and outputs _calgrps.fits files in a
    specified output directory.

    Parameters
    ----------
    input_dir : list
        List of _ramp.fits file directories.
    output_dir : str
        Output directory for _calgrps.fits files.
    filename_base : str
        Filename base for _ramp.fits files,
        e.g. "jw01242004001_03102_00001_"
    tframe : float
        Frame time in seconds?  # TODO check this, ask Dori
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # string manip to make sure we have the right format
    if input_dir[-1] != "/":
        input_dir += "/"
    if output_dir[-1] != "/":
        output_dir += "/"

    # OPENS THE LINEARIZED RAMP FILE
    file_t = os.path.join(input_dir, filename_base + "nis_ramp.fits")

    # EXTRACTS THE DATA FROM THE LINEARIZED RAMP FILE
    hdu = fits.open(file_t)
    cube_t = hdu[1].data
    cube_3t = hdu[3].data
    cube_4t = hdu[4].data
    print(cube_t.shape)

    # number of groups
    ngroups = cube_t.shape[1]
    npix = 80

    out_cube_data = np.zeros(shape=(ngroups, npix, npix))

    # LOOPS OVER GROUPS
    for i in tqdm(range(1, ngroups + 1)):
        # GRABS THE DATA FROM ONE GROUP
        hdu1 = fits.open(file_t)

        # load data for each individual group
        hdu1[1].data = cube_t[:, i - 1 : i, :, :]
        hdu1[3].data = cube_3t[:, i - 1 : i, :, :]
        hdu1[4].data = cube_4t[:, i - 1 : i, :, :]

        # CREATES A NEW UNCAL FILE FOR EACH GROUP
        file_t2 = os.path.join(output_dir, filename_base + str(i) + "_nis_uncal.fits")

        hdu1.writeto(file_t2, overwrite=True)
        hdu1.close()

        # RUNS THE FIRST STAGE OF THE PIPELINE ON THE NEW UNCAL FILE
        pl1 = Detector1Pipeline()
        pl1.save_results = True
        pl1.output_dir = str(output_dir)
        pl1.ipc.skip = True
        pl1.save_calibrated_ramp = False

        # run pipeline from linearized ramp step
        pl1.lin_ramp_only = False
        pl1.from_lin_ramp = True
        pl1.run(str(file_t2))

        # RUNS THE SECOND STAGE OF THE PIPELINE ON THE NEW CALIBRATED FILE
        pl2 = Image2Pipeline()
        pl2.save_results = True
        pl2.output_dir = output_dir
        pl2.photom.skip = True
        pl2.flat_field.skip = True  # skip flat field
        pl2.bkg_subtract.skip = True  # skip background subtraction
        pl2.resample.skip = True
        fitsfile = file_t2.replace("_uncal.fits", "_rateints.fits")
        pl2.run(fitsfile)

        # correct the units in the calints files
        file_o = os.path.join(output_dir, filename_base + str(i) + "_nis_calints.fits")

        hdu_o = fits.open(file_o)

        # convert science data to e-
        sci = hdu_o["SCI"].data * tframe

        # sigma clip across integrations to remove cosmic rays, etc
        data_sc = astropy.stats.sigma_clip(sci, axis=0, masked=True)
        data_mean = np.nanmean(data_sc, axis=0)

        hdu_o["SCI"].data = np.array(data_mean)

        # APPENDING TO THE OUTPUT CUBE
        out_cube_data[i - 1, :, :] = np.array(data_mean)

        hdu_o.writeto(file_o, overwrite=True)
        # for last group we want to keep it open so we can easily write out to a calgrps file
        if i != range(1, ngroups + 1)[-1]:
            hdu_o.close()
            os.remove(file_o)

        # actively delete files we don't need
        os.remove(file_t2)  # REMOVES THE UNCAL FILE GENERATED FOR EACH GROUP
        file_t4 = os.path.join(output_dir, filename_base + str(i) + "_nis_rate.fits")
        os.remove(file_t4)
        file_t5 = os.path.join(
            output_dir, filename_base + str(i) + "_nis_rateints.fits"
        )
        os.remove(file_t5)

    file_calgrps = os.path.join(
        output_dir,
        filename_base + "nis_calgrps.fits",
    )

    hdu_o["SCI"].data = out_cube_data

    hdu_o.writeto(file_calgrps, overwrite=True)

    hdu_o.close()
    os.remove(file_o)
