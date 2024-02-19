import os
import numpy as np
import astropy
from astropy.io import fits
from jwst.pipeline import Detector1Pipeline
import matplotlib.pyplot as plt


def process_stage0(directory, output_dir='stage1/'):
    # string manip to make sure we have the right format
    if directory[-1] != "/":
        directory += "/"
    if output_dir[-1] != "/":
        output_dir += "/"
    
    # Get the files
    files = [directory + f for f in os.listdir(directory) if f.endswith("_uncal.fits")]

    # Check if there are any files to process
    if len(files) == 0:
        print("No _uncal.fits files found, no processing done.")
        return

    # Get the file paths
    paths = files[0].split('/')
    base_path = '/'.join(paths[:-2]) + '/'
    output_path = base_path + output_dir

    # Check whether the specified output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    print("Running stage 1...")
    for file_path in files:
        file_name = file_path.split('/')[-1]
        file_root = '_'.join(file_name.split('_')[:-2])

        # # Check whether the specified output directory exists, and get output path
        # output_path, file_root = check_directories(file_path, output_dir)
        final_output_path = output_path + file_root + "_nis_ramp.fits"

        # Check if the file is a NIS_AMI file
        if os.path.exists(final_output_path):
            print("File already exists, skipping...")
            continue

        # Check if the file is a NIS_AMI file
        file = fits.open(file_path)
        if file[0].header["EXP_TYPE"] != "NIS_AMI":
            print("Not a NIS_AMI file, skipping...")
            continue
        
        # Run stage 1
        pl1 = Detector1Pipeline()
        pl1.output_dir = str(output_path)
        
        pl1.save_results = True # what does this do?
        pl1.save_calibrated_ramp = True  # save the output

        # These are all the ones that are run at present, in order
        pl1.dq_init.skip = False
        pl1.saturation.skip = False
        pl1.ipc.skip = False
        pl1.superbias.skip = False
        pl1.refpix.skip = False
        pl1.linearity.skip = False
        pl1.persistence.skip = False
        pl1.dark_current.skip = False
        pl1.charge_migration.skip = True
        pl1.jump.skip = False
        pl1.ramp_fit.skip = True

        pl1.run(str(file_path))  # run pipeline from uncal file
    
    print("Done\n")

    return output_path

def process_stage1(directory, output_dir='calgrps/', refpix_correction=0):
    """
    ref_pix_correction: int
        What reference pixel correction to apply. 0 for none, 1 for first, 2 for second.
        'first' subtracts off the reference pixel value for _each group and 
        integration_, which is then masked and sigma clipped. 'second' performs the 
        masking and sigma clipping on both the data and reference pixel first, and then
        subtracts the single reference pixel value for each group and column from the 
        cleaned data.
    """
    if refpix_correction not in [0, 1, 2]:
        raise ValueError("ref_pix_correction must be 0, 1, or 2.")
    # string manip to make sure we have the right format
    if directory[-1] != "/":
        directory += "/"
    if output_dir[-1] != "/":
        output_dir += "/"
    
    # Get the files
    files = [directory + f for f in os.listdir(directory) if f.endswith("_ramp.fits")]

    # Check if there are any files to process
    if len(files) == 0:
        print("No _ramp.fits files found, no processing done.")
        return
    
    # Get the file paths
    paths = files[0].split('/')
    base_path = '/'.join(paths[:-2]) + '/'
    output_path = base_path + output_dir

    # Check whether the specified output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over files
    print("Running calgrps processing...")
    for file_path in files:
        file_name = file_path.split('/')[-1]
        file_root = '_'.join(file_name.split('_')[:-2])

        # Check if the file is a NIS_AMI file
        final_output_path = output_path + file_root + "_nis_calgrps.fits"
        if os.path.exists(final_output_path):
            print("File already exists, skipping...")
            continue

        # Check if the file is a NIS_AMI file
        file = fits.open(file_path)
        if file[0].header["EXP_TYPE"] != "NIS_AMI":
            print("Not a NIS_AMI file, skipping...")
            continue

        # Get the bits
        dq = np.array(file['PIXELDQ'].data) > 0
        group_dq = np.array(file['GROUPDQ'].data) > 0
        electrons = np.array(file['SCI'].data) # * file[0].header["TFRAME"]

        # Reference pixel correction 1
        if refpix_correction == 1:
            electrons += electrons[:, :, 4:5, :]

        # Clean and clip
        full_dq = group_dq | dq[None, None, ...]
        cleaned = np.where(full_dq, np.nan, electrons)

        # Mask the invalid values and sigma clip
        masked = np.ma.masked_invalid(cleaned, copy=True)
        masked_clipped = astropy.stats.sigma_clip(masked, axis=0, sigma=3)

        # Fill the masked values with nans
        cleaned_ramp = np.ma.filled(masked_clipped, fill_value=np.nan)

        # TODO: Fit gaussian to the ramp value? - Better mean and error?
        # Get the ramp and error 
        # Mean after sigma clipping - 'Robust mean'
        # This is quantised data, so mean is better than median
        ramp = np.nanmean(cleaned_ramp, axis=0)
        err = np.nanstd(cleaned_ramp, axis=0)

        # Reference pixel correction 2
        if refpix_correction == 2:
            refpix = electrons[:, :, 4:5, :]
            refpix_clipped = astropy.stats.sigma_clip(refpix, axis=0, sigma=3)
            cleaned_refpix = np.ma.filled(refpix_clipped, fill_value=np.nan)
            ramp += np.nanmedian(cleaned_refpix, axis=0)

        # Write to file
        file["SCI"].data = ramp
        file["ERR"].data = err

        # Save as calgrp
        file_calgrps = os.path.join(final_output_path)
        file.writeto(file_calgrps, overwrite=True)
        file.close()

    print("Done\n")
    return output_path