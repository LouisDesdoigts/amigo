import os
import shutil
import jax.numpy as np
import numpy as onp
from astropy.io import fits
from astropy.stats import sigma_clip
import pkg_resources as pkg
from .tqdm import tqdm


def delete_contents(path):
    for file_name in os.listdir(path):
        file_path = os.path.join(path, file_name)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


def process_calslope(
    directory,
    output_dir="calslope/",
    sigma=5,
    chunk_size=0,
    n_groups=None,  # how many groups of the ramp to use
    clean_dir=True,
):
    """
    Chunk size determines the maximum number of integrations in a 'chunk'. Each chunk
    is saved to its own file with an integer extension added. This breaks the data set
    into smaller time series to help avoid issues with any time-variation in the data.
    A chunk_size of zero will do no chunking and process the data all in one.

    This will (presently) always reprocess data

    if clean_dir is True, the existisng contents of the output_dir will be deleted prior
    to procesing, so ensure no old files are hanging around
    """
    if directory[-1] != "/":
        directory += "/"
    if output_dir[-1] != "/":
        output_dir += "/"

    # Get the files
    files = [directory + f for f in os.listdir(directory) if f.endswith("_uncal.fits")]

    # Check if there are any files to process
    if len(files) == 0:
        print("No _ramp.fits files found, no processing done.")
        return

    # Get the file paths
    paths = files[0].split("/")
    base_path = "/".join(paths[:-2]) + "/"
    output_path = base_path + output_dir

    # Check whether the specified output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Clear the existing files (since we might use different chunk sizes, and we do not
    # want to have old files hang around)
    if clean_dir:
        print("Cleaning existing directory")
        delete_contents(output_path)

    # Iterate over files
    print("Running calslope processing...")
    for file_path in tqdm(files):

        file = fits.open(file_path)

        # Check if the file is a NIS_AMI file
        if file[0].header["EXP_TYPE"] != "NIS_AMI":
            print("Not a NIS_AMI file, skipping...")
            continue

        # Skip single group files
        if file[0].header["NGROUPS"] == 1:
            print("Only one group, skipping...")
            continue

        # Get the data
        data = np.array(file["SCI"].data, int)
        file.close()

        if chunk_size == 0:
            chunks = [data]
            nchunks = 1
        else:
            nints = data.shape[0]
            if nints < chunk_size:
                nchunks = 1
            else:
                nchunks = np.round(nints / chunk_size).astype(int)
            chunks = np.array_split(data, nchunks)
            print(f"Breaking into {nchunks} chunks")

        # Get the root of the file name
        file_name = file_path.split("/")[-1]
        file_root = "_".join(file_name.split("_")[:-2])

        for i, chunk in enumerate(chunks):

            # Check if the file is a NIS_AMI file
            file_name = file_root + f"_{i+1:0{4}}" + "_nis_calslope.fits"
            file_calslope = os.path.join(output_path + file_name)

            # Create the new file
            shutil.copy(file_path, file_calslope)

            # Open new file
            # file = fits.open(file_calslope, mode="update")
            file = fits.open(file_calslope)

            # Remove the redundant or undesired extensions
            del file["SCI"]
            del file["GROUP"]
            del file["INT_TIMES"]

            # Update the various headers
            file[0].header["NCHUNKS"] = int(nchunks)
            file[0].header["CHUNK"] = i + 1
            file[0].header["CHUNKSZ"] = int(chunk_size)
            file[0].header["NINTS"] = int(chunk.shape[0])
            file[0].header["FILENAME"] = file_name
            file[0].header["SIGMA"] = sigma

            # Process the chunk
            file = process_data(file, chunk, sigma=sigma)

            # Save as calslope
            file.writeto(file_calslope, overwrite=True)
            file.close()

    print("Done\n")
    return output_path


def apply_sigma_clip(data, sigma=5.0, axis=0):
    """NOTE: casts bad values to nan, so output must be float array"""
    # Mask invalid values (nans, infs, etc.)
    masked = onp.ma.masked_invalid(data, copy=True)

    # Apply sigma clipping
    clipped = sigma_clip(masked, axis=axis, sigma=sigma)

    # Fill clipped/invalid values with -1
    data = np.array(onp.ma.filled(clipped, fill_value=-1), dtype=float)

    # Cast bad values to nan now that it is guaranteed a float array
    return data.at[np.where(data == -1.0)].set(np.nan)


def calc_mean_and_standard_error(data, axis=0):
    support = np.asarray(~np.isnan(data), int).sum(axis=axis)
    mean = np.nanmean(data, axis=axis)
    std_err = np.nanstd(data, axis=axis) / np.sqrt(support)
    return mean, std_err, support


def process_data(file, data, sigma=5.0):
    """
    Processes the data and saves the outputs to the file

    Note we sigma clip the data first to catch vary large or small values after things
    like cosmic ray hits, etc. Then we take the slopes and sigma clip those to catch
    any outliers that might have been missed in the first pass. We then calculate the
    mean and standard error of the ramp and the slope.
    """

    # Clip the data first
    clipped_data = apply_sigma_clip(data, sigma=sigma, axis=0)

    # Calculate the ramp mean and standard error
    ramp, ramp_err, ramp_support = calc_mean_and_standard_error(clipped_data, axis=0)

    # Calculate the slopes and sigma clip
    slopes = np.diff(clipped_data, axis=1)
    clipped_slopes = apply_sigma_clip(slopes, sigma=sigma, axis=0)

    # Calculate the slope mean and standard error
    slope, slope_err, slope_support = calc_mean_and_standard_error(clipped_slopes, axis=0)

    # Get the bad pixel array
    badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy")).astype(int)

    # Save the Outputs
    header = fits.Header()
    header["EXTNAME"] = "RAMP"
    file.append(fits.ImageHDU(data=ramp, header=header))

    header = fits.Header()
    header["EXTNAME"] = "RAMP_ERR"
    file.append(fits.ImageHDU(data=ramp_err, header=header))

    header = fits.Header()
    header["EXTNAME"] = "RAMP_SUP"
    file.append(fits.ImageHDU(data=ramp_support, header=header))

    header = fits.Header()
    header["EXTNAME"] = "SLOPE"
    file.append(fits.ImageHDU(data=slope, header=header))

    header = fits.Header()
    header["EXTNAME"] = "SLOPE_ERR"
    file.append(fits.ImageHDU(data=slope_err, header=header))

    header = fits.Header()
    header["EXTNAME"] = "SLOPE_SUP"
    file.append(fits.ImageHDU(data=slope_support, header=header))

    header = fits.Header()
    header["EXTNAME"] = "BADPIX"
    file.append(fits.ImageHDU(data=badpix, header=header))

    # Move the ASDF extention to the end
    file.append(file.pop("ASDF"))

    return file
