import os
import jax.numpy as np
import shutil
import numpy as onp
import astropy
from astropy.io import fits
from jwst.pipeline import Detector1Pipeline
from tqdm.notebook import tqdm


def process_uncal(directory, output_dir="stage1/", verbose=False, AMI_only=True, reprocess=False):
    # string manip to make sure we have the right format
    if directory[-1] != "/":
        directory += "/"
    if output_dir[-1] != "/":
        output_dir += "/"

    # Get the files
    files = [directory + f for f in os.listdir(directory) if f.endswith("_uncal.fits")]

    # Check if there are any files to process
    if len(files) == 0:
        if verbose:
            print("No _uncal.fits files found, no processing done.")
        return

    # Get the file paths
    paths = files[0].split("/")
    base_path = "/".join(paths[:-2]) + "/"
    output_path = base_path + output_dir

    # Check whether the specified output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if verbose:
        print("Running stage 1...")
    for file_path in files:
        file_name = file_path.split("/")[-1]
        file_root = "_".join(file_name.split("_")[:-2])

        # # Check whether the specified output directory exists, and get output path
        # output_path, file_root = check_directories(file_path, output_dir)
        final_output_path = output_path + file_root + "_nis_ramp.fits"

        # Check if the file is a NIS_AMI file
        if os.path.exists(final_output_path):
            if verbose:
                print("File already exists, skipping...")
            continue

        # Check if the file is a NIS_AMI file
        file = fits.open(file_path)
        if AMI_only and file[0].header["EXP_TYPE"] != "NIS_AMI":
            if verbose:
                print("Not a NIS_AMI file, skipping...")
            continue

        # Run stage 1
        pl1 = Detector1Pipeline()
        pl1.output_dir = str(output_path)

        pl1.save_results = True  # what does this do?
        pl1.save_calibrated_ramp = True  # save the output

        # These are all the ones that are run at present, in order
        pl1.dq_init.skip = False  # This is run
        pl1.saturation.skip = False  # This is run
        pl1.ipc.skip = True
        pl1.superbias.skip = False  # This is run
        pl1.refpix.skip = True
        pl1.linearity.skip = True
        pl1.persistence.skip = True
        pl1.dark_current.skip = True
        pl1.charge_migration.skip = True
        pl1.jump.skip = False  # This is run
        pl1.ramp_fit.skip = True

        pl1.run(str(file_path))  # run pipeline from uncal file

    if verbose:
        print("Done\n")

    return output_path


def sigma_clip(array, sigma=5.0, axis=0):
    masked = onp.ma.masked_invalid(array, copy=True)
    clipped = astropy.stats.sigma_clip(masked, axis=axis, sigma=sigma)
    return onp.ma.filled(clipped, fill_value=onp.nan)


# def rebuild_ramps(cleaned_slopes):
#     ngroups = cleaned_slopes.shape[1]

#     # dx is defined to be 1, so the slope _is the counts_
#     counts = 0
#     cleaned_counts = []
#     for k in range(ngroups):
#         counts += cleaned_slopes[:, k]
#         cleaned_counts.append(counts)
#     cleaned_counts = np.array(cleaned_counts)

#     # This output has the dimension of the groups and ints swapped
#     return np.swapaxes(cleaned_counts, 0, 1)


def nan_dqd(file, n_groups: int = None, dq_thresh: float = 0.0):
    # Get the bits
    dq = np.array(file["PIXELDQ"].data) > dq_thresh
    group_dq = np.array(file["GROUPDQ"].data) > 0
    electrons = np.array(file["SCI"].data)

    # Nan the bad bits
    full_dq = group_dq | dq[None, None, ...]

    if n_groups is None:
        return np.where(full_dq, np.nan, electrons)

    # for truncating the top of the ramp
    elif isinstance(n_groups, int):
        return np.where(full_dq[:, :n_groups], np.nan, electrons[:, :n_groups])

    # # Mask the invalid values and sigma clip
    # return onp.ma.masked_invalid(cleaned, copy=True)


# def group_fit(cleaned_ramps, lower_bound=False):

#     # Mask the invalid values, sigma clip, and set back to nans for jax
#     masked = onp.ma.masked_invalid(cleaned_ramps, copy=True)
#     masked_clipped = astropy.stats.sigma_clip(masked, axis=0, sigma=3)
#     cleaned = onp.ma.filled(masked_clipped, fill_value=np.nan)

#     # Get the support of the data - ie how many integrations contribute to the data
#     support = np.asarray(~np.isnan(cleaned), int).sum(0)

#     # Mean after sigma clipping - The 'robust mean', better for quantised data
#     ramp = np.nanmean(cleaned, axis=0)

#     # We dont want the error of the mean, we want the _STANDARD ERROR OF THE MEAN_,
#     # ie scaled by the sqrt of the number of samples
#     var = np.nanvar(cleaned, axis=0)
#     # if lower_bound:
#     #     var = np.maximum(var, np.nanmean(cleaned, axis=0))
#     var /= support

#     return ramp, var


def calc_mean_and_var(data, axis=0):
    # Get the support of the data - ie how many integrations contribute to the data
    support = np.asarray(~np.isnan(data), int).sum(axis)

    # Mean after sigma clipping - The 'robust mean', better for quantised data
    mean = np.nanmean(data, axis=axis)

    # We dont want the error of the mean, we want the _STANDARD ERROR OF THE MEAN_,
    # ie scaled by the sqrt of the number of samples
    var = np.nanvar(data, axis=axis)
    var /= support

    return mean, var


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
    sigma=0,
    chunk_size=0,
    n_groups=None,  # how many groups of the ramp to use
    dq_thresh=0.0,  # threshold value for the PIXELDQ flags
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
    files = [directory + f for f in os.listdir(directory) if f.endswith("_ramp.fits")]

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

    # Clear the existsing files (since we might use different chunk sizes, and we do not
    # want to have old files hang around)
    if clean_dir:
        print("Cleaning existing directory")
        delete_contents(output_path)

    # Iterate over files
    print("Running calslope processing...")
    for file_path in tqdm(files):
        file_name = file_path.split("/")[-1]
        file_root = "_".join(file_name.split("_")[:-2])

        file = fits.open(file_path)

        # Check if the file is a NIS_AMI file
        if file[0].header["EXP_TYPE"] != "NIS_AMI":
            print("Not a NIS_AMI file, skipping...")
            continue

        # Skip single group files
        if file[0].header["NGROUPS"] == 1:
            print("Only one group, skipping...")
            continue

        print(file[0].header["TARGPROP"], end=" ")
        print(file[0].header["NINTS"])

        # Get the data
        data = nan_dqd(file, dq_thresh=dq_thresh, n_groups=n_groups)
        file.close()

        if chunk_size == 0:
            chunks = [data]
        else:
            nints = data.shape[0]
            if nints < chunk_size:
                nchunks = 1
            else:
                nchunks = np.round(nints / chunk_size).astype(int)
            chunks = np.array_split(data, nchunks)

        print(f"Breaking into {nchunks} chunks")

        for i, chunk in enumerate(chunks):
            # Check if the file is a NIS_AMI file
            file_name = file_root + f"_{i+1:0{4}}" + "_nis_calslope.fits"
            file_calslope = os.path.join(output_path + file_name)

            # Create the new file
            shutil.copy(file_path, file_calslope)

            # Open new file
            file = fits.open(file_calslope, mode="update")

            # Remove the redundant extensions
            del file["GROUPDQ"]
            del file["ERR"]
            del file["GROUP"]
            del file["INT_TIMES"]

            # Update the various headers
            file[0].header["NCHUNKS"] = int(nchunks)
            file[0].header["CHUNK"] = i + 1
            file[0].header["CHUNKSZ"] = int(chunk_size)
            file[0].header["NINTS"] = int(chunk.shape[0])
            file[0].header["FILENAME"] = file_name
            file[0].header["SIGMA"] = sigma

            # Sigma clip the data, not the slopes. We sigma clip the data since the detector
            # can not distinguish between real signal and bias, so its value couples
            # through pixel non-linearities (ie pixel response, BFE)
            if sigma > 0:
                chunk = sigma_clip(chunk, sigma=sigma)

            # Get slopes
            slopes = np.diff(chunk, axis=1)
            slope, slope_var = calc_mean_and_var(slopes)

            # Zero-point - We may actually want to track this to feed into the
            # forwards model. The bias/zero point will couple into the BFE, and so exposures
            # the 'zero-point' of a dim exposure will be different to a bright exposure. As
            # such we need to track this. With this zero-point, we theoretically should be
            # able to fully re-build the data
            zero_point, zero_point_var = calc_mean_and_var(chunk[:, 0])

            # Save the data
            file["SCI"].data = slope

            # Save the variance as a separate extension
            header = fits.Header()
            header["EXTNAME"] = "SCI_VAR"
            file.append(fits.ImageHDU(data=slope_var, header=header))

            # Save the zero point as a separate extension
            header = fits.Header()
            header["EXTNAME"] = "ZPOINT"
            file.append(fits.ImageHDU(data=zero_point, header=header))

            # Save the zero point variance as a separate extension
            header = fits.Header()
            header["EXTNAME"] = "ZPOINT_VAR"
            file.append(fits.ImageHDU(data=zero_point_var, header=header))

            # Save as calslope
            file.writeto(file_calslope, overwrite=True)
            file.close()

    print("Done\n")
    return output_path
