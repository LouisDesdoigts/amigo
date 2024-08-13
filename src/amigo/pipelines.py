import os
import shutil
import jax.numpy as np
from tqdm.notebook import tqdm
from astropy.io import fits
from .misc import apply_sigma_clip, calc_mean_and_std_var


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

        # Get the data
        data = np.array(file["SCI"].data)
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
                chunk = apply_sigma_clip(chunk, sigma=sigma)

            # Get slopes
            slopes = np.diff(chunk, axis=1)
            slope, slope_var = calc_mean_and_std_var(slopes)

            # Zero-point - We may actually want to track this to feed into the
            # forwards model. The bias/zero point will couple into the BFE, and so exposures
            # the 'zero-point' of a dim exposure will be different to a bright exposure. As
            # such we need to track this. With this zero-point, we theoretically should be
            # able to fully re-build the data
            zero_point, zero_point_var = calc_mean_and_std_var(chunk[:, 0])

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
