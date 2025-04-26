import os
import shutil
import jax.numpy as np
import numpy as onp
from jax import vmap
from astropy.io import fits
from astropy.stats import sigma_clip
import pkg_resources as pkg
from .misc import tqdm


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
    input_dir,
    output_dir="calslope/",
    sigma=3.0,
    # chunk_size=0,
    # n_groups=None,  # how many groups of the ramp to use NOTE Does nothing rn
    clean_dir=True,
    flat=False,
    correct_ADC=True,
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
    if input_dir[-1] != "/":
        input_dir += "/"
    if output_dir[-1] != "/":
        output_dir += "/"

    # Get the files (flats have different extension)
    if flat:
        files = [input_dir + f for f in os.listdir(input_dir)]
    else:
        files = [input_dir + f for f in os.listdir(input_dir) if f.endswith("_uncal.fits")]

    # Check if there are any files to process
    if len(files) == 0:
        print("No _ramp.fits files found, no processing done.")
        return

    # Check whether the specified output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Clear the existing files (since we might use different chunk sizes, and we do not
    # want to have old files hang around)
    if clean_dir:
        print("Cleaning existing directory")
        delete_contents(output_dir)

    # Iterate over files
    print("Running calslope processing...")
    for file_path in tqdm(files):

        try:
            file = fits.open(file_path)
        except OSError as e:
            print(f"Skipped, reason: {e}")
            continue

        # Check if the file is a NIS_AMI file
        if file[0].header["EXP_TYPE"] not in ["NIS_LAMP", "NIS_AMI"]:
            print("Not a NIS_AMI or flat file, skipping...")
            continue

        # Skip single group files
        if file[0].header["NGROUPS"] == 1:
            print("Only one group, skipping...")
            continue

        # Get the data (cast to float so we can nan the bad pixels)
        data = np.array(file["SCI"].data, float)

        # Get the filter and ngroups (for flat field files)
        filt = file["PRIMARY"].header["FILTER"]
        ngroups = file["PRIMARY"].header["NGROUPS"]

        # Copying header information
        sci_header = file["SCI"].header.copy(strip=True)
        sci_header.remove("EXTNAME")

        # Close the file
        file.close()

        # Get the root of the file name
        file_name = file_path.split("/")[-1]
        file_root = "_".join(file_name.split("_")[:-2])

        # Check if the file is a NIS_AMI file
        if flat:
            file_name = f"flat_{filt}_{ngroups}_nis_calslope.fits"
        else:
            file_name = file_root + "_nis_calslope.fits"
        file_calslope = os.path.join(output_dir + file_name)

        # Create the new file
        shutil.copy(file_path, file_calslope)

        # Open new file and process the data
        file = fits.open(file_calslope)
        file = process_data(file, data, sigma=sigma, correct_ADC=correct_ADC, flat=flat)

        # Remove the redundant or undesired extensions
        del file["SCI"]
        del file["GROUP"]
        del file["INT_TIMES"]

        # Update the various headers
        file[0].header["NINTS"] = int(data.shape[0])
        file[0].header["FILENAME"] = file_name
        file[0].header["SIGMA"] = sigma
        file[0].header["ADC_CAL"] = correct_ADC
        file[0].header.extend(sci_header)  # Copy the science header

        # Save as calslope
        file.writeto(file_calslope, overwrite=True)
        file.close()

    print("Done\n")


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


def rebuild_ramps(data, slopes):
    clean_ramp = np.cumsum(slopes, axis=1)
    return np.concatenate([data[:, :1], data[:, :1] + clean_ramp], axis=1)


def process_data(file, data, sigma=3.0, correct_ADC=True, flat=False):
    """
    Processes the data and saves the outputs to the file

    Note we sigma clip the data first to catch vary large or small values after things
    like cosmic ray hits, etc. Then we take the slopes and sigma clip those to catch
    any outliers that might have been missed in the first pass. We then calculate the
    mean and standard error of the ramp and the slope.
    """
    # ADC correction
    if correct_ADC:
        amp, period = 2, 1024
        data = data - amp * np.sin(2 * np.pi * np.nanmean(data, axis=0) / period)

    badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy"))
    # badpix = badpix.at[:5, :].set(True)
    data = data.at[:, :, badpix].set(np.nan)

    if flat:
        slopes = np.diff(data, axis=1)
        slopes = apply_sigma_clip(slopes, axis=(0, 1, 2, 3), sigma=sigma)
        data = rebuild_ramps(data, slopes)
    else:
        data = apply_sigma_clip(data, axis=0, sigma=sigma)

    # Sigma clip the slopes
    slopes = np.diff(data, axis=1)
    slopes = apply_sigma_clip(slopes, axis=0, sigma=sigma)

    # Rebuild the data with the nan'd values cut upper ramps
    data = rebuild_ramps(data, slopes)
    slopes = np.diff(data, axis=1)

    # Return the values
    return update_headers(file, slopes)


def calc_mean_and_cov(data):
    # Get the pixel support and mean
    support = np.asarray(~np.isnan(data), int).sum(axis=0)
    mean = np.nanmean(data, axis=0)

    # Get the shapes we need
    npix = data.shape[2]
    ngroups = data.shape[1]

    # Calculate the covariance matrix
    group_data = np.swapaxes(data, 0, 1)
    group_vec = group_data.reshape(*group_data.shape[:2], -1)
    data_cov_vec = vmap(nancov, in_axes=-1, out_axes=-1)(group_vec)
    cov = data_cov_vec.reshape(ngroups, ngroups, npix, npix)

    # Return the bits
    return mean, cov, support


def nancov(X):
    """Compute nan-aware covariance and ensure PSD."""
    cov = calc_nancov(X, eps=0)  # Compute covariance without regularization
    return make_psd(cov, eps=1e-6)  # Ensure positive semi-definite


def calc_nancov(X, eps=1e-6):
    """Compute covariance while ignoring NaNs."""
    mask = np.isnan(X)
    valid_counts = np.sum(~mask, axis=1, keepdims=True)
    mean = np.nansum(X, axis=1, keepdims=True) / valid_counts
    X_centered = np.where(mask, 0, X - mean)
    cov_matrix = (X_centered @ X_centered.T) / (valid_counts @ valid_counts.T - 1)
    return cov_matrix + eps * np.eye(cov_matrix.shape[0])


def make_psd(A, eps=1e-6):
    """Ensure matrix A is positive semi-definite by symmetrizing and clipping small eigenvalues."""
    B = (A + A.T) / 2  # Force symmetry
    eigvals, eigvecs = np.linalg.eigh(B)  # Get eigenvalues & eigenvectors

    eigvals = np.clip(eigvals, eps, None)  # Ensure all eigenvalues are >= eps
    return eigvecs @ np.diag(eigvals) @ eigvecs.T  # Reconstruct matrix


def update_headers(
    file,
    data,
    slopes,
    # ramp,
    # ramp_cov,
    # ramp_support,
    # slope,
    # slope_cov,
    # slope_support,
    # badpix,
):

    badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy"))

    # Calculate the ramp mean and covariance
    ramp, ramp_cov, ramp_support = calc_mean_and_cov(data)

    # Calculate the slope mean and covariance
    slopes, slope_cov, slope_support = calc_mean_and_cov(slopes)

    # Save the Outputs
    header = fits.Header()
    header["EXTNAME"] = "RAMP"
    file.append(fits.ImageHDU(data=ramp, header=header))

    header = fits.Header()
    header["EXTNAME"] = "RAMP_COV"
    file.append(fits.ImageHDU(data=ramp_cov, header=header))

    header = fits.Header()
    header["EXTNAME"] = "RAMP_SUP"
    file.append(fits.ImageHDU(data=ramp_support, header=header))

    header = fits.Header()
    header["EXTNAME"] = "SLOPE"
    file.append(fits.ImageHDU(data=slopes, header=header))

    header = fits.Header()
    header["EXTNAME"] = "SLOPE_COV"
    file.append(fits.ImageHDU(data=slope_cov, header=header))

    header = fits.Header()
    header["EXTNAME"] = "SLOPE_SUP"
    file.append(fits.ImageHDU(data=slope_support, header=header))

    header = fits.Header()
    header["EXTNAME"] = "BADPIX"
    file.append(fits.ImageHDU(data=badpix, header=header))

    # Move the ASDF extention to the end
    file.append(file.pop("ASDF"))

    return file
