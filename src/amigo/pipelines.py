import os
import shutil
import jax.numpy as np
import numpy as onp
from jax import vmap
from astropy.io import fits
from astropy.stats import sigma_clip
import pkg_resources as pkg
from .misc import tqdm
from .stats import build_cov


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
    sigma=3,
    chunk_size=0,
    n_groups=None,  # how many groups of the ramp to use NOTE Does nothing rn
    clean_dir=True,
    flat_field=False,
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
    if directory[-1] != "/":
        directory += "/"
    if output_dir[-1] != "/":
        output_dir += "/"

    # Get the files (flats have different extension)
    if flat_field:
        files = [directory + f for f in os.listdir(directory)]
    else:
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

        try:
            file = fits.open(file_path)  # , ignore_missing_simple=True)
        except OSError:
            print("Skipped")
            continue

        # Check if the file is a NIS_AMI file
        if file[0].header["EXP_TYPE"] not in ["NIS_LAMP", "NIS_AMI"]:
            print("Not a NIS_AMI or flat file, skipping...")
            continue

        # Skip single group files
        if file[0].header["NGROUPS"] == 1:
            print("Only one group, skipping...")
            continue

        # Get the data
        data = np.array(file["SCI"].data, int)

        # Get the filter and ngroups (for flat field files)
        filt = file["PRIMARY"].header["FILTER"]
        ngroups = file["PRIMARY"].header["NGROUPS"]

        # Copying header information
        sci_header = file["SCI"].header.copy(strip=True)
        sci_header.remove("EXTNAME")

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
            if flat_field:
                file_name = f"flat_{filt}_{ngroups}_nis_calslope.fits"
            else:
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
            file[0].header.extend(sci_header)  # Copy the science header

            if correct_ADC:
                file[0].header["ADC_CAL"] = True

            # Process the chunk
            if flat_field:
                file = process_flat(file, chunk)
            else:
                # file = process_data(file, chunk, sigma=sigma)
                file = process_data_new(file, chunk, sigma=sigma, correct_ADC=correct_ADC)

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


def rebuild_ramps(data, slopes):
    clean_ramp = np.cumsum(slopes, axis=1)
    return np.concatenate([data[:, :1], data[:, :1] + clean_ramp], axis=1)


def process_flat(file, data):
    """Processes the flat field data"""

    # Nan the insensitive edge pixels
    data = data.at[:, :, :5, :].set(np.nan)

    # Sigma clip the slopes
    slopes = np.diff(data, axis=1)
    slopes = apply_sigma_clip(slopes, axis=(0, 1, 2, 3), sigma=3)

    # Rebuild the data with the nan'd values cut upper ramps
    data = rebuild_ramps(data, slopes)

    # ngroups = data.shape[1]

    # Get the stats
    slopes = np.diff(data, axis=1)
    slope_supp = (~np.isnan(slopes)).astype(int).sum(0)
    min_slope_supp = slope_supp.min(0)

    # Nuke the pixels with low support
    slopes = np.where(min_slope_supp < 5, np.nan, slopes)
    data = rebuild_ramps(data, slopes)

    # Calculate the ramp mean and standard error
    ramp, ramp_err, ramp_support = calc_mean_and_std_cov(data, psd_safe=True)
    # ramp_err *= np.eye(ngroups)[..., None, None]
    # ramp, ramp_err, ramp_support = nancov_psd(data)

    # Calculate the slopes and sigma clip
    slope = np.diff(data, axis=1)

    # Calculate the slope mean and standard error
    # slope, slope_err, slope_support = calc_mean_and_std_cov(slope, psd_safe=True)
    slope, slope_err, slope_support = get_slope_stats(
        slope, var_lower_bound=True, lower_bound=0.95
    )
    # slope_err *= np.eye(ngroups - 1)[..., None, None]
    # slope, slope_err, slope_support = nancov_psd(slope)

    # Nuke the badpixels
    im = np.nanmean(np.nanmean(slopes, axis=0), axis=0)
    im = np.where(im == 0, np.nan, im)
    im = apply_sigma_clip(im, sigma=5, axis=(0, 1))
    badpix = np.isnan(im).astype(int)

    # Return the values
    return update_headers(
        file, ramp, ramp_err, ramp_support, slope, slope_err, slope_support, badpix
    )


def weighted_least_squares(x, y, C):
    """
    Perform a weighted least squares fit of a straight line y = a*x + b
    given data points (x, y) and covariance matrix C.

    Parameters:
    x : array-like, shape (N,)
        Independent variable.
    y : array-like, shape (N,)
        Dependent variable.
    C : array-like, shape (N, N)
        Covariance matrix of y values.

    Returns:
    a, b : tuple
        Best-fit slope and intercept.
    """
    C_inv = np.linalg.inv(C)  # Invert the covariance matrix

    # Design matrix
    X = np.vstack([x, np.ones_like(x)]).T  # Shape (N, 2)

    # Compute (X^T C^-1 X)^{-1} X^T C^-1 y
    beta = np.linalg.inv(X.T @ C_inv @ X) @ (X.T @ C_inv @ y)

    return beta


def slope_least_squares(slopes, cov):
    # slopes: (nslopes, npix, npix)
    # covariances: (nslopes, nslopes, npix, npix)

    nslopes = slopes.shape[0]
    npix = slopes.shape[-1]

    # The x-location of the slope measurement. We add 1.5 since the slope values are
    # measured at the _mid point_ of the groups
    # x_slopes = np.arange(nslopes) + 1.5
    x_slopes = np.arange(nslopes) + 1.0

    # Convert the data into a vector to make vmap easier
    slopes_vec = slopes.reshape(nslopes, -1)
    cov_vec = cov.reshape(nslopes, nslopes, -1)

    # Perform the fit over the pixel vectors
    fit_fn = lambda slope, cov: weighted_least_squares(x_slopes, slope, cov)
    lin_vec, const_vec = vmap(fit_fn, in_axes=-1, out_axes=-1)(slopes_vec, cov_vec)

    # Return the fit values back to an image
    linear = lin_vec.reshape(npix, npix)
    constant = const_vec.reshape(npix, npix)
    return linear, constant


def get_slope_stats(slopes, var_lower_bound=True, lower_bound=0.95):
    """
    Slope data has an expected covariance matrix structure that we can leverage to get
    an estimate of the variance that we can use as a lower-bound in order to correct
    biased variances arising from non-linear gain and low number statistics.

    Remember to add the 2 * read variance to the variance lower bound!
    """
    # Get the data shapes
    nints, nslopes = slopes.shape[:2]

    # Get the pixel support and mean
    support = np.asarray(~np.isnan(slopes), int).sum(axis=0)

    # Get the covariance matrix support - slightly more complex than it seems since the
    # the off diagonal terms are constructed from two different reads, which can both
    # have different support values. Here I simply take the mean support over both
    # reads, constructed in such a way to match the entries of the covariance matrix.
    cov_support = (support[None, ...] + support[:, None, ...]) / 2

    # Get the mean values
    mean = np.nanmean(slopes, axis=0)

    # Get the raw variance and paste into a diagonal covariance matrix
    var = np.nanvar(slopes, axis=0)

    # Get the read noise to populate the covariance matrix
    read_std = np.load(pkg.resource_filename(__name__, "data/SUB80_readnoise.npy"))

    # Build the covariance matrix
    cov = build_cov(var, read_std) / cov_support

    if var_lower_bound:
        # Now we do a linear least-squares fit to the slope data using the covariance
        # matrix. The constant term is then our _linear_ accumulation rate, which can be
        # used to set a lower bound on the variance
        linear, constant = slope_least_squares(mean, cov)

        # Constant is now our estimate of the linear per-pixel flux, which we can use for a
        # lower bound estimate. We apply 2x the read variance since each slope is measured
        # through two different reads (grp_i - grp_i+1)
        var_lb = lower_bound * ((2 * np.square(read_std)) + constant) / nints

        # correct the variance
        corrected_var = np.maximum(var, var_lb[None, ...])
        cov = build_cov(corrected_var, read_std) / cov_support
    return mean, cov, support


def calc_mean_and_std_cov(data, psd_safe=False):
    """
    do we want to set a lower-bound on the variance due to the non-linear gain

    Least squares linear fit to the slopes gives a quadratic
    """
    # Get the pixel support and mean
    support = np.asarray(~np.isnan(data), int).sum(axis=0)
    mean = np.nanmean(data, axis=0)
    # var =

    # Get the shapes we need
    npix = data.shape[2]
    ngroups = data.shape[1]

    # Calculate the ramp covariances
    group_data = np.swapaxes(data, 0, 1)
    group_vec = group_data.reshape(*group_data.shape[:2], -1)
    # data_cov_vec = vmap(np.cov, in_axes=-1, out_axes=-1)(group_vec)

    if psd_safe:
        data_cov_vec = vmap(nancov_psd, in_axes=-1, out_axes=-1)(group_vec)
    else:
        data_cov_vec = vmap(nancov, in_axes=-1, out_axes=-1)(group_vec)

    cov = data_cov_vec.reshape(ngroups, ngroups, npix, npix)

    # Return the bits
    return mean, cov, support


# def calc_mean_and_std_cov(data, psd_safe=False):
#     # Get the pixel support and mean
#     support = np.asarray(~np.isnan(data), int).sum(axis=0)
#     mean = np.nanmean(data, axis=0)

#     # Get the shapes we need
#     npix = data.shape[2]
#     ngroups = data.shape[1]

#     # Calculate the ramp covariances
#     group_data = np.swapaxes(data, 0, 1)
#     group_vec = group_data.reshape(*group_data.shape[:2], -1)
#     # data_cov_vec = vmap(np.cov, in_axes=-1, out_axes=-1)(group_vec)

#     if psd_safe:
#         data_cov_vec = vmap(nancov_psd, in_axes=-1, out_axes=-1)(group_vec)
#     else:
#         data_cov_vec = vmap(nancov, in_axes=-1, out_axes=-1)(group_vec)

#     cov = data_cov_vec.reshape(ngroups, ngroups, npix, npix)

#     # Return the bits
#     return mean, cov, support


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
    slope = np.diff(clipped_data, axis=1)
    clipped_slope = apply_sigma_clip(slope, sigma=sigma, axis=0)

    # Calculate the slope mean and standard error
    slope, slope_err, slope_support = calc_mean_and_standard_error(clipped_slope, axis=0)

    # Get the bad pixel array
    badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy")).astype(int)

    return update_headers(
        file, ramp, ramp_err, ramp_support, slope, slope_err, slope_support, badpix
    )


def nancov(X, eps=1e-6):
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


def nancov_psd(X):
    """Compute nan-aware covariance and ensure PSD."""
    cov = nancov(X, eps=0)  # Compute covariance without regularization
    return make_psd(cov)


def process_data_new(file, data, sigma=3.0, correct_ADC=True):
    """
    Processes the data and saves the outputs to the file

    Note we sigma clip the data first to catch vary large or small values after things
    like cosmic ray hits, etc. Then we take the slopes and sigma clip those to catch
    any outliers that might have been missed in the first pass. We then calculate the
    mean and standard error of the ramp and the slope.
    """

    # Sigma clip the slopes
    slopes = np.diff(data, axis=1)
    slopes = apply_sigma_clip(slopes, axis=0, sigma=sigma)

    # Rebuild the data with the nan'd values cut upper ramps
    data = rebuild_ramps(data, slopes)

    # ADC correction
    if correct_ADC:
        amp, period = 2, 1024
        data = data - amp * np.sin(2 * np.pi * np.nanmean(data, axis=0) / period)

    # Calculate the ramp support and covariance
    ramp, ramp_cov, ramp_support = calc_mean_and_std_cov(data)

    # Calculate the slope mean and standard error
    slopes = np.diff(data, axis=1)
    # slope, slope_cov, slope_support = calc_mean_and_std_cov(slopes)
    slope, slope_cov, slope_support = get_slope_stats(
        slopes, var_lower_bound=True, lower_bound=0.95
    )

    # Get the bad pixel array
    badpix = np.load(pkg.resource_filename(__name__, "data/badpix.npy")).astype(int)

    return update_headers(
        file, ramp, ramp_cov, ramp_support, slope, slope_cov, slope_support, badpix
    )


def update_headers(
    file,
    ramp,
    ramp_err,
    ramp_support,
    slope,
    slope_err,
    slope_support,
    badpix,
):
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
