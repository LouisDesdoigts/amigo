import os
import jax.numpy as np
from jax import vmap
import numpy as onp
import astropy
from astropy.io import fits
from jwst.pipeline import Detector1Pipeline


def process_stage0(directory, output_dir="stage1/", verbose=False):
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
        if file[0].header["EXP_TYPE"] != "NIS_AMI":
            if verbose:
                print("Not a NIS_AMI file, skipping...")
            continue

        # Run stage 1
        pl1 = Detector1Pipeline()
        pl1.output_dir = str(output_path)

        pl1.save_results = True  # what does this do?
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

    if verbose:
        print("Done\n")

    return output_path


def im2vec(im):
    return im.reshape(len(im), -1).T


def vec2im(vec, npix):
    return vec.T.reshape(npix, npix)


def least_sq(x, y):
    A = np.vstack([x, np.ones(len(x))]).T
    m, b = np.linalg.lstsq(A, y, rcond=None)[0]
    return m, b


def slope_im(im, groups):
    npix = im.shape[-1]
    ms, bs = vmap(least_sq, (None, 0))(groups, im2vec(im))
    return vec2im(ms, npix), vec2im(bs, npix)


def estimate_biases(data, ngroups=2):
    groups = np.arange(1, 1 + ngroups)
    first_groups = data[:, :ngroups]
    slopes, biases = vmap(slope_im, (0, None))(first_groups, groups)
    return biases


def subtract_bias(data):
    groups = np.arange(1, 3)
    first_groups = data[:, :2]
    slopes, biases = vmap(slope_im, (0, None))(first_groups, groups)
    return data - biases[:, None, :, :], biases

    # def clean_slopes(bias_cleaned_data):
    #     nints, ngroups, npix = bias_cleaned_data.shape[:3]

    #     full_zero_ramp = np.zeros((nints, ngroups + 1, npix, npix))
    #     full_ramp = full_zero_ramp.at[:, 1:].set(bias_cleaned_data)

    #     cleaned_slopes = []
    #     for k in range(ngroups):
    #         group_vals = full_ramp[:, k : k + 2]
    #         groups = np.arange(k, k + 2)

    #         slopes, biases = vmap(slope_im, (0, None))(group_vals, groups)
    #         clipped = astropy.stats.sigma_clip(slopes, axis=0, sigma=3)
    #         cleaned_slopes.append(onp.ma.filled(clipped, fill_value=onp.nan))

    #     # This output has the dimension of the groups and ints swapped
    #     return np.swapaxes(np.array(cleaned_slopes), 0, 1)


def sigma_clip(array, sigma=5.0):
    masked = onp.ma.masked_invalid(array, copy=True)
    clipped = astropy.stats.sigma_clip(masked, axis=0, sigma=sigma)
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


def nan_dqd(file):
    # Get the bits
    dq = np.array(file["PIXELDQ"].data) > 0
    group_dq = np.array(file["GROUPDQ"].data) > 0
    electrons = np.array(file["SCI"].data)

    # Nan the bad bits
    full_dq = group_dq | dq[None, None, ...]
    return np.where(full_dq, np.nan, electrons)
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


def calc_mean_and_var(data):
    # # Mask the invalid values, sigma clip, and set back to nans for jax
    # masked = onp.ma.masked_invalid(data, copy=True)
    # masked_clipped = astropy.stats.sigma_clip(masked, axis=0, sigma=sigma)
    # cleaned = onp.ma.filled(masked_clipped, fill_value=np.nan)
    # cleaned = sigma_clip(data, sigma=sigma)

    # Get the support of the data - ie how many integrations contribute to the data
    support = np.asarray(~np.isnan(data), int).sum(0)

    # Mean after sigma clipping - The 'robust mean', better for quantised data
    mean = np.nanmean(data, axis=0)

    # We dont want the error of the mean, we want the _STANDARD ERROR OF THE MEAN_,
    # ie scaled by the sqrt of the number of samples
    var = np.nanvar(data, axis=0)
    var /= support

    return mean, var


def bias_careful_mean_and_var(data, bias):

    # Get the support of the data - ie how many integrations contribute to the data
    support = np.asarray(~np.isnan(data), int).sum(0)

    # Now subtract bias to avoid skewing the variances
    data -= bias[None, None, :, :]

    # Mean after sigma clipping - The 'robust mean', better for quantised data
    mean = np.nanmean(data, axis=0)

    # # Now subtract bias to avoid skewing the variances
    # mean -= bias[None, :, :]

    # Take variance from the non-bias subtracted data
    var = np.nanvar(data, axis=0)
    var /= support

    return mean, var


def process_stage1(directory, output_dir="calgrps/", sigma=0, method=0):
    # def process_stage1(directory, output_dir="calgrps/", refpix_correction=0):
    """
        ref_pix_correction: int
            What reference pixel correction to apply. 0 for none, 1 for first, 2 for second.
            'first' subtracts off the reference pixel value for _each group and
            integration_, which is then masked and sigma clipped. 'second' performs the
            masking and sigma clipping on both the data and reference pixel first, and then
            subtracts the single reference pixel value for each group and column from the
            cleaned data.

    lower_bound: bool
        Assuming we have correctly calibrated the ramp zero point (ie bias), the
        minimum value the variance should be able to take is:
            (psf var + read var) / nints
        However, due to the sigma clipping in the pipeline, when we have small nints,
        this can result in variances that are much smaller than this theoretical
        minimum value. To correct for this, we can set a lower bound to be the minimum
        value of the mean of the data, and the theoretical minimum value.
    """
    # if refpix_correction not in [0, 1, 2]:
    #     raise ValueError("ref_pix_correction must be 0, 1, or 2.")
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
    paths = files[0].split("/")
    base_path = "/".join(paths[:-2]) + "/"
    output_path = base_path + output_dir

    # Check whether the specified output directory exists
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    # Iterate over files
    print("Running calgrps processing...")
    for file_path in files:
        file_name = file_path.split("/")[-1]
        file_root = "_".join(file_name.split("_")[:-2])

        # Check if the file is a NIS_AMI file
        final_output_path = output_path + file_root + "_nis_calgrps.fits"
        if os.path.exists(final_output_path):
            print("File already exists, skipping...")
            continue
        file_calgrps = os.path.join(final_output_path)

        # Create the new file
        import shutil

        shutil.copy(file_path, file_calgrps)

        # Check if the file is a NIS_AMI file
        file = fits.open(file_calgrps, mode="update")
        if file[0].header["EXP_TYPE"] != "NIS_AMI":
            print("Not a NIS_AMI file, skipping...")
            continue

        # Get the data with the DQ flags nan'd
        data = nan_dqd(file)
        # print(np.sum(np.isnan(raw_data)) / raw_data.size)

        if data.shape[1] == 1:
            print("Only one group, skipping...")
            continue

        # # Subtract the bias estimate from the data
        # # (I think this is somewhat redundant if we are only working from slopes)
        # # Actually, this affects the lower_bound flag, as the lower bound on the
        # # variance is the mean of the data
        # bias_subtracted_data, biases = subtract_bias(cleaned_data)

        # # # Sigma clip the biases
        # # cleaned_biases = clean_biases(biases)

        # # Estimate the bias and error
        # biases, bias_var = calc_mean_and_var(biases, sigma=sigma)

        # # # Sigma clip the slopes
        # # cleaned_slopes = clean_slopes(bias_subtracted_data)

        # # # Rebuild the 'clean' ramps
        # # cleaned_ramps = rebuild_ramps(cleaned_slopes)

        # # Estimate the ramp and error
        # ramp, ramp_var = calc_mean_and_var(bias_subtracted_data, sigma=sigma)

        """
        OLD METHOD ABOVE

        NEW PLAN

        Dont subtract the biases at the start, subtract them at the end!

        Take the means and variances of each pixel group value
        Take the bias as the slope fit of the resulting first two groups
        Subtract the bias from the data
        """
        # # Get data and subtract the bias
        # # TODO: Should I sigma clip at the outset here too?
        # biases = estimate_biases(data)
        # biases = sigma_clip(biases, sigma=sigma)
        # bias, bias_var = calc_mean_and_var(biases)

        # # Subtract mean from data
        # # data = data - biases[:, None, :, :]
        # data = sigma_clip(data, sigma=sigma)
        # ramp, ramp_var = bias_careful_mean_and_var(data, bias)

        # data, biases = subtract_bias(raw_data)
        # print(np.sum(np.isnan(data)) / data.size)

        # print(biases.shape)
        # print(data.shape)
        # inds = np.where(np.isnan(biases))
        # data = data.at[inds[0], :, *inds[1:]].set(np.nan)
        # print(np.sum(np.isnan(data)) / data.size)

        # Estimate ramp, bias and variances
        # print(np.sum(np.isnan(ramp)) / ramp.size)

        # slope, bias = slope_im(ramp[:2], np.arange(1, 3))

        # # Estimate bias variance
        # _, biases = vmap(slope_im, (0, None))(data[:, :2], np.arange(1, 3))

        # # Subtract the bias from the data
        # ramp -= bias

        # def calc_slopes(data):
        # nints, ngroups, npix = data.shape[:3]

        if method == 0:
            slopes = np.diff(data, axis=1)
            if sigma > 0:
                slopes = sigma_clip(slopes, sigma=sigma)
            slope, slope_var = calc_mean_and_var(slopes)
        elif method == 1:

            slopes = np.diff(data, axis=1)
            if sigma > 0:
                slopes = sigma_clip(slopes, sigma=sigma)
            n = len(slopes) // 2
            data_first_half = slopes[:n]
            data_second_half = slopes[n:]
            slope1, slope1_var = calc_mean_and_var(data_first_half)
            slope2, slope2_var = calc_mean_and_var(data_second_half)

            # Slope of slope (lol)
            from .misc import slope_im

            ms, bs = slope_im(slope1)
            xs = np.arange(len(slope1)) + 1
            ys = ms * xs[:, None, None] + bs
            clean_slope2 = slope2 - slope1 + ys

            # Slope of slope (lol)
            ms, bs = slope_im(slope2)
            xs = np.arange(len(slope2)) + 1
            ys = ms * xs[:, None, None] + bs
            clean_slope1 = slope1 - slope2 + ys

            slope = (clean_slope1 + clean_slope2) / 2
            slope_var = (slope1_var + slope2_var) / 2

            print("Slopes are cleaned")

        elif method == 2:
            from .misc import slope_im

            # ngroups = data.shape[1]
            # nslopes = ngroups - 1

            # slopes = np.diff(data, axis=1)
            # ms, bs = slope_im(slopes)
            # if sigma > 0:
            #     ms = sigma_clip(ms, sigma=sigma)
            #     bs = sigma_clip(bs, sigma=sigma)
            # m, _ = calc_mean_and_var(ms)
            # b, _ = calc_mean_and_var(bs)

            # xs = np.arange(nslopes) + 1
            # slope = m * xs[:, None, None] + b
            # _, slope_var = calc_mean_and_var(slopes)

            nints = data.shape[0]
            ngroups = data.shape[1]
            nslopes = ngroups - 1

            slopes = np.diff(data, axis=1)
            if sigma > 0:
                slopes = sigma_clip(slopes, sigma=sigma)
            _, slope_var = calc_mean_and_var(slopes)

            ms, bs = vmap(slope_im)(slopes)
            m, _ = calc_mean_and_var(ms)
            b, _ = calc_mean_and_var(bs)

            xs = np.arange(nslopes) + 1
            slope = m * xs[:, None, None] + b

            # Slight magic numbers here
            nan_mask = np.where(slope_var * nints > (2.0 * slope + 14**2))
            slope = slope.at[nan_mask].set(np.nan)
            slope_var = slope_var.at[nan_mask].set(np.nan)

        elif method == 3:

            clean_data = sigma_clip(data, sigma=sigma)
            ramp, ramp_var = calc_mean_and_var(clean_data)
            slope = np.diff(ramp, axis=0)

            nslope = slope.shape[0]
            slope_var = np.array([ramp_var[i] + ramp_var[i + 1] for i in range(nslope)])
        else:
            raise ValueError("Invalid method")

        # full_zero_ramp = np.zeros((nints, ngroups + 1, npix, npix))
        # full_ramp = full_zero_ramp.at[:, 1:].set(data)

        # cleaned_slopes = []
        # for k in range(ngroups):
        #     group_vals = full_ramp[:, k : k + 2]
        #     groups = np.arange(k, k + 2)

        #     slopes, biases = vmap(slope_im, (0, None))(group_vals, groups)
        #     clipped = astropy.stats.sigma_clip(slopes, axis=0, sigma=3)
        #     cleaned_slopes.append(onp.ma.filled(clipped, fill_value=onp.nan))

        # # This output has the dimension of the groups and ints swapped
        # return np.swapaxes(np.array(cleaned_slopes), 0, 1)

        # if lower_bound:

        #     # Load the read noise
        #     import pkg_resources as pkg

        #     file_path = pkg.resource_filename(__name__, "data/SUB80_readnoise.npy")
        #     read_var = np.load(file_path)[None, ...] ** 2

        #     # Get the number of integrations
        #     nints = data.shape[0]

        #     # Get the minimum variance
        #     min_var = (ramp + read_var) / nints

        #     bad_vars = ramp_var < min_var
        #     ramp_var = np.where(bad_vars, min_var, ramp_var)
        #     nbad = bad_vars.sum()
        #     print(
        #         f"Corrected {nbad} variances to the minimum value, ~{100 * nbad / ramp_var.size:.2f}% of all pixel reads."
        #     )

        # Write to file
        # print(np.sum(np.isnan(ramp)) / ramp.size)
        # print(np.nansum(ramp))
        # print(np.nansum(ramp_var))
        # print()
        # file["SCI"].data = ramp
        file["SCI"].data = slope

        # Save the biases as a separate extension
        header = fits.Header()
        header["EXTNAME"] = "SCI_VAR"
        # file.append(fits.ImageHDU(data=ramp_var, header=header))
        file.append(fits.ImageHDU(data=slope_var, header=header))

        # # Save the biases as a separate extension
        # header = fits.Header()
        # header["EXTNAME"] = "BIAS"
        # file.append(fits.ImageHDU(data=bias, header=header))

        # # Save the bias variance as a separate extension
        # header = fits.Header()
        # header["EXTNAME"] = "BIAS_VAR"
        # file.append(fits.ImageHDU(data=bias_var, header=header))

        # Save as calgrp
        file.writeto(file_calgrps, overwrite=True)
        file.close()

    print("Done\n")
    return output_path
