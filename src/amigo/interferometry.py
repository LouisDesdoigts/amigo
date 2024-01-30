import jax.numpy as np
from tqdm.notebook import tqdm
import dLux.utils as dlu
from jax import vmap


def dsamp(arr, k):
    return arr.reshape(-1, k).mean(1)


def osamp_freqs(n, dx, osamp):
    df = 1 / (n * dx)
    odf = df / osamp

    if n % 2 == 0:
        start = -1 / (2 * dx)
        end = (n - 2) / (2 * dx * n)
    else:
        start = (1 - n) / (2 * n * dx)
        end = (n - 1) / (2 * n * dx)

    ostart = start + (odf - df) / 2
    oend = end + (df - odf) / 2
    return np.linspace(ostart, oend, n * osamp, endpoint=True)


# ns = [1, 2, 3, 4, 5, 6, 7]
# dxs = [1, 2, 3, 4, 5]
# ps = [1, 2, 3, 4, 5]

# TODO: Make this a test
# for n in ns:
#     for dx in dxs:
#         for p in ps:
#             base_freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
#             freqs = osamp_freqs(n, dx, p)
#             print(n, dx, np.isclose(base_freqs, dsamp(freqs, p)))


# import numpy as np


def pairwise_vectors(points):
    """
    Generates a non-redundant list of the pairwise vectors connecting each point in an array of (x,y) points,
    ordered ascendingly by the length of the vector.

    Args:
        points (ndarray): An array of shape (n, 2) containing the (x,y) coordinates of the points.

    Returns:
        list: A list of tuples containing the pairwise vectors connecting each point, ordered ascendingly by the length of the vector.
    """
    # Compute the pairwise vectors between each point
    vectors = points[:, np.newaxis] - points

    # Compute the lengths of the pairwise vectors
    lengths = np.sqrt(np.sum(vectors**2, axis=-1))

    # Create a list of non-redundant pairwise vectors
    pairwise_vectors = []
    for i in range(vectors.shape[0]):
        for j in range(i + 1, vectors.shape[1]):
            pairwise_vectors.append((vectors[i, j], i, j))

    # Sort the pairwise vectors by length
    # TODO: Replace with an argsort?
    pairwise_vectors.sort(key=lambda x: lengths[x[1], x[2]])
    # return pairwise_vectors

    vecs = []
    for vec in pairwise_vectors:
        v = vec[0]
        vecs.append([v[1], v[0]])
    return np.array(vecs)


# def hex_from_bls(coords, bl, rmax):
def hex_from_bls(bl, coords, rmax):
    # coords = dlu.translate_coords(coords, np.array([-bl[0], bl[1]]))
    coords = dlu.translate_coords(coords, np.array(bl))
    return dlu.reg_polygon(coords, rmax, 6)


def get_baselines(holes):
    # Get the baselines in m/wavelength (I do not know how this works)
    hole_mask = np.where(~np.eye(holes.shape[0], dtype=bool))
    thisu = (holes[:, 0, None] - holes[None, :, 0]).T[hole_mask]
    thisv = (holes[:, 1, None] - holes[None, :, 1]).T[hole_mask]
    return np.array([thisv, thisu]).T


def uv_hex_mask(
    holes,
    f2f,
    wavelength,
    psf_pscale,
    psf_npix,
    psf_oversample,
    uv_pad,
    mask_pad,
    verbose=False,
):
    """
    Holes: Hole positions, meters
    f2f: Hexagonal hole flat to flat distance, meters
    wavelength: Wavelength, meters
    psf_pscale: psf pixel scale, arcsec/pix
    psf_npix: psf npixels
    psf_oversample: oversampling of the psf
    uv_pad: padding before transforming to the uv plane
    mask_pad: mask calculation padding (ie mask oversample)
    """
    psf_npix *= psf_oversample

    # Correctly centred over sampled corods
    dx = dlu.arcsec2rad(psf_pscale) / psf_oversample
    shifted_coords = osamp_freqs(psf_npix * uv_pad, dx, mask_pad)
    uv_coords = np.array(np.meshgrid(shifted_coords, shifted_coords))

    # Do this outside so we can scatter plot the baseline vectors over the psf splodges
    hbls = pairwise_vectors(holes) / wavelength

    # Hole parameters
    rmax = f2f / np.sqrt(3)
    rmax_in = 2 * rmax / wavelength  # x2 because size doubles through a correlation

    # Get splodge masks and append DC term
    uv_hexes = []
    uv_hexes_conj = []

    # Baselines
    if verbose:
        looper = tqdm(hbls)
    else:
        looper = hbls

    for bl in looper:
        uv_hexes.append(hex_from_bls(bl, uv_coords, rmax_in))
        uv_hexes_conj.append(hex_from_bls(-1 * bl, uv_coords, rmax_in))
    uv_hexes = np.array(uv_hexes)
    uv_hexes_conj = np.array(uv_hexes_conj)

    dc_hex = np.array([hex_from_bls([0, 0], uv_coords, rmax_in)])

    hexes = np.concatenate([dc_hex, uv_hexes, uv_hexes_conj])

    # Normalise
    norm_hexes = dlu.nandiv(hexes, hexes.sum(0), 0.0)
    dsampler = vmap(lambda arr: dlu.downsample(arr, mask_pad))
    return dsampler(norm_hexes)


def get_AMI_splodge_mask(tel, wavelengths, calc_pad=2, pad=2, verbose=True, f2f=0.82):
    from nrm_analysis.misctools import mask_definitions

    # Take holes from ImPlaneIA
    holes = mask_definitions.jwst_g7s6c()[1]

    # Get values from telescope
    oversample = tel.oversample
    psf_npix = tel.psf_npixels
    pscale = tel.psf_pixel_scale

    if isinstance(wavelengths, float):
        wavelengths = [wavelengths]

    if verbose:
        looper = tqdm(wavelengths)
    else:
        looper = wavelengths

    # Now we calculate the masks
    masks = []
    for wl in looper:
        masks.append(
            uv_hex_mask(holes, f2f, wl, pscale, psf_npix, oversample, pad, calc_pad)
        )

    return np.squeeze(np.array(masks))
