import jax.numpy as np
from tqdm.notebook import tqdm
import dLux.utils as dlu
from jax import vmap


# Mask generation and baselines
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


def pairwise_vectors(points):
    """
    Generates a non-redundant list of the pairwise vectors connecting each point in an
    array of (x,y) points, ordered ascendingly by the length of the vector.

    Args:
        points (ndarray): An array of shape (n, 2) containing the (x,y) coordinates
        of the points.

    Returns:
        list: A list of tuples containing the pairwise vectors connecting each point,
        ordered ascendingly by the length of the vector.
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
    # # TODO: Replace with an argsort?
    # pairwise_vectors.sort(key=lambda x: lengths[x[1], x[2]])

    pairwise_vectors = np.array(pairwise_vectors)
    lengths_key = [lengths[x[1], x[2]] for x in pairwise_vectors]
    indices = np.argsort(lengths_key)

    # Now you can use these indices to sort pairwise_vectors and any other list in the same way
    sorted_pairwise_vectors = pairwise_vectors[indices]

    vecs = []
    # for vec in pairwise_vectors:
    for vec in sorted_pairwise_vectors:
        vecs.append(vec[0])
    return np.array(vecs)


def get_baselines_and_inds(holes):
    """Better version of pairwise_vectors that returns hole indicies too"""
    pairwise_vectors = []
    hole_inds = []
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            pairwise_vectors.append(holes[i] - holes[j])
            hole_inds.append((i, j))
    return np.array(pairwise_vectors), np.array(hole_inds)


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
    # return np.array([thisv, thisu]).T
    return np.array([thisu, thisv]).T


def uv_hex_mask(
    holes,
    f2f,
    tf,
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

    # Apply the mask transformations
    tf = tf.set("translation", np.zeros(2))  # Enforce paraxial splodges (since they are)
    uv_coords = tf.apply(uv_coords)

    # Do this outside so we can scatter plot the baseline vectors over the psf splodges
    # hbls = pairwise_vectors(holes) / wavelength
    hbls, inds = get_baselines_and_inds(holes)
    hbls /= wavelength

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


# Visibility modelling
def to_uv(psf):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))


def from_uv(uv):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uv)))


def splodge_mask(mask, vis):
    if len(vis) == 22:  # Includes DC term already
        coeffs = np.ones(2 * len(vis) - 1, complex)
        dc = np.array([vis[0]])
        coeffs = np.concatenate([dc, vis[1:], vis[1:].conj()])
    else:
        coeffs = np.ones(2 * len(vis) + 1, complex)
        coeffs = coeffs.at[1:].set(np.concatenate([vis, vis.conj()]))
    return dlu.eval_basis(mask, coeffs)


def apply_visibilities(psf, mask, vis):
    # Get splodge mask and inverse
    splodges = splodge_mask(mask, vis)
    inv_splodge_support = np.abs(1 - mask.sum(0))

    # We dont use np.where here because we have soft edges on the boundary of the mask
    return from_uv(to_uv(psf) * (splodges + inv_splodge_support))


# def apply_visibilities(psf, mask, vis):
#     # Get splodge mask and inverse
#     splodges = splodge_mask(mask, vis)
#     # inv_splodge_support = np.abs(1 - mask.sum(0))

#     # We dont use np.where here because we have soft edges on the boundary of the mask
#     return from_uv(to_uv(psf) * splodges)


def visibilities(amplitudes, phases):
    return amplitudes * np.exp(1j * phases)


def uv_model(vis, psfs, mask, cplx=False):
    # Get the sizes
    npix = psfs.shape[-1]
    npix_pad = mask.shape[-1]

    # Pad, apply the splodges, and cut
    psfs_pad = vmap(lambda x: dlu.resize(x, npix_pad))(psfs)
    cplx_psfs_pad = vmap(apply_visibilities, (0, 0, None))(psfs_pad, mask, vis)
    cplx_psfs = vmap(lambda x: dlu.resize(x, npix))(cplx_psfs_pad)

    # Return complex or magnitude
    if cplx:
        return cplx_psfs
    return np.abs(cplx_psfs)


def applied_splodges(masks, vis):
    return vmap(splodge_mask, (0, None))(masks, vis)


#
