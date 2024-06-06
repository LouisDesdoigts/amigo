import jax.numpy as np
from tqdm.notebook import tqdm
import dLux.utils as dlu
from jax import vmap, Array
import zodiax as zdx
from dLuxWebbpsf.basis import get_noll_indices


class UVHexikes(zdx.Base):
    basis: Array
    weight: Array
    support: Array
    inv_support: Array

    def __init__(self, basis, weight, support):
        self.basis = basis
        self.weight = weight
        self.support = support

        # calculating the inverse support mask
        self.inv_support = 1.0 - support


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


def hex_from_bls(bl, coords, rmax):
    # coords = dlu.translate_coords(coords, np.array([-bl[0], bl[1]]))
    coords = dlu.translate_coords(coords, np.array(bl))
    return dlu.reg_polygon(coords, rmax, 6)


def hexikes_from_bls(bl, coords, rmax, radial_orders=None, noll_indices=None):
    noll_indices = get_noll_indices(radial_orders, noll_indices)

    coords = dlu.translate_coords(coords, np.array(bl))
    hexagon = dlu.reg_polygon(coords, rmax, 6)
    return hexagon[None, ...] * dlu.zernike_basis(noll_indices, coords, 2 * rmax)


def get_baselines(holes):
    # Get the baselines in m/wavelength (I do not know how this works)
    hole_mask = np.where(~np.eye(holes.shape[0], dtype=bool))
    thisu = (holes[:, 0, None] - holes[None, :, 0]).T[hole_mask]
    thisv = (holes[:, 1, None] - holes[None, :, 1]).T[hole_mask]
    # return np.array([thisv, thisu]).T
    return np.array([thisu, thisv]).T


def build_hexikes(
    holes,
    f2f,
    tf,
    wavelength,
    psf_pscale,
    psf_npix,
    psf_oversample,
    uv_pad,
    mask_pad,
    radial_orders=None,
    noll_indices=None,
    crop_npix=None,
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

    # cropping
    if crop_npix is not None:
        uv_coords = vmap(dlu.resize, (0, None))(uv_coords, crop_npix)

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
    uv_hexikes = []
    uv_hexikes_conj = []

    # Baselines
    if verbose:
        looper = tqdm(hbls)
    else:
        looper = hbls
    for bl in looper:
        uv_hexikes.append(
            hexikes_from_bls(
                bl, uv_coords, rmax_in, radial_orders=radial_orders, noll_indices=noll_indices
            )
        )
        uv_hexikes_conj.append(
            hexikes_from_bls(
                -1 * bl, uv_coords, rmax_in, radial_orders=radial_orders, noll_indices=noll_indices
            )
        )  # Conjugate

    dc_hex = np.array(
        [
            hexikes_from_bls(
                [0, 0], uv_coords, rmax_in, radial_orders=radial_orders, noll_indices=noll_indices
            )
        ]
    )
    hexikes = np.concatenate([dc_hex, np.array(uv_hexikes), np.array(uv_hexikes_conj)])

    # Normalising
    weight_mask = hexikes.sum(0)[0]  # grabbing piston

    # reshaping to vmap over both wavelength and baselines
    vmapped_shapes = hexikes.shape[:2]
    npix_in = hexikes.shape[2:]
    dsampler = vmap(lambda arr: dlu.downsample(arr, mask_pad))  # vmap function
    hex_mask = dsampler(hexikes.reshape(-1, *npix_in))  # vmapping
    hex_mask = hex_mask.reshape(*vmapped_shapes, *hex_mask.shape[1:])  # reshaping back

    # downsampling weights and support
    weight_mask = dlu.downsample(weight_mask, mask_pad)
    support_mask = dlu.nandiv(hex_mask[:, 0], weight_mask[None, ...], 0.0).sum(0)

    return hex_mask, weight_mask, support_mask


# Visibility modelling
def to_uv(psf):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))


def from_uv(uv):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uv)))


def splodge_mask(basis, vis):
    n_vis = vis.shape[0]
    n_zernikes = vis.shape[1]

    if n_vis == 22:  # Includes DC term already
        dc = np.array([vis[0]])
        coeffs = np.concatenate([dc, vis[1:], vis[1:].conj()])
    elif n_vis == 21:
        dc = np.array([1] + (n_zernikes - 1) * [0], complex)[None]
        coeffs = np.concatenate([dc, vis, vis.conj()])

    return dlu.eval_basis(basis, coeffs)


def apply_visibilities(psf, vis, basis, weights, inv_support, pad_to=None):
    # normalise the basis
    basis = dlu.nandiv(basis, weights, 0.0)

    # zero padding to correct size
    if pad_to is not None:
        # padding normalised basis
        vmapped_shapes = basis.shape[:2]
        npix_in = basis.shape[2:]
        padder = vmap(lambda arr: dlu.resize(arr, pad_to))  # vmap function
        basis = padder(basis.reshape(-1, *npix_in))  # vmapping
        basis = basis.reshape(*vmapped_shapes, *basis.shape[1:])  # reshaping back

        # padding inverse support mask with ones
        inv_support = np.pad(
            inv_support,
            (pad_to - inv_support.shape[0]) // 2,
            constant_values=1.0,
        )

    # We dont use np.where here because we have soft edges on the boundary of the mask
    return from_uv(to_uv(psf) * (splodge_mask(basis, vis) + inv_support))


def visibilities(amplitudes, phases):
    return amplitudes * np.exp(1j * phases)


def uv_model(vis, psfs, hexikes, cplx=False, pad=2):
    # Get the sizes
    npix = psfs.shape[-1]
    npix_pad = pad * npix  # array size to pad mask and psfs to

    # unpacking from hexikes
    basis = hexikes.basis
    weights = hexikes.weight
    inv_support = hexikes.inv_support

    # Pad, apply the splodges, and crop
    psfs_pad = vmap(lambda x: dlu.resize(x, npix_pad))(psfs)
    vis_applyer = vmap(apply_visibilities, (0, None, 0, 0, 0, None))
    cplx_psfs_pad = vis_applyer(psfs_pad, vis, basis, weights, inv_support, npix_pad)
    cplx_psfs = vmap(lambda x: dlu.resize(x, npix))(cplx_psfs_pad)

    # Return complex or magnitude
    if cplx:
        return cplx_psfs
    return np.abs(cplx_psfs)


def applied_splodges(masks, vis):
    return vmap(splodge_mask, (0, None))(masks, vis)


#
