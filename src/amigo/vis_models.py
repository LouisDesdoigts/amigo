import zodiax as zdx
import jax.numpy as np
import interpax as ipx
from jax import vmap


# Mask generation and baselines
def osamp_freqs(n, dx, osamp=1):
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
    """Better version of pairwise_vectors that returns hole indices too"""
    pairwise_vectors = []
    hole_inds = []
    for i in range(len(holes)):
        for j in range(i + 1, len(holes)):
            pairwise_vectors.append(holes[i] - holes[j])
            hole_inds.append((i, j))
    return np.array(pairwise_vectors), np.array(hole_inds)


def get_knot_coords(r_max, x_osamp, y_osamp, x_pad, y_pad):
    # Minimum number of points required to sample the splodge centers
    rx_pts, ry_pts = 4, 7

    # Get all the annoying bits
    true_rx, true_ry = x_osamp * (rx_pts + x_pad), y_osamp * (ry_pts + y_pad)
    x_pts, y_pts = 2 * true_rx + 1, 2 * true_ry + 1
    dx, dy = 1 / (rx_pts * x_osamp), 1 / (ry_pts * y_osamp)
    dx_pad, dy_pad = (x_osamp * x_pad * dx), (y_osamp * y_pad * dy)

    # Generate the knot coordinates
    xs = r_max * np.linspace(-1 - dx_pad, 1 + dx_pad, x_pts)
    ys = r_max * np.linspace(-1 - dy_pad, 1 + dy_pad, y_pts)
    return xs, ys


def build_vis_pts(amp_vec, pha_vec, shape):
    vis_vec = amp_vec * np.exp(1j * pha_vec)
    dc = np.array([np.exp(0j)])
    return np.concatenate([vis_vec, dc, vis_vec.conj()[::-1]]).reshape(shape)


def get_mean_wavelength(wavels, weights):
    """Get the spectrally weighted mean wavelength"""
    return ((wavels * weights).sum() / weights.sum()).mean()


def get_uv_coords(wavel, pixel_scale, full_size, crop_size):
    """Assumes pixel scale is in radians"""
    dx = pixel_scale / wavel
    crop_to = lambda arr, npix: arr[(len(arr) - npix) // 2 : (len(arr) + npix) // 2]
    u_coords = crop_to(osamp_freqs(full_size, dx), crop_size)
    uv_coords = np.meshgrid(u_coords, u_coords)
    return uv_coords


def sample_spline(image, knots, sample_coords):
    xs, ys = knots
    xpts, ypts = sample_coords.reshape(2, -1)

    # NOTE: Extrapolation is used since the outer edges get cut with a hard edge
    # and there are some pixels in the support outside the edge points
    return ipx.interp2d(ypts, xpts, ys[:, 0], xs[0], image, method="cubic2", extrap=True).reshape(
        sample_coords[0].shape
    )


def to_uv(psf):
    return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))


def from_uv(uv):
    return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uv)))


class SplineVis(zdx.Base):
    """knots is (x, y) indexed."""

    bls: np.ndarray
    knots: np.ndarray
    bls_inds: np.ndarray
    hole_inds: np.ndarray
    bls_map: np.ndarray

    def __init__(self, optics, x_osamp=3, y_osamp=2, x_pad=1, y_pad=2):

        # Get the baseline coordinates OTF coordinates
        cen_holes = optics.holes - optics.holes.mean(0)[None, :]
        bls, hole_inds = get_baselines_and_inds(cen_holes)
        self.bls = bls
        self.hole_inds = hole_inds

        # Define number of control points and add padding to control outer behaviour
        xs, ys = get_knot_coords(bls.max(), x_osamp, y_osamp, x_pad, y_pad)
        self.knots = np.array(np.meshgrid(xs, ys))

        def nearest_fn(pt, coords):
            dist = np.hypot(*(coords - pt[:, None, None]))
            return dist == dist.min()

        is_near = vmap(nearest_fn, (0, None))(bls, self.knots)
        self.bls_map = np.sum(is_near.astype(int), 0)
        self.bls_inds = np.array(np.where(self.bls_map))
