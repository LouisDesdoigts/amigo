import zodiax as zdx
import jax.numpy as np
import interpax as ipx
from jax import vmap
from .interferometry import get_baselines_and_inds, osamp_freqs
from .files import calc_splodge_masks


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


class SplineVis(zdx.Base):
    """knots is (x, y) indexed."""

    bls: np.ndarray
    knots: np.ndarray
    bls_inds: np.ndarray
    hole_inds: np.ndarray
    bls_map: np.ndarray

    def __init__(self, optics, x_osamp=2, y_osamp=1, x_pad=1, y_pad=2):

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


class VisModel(zdx.Base):
    basis: np.ndarray
    weight: np.ndarray
    support: np.ndarray
    inv_support: np.ndarray

    def __init__(
        self,
        exposures,
        optics,
        uv_pad=2,
        calc_pad=3,
        crop_npix=None,
        radial_orders=None,
        hexike_cache="files/uv_hexikes",
        verbose=False,
        recalculate=False,
        nwavels=9,
    ):
        """
        Note caches masks to disk for faster loading. The cache is indexed _relative_ to
        where the file is run from.
        """
        bases, weights, supports, inv_supports = calc_splodge_masks(
            exposures,
            optics,
            uv_pad=uv_pad,
            calc_pad=calc_pad,
            crop_npix=crop_npix,
            radial_orders=np.arange(radial_orders),
            hexike_cache=hexike_cache,
            verbose=verbose,
            recalculate=recalculate,
        )

        self.basis = bases
        self.weight = weights
        self.support = supports
        self.inv_support = inv_supports
