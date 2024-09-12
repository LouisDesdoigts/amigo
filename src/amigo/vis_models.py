import equinox as eqx
import zodiax as zdx
import jax.numpy as np
import jax.tree_util as jtu
import dLux.utils as dlu
import interpax as ipx
from jax import vmap
from .misc import nearest_fn
from jax.scipy.signal import correlate


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
    return np.array(uv_coords)


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

    uv_pad: int = eqx.field(static=True)
    crop_size: int = eqx.field(static=True)
    per_wavelength: bool = eqx.field(static=True)
    bls: np.ndarray
    knots: np.ndarray
    bls_inds: np.ndarray
    hole_inds: np.ndarray
    bls_map: np.ndarray
    uv_coords: dict
    otf_masks: dict

    def __init__(
        self,
        optics,
        x_osamp=3,
        y_osamp=2,
        x_pad=2,
        y_pad=2,
        uv_pad=2,
        crop_size=160,
        per_wavelength=True,
    ):

        self.per_wavelength = per_wavelength
        self.uv_pad = uv_pad
        self.crop_size = crop_size

        # Get the baseline coordinates OTF coordinates
        cen_holes = optics.holes - optics.holes.mean(0)[None, :]
        bls, hole_inds = get_baselines_and_inds(cen_holes)
        self.bls = bls
        self.hole_inds = hole_inds

        # Define number of control points and add padding to control outer behaviour
        xs, ys = get_knot_coords(self.bls.max(), x_osamp, y_osamp, x_pad, y_pad)

        # TODO: Very dumb hack to be removed
        if x_pad == 2:
            xs = xs[1:-1]
        self.knots = np.array(np.meshgrid(xs, ys))

        is_near = vmap(nearest_fn, (0, None))(bls, self.knots)
        self.bls_map = np.sum(is_near.astype(int), 0)
        self.bls_inds = np.array(np.where(self.bls_map))

        pscale = dlu.arcsec2rad(optics.psf_pixel_scale / optics.oversample)
        npix_pad = self.uv_pad * optics.psf_npixels * optics.oversample

        uv_coords = {}
        for key, (wavels, weights) in optics.filters.items():
            if per_wavelength:
                coord_fn = lambda lam: get_uv_coords(lam, pscale, npix_pad, crop_size)
                vals = vmap(coord_fn)(wavels)
                uv_coords[key] = vals
            else:
                lam = get_mean_wavelength(wavels, weights)
                uv_coords[key] = get_uv_coords(lam, pscale, npix_pad, crop_size)

        self.uv_coords = uv_coords

        ###
        ### Calculate the _per baseline_ splodge masks
        ###

        def get_hole_mask(pt, mask, coords, k=100):
            inds = np.where(nearest_fn(pt, coords))
            i, j = inds[0][0], inds[1][0]
            sy, ey, sx, ex = i - k, i + k, j - k, j + k
            return mask.at[sy:ey, sx:ex].set(True)

        holes = optics.holes
        mask = optics.calc_mask(optics.wf_npixels, optics.diameter)
        coords = dlu.pixel_coords(optics.wf_npixels, optics.diameter)

        splodge_masks = {}
        for i in range(len(holes)):
            for j in range(len(holes)):
                if i == j:
                    continue
                pt1, pt2 = holes[i], holes[j]
                hole_mask = np.zeros_like(mask, bool)
                hole_mask = get_hole_mask(pt1, hole_mask, coords)
                hole_mask = get_hole_mask(pt2, hole_mask, coords)

                reduced_mask = mask * hole_mask
                corr = correlate(reduced_mask, reduced_mask, method="fft")
                corr /= corr.max()
                splodge_masks[(i, j)] = corr > 1e-3

        ###
        ### Calculate the OTF masks
        ###

        bls_coords = dlu.pixel_coords(2 * optics.wf_npixels, 2 * optics.diameter)

        def interp(knots, sample_coords, values):
            xs, ys = knots
            xpts, ypts = sample_coords.reshape(2, -1)

            return ipx.interp2d(
                ypts, xpts, ys[:, 0], xs[0], values, method="linear", extrap=True
            ).reshape(sample_coords[0].shape)

        def calculate_otf_mask(uv_coords):
            dsamp = np.ceil(bls_coords.shape[-1] / uv_coords.shape[-1])
            npix_out = int(crop_size * dsamp)

            ruv = uv_coords.shape[1] * (uv_coords[1, 1, 0] - uv_coords[1, 0, 0])
            rbls = bls_coords.shape[1] * (bls_coords[1, 1, 0] - bls_coords[1, 0, 0])

            sample_coords = dlu.pixel_coords(npix_out, ruv)
            knots = dlu.pixel_coords(len(splodge_masks[(0, 1)]), rbls)

            resample_fn = eqx.filter_jit(
                lambda mask: dlu.downsample(
                    np.where(interp(knots, sample_coords, mask) > 0.5, 1, 0), int(dsamp)
                )
            )

            full_mask = jtu.tree_map(resample_fn, splodge_masks)
            return np.array(jtu.tree_leaves(full_mask)).sum(0)

        otf_masks = {}
        for filt, uv_coords in self.uv_coords.items():

            if per_wavelength:
                otf_masks[filt] = vmap(calculate_otf_mask)(uv_coords)
            else:
                otf_masks[filt] = calculate_otf_mask(uv_coords)
        self.otf_masks = otf_masks

    def apply_vis(self, psfs, vis_pts, filter):
        otf_mask = self.otf_masks[filter]

        if self.per_wavelength:
            psf_fn = lambda psf, vis_map, otf_mask: apply_vis(psf, vis_map, otf_mask, self.uv_pad)
            vis_map = self.get_vis_map(vis_pts, filter)
            psf = vmap(psf_fn)(psfs, vis_map, otf_mask).sum(0)
        else:
            psf = apply_vis(psf.sum(0), vis_map, otf_mask, self.uv_pad)
        return psf

    def get_vis_map(self, vis_pts, filter):
        """Interpolates the visibility knots onto the UV coordinates."""
        uv_coords = self.uv_coords[filter]
        interp_fn = lambda im, coords: sample_spline(im, self.knots, coords)

        if self.per_wavelength:
            amp_map = vmap(interp_fn, (None, 0))(np.abs(vis_pts), uv_coords)
            phase_map = vmap(interp_fn, (None, 0))(np.angle(vis_pts), uv_coords)
        else:
            amp_map = interp_fn(np.abs(vis_pts), uv_coords)
            phase_map = interp_fn(np.angle(vis_pts), uv_coords)
        return np.maximum(amp_map, 0) * np.exp(1j * phase_map)


def apply_vis(psf, vis_map, otf_mask, uv_pad):
    npix = psf.shape[-1]
    npix_pad = uv_pad * psf.shape[-1]
    npix_crop = otf_mask.shape[-1]
    c, s = npix_pad // 2, npix_crop // 2

    full_splodges = to_uv(dlu.resize(psf, npix_pad))
    splodges = dlu.resize(full_splodges, npix_crop)
    masked = np.where(otf_mask > 0.5, splodges * vis_map, splodges)
    applied = full_splodges.at[c - s : c + s, c - s : c + s].set(masked)

    return dlu.resize(np.abs(from_uv(applied)), npix)
