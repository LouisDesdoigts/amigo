import jax
import jax.numpy as np
import dLux as dl
import dLux.utils as dlu


### Sub-propagations ###
def transfer(coords, npixels, wavelength, pscale, distance):
    """
    The optical transfer function (OTF) for the gaussian beam.
    Assumes propagation is along the axis.
    """
    scaling = npixels * pscale**2
    rho_sq = ((coords / scaling) ** 2).sum(0)
    return np.exp(-1.0j * np.pi * wavelength * distance * rho_sq)


def _fft(phasor):
    return 1 / phasor.shape[0] * np.fft.fft2(phasor)


def _ifft(phasor):
    return phasor.shape[0] * np.fft.ifft2(phasor)


def plane_to_plane(wf, distance):
    tf = transfer(wf.coordinates, wf.npixels, wf.wavelength, wf.pixel_scale, distance)
    phasor = _fft(wf.phasor)
    phasor *= np.fft.fftshift(tf)
    phasor = _ifft(phasor)
    return phasor


class FreeSpace(dl.layers.optics.OpticalLayer):
    distance: jax.Array

    def __init__(self, dist):
        self.distance = np.asarray(dist, float)

    def apply(self, wf):
        phasor_out = plane_to_plane(wf, self.distance)
        return wf.set(
            ["amplitude", "phase"], [np.abs(phasor_out), np.angle(phasor_out)]
        )


from nrm_analysis.misctools import mask_definitions


class DynamicAMI(dl.layers.optical_layers.OpticalLayer):
    holes: jax.Array
    f2f: jax.Array
    transformation: dl.CoordTransform
    normalise: bool

    def __init__(self, f2f=0.82, normalise=False):
        self.f2f = np.asarray(f2f, float)
        self.transformation = dl.CoordTransform((0.0, 0.0), 0.0, (1.0, 1.0), (0.0, 0.0))
        holes = mask_definitions.jwst_g7s6c()[1]
        self.holes = np.roll(holes, 1, -1)
        self.normalise = normalise

    def gen_AMI(self, npix, diameter):
        rmax = self.f2f / np.sqrt(3)
        coords = dlu.pixel_coords(npix, diameter)
        pscale = diameter / npix

        # Hard-coded offsets to match initial positioning to webbpsf
        sx = +21 * (diameter / npix)
        sy = -13 * (diameter / npix)
        coords = dlu.translate_coords(coords, np.array([sx, sy]))

        def hole_fn(hole):
            coords_in = dlu.translate_coords(coords, hole)
            coords_in = self.transformation.apply(coords_in)
            return dlu.soft_reg_polygon(coords_in, rmax, 6, pscale)

        return jax.vmap(hole_fn)(self.holes).sum(0)

    def apply(self, wavefront):
        wavefront = wavefront * self.gen_AMI(wavefront.npixels, wavefront.diameter)
        if self.normalise:
            return wavefront.normalise()
        return wavefront()

    def __getattr__(self, key):
        if hasattr(self.transformation, key):
            return getattr(self.transformation, key)
        else:
            raise AttributeError(f"Interpolator has no attribute {key}")


from jax.scipy.ndimage import map_coordinates


def arr2pix(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return pscale * (coords - shift)


def pix2arr(coords, pscale=1):
    n = coords.shape[-1]
    shift = (n - 1) / 2
    return (coords / pscale) + shift


class Interpolator(dl.layers.unified_layers.UnifiedLayer):
    transform: dl.CoordTransform
    order: int
    layer: None

    def __init__(self, layer, transform, order=1):
        self.transform = transform
        self.layer = layer
        self.order = int(order)

    # def __getattribute__(self, name):
    #     output = super().__getattribute__(name)
    #     if self._check(output):
    #         print("getattribute")
    #         return self.interpolate(output)
    #     return output

    def __getattr__(self, key):
        if hasattr(self.layer, key):
            # output = getattr(self.layer, key)
            # if self._check(output):
            #     print("getattr")
            #     return self.interpolate(output)
            # else:
            #     return output
            return getattr(self.layer, key)
        elif hasattr(self.transform, key):
            return getattr(self.transform, key)
        else:
            raise AttributeError(f"Interpolator has no attribute {key}")

    def _check(self, item):
        if (
            isinstance(item, jax.Array)  # Array check
            and item.ndim == 2  # 2D check
            and item.shape[0] == item.shape[1]  # Square check
        ):
            return True
        return False

    def orig_coords(self, arr):
        # Generate paraxial coords with pixel scale of 1
        return dlu.pixel_coords(arr.shape[0], arr.shape[0])

    def interp_coords(self, arr):
        # Apply the transformation
        return self.transform.apply(self.orig_coords(arr))

    def pix_coords(self, arr):
        # Convert from pixel to array coords
        coords = pix2arr(self.interp_coords(arr))

        # indexing convention swap: (x, y) -> (i, j)
        return np.array([coords[1], coords[0]])

    def interpolate(self, arr):
        return map_coordinates(arr, self.pix_coords(arr), 1)

    @property
    def transformed(self):
        fn = lambda leaf: self.interpolate(leaf) if self._check(leaf) else leaf
        return jax.tree_util.tree_map(fn, self.layer)

    def apply(self, wavefront):
        return self.transformed.apply(wavefront)


class Null(dl.layers.unified_layers.UnifiedLayer):
    def apply(self, inputs):
        return inputs
