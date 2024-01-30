import jax
import jax.numpy as np
from jax import vmap
import dLux as dl
import dLux.utils as dlu
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

        # Shift the coordinates for each hole
        shifted_coords = vmap(dlu.translate_coords, (None, 0))(coords, self.holes)

        # Rotate, shear and compress coordinates
        transformed_coords = vmap(self.transformation.apply)(shifted_coords)

        # Generate the hexagons
        hex_fn = lambda coords: dlu.soft_reg_polygon(coords, rmax, 6, pscale)
        return vmap(hex_fn)(transformed_coords).sum(0)

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
