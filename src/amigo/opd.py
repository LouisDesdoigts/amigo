"""Named OPD components evaluated on fixed aperture windows.

All components resolve to (hole, row, column) arrays in metres. Geometry,
constraint construction and filter/visit sharing belong to the caller; evaluating
an OPD neither rebuilds its basis nor removes any modes.
"""

import jax.numpy as np
from jax import Array, vmap
import dLux.utils as dlu
import zodiax as zdx


def sum_opds(opds, shape):
    """Resolve named components on the aperture's fixed windows, in metres."""
    total = np.zeros(shape)
    for component in (opds or {}).values():
        stamps = np.asarray(zdx.resolve(component))
        if stamps.shape != shape:
            raise ValueError(f"OPD components must match aperture windows {shape}")
        total = total + stamps
    return total


class FourierOPD(zdx.Transform):
    """Per-hole Fourier coefficients evaluated using dLux's cached kernels.

    ``x`` is an array or bound Map resolving to (hole, mode_x, mode_y), in metres.
    ``npix`` and ``n_modes`` follow dLux's (x, y) order. Output stamps follow image
    (row, column) order. The Fourier period spans the fixed window on each axis;
    changing window size therefore changes the physical basis.
    """

    kx: Array
    ky: Array

    def __init__(self, n_modes, npix, x=None, *, alias=None):
        self.x = x
        self.alias = alias
        self.kx, self.ky = dlu.fourier_kernels(n_modes, npix)

    def fwd(self, x, **context):
        if x.ndim != 3 or x.shape[1:] != (self.kx.shape[1], self.ky.shape[1]):
            raise ValueError("Fourier coefficients must have shape (hole, mode_x, mode_y)")

        # dLux evaluates in (x, y); aperture stamps are indexed (row, column).
        stamps = vmap(dlu.eval_fourier_basis, (0, None, None))(x, self.kx, self.ky)
        return np.swapaxes(stamps, -1, -2)


class BasisOPD(zdx.Transform):
    """Evaluate independent per-hole coefficients on sampled local bases.

    ``basis`` has shape (hole, mode, row, column); ``x`` resolves to (hole, mode).
    A bound Map in ``x`` can couple holes or remove redundant coordinates. This
    contraction avoids storing a mostly-zero block-diagonal sampled basis.
    """

    basis: Array

    def __init__(self, basis, x=None, *, alias=None):
        self.basis = np.asarray(basis)
        if self.basis.ndim != 4:
            raise ValueError("Local bases must have shape (hole, mode, row, column)")
        self.x = x
        self.alias = alias

    def fwd(self, x, **context):
        if x.shape != self.basis.shape[:2]:
            raise ValueError("Local coefficients must have shape (hole, mode)")
        return np.einsum("hm,hmij->hij", x, self.basis)


def zernike_opd(coords, indices, diameter, x=None, *, per_hole=False, polike=False):
    """Build sampled Zernikes once on fixed window coordinates, in metres.

    ``coords`` has shape (hole, 2, row, column). For global Zernikes, provide
    pupil-centred coordinates and the pupil diameter. For per-hole Zernikes,
    provide coordinates relative to each reference hole and the hole diameter.
    ``indices`` are Noll indices; ``polike=True`` uses hexagonal polynomials.
    No piston/tilt constraints are imposed here.

    Global coefficients have shape (mode,); local coefficients (hole, mode).
    Either may instead be a bound coefficient Map, including unit conversion.
    """
    coords = np.asarray(coords)
    if coords.ndim != 4 or coords.shape[1] != 2:
        raise ValueError("Coordinates must have shape (hole, 2, row, column)")

    # Sampling and coordinate choices are fixed at construction, not during fits.
    if polike:
        basis_fn = lambda xy: dlu.polike_basis(6, indices, xy, diameter)
    else:
        basis_fn = lambda xy: dlu.zernike_basis(indices, xy, diameter)
    basis = vmap(basis_fn)(coords)
    if per_hole:
        x = np.zeros(basis.shape[:2]) if x is None else x
        return BasisOPD(basis, x=x)

    # One shared vector evaluates the global modes across every aperture window.
    x = np.zeros(basis.shape[1]) if x is None else x
    return zdx.Map(x=x, M=np.moveaxis(basis, 1, 0))
