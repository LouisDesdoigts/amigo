"""Named OPD components evaluated on fixed aperture windows.

All components resolve to (hole, row, column) arrays in metres. Geometry,
constraint construction and filter/visit sharing belong to the caller; evaluating
an OPD neither rebuilds its basis nor removes any modes.
"""

from copy import copy
import equinox as eqx
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

    ``basis`` is either shared (mode, row, column) or sampled separately on
    each hole (hole, mode, row, column). ``x`` resolves to (hole, mode).
    A bound Map in ``x`` can couple holes or remove redundant coordinates. This
    contraction avoids storing a mostly-zero block-diagonal sampled basis.
    """

    basis: Array

    def __init__(self, basis, x=None, *, alias=None):
        self.basis = np.asarray(basis)
        if self.basis.ndim not in (3, 4):
            raise ValueError("Basis must have shape (mode, row, column), optionally per hole")
        self.x = x
        self.alias = alias

    def fwd(self, x, **context):
        if x.ndim != 2 or x.shape[1] != self.basis.shape[-3]:
            raise ValueError("Local coefficients must have shape (hole, mode)")
        if self.basis.ndim == 3:
            return np.einsum("hm,mij->hij", x, self.basis)
        if x.shape[0] != self.basis.shape[0]:
            raise ValueError("Coefficients and basis must have the same number of holes")
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


# Construction-time coefficient coordinates


def opd_basis(component, shape):
    """Sample a linear evaluator in its native coefficient coordinates.

    ``shape`` is the native coefficient shape, before any input Map. Rows of the
    result are unit-coefficient OPD stamps. Evaluators are linear, so sample
    coefficient unit vectors directly instead of differentiating image outputs.
    This setup calculation is not part of propagation or optimisation.
    """
    size = 1
    for dimension in shape:
        size *= dimension
    coefficients = np.eye(size).reshape((size, *shape))
    return vmap(component.fwd)(coefficients)


def pupil_metric(weights):
    """Normalise explicit nonnegative pupil weights to unit total weight.

    Supply (hole, row, column), or (realisation, hole, row, column). Each
    realisation contributes equally after its own normalisation. For intensity
    weighting, pass squared amplitude transmissions; no weighting is inferred.
    """
    weights = np.asarray(weights, float)
    if weights.ndim == 3:
        weights = weights[None]
    if weights.ndim != 4 or not bool(np.all(np.isfinite(weights) & (weights >= 0))):
        raise ValueError("Pupil weights must be finite, nonnegative window arrays")
    totals = weights.sum(axis=(1, 2, 3), keepdims=True)
    if not bool(np.all(np.isfinite(totals) & (totals > 0))):
        raise ValueError("Every pupil realisation must have finite, nonzero total weight")
    return (weights / totals).mean(axis=0)


def opd_map(basis, weights, *, null=None, normalise=True, rtol=None):
    """Build reduced coefficient coordinates under a fixed pupil metric.

    ``basis`` is (coefficient, hole, row, column); ``null`` is an optional stack
    of OPD patterns with the same windows. First constrain their weighted overlap
    to zero, then discard unobservable directions. With ``normalise=True``, the
    retained eigenmodes have unit weighted RMS and are mutually orthogonal.

    Returns a bound Map from reduced coordinates to the flat native coefficient
    vector. Reshape its M output axes to the native coefficient shape when needed.
    Values are dimensionless unless a unit conversion is explicitly added to s.
    Construction is not JIT-compatible: rank selection changes the output size.
    Constraints hold under the averaged reference metric, not separately under
    every realisation. No basis components are subtracted outside its own span.
    """
    basis = np.asarray(basis, float)
    metric = pupil_metric(weights)
    if basis.ndim != 4 or basis.shape[1:] != metric.shape or not bool(np.all(np.isfinite(basis))):
        raise ValueError("Basis must contain finite modes on the weighted windows")
    if basis.shape[0] == 0:
        raise ValueError("Supply at least one basis mode")
    rtol = 100 * np.finfo(basis.dtype).eps if rtol is None else float(rtol)
    if not 0 < rtol < 1:
        raise ValueError("rtol must lie between zero and one")

    # Work with compact coefficient matrices; the pupil metric stays fixed.
    modes = basis.reshape(basis.shape[0], -1) * np.sqrt(metric.ravel())
    mapping = np.eye(len(modes))
    if null is not None:
        null = np.asarray(null, float)
        if null.ndim != 4 or null.shape[1:] != metric.shape or not bool(np.all(np.isfinite(null))):
            raise ValueError("Null patterns must contain finite modes on the same windows")
        patterns = null.reshape(len(null), -1) * np.sqrt(metric.ravel())
        # Remove dependence on the units chosen for each constraint pattern.
        norms = np.linalg.norm(patterns, axis=1)
        patterns = patterns / np.where(norms > 0, norms, 1)[:, None]
        overlap = patterns @ modes.T
        _, values, vh = np.linalg.svd(overlap, full_matrices=True)
        rank = int(np.sum(values > rtol * np.linalg.norm(modes)))
        mapping = vh[rank:]
    if not len(mapping):
        raise ValueError("Constraints remove the entire coefficient space")

    # Diagonalise within the allowed space, never by changing the forward model.
    allowed = mapping @ modes
    gram = allowed @ allowed.T
    values, vectors = np.linalg.eigh(gram)
    keep = values > rtol * np.maximum(values[-1], 0)
    if not bool(np.any(keep)):
        raise ValueError("No observable OPD modes remain under this pupil metric")
    mapping = vectors[:, keep].T @ mapping
    if normalise:
        mapping = mapping / np.sqrt(values[keep])[:, None]
    return zdx.Map(x=np.zeros(len(mapping)), M=mapping)


def piston_tip_tilt(coords):
    """Global piston, x and y patterns on pupil-centred window coordinates."""
    coords = np.asarray(coords)
    if coords.ndim != 4 or coords.shape[1] != 2:
        raise ValueError("Coordinates must have shape (hole, 2, row, column)")
    return np.stack([np.ones_like(coords[:, 0]), coords[:, 0], coords[:, 1]])


# Aperture-owned construction recipes


def reference_pupil(aperture):
    """Amplitude transmission and global coordinates on fixed pupil windows."""
    from .optical_models import StaticApertureMask, crop_windows, window_centers

    if isinstance(aperture, StaticApertureMask):
        transmission = aperture.transmission
        if transmission.ndim == 2:
            transmission = crop_windows(transmission, aperture.corners, aperture.size)
    else:
        transmission = aperture.sparse_apertures()
    if aperture.amp_basis is not None:
        transmission = transmission * (1 + vmap(dlu.eval_basis)(
            aperture.amp_basis, aperture.amp_coeffs))

    pixel_scale = aperture.diameter / aperture.npixels
    local = dlu.pixel_coords(aperture.size, aperture.size * pixel_scale)
    centers = window_centers(aperture.corners, aperture.size,
                             aperture.npixels, aperture.diameter)
    return transmission, local[None] + centers[:, :, None, None]


def ptt_defocus(coords):
    """Global piston, tip, tilt and quadratic pupil defocus patterns.

    This is a pupil-OPD constraint, not an exact response to axial displacement
    in the propagator. A local-plane basis removes only its projected overlap.
    """
    ptt = piston_tip_tilt(coords)
    return np.concatenate([ptt, (coords ** 2).sum(axis=1)[None]])


class OPDSetup(zdx.Base):
    """Construction settings consumed by the aperture, not runtime fit state."""

    sharing: tuple = eqx.field(default=("program", "filter"), kw_only=True, static=True)
    weighting: str = eqx.field(default="intensity", kw_only=True, static=True)
    references: tuple | None = eqx.field(default=None, kw_only=True)
    rtol: float | None = eqx.field(default=None, kw_only=True, static=True)

    def build(self, aperture, extra_null=None):
        """Sample, constrain and normalise once; bind nanometre coefficients."""
        _, coords = reference_pupil(aperture)
        references = (aperture,) if self.references is None else self.references
        weights = []
        for reference in references:
            if (reference.size != aperture.size or reference.npixels != aperture.npixels
                    or reference.diameter != aperture.diameter
                    or not bool(np.array_equal(reference.corners, aperture.corners))):
                raise ValueError("OPD reference apertures must share fixed pupil windows")
            amplitude, _ = reference_pupil(reference)
            if self.weighting == "intensity":
                weights.append(amplitude ** 2)
            elif self.weighting == "uniform":
                weights.append((amplitude != 0).astype(float))
            else:
                raise ValueError("OPD weighting must be 'intensity' or 'uniform'")
        if not weights:
            raise ValueError("Supply at least one reference aperture")

        component, shape = self.component(aperture)
        basis = opd_basis(component, shape)
        null = ptt_defocus(coords)
        if extra_null is not None:
            null = np.concatenate([null, extra_null])
        mapping = opd_map(basis, np.stack(weights), null=null, rtol=self.rtol)
        mapping = mapping.set(["M", "s"], [mapping.M.reshape(-1, *shape), np.array(1e-9)])
        return component.set("x", mapping)


class PTT(OPDSetup):
    """Residual-fit local planes, excluding global piston/tip/tilt/defocus.

    Every hole uses the same three polynomials; only its coefficients differ.
    Construction mixes those coefficients into reduced, RMS-normalised modes.
    """

    def component(self, aperture):
        local = dlu.pixel_coords(aperture.size, 2)
        basis = np.stack([np.ones_like(local[0]), local[0], local[1]])
        return BasisOPD(basis), (len(aperture.corners), 3)


class Fourier(OPDSetup):
    """Calibration Fourier modes excluding global piston/tip/tilt/defocus.

    This is an alternative to the residual PTT fit, not a joint PTT+Fourier
    model. It does not remove every per-hole PTT pattern from the Fourier span.
    """

    n_modes: int = 5

    def component(self, aperture):
        component = FourierOPD(self.n_modes, aperture.size)
        return component, (len(aperture.corners), self.n_modes, self.n_modes)


def build_opds(aperture, sharing=None):
    """Resolve recipes and add coefficient aliases and sharing defaults."""
    opds = dict(aperture.opds or {})
    aliases = dict(aperture.alias or ())
    groups = {}
    for name, component in opds.items():
        if isinstance(component, OPDSetup):
            opds[name] = component.build(aperture)
            alias = name + "_coeffs"
            target = f"opds.{name}.x.x"
            if alias in aliases and aliases[alias] != target:
                raise ValueError(f"Generated OPD alias {alias!r} conflicts with an existing alias")
            aliases[alias] = target
            groups[alias] = component.sharing
    if sharing is not None:
        groups.update(sharing)
    groups = tuple((name, (fields,) if isinstance(fields, str) else tuple(fields))
                   for name, fields in groups.items())
    return opds or None, aliases or None, groups or (() if sharing is not None else None)


# Default calibration/science components

DEFAULT_FILTER_MODES = (5, 6, 7, 8, 9, 10)


class OPDCalibration(zdx.Base):
    """Keep the reference aperture, its exact bases/Maps and grouped coefficients.

    This is an in-memory calibration record, not just a coefficient vector.
    The reference aperture also preserves the pupil geometry used for construction.
    """

    aperture: zdx.Module
    coefficients: dict
    filters: tuple = eqx.field(default=(), static=True, kw_only=True)

    def _bound(self, name, coefficients):
        """Expose a component with its recovered input and original basis/Map."""
        return self.aperture.opds[name].set("x.x", coefficients)

    @property
    def fourier(self):
        """The global calibrated Fourier component, independently evaluable.

        Explicitly grouped Fourier fits return a dictionary of bound components
        instead. No OPD images are collapsed into a combined reference array.
        """
        values = self.coefficients.get("fourier_coeffs", self.aperture.fourier_coeffs)
        if isinstance(values, dict):
            if set(values) == {"shared"}:
                values = values["shared"]
            else:
                return {key: self._bound("fourier", value) for key, value in values.items()}
        return self._bound("fourier", values)

    @property
    def filter_opds(self):
        """Per-filter Zernike components with their calibrated coefficients.

        The returned components share the exact sampled basis and coefficient
        mapping; each can be evaluated or inspected separately from Fourier.
        """
        values = self.coefficients.get("filter_coeffs", self.aperture.filter_coeffs)
        if not isinstance(values, dict):
            values = {filt: values for filt in self.filters}
        return {key: self._bound("filter", value) for key, value in values.items()}


def project_opd(basis, null, weights):
    """Remove weighted null patterns from sampled shapes, preserving their span
    modulo those patterns. Unlike opd_map, this changes the sampled basis itself.
    """
    root_weight = np.sqrt(pupil_metric(weights)).ravel()
    modes = basis.reshape(len(basis), -1)
    patterns = null.reshape(len(null), -1)
    overlap = (modes * root_weight) @ np.linalg.pinv(patterns * root_weight)
    return (modes - overlap @ patterns).reshape(basis.shape)


def standard_opds(aperture, fourier_modes=5, filter_modes=DEFAULT_FILTER_MODES):
    """Construct all three default components once, independent of fit flags.

    Global Zernikes own their gauge-removed sampled span. Fourier excludes that
    span as well as global PTT/defocus. Residual PTT only excludes global
    PTT/defocus: it is intended for fitting after the calibration is fixed.
    """
    if not filter_modes or any(j < 5 for j in filter_modes) or len(set(filter_modes)) != len(filter_modes):
        raise ValueError("filter_modes must contain distinct Noll indices >= 5")
    amplitude, coords = reference_pupil(aperture)
    weights = amplitude ** 2

    # Sparse pupils break the full-circle orthogonality of ordinary Zernikes.
    # Remove global gauge patterns from the shapes before normalising the span.
    global_basis = zernike_opd(coords, filter_modes, aperture.diameter).M
    global_basis = project_opd(global_basis, ptt_defocus(coords), weights)
    mapping = opd_map(global_basis, weights).set("s", np.array(1e-9))
    global_opd = zdx.Map(x=mapping, M=global_basis)

    # Both calibration terms then occupy disjoint spaces in the same metric.
    fourier = Fourier(n_modes=fourier_modes).build(aperture, extra_null=global_basis)
    ptt = PTT().build(aperture)
    opds = {"fourier": fourier, "filter": global_opd, "ptt": ptt}
    aliases = {name + "_coeffs": f"opds.{name}.x.x" for name in opds}
    sharing = (("fourier_coeffs", ()), ("filter_coeffs", ("filter",)),
               ("ptt_coeffs", ("program", "filter")))
    # Alias/sharing metadata are static Equinox fields, not fitted leaves.
    result = copy(aperture)
    object.__setattr__(result, "opds", opds)
    object.__setattr__(result, "alias", tuple(sorted({**dict(aperture.alias or ()), **aliases}.items())))
    object.__setattr__(result, "opd_sharing", sharing)
    result.validate_aliases()
    return result


def coefficient_value(aperture, name, coefficients, exp):
    """Select fixed or fitted values; populated groups never silently fall back."""
    value = coefficients.get(name)
    if value is None or (isinstance(value, dict) and not value):
        return aperture.get(name)
    if isinstance(value, dict):
        fields = dict(aperture.opd_sharing)[name]
        key = opd_key(exp, fields)
        if key not in value:
            raise ValueError(f"No calibrated coefficients for {name!r}, group {key!r}")
        value = value[key]
    if np.shape(value) != aperture.get(name).shape:
        raise ValueError(f"OPD coefficients for {name!r} have the wrong shape")
    return np.asarray(value)


# Exposure selection; basis definitions remain on the aperture.


def opd_key(exp, fields):
    """Select an explicit sharing group from exposure metadata."""
    if not fields:
        return "shared"
    try:
        values = [str(exp[field] if isinstance(exp, dict) else getattr(exp, field)) for field in fields]
    except (KeyError, AttributeError) as error:
        raise ValueError(f"Selecting this OPD requires exposure fields {fields}") from error
    if any("." in value or ":" in value for value in values):
        raise ValueError("OPD group values cannot contain '.' or ':'")
    return ":".join(values)


def init_opds(aperture, exposures, groups, coefficients=None):
    """Initialise grouped fit values from the aperture's coefficient aliases."""
    coefficients = {} if coefficients is None else coefficients
    aliases = dict(aperture.alias or ())
    targets = []
    values = {}
    for name, fields in groups:
        if name not in aliases or not aliases[name].startswith("opds."):
            raise ValueError(f"{name!r} must be an alias to an OPD coefficient input")
        target = aliases[name]
        if any(target == other or target.startswith(other + ".") or other.startswith(target + ".")
               for other in targets):
            raise ValueError("OPD fit aliases must select distinct coefficient inputs")
        targets.append(target)
        value = aperture.get(name)
        if not isinstance(value, Array):
            raise ValueError(f"{name!r} must select a coefficient array")
        values[name] = {opd_key(exp, fields): coefficient_value(aperture, name, coefficients, exp)
                        for exp in exposures}
    return values


def apply_opds(aperture, values, groups, exp):
    """Inject this exposure's selected coefficients, retaining fixed mappings."""
    for name, fields in groups:
        value = coefficient_value(aperture, name, values, exp)
        aperture = aperture.set(name, value)
    return aperture
