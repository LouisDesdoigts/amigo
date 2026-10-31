"""Compact global or per-hole Fourier OPD evaluated only on pupil stamps."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import zodiax as zdx

from ._fourier import _to_xy, eval_fourier_basis, fourier_kernel_at_coordinates


class FourierWavefront(zdx.Base):
    """Fixed Fourier coordinates with global pupil gauges removed.

    Kernels and native coefficient grids use live dLux's (x,y) ordering.
    Output stamps use AMIGO's (hole,row,column) ordering. One flat fitted
    vector maps to either a shared grid or independent grids per hole.
    Each projected coordinate has unit RMS on the reference pupil, so fitted
    values are nm per coordinate. Correlated modes mean coefficient-vector
    norm is NOT the RMS of their combined OPD.

    Gauge projection/normalization uses fixed reference support. Evaluation
    continues over the entire stamp, allowing aperture edges to move inside
    those fixed windows. Never rebuild the coordinates during an optimizer step.
    """
    kernels_x: jax.Array
    kernels_y: jax.Array
    pupil_x: jax.Array
    pupil_y: jax.Array
    gauge_map: jax.Array
    indices: jax.Array
    scales: jax.Array
    support: jax.Array
    corners: jax.Array
    layout: str = eqx.field(static=True)
    coefficient_shape: tuple = eqx.field(static=True)
    pupil_pixels: int = eqx.field(static=True)
    diameter_m: float = eqx.field(static=True)
    period_m: float = eqx.field(static=True)

    def __init__(self, corners, support, pupil_pixels, diameter_m, *,
                 n_modes=7, layout="per_hole", period_m=None):
        """Build small kernels; n_modes counts Fourier functions per (x,y) axis."""
        if not jax.config.jax_enable_x64:
            raise ValueError("Fourier wavefront construction requires JAX x64")
        if layout not in ("global", "per_hole"):
            raise ValueError("Fourier layout must be global or per_hole")
        counts = (n_modes, n_modes) if isinstance(n_modes, int) else n_modes
        if (not isinstance(counts, (list, tuple)) or len(counts) != 2 or
                any(isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1 for n in counts)):
            raise ValueError("Fourier n_modes must contain two positive integers")
        modes_x, modes_y = _to_xy(n_modes, "n_modes")
        corners = np.asarray(corners)
        support = np.asarray(support, dtype=bool)
        if (support.ndim != 3 or corners.shape != (len(support), 2)
                or not np.issubdtype(corners.dtype, np.integer) or not support.any()):
            raise ValueError("Declare integer (x,y) corners and nonempty hole-stamp support")
        holes, rows, columns = support.shape
        if (not isinstance(pupil_pixels, int) or pupil_pixels < 1
                or not np.isfinite(diameter_m) or diameter_m <= 0):
            raise ValueError("Declare positive pupil sampling and diameter")
        if (np.any(corners < 0) or np.any(corners[:, 0]+columns > pupil_pixels)
                or np.any(corners[:, 1]+rows > pupil_pixels)):
            raise ValueError("Fourier stamp windows extend outside the pupil")
        if any(not support[h].any() for h in range(holes)):
            raise ValueError("Every reference hole must have illuminated support")
        period_m = float(period_m if period_m is not None else
                         (diameter_m if layout == "global" else
                          max(rows, columns)*diameter_m/pupil_pixels))
        if not np.isfinite(period_m) or period_m <= 0:
            raise ValueError("Fourier period must be finite and positive")
        # AMIGO's even-sized pixel coordinates are centered between pixels.
        # Explicit coordinates preserve this convention and the global origin.
        x = (corners[:, 0, None]+np.arange(columns)-(pupil_pixels-1)/2)/pupil_pixels
        y = (corners[:, 1, None]+np.arange(rows)-(pupil_pixels-1)/2)/pupil_pixels
        if layout == "global":
            sample_x, sample_y = x*diameter_m/period_m, y*diameter_m/period_m
        else:
            sample_x = np.broadcast_to((np.arange(columns)-(columns-1)/2)*diameter_m/pupil_pixels/period_m, x.shape)
            sample_y = np.broadcast_to((np.arange(rows)-(rows-1)/2)*diameter_m/pupil_pixels/period_m, y.shape)
        kx = np.stack([np.asarray(fourier_kernel_at_coordinates(modes_x, a)) for a in sample_x])
        ky = np.stack([np.asarray(fourier_kernel_at_coordinates(modes_y, a)) for a in sample_y])
        shape = (modes_x, modes_y) if layout == "global" else (holes, modes_x, modes_y)
        # Only four polynomial stamps are temporary setup arrays. No sampled
        # (hole,mode,row,column) Fourier tensor is ever constructed.
        xx, yy = np.broadcast_arrays(x[:, None, :], y[:, :, None])
        terms = np.stack([np.ones_like(xx), xx, yy, xx**2+yy**2])
        count = support.sum()
        sampled_terms = terms[:, support]
        gram = sampled_terms@sampled_terms.T/count
        contraction = "hxi,hyj,hyx->ij" if layout == "global" else "hxi,hyj,hyx->hij"
        cross = np.stack([np.einsum(contraction, kx, ky, term*support, optimize=True).ravel()/count
                          for term in terms])
        gauge_map = np.linalg.pinv(gram, rcond=1e-12)@cross
        raw_norm = np.einsum(contraction, kx**2, ky**2, support, optimize=True).ravel()/count
        projected_norm = raw_norm-np.sum(cross*gauge_map, axis=0)
        indices = np.flatnonzero(projected_norm > 1e-12)
        # Per-hole DCs retain relative piston. Fix only the first raw DC to
        # remove the one remaining redundant common-piston coordinate.
        indices = indices[indices != 0]
        if not len(indices):
            raise ValueError("No Fourier directions survive the pupil gauge projection")
        self.kernels_x, self.kernels_y = jnp.asarray(kx), jnp.asarray(ky)
        self.pupil_x, self.pupil_y = jnp.asarray(x), jnp.asarray(y)
        self.gauge_map, self.indices = jnp.asarray(gauge_map), jnp.asarray(indices)
        self.scales = jnp.asarray(1/np.sqrt(projected_norm[indices]))
        self.support, self.corners = jnp.asarray(support), jnp.asarray(corners)
        self.layout, self.coefficient_shape = layout, shape
        self.pupil_pixels, self.diameter_m, self.period_m = pupil_pixels, float(diameter_m), period_m

    @property
    def n_coefficients(self):
        return len(self.indices)

    @property
    def shape(self):
        """Logical mode/stamp dimensions, without sampled mode allocation."""
        return (len(self.corners), self.n_coefficients,
                self.kernels_y.shape[1], self.kernels_x.shape[1])

    @property
    def metadata(self):
        return {"schema": "amigo-fourier-wavefront-v1", "layout": self.layout,
                "modes_xy": [self.kernels_x.shape[-1], self.kernels_y.shape[-1]],
                "period_m": self.period_m, "diameter_m": self.diameter_m,
                "coordinates": "nm per unit reference-pupil RMS coordinate; correlated modes",
                "optical_sampling": {"pupil_pixels": self.pupil_pixels}}

    def signature(self):
        """Hash the actual kernels, gauge, support and normalization for resume."""
        import hashlib
        arrays = (self.kernels_x, self.kernels_y, self.pupil_x, self.pupil_y,
                  self.gauge_map, self.indices, self.scales, self.support, self.corners)
        digest = hashlib.sha256()
        for value in arrays:
            value = np.asarray(value)
            digest.update(str((value.shape, value.dtype.str)).encode())
            digest.update(value.tobytes())
        return {**self.metadata, "basis_sha256": digest.hexdigest(),
                "mode_stamp_shape": list(self.shape)}

    def save(self, path):
        """Save compact arrays for exact reconstruction without an optical fit."""
        import json
        from pathlib import Path
        path = Path(path)
        temporary = path.with_suffix(".tmp.npz")
        names = ("kernels_x", "kernels_y", "pupil_x", "pupil_y", "gauge_map",
                 "indices", "scales", "support", "corners")
        np.savez_compressed(temporary, **{name: np.asarray(getattr(self, name)) for name in names},
                            metadata=json.dumps(self.signature()))
        temporary.replace(path)

    @classmethod
    def load(cls, path):
        import json
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"]))
            if metadata.get("schema") != "amigo-fourier-wavefront-v1":
                raise ValueError("Unsupported Fourier wavefront artifact")
            basis = cls(data["corners"], data["support"], metadata["optical_sampling"]["pupil_pixels"],
                        metadata["diameter_m"], n_modes=metadata["modes_xy"],
                        layout=metadata["layout"], period_m=metadata["period_m"])
            names = ("kernels_x", "kernels_y", "pupil_x", "pupil_y", "gauge_map",
                     "indices", "scales", "support", "corners")
            for name in names:
                if data[name].shape != getattr(basis, name).shape or not np.isfinite(data[name]).all():
                    raise ValueError("Invalid Fourier wavefront artifact arrays")
            basis = basis.set(list(names), [jnp.asarray(data[name]) for name in names])
            if basis.signature()["basis_sha256"] != metadata["basis_sha256"]:
                raise ValueError("Fourier wavefront artifact checksum mismatch")
            return basis

    def evaluate(self, coefficients):
        """Return seven (or the declared number of) OPD stamps in metres."""
        coefficients = jnp.asarray(coefficients)
        if coefficients.shape != (self.n_coefficients,):
            raise ValueError("Fourier WFE coefficients must match the flat fitted layout")
        raw = jnp.zeros(int(np.prod(self.coefficient_shape))).at[self.indices].set(coefficients*self.scales)
        grids = raw.reshape(self.coefficient_shape)
        if self.layout == "global":
            stamps = jax.vmap(lambda kx, ky: eval_fourier_basis(grids, kx, ky).T)(self.kernels_x, self.kernels_y)
        else:
            stamps = jax.vmap(lambda c, kx, ky: eval_fourier_basis(c, kx, ky).T)(grids, self.kernels_x, self.kernels_y)
        g = self.gauge_map@raw
        x, y = self.pupil_x[:, None, :], self.pupil_y[:, :, None]
        correction = g[0]+g[1]*x+g[2]*y+g[3]*(x*x+y*y)
        return 1e-9*(stamps-correction)
