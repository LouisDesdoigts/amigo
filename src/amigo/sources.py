import jax
import jax.numpy as np
import dLux as dl
from jax import vmap
import dLux.utils as dlu
import equinox as eqx


def planck(wav, T):
    h = 6.626e-34
    c = 3.0e8
    k = 1.38e-23
    a = 2.0 * h * c**2
    b = h * c / (wav * k * T)
    intensity = a / ((wav**5) * (np.exp(b) - 1.0))
    return intensity


# class PlanckSpectrum(dl.sources.BaseSpectrum):
#     wavelengths: jax.Array
#     Teff: float

#     def __init__(self, wavelengths, Teff):
#         self.wavelengths = np.asarray(wavelengths, float)
#         self.Teff = float(Teff)

#     @property
#     def weights(self):
#         weights = planck(self.wavelengths, self.Teff)
#         return weights / weights.sum()


class PlanckSource(dl.BaseSource):
    wavelengths: jax.Array
    Teff: jax.Array
    position: jax.Array
    flux: jax.Array

    def __init__(self, wavelengths, Teff, position=np.zeros(2), flux=np.ones(1)):
        self.wavelengths = np.asarray(wavelengths, float)
        self.Teff = np.asarray(Teff, float)
        self.position = np.asarray(position, float)
        self.flux = np.asarray(flux, float)

    def normalise(self):
        pass

    @property
    def weights(self):
        weights = planck(self.wavelengths * 1e-6, self.Teff)
        return weights / weights.sum()

    def model(self, optics, return_wf=False, return_psf=False):
        weights = self.weights * self.flux
        wls = self.wavelengths * 1e-6
        pos = dlu.arcsec2rad(self.position)
        return optics.propagate(wls, pos, weights, return_wf, return_psf)


class UVSource(dl.BaseSource):
    wavelengths: jax.Array
    weights: jax.Array
    position: jax.Array  # arcsec
    flux: jax.Array
    mask: jax.Array
    amplitudes: jax.Array
    phases: jax.Array
    pad: int

    def __init__(
        self,
        wavelengths,  # Wavelengths in meters
        mask,
        position=np.zeros(2),  # Source position in arcsec
        flux=1,  # Source Flux
        pad=2,  # UV-transform padding factor
        weights=None,  # Spectral weights
    ):
        """
        Assumes the last term in the mask is the DC term, and that the first half of the
        array is the positive frequencies and the second half is the negative baselines.
        """
        # Set up wavelengths and weights
        self.wavelengths = np.asarray(wavelengths, float)
        if weights is None:
            weights = np.ones_like(self.wavelengths)
        self.weights = np.asarray(weights, float)

        # Ensure wavelengths and weights are the same shape
        if self.wavelengths.shape != self.weights.shape:
            raise ValueError(
                f"Shape mismatch between wavelengths ({self.wavelengths.shape}) and "
                f"weights ({self.weights.shape})"
            )

        # Set up position and flux
        self.position = np.asarray(position, float)
        self.flux = np.asarray(flux, float)
        if self.position.shape != (2,):
            raise ValueError(
                f"Position must be a 2-element array, not {self.position.shape}"
            )

        self.pad = int(pad)
        self.mask = mask

        # Construct amplitudes and phases
        N = (self.mask.shape[1] - 1) // 2  # Should always be even after -1, because DC
        # self.amplitudes = np.ones(N + 1)  # +1 for dc term
        # self.phases = np.zeros(N + 1)  # +1 for dc term
        self.amplitudes = np.ones(N)
        self.phases = np.zeros(N)

    def _to_uv(self, psf):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))

    def _from_uv(self, uv):
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uv)))

    # TODO: Rename apply visibilites, take splodge and inv from self
    def _apply_splodge(self, psf, splodge, inv_splodge_support):
        cplx = self._to_uv(psf)
        splodged_cplx = cplx * splodge
        inv_splodge_cplx = cplx * inv_splodge_support
        return self._from_uv(splodged_cplx + inv_splodge_cplx)

    @property
    def visibilities(self):
        return self.amplitudes * np.exp(1j * self.phases)

    def normalise(self):
        return self
        # return self.divide("weights", self.weights.sum())

    @property
    def N(self):
        # return len(self.amplitudes) - 1  # -1 for dc term
        return len(self.amplitudes)

    @property
    def splodges(self):
        # Get the components of the calculation
        dc_mask = self.mask[:, 0]
        mask = self.mask[:, 1 : self.N + 1]
        conj_mask = self.mask[:, -self.N :]
        vis = self.visibilities

        # Get the splodges
        dot = lambda a, b: dlu.eval_basis(a, b)
        splodge_fn = lambda dc_mask, mask, conj_mask: (
            # dc_mask * vis[0] + dot(mask, vis[1:]) + dot(conj_mask, vis[1:].conj())
            dc_mask
            + dot(mask, vis)
            + dot(conj_mask, vis.conj())
        )
        return vmap(splodge_fn)(dc_mask, mask, conj_mask)

    @property
    def inv_splodges_support(self):
        return np.abs(1 - self.mask.sum(1))

    @property
    def weights(self):
        weights = planck(self.wavelengths, self.Teff)
        return weights / weights.sum()

    # TODO: Allow for return wf and psf
    def model(self, optics, return_wf=False, return_psf=False):
        """ """
        # Normalise
        self = self.normalise()

        # Calculate the PSF
        pos_rad = dlu.arcsec2rad(self.position)
        wgts = self.flux * self.weights
        wfs = optics.propagate(self.wavelengths, pos_rad, wgts, return_wf=True)
        psfs = wfs.psf

        # Apply padding
        npix = self.pad * psfs[0].shape[0]
        padded = vmap(lambda x: dlu.resize(x, npix))(psfs)

        # Shape check
        if padded[0].shape != self.mask.shape[-2:]:
            raise ValueError(
                f"PSF shape {padded[0].shape} does not match mask shape "
                f"{self.mask.shape}. This is likely because the wrong `npix` or "
                "`oversample` value was provided to the constructor."
            )

        splodges = self.splodges
        inv_splodges_support = self.inv_splodges_support
        cplx_psfs = []
        for i in range(len(padded)):
            cplx_psfs.append(
                self._apply_splodge(padded[i], splodges[i], inv_splodges_support[i])
            )
        cplx_psfs = np.array(cplx_psfs)
        cplx_psfs = vmap(lambda x: dlu.resize(x, psfs.shape[-1]))(cplx_psfs)

        # Return wf
        if return_wf:
            return wfs.set(
                ["amplitude", "phase"], [np.abs(cplx_psfs), np.angle(cplx_psfs)]
            )
        if return_psf:
            return eqx.filter_vmap(dl.PSF)(np.abs(cplx_psfs), wfs.pixel_scale)

        return np.abs(cplx_psfs).sum(0)


class PlanckUVSource(dl.BaseSource):
    Teff: jax.Array
    position: jax.Array  # arcsec
    flux: jax.Array
    mask: jax.Array
    amplitudes: jax.Array
    phases: jax.Array
    pad: int

    def __init__(
        self,
        wavelengths,  # Wavelengths in meters
        mask,
        Teff=4500,  # Source temperature (K)
        position=np.zeros(2),  # Source position in arcsec
        flux=1,  # Source Flux
        pad=2,  # UV-transform padding factor
        weights=None,  # Spectral weights
    ):
        """
        Assumes the last term in the mask is the DC term, and that the first half of the
        array is the positive frequencies and the second half is the negative baselines.
        """
        # Set up wavelengths and weights
        self.wavelengths = np.asarray(wavelengths, float)
        self.Teff = np.asarray(Teff, float)
        # if weights is None:
        #     weights = np.ones_like(self.wavelengths)
        # self.weights = np.asarray(weights, float)

        # # Ensure wavelengths and weights are the same shape
        # if self.wavelengths.shape != self.weights.shape:
        #     raise ValueError(
        #         f"Shape mismatch between wavelengths ({self.wavelengths.shape}) and "
        #         f"weights ({self.weights.shape})"
        #     )

        # Set up position and flux
        self.position = np.asarray(position, float)
        self.flux = np.asarray(flux, float)
        if self.position.shape != (2,):
            raise ValueError(
                f"Position must be a 2-element array, not {self.position.shape}"
            )

        self.pad = int(pad)
        self.mask = mask

        # Construct amplitudes and phases
        N = (self.mask.shape[1] - 1) // 2  # Should always be even after -1, because DC
        # self.amplitudes = np.ones(N + 1)  # +1 for dc term
        # self.phases = np.zeros(N + 1)  # +1 for dc term
        self.amplitudes = np.ones(N)
        self.phases = np.zeros(N)

    def _to_uv(self, psf):
        return np.fft.fftshift(np.fft.fft2(np.fft.fftshift(psf)))

    def _from_uv(self, uv):
        return np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(uv)))

    # TODO: Rename apply visibilites, take splodge and inv from self
    def _apply_splodge(self, psf, splodge, inv_splodge_support):
        cplx = self._to_uv(psf)
        splodged_cplx = cplx * splodge
        inv_splodge_cplx = cplx * inv_splodge_support
        return self._from_uv(splodged_cplx + inv_splodge_cplx)

    @property
    def visibilities(self):
        return self.amplitudes * np.exp(1j * self.phases)

    def normalise(self):
        return self
        # return self.divide("weights", self.weights.sum())

    @property
    def N(self):
        # return len(self.amplitudes) - 1  # -1 for dc term
        return len(self.amplitudes)

    @property
    def splodges(self):
        # Get the components of the calculation
        dc_mask = self.mask[:, 0]
        mask = self.mask[:, 1 : self.N + 1]
        conj_mask = self.mask[:, -self.N :]
        vis = self.visibilities

        # Get the splodges
        dot = lambda a, b: dlu.eval_basis(a, b)
        splodge_fn = lambda dc_mask, mask, conj_mask: (
            # dc_mask * vis[0] + dot(mask, vis[1:]) + dot(conj_mask, vis[1:].conj())
            dc_mask
            + dot(mask, vis)
            + dot(conj_mask, vis.conj())
        )
        return vmap(splodge_fn)(dc_mask, mask, conj_mask)

    @property
    def inv_splodges_support(self):
        return np.abs(1 - self.mask.sum(1))

    # TODO: Allow for return wf and psf
    def model(self, optics, return_wf=False, return_psf=False):
        """ """
        # Normalise
        self = self.normalise()

        # Calculate the PSF
        pos_rad = dlu.arcsec2rad(self.position)
        wgts = self.flux * self.weights
        wfs = optics.propagate(self.wavelengths, pos_rad, wgts, return_wf=True)
        psfs = wfs.psf

        # Apply padding
        npix = self.pad * psfs[0].shape[0]
        padded = vmap(lambda x: dlu.resize(x, npix))(psfs)

        # Shape check
        if padded[0].shape != self.mask.shape[-2:]:
            raise ValueError(
                f"PSF shape {padded[0].shape} does not match mask shape "
                f"{self.mask.shape}. This is likely because the wrong `npix` or "
                "`oversample` value was provided to the constructor."
            )

        splodges = self.splodges
        inv_splodges_support = self.inv_splodges_support
        cplx_psfs = []
        for i in range(len(padded)):
            cplx_psfs.append(
                self._apply_splodge(padded[i], splodges[i], inv_splodges_support[i])
            )
        cplx_psfs = np.array(cplx_psfs)
        cplx_psfs = vmap(lambda x: dlu.resize(x, psfs.shape[-1]))(cplx_psfs)

        # Return wf
        if return_wf:
            return wfs.set(
                ["amplitude", "phase"], [np.abs(cplx_psfs), np.angle(cplx_psfs)]
            )
        if return_psf:
            return eqx.filter_vmap(dl.PSF)(np.abs(cplx_psfs), wfs.pixel_scale)

        return np.abs(cplx_psfs).sum(0)
