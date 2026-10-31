"""Functional Fourier helpers copied from live dLux utils, not overhaul.

Upstream: LouisDesdoigts/dLux, main 285ea0019a13951fa815a4700978a81c0ba6ba1e.
Source: src/dLux/utils/fourier.py, SHA256
ac28c10b91103f2c2f7855e76fcfe46a411c389eb4309ddd463919adbbfeb1e0.
The released (x,y) ordering and positive MFT phase are retained. Only the
unit-weight MFT kernel is inlined to avoid adding abcdLux or upgrading dLux.
Arbitrary-coordinate evaluation extends the same kernel for pupil stamps.

BSD 3-Clause License

Copyright (c) 2022, Louis Desdoigts
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""
from jax import Array
import jax.numpy as np

def _to_xy(value: int | tuple[int], name: str) -> tuple[int]:
    """
    Casts an integer or tuple input to an `(x, y)` tuple.

    Parameters
    ----------
    value : int | tuple[int]
        The input value.
    name : str
        The name of the input, used for error messages.

    Returns
    -------
    value : tuple[int]
        The cast input value.
    """
    if isinstance(value, int):
        value = (value, value)
    elif not isinstance(value, (tuple, list)) or len(value) != 2:
        raise TypeError(f"{name} must be an int or length-2 tuple, got {type(value)}.")

    value = (int(value[0]), int(value[1]))
    if value[0] <= 0 or value[1] <= 0:
        raise ValueError(f"{name} must contain positive integers, got {value}.")

    return value


def _fourier_mode_order(n_modes: int) -> Array:
    """
    Calculates the Fourier mode ordering for the complex exponential basis.

    Parameters
    ----------
    n_modes : int
        The number of Fourier modes.

    Returns
    -------
    modes : Array
        The ordered Fourier mode indices.
    """
    if n_modes % 2 == 1:
        kmax = (n_modes - 1) // 2
        modes = [0]
        for m in range(1, kmax + 1):
            modes.extend([m, -m])
    else:
        kmax = n_modes // 2
        modes = [0]
        for m in range(1, kmax):
            modes.extend([m, -m])
        modes.append(kmax)

    return np.asarray(modes)


def _map_to_real(kernel: Array) -> Array:
    """
    Maps a complex exponential Fourier kernel to the real Fourier basis.

    The input kernel is assumed to use the mode ordering
    `[0, +1, -1, +2, -2, ...]`. The returned kernel uses
    `[DC, cos1, sin1, cos2, sin2, ...]`.

    Parameters
    ----------
    kernel : Array
        The complex exponential Fourier kernel.

    Returns
    -------
    kernel : Array
        The real Fourier kernel.
    """
    n_modes = kernel.shape[-1]

    # Initialise the basis transform matrix and set the DC term
    T = np.zeros((n_modes, n_modes), dtype=complex)
    T = T.at[0, 0].set(1.0)

    # Determine number of harmonic pairs and Nyquist handling.
    if n_modes % 2 == 0:
        n_harm = n_modes // 2 - 1
        has_nyq = True
    else:
        n_harm = (n_modes - 1) // 2
        has_nyq = False

    if n_harm > 0:
        k = np.arange(1, n_harm + 1)
        half = np.array(0.5, dtype=complex)
        half_j = np.array(0.5j, dtype=complex)

        # Get the indicies
        ip = 2 * k - 1
        im = 2 * k

        # Set the transform for each harmonic pair
        T = T.at[ip, ip].set(half)
        T = T.at[ip, im].set(-half_j)
        T = T.at[im, ip].set(half)
        T = T.at[im, im].set(half_j)

    # Nyquist mode (even n): purely cosine, maps directly.
    if has_nyq:
        T = T.at[n_modes - 1, n_modes - 1].set(1.0)

    return (kernel @ T).real


def fourier_kernel_1d(n_modes: int, npix: int, scale: float = 1.0) -> Array:
    """
    Calculates a cached 1D Fourier basis evaluation kernel.

    A unit-amplitude mode with `scale=1` evaluates to values between `-1` and `+1`.

    Parameters
    ----------
    n_modes : int
        The number of Fourier modes.
    npix : int
        The number of output pixels.
    scale : float = 1.0
        The output amplitude scaling.

    Returns
    -------
    kernel : Array
        The cached Fourier kernel.
    """
    # Calculate the Fourier mode indices.
    modes = _fourier_mode_order(n_modes)

    # Calculate the output coordinates.
    coords = (np.arange(npix) - npix // 2) / npix

    # Calculate the complex exponential kernel.
    kernel = np.exp(2j * np.pi * coords[:, None] * modes[None, :])

    # Map the kernel to the real Fourier basis.
    return scale * _map_to_real(kernel)


def fourier_kernels(
    n_modes: int | tuple[int],
    npix: int | tuple[int],
    scale: float = 1.0,
) -> tuple[Array, Array]:
    """
    Calculates the cached 2D Fourier basis evaluation kernels.

    The input `n_modes` and `npix` are interpreted in `(x, y)` order. If an
    integer is provided it is cast to `(value, value)`. The returned kernels satisfy

    `output = Kx @ coefficients @ Ky.T`

    where `coefficients` has shape `(n_modes_x, n_modes_y)` and `output` has shape
    `(npix_x, npix_y)`.

    Parameters
    ----------
    n_modes : int | tuple[int]
        The number of Fourier modes in `(x, y)`.
    npix : int | tuple[int]
        The output number of pixels in `(x, y)`.
    scale : float = 1.0
        The per-axis output amplitude scaling.

    Returns
    -------
    Kx : Array
        The x-axis Fourier kernel.
    Ky : Array
        The y-axis Fourier kernel.
    """
    # Unpack the number of modes and pixels for each dimension.
    n_modes_x, n_modes_y = _to_xy(n_modes, "n_modes")
    npix_x, npix_y = _to_xy(npix, "npix")

    # Calculate the Fourier kernels for each dimension.
    Kx = fourier_kernel_1d(n_modes_x, npix_x, scale)
    Ky = fourier_kernel_1d(n_modes_y, npix_y, scale)

    return Kx, Ky


def eval_fourier_basis(coefficients: Array, Kx: Array, Ky: Array) -> Array:
    """
    Evaluates a 2D real Fourier basis using cached kernels.

    The coefficient array is assumed to be ordered as `(x, y)`, with shape
    `(n_modes_x, n_modes_y)`. The returned output is also ordered as `(x, y)`,
    with shape `(npix_x, npix_y)`.

    Parameters
    ----------
    coefficients : Array
        The Fourier coefficients.
    Kx : Array
        The x-axis Fourier kernel.
    Ky : Array
        The y-axis Fourier kernel.

    Returns
    -------
    output : Array
        The evaluated Fourier basis.
    """
    return Kx @ coefficients @ Ky.T


def fourier_kernel_at_coordinates(n_modes, coordinates, scale=1.0):
    """Same live dLux kernel at explicit coordinates measured in periods."""
    modes = _fourier_mode_order(n_modes)
    coordinates = np.asarray(coordinates)
    kernel = np.exp(2j * np.pi * coordinates[:, None] * modes[None, :])
    return scale * _map_to_real(kernel)
