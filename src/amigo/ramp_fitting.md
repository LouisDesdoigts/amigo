# Forwards Model Ramp Fitting: The Theory

Primary Source: <https://www.stsci.edu/files/live/sites/www/files/home/roman/_documents/Roman-STScI-000394_DeterminingTheBestFittingSlope.pdf>

For some data with shape `(ngroups, npix, npix)` and model that outputs a ramp of the same shape, we must build a covariance matrix $\vec{\Sigma}$ with shape `(ngroups, ngroups, npix, npix)`.

Examining this at the single-pixel level, we need to find $\Sigma$ where

$$
\Sigma =
\begin{bmatrix}
Var(t_0) & Cov(t_1, t_0) & \cdots & Cov(t_n, t_0) \\
Cov(t_1, t_0) & Var(t_1) &  & \vdots \\
\vdots &  & \ddots &  \\
Cov(t_n, t_0) & \cdots &  & Var(t_n)
\end{bmatrix}
$$

This matrix is actually the combination of two different matrices, $\Sigma_{read}$ & $\Sigma_{photon}$.

---

### $\Sigma_{read}$

The read noise matrix is simple as each measurement is completely independent from each-other, meaning it is diagonal with the same value along the diagonal. So for some pixel, we have:

$$
\Sigma_{read} =
\begin{bmatrix}
\sigma_{read}^2 & 0 & \cdots & 0 \\
0 & \sigma_{read}^2 &  & \vdots \\
\vdots &  & \ddots &  \\
0 & \cdots &  & \sigma_{read}^2
\end{bmatrix}
$$

The values along the diagonal change for each pixel, but is a constant value that is pre-measured.

---

### $\Sigma_{photon}$

Note: $t_i$ = $t_g * i$ where $t_g$ is the group time and $i$ is the group number.

The photon noise matrix is more complicated as each measurement is not independent from each-other, since each new measurement is the sum of the previous measurement and the new photons.

In the simple case where each pixel has a linear response to incident photons, we have $f * t_g * i$ photons in each group.

In this case the variance and covariance are then simply:

$$Var(t_i) = f * t_g * i = f * t_i$$

$$Cov(t_i, t_j) = f * t_g * min(i, j) = f * min(t_i, t_j)$$

In this case the matrix is then:

$$
\Sigma_{photon} =
\begin{bmatrix}
f * t_0 & f * t_0 & \cdots & f * t_0 \\
f * t_0 & f * t_1 &  & \vdots \\
\vdots &  & \ddots &  \\
f * t_0 & \cdots &  & f * t_n
\end{bmatrix}
$$

Simple, right? Not so fast cowboy. This is only true if the electron count-rate is linear with photons. In reality, charge migration effects cause the electron count-rate to be non-linear with photons.

---

### A Note on Integrations

The above is true for a single integration, but we fit multiple integrations in order to be robust to cosmic rays and other sources of noise. The statistics here are simple, as the variance of the sum of two independent variables is the sum of the variances,
