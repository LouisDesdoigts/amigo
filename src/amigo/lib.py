import jax
import jax.numpy as np
import jax.scipy as jsp
import dLux as dl
import dLux.utils as dlu
import equinox as eqx
from jax.scipy.signal import convolve
from matplotlib import colormaps
from jax import vmap

import optax

import zodiax as zdx
import tqdm.notebook as tqdm

