import jax.numpy as np
import matplotlib.pyplot as plt

# from interferometry import pairwise_vectors, osamp_freqs
from matplotlib import colormaps

inferno = colormaps["inferno"]
seismic = colormaps["seismic"]


def plot_losses(losses, start, stop=-1):
    plt.figure(figsize=(16, 5))
    plt.subplot(1, 2, 1)
    plt.title("Full Loss")
    plt.plot(losses)

    if start >= len(losses):
        start = 0
    last_losses = losses[start:stop]
    n = len(last_losses)
    plt.subplot(1, 2, 2)
    plt.title(f"Final {n} Losses")
    plt.plot(np.arange(start, start + n), last_losses)

    plt.tight_layout()
    plt.show()


#
