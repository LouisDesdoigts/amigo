# Installation

Amigo should be installed from source. Clone the repository and install the package with pip:

```bash
git clone https://github.com/LouisDesdoigts/amigo.git
cd amigo
pip install .
```

It has been developed and locally tested in Python 3.11 & 3.12 on both CPU and GPU. Amigo is currently tested for jax=0.7.2 

If you wish to run Amigo on GPU, make sure install the appropriate version of JAX! For CUDA 12, use:

```bash
pip install -U "jax[cuda12]==0.7.2"
```

For more details on JAX installation, refer to the [official JAX installation guide](https://docs.jax.dev/en/latest/installation.html#installation).

If you want to run the notebooks, you will also need to install Jupyter and IPython kernel:

```bash
pip install jupyter notebook ipykernel
```

Note that python 3.14.0 has had some recent issues with ipython, so you may need to using 3.13 to run the notebooks.

Amigo also requires a set of calibration files, which will be available publicly soon. In the meantime, please [contact me](mailto:louis.desdoigts@gmail.com) directly to obtain them.