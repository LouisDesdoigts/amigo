# AMIGO

Aperture Masking Interferometry Generative Observations for JWST.

---

This repo is currently under development and subject to change.

Much of the code is eventually be split into either dLux (ie the interferometry.py file) or Zodiax (most of the fitting.py and FIM.py files).

The amgio pipeline requires _custom file processing_, which is covered by the pipelines.py file. More details on this later.

To see example usage of the pipeline, see the [amigo_notebooks](https://github.com/LouisDesdoigts/amigo_notebooks) repo. It is currently private, so just message me for access.

### Installation

**Python version**

Development is being done in 3.12 and takes advantage of its new syntax, so amigo should be run in 3.12. However the pipeline module uses [jwst](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html), which can only be run on python 3.9 - 3.11. Because of this, it is not listed as a dependency, and the pipeline should be run for a different environment with python <= 3.11, with the jwst package installed.

To create a fresh environment for the pipeline, run:

```
conda create -n amigo python=3.xx
conda activate amigo
```

Then clone the repository and install:

```
git clone https://github.com/LouisDesdoigts/amigo.git
pip install .
```

**Dependencies**

Core:

- [Jax](https://github.com/google/jax)
- [Equinox]
- [Zodiax]
- [dLux]

Other:

- [xara](https://github.com/fmartinache/xara) for initialisation of the source position.
- [pyia](https://github.com/adrn/pyia) for initialisation of the source Teffs.
- [astroquery](https://github.com/astropy/astroquery) for initialisation of the source Teffs.
- [Webbpsf](https://github.com/spacetelescope/webbpsf) for initialisation of the measured WFE at closes observation time.
- [dLuxWebbPSF](https://github.com/itroitskaya/dLuxWebbpsf) For the JWST primary mirror aberration model + cubic spline interpolation for detector rotation. If running on the latest jax you may need to use [this branch](https://github.com/itroitskaya/dLuxWebbpsf/pull/24)
- [jwst](https://jwst-pipeline.readthedocs.io/en/latest/getting_started/install.html) for the pipeline module -> **Read the note above about python version**

And never forget, AMI go brrrrrrrr
