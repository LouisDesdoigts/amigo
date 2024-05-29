# AMIGO

Aperture Masking Interferometry Generative Observations for JWST.

This repo is currently under development and subject to change.

Much of the code is eventually be split into either dLux (ie the interferometry.py file) or Zodiax (most of the fitting.py and FIM.py files).

### Installation

**Python version**

Amigo requires python 3.11 due to a complex set of dependencies (various packages like `jwst` and `webbpsf` set annoying requirements, despite being minor dependencies).

To run amigo is it recommended to create a fresh environment. To do so using conda, run:

```
conda create -n amigo python=3.11
conda activate amigo
```

Then clone the repository and install:

```
git clone https://github.com/LouisDesdoigts/amigo.git
pip install .
```

### Data Processing

Amigo requires a custom set of pipeline steps out of the JWST pipeline. There are functions set up to make this easy to do:

```python
from amigo.pipelines import process_uncal, process_calslope
import os

os.environ["CRDS_PATH"] = "/path/to/crds_cache"
dirs = ["/path/to/uncal/data/"]

for directory in dirs:
    stage1_dir = process_uncal(directory, "stage1", verbose=False)
    calslope_dir = process_calslope(stage1_dir)
```

It's that simple (in theory). In practice the jwst pipeline can be difficult to get installed, this is left as an exercise for the user ;).

And never forget, AMI go brrrrrrrr
