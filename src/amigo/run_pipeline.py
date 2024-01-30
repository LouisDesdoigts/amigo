"""
Run file from command line as:
`python run_pipeline.py <CRDS_PATH> <stage0_dir>`

E.g.
`python run_pipeline.py /Users/mcha5804/JWST/crds_cache/ /Users/mcha5804/JWST/COMM1093/stage0/`

Recommended usage:
Create a new directory; say, e.g. `./COMM1093/`
with a subdirectory called stage0 (e.g. `./COMM1093/stage0`).
Place all uncal files in the stage0 directory and run the script.
This will create a stage1 and stage2 directory within the ./COMM1093/
for example. The stage1 directory will contain the ramp files and the
stage2 directory will contain the calgrps files.

If you're unsure what the CRDS path is, see the jwst docs:
https://jwst-pipeline.readthedocs.io/en/latest/jwst/user_documentation/reference_files_crds.html#crds

Good luck on that journey...
"""

# imports
import os
import sys
from tqdm import tqdm
from pipelines import generate_ramps, generate_calgrps

#  set paths for jwst pipeline environment variables
os.environ["CRDS_PATH"] = sys.argv[1]
os.environ["CRDS_SERVER_URL"] = "https://jwst-crds.stsci.edu"

# set directories for file management
if sys.argv[2][-1] == "/":
    stage0_dir = sys.argv[2]
else:
    stage0_dir = sys.argv[2] + "/"

parent_dir = os.path.dirname(stage0_dir[:-1])
stage1_dir = parent_dir + "/stage1/"
stage2_dir = parent_dir + "/stage2/"

# RUNNING PIPELINE 1
print("Running Pipeline 1")
# getting list of uncal files in directory
uncal_files = []
for filename in os.listdir(stage0_dir):
    if filename.endswith("_uncal.fits"):
        uncal_files.append(stage0_dir + filename)

# running pipeline 1 on list of uncal files
generate_ramps(uncal_files, stage1_dir)

# RUNNING PIPELINE 2
print("Running Pipeline 2")
for filename in tqdm(os.listdir(stage1_dir)):
    if filename.endswith("nis_ramp.fits"):
        generate_calgrps(
            input_dir=stage1_dir,
            output_dir=stage2_dir,
            filename_base=filename[:-13],
            tframe=0.07544,
        )

print("Done!")
