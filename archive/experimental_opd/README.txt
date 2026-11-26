Archived experimental OPD implementation

These files were moved intact from src/amigo during the opds consolidation.
They are retained for historical analyses, not imported by the active package.

fourier_wavefront.py combined Fourier evaluation, reference support, gauge
removal, normalisation and serialization. _fourier.py included copied Fourier
utilities and arbitrary-coordinate kernels.

The active evaluator is src/amigo/opd.py and uses dLux Fourier kernels.
It does not automatically reproduce the old gauge, normalisation or Fourier
period/coordinate choices. Historical coefficient files must not be supplied
to it without explicitly reconstructing their mapping and sampling conventions.
Constraint/eigenbasis construction remains separate from evaluation.
