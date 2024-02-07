# DistanceDistributionDescriptors
Smeared radial distribution 3D descriptors

## Description
These distance-based molecular 3D descriptors are inspired by those of [SchNet](https://github.com/atomistic-machine-learning/schnetpack/), with the additional summation over identical atom pairs. As such, these descriptors are invariant by global translation, rotation and inversion, as well as permutation of equivalent  atoms (i.e. same atomic number).

More details can be found in the related paper: _coming soon_

## Usage
```
usage: src/ddd.py [-h] -i INPUT [-o OUTPUT] [--min-cutoff MIN_CUTOFF] [--cutoff CUTOFF]
                  [--nb-basis-functions NB_BASIS_FUNCTIONS] [--centers CENTERS]
                  [--widths WIDTHS] [--basis-function BASIS_FUNCTION]
                  [--atomic-types ATOMIC_TYPES]

optional arguments:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Input file path (any format recognized by ASE) (default: None)
  -o OUTPUT, --output OUTPUT
                        Output file path (prints to standard output if unspecified or empty) (default: None)
  --min-cutoff MIN_CUTOFF
                        Minimum distance cutoff (Angstroms) (default: 0)
  --cutoff CUTOFF       Maximum distance cutoff (Angstroms). Default: 10 Bohr (default: 5.291772105638412)
  --nb-basis-functions NB_BASIS_FUNCTIONS, --size NB_BASIS_FUNCTIONS
                        Number of basis functions for describing each atomic pair (default: 20)
  --centers CENTERS     Custom centers to use (JSON format, list of Angstroms) (default: None)
  --widths WIDTHS       Custom widths to use (JSON format, list of Angstroms) (default: None)
  --basis-function BASIS_FUNCTION
                        Basis function ("gaussian_basic" or "gaussian_normalized") (default: gaussian_basic)
  --atomic-types ATOMIC_TYPES
                        Atomic types to include in the descriptors (JSON format, list of atomic numbers). Default: the atomic types of the input (default: None)
```
