#!/usr/bin/env python3

import torch
import ase.io
from ase.units import Bohr
import argparse
import json
import sys
from typing import Union, Sequence, Callable

def gaussian_basic(
    distances: Union[float, Sequence[float]],
    centers: Sequence[float],
    widths: Sequence[float],
) -> torch.Tensor:
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances)
    if not isinstance(centers, torch.Tensor):
        centers = torch.tensor(centers)
    if not isinstance(widths, torch.Tensor):
        widths = torch.tensor(widths)
    
    return(torch.exp(-(distances.view(-1, 1) - centers)**2/widths))

def gaussian_normalized(
    distances: Union[float, Sequence[float]],
    centers: Sequence[float],
    widths: Sequence[float],
) -> torch.Tensor:
    if not isinstance(distances, torch.Tensor):
        distances = torch.tensor(distances)
    if not isinstance(centers, torch.Tensor):
        centers = torch.tensor(centers)
    if not isinstance(widths, torch.Tensor):
        widths = torch.tensor(widths)
    
    sqrt_2_pi = 2.5066282746310002
    return(1/(widths*sqrt_2_pi)*torch.exp(-1/2*((distances.view(-1, 1) - centers)/widths)**2))

class DDD(object):
    r"""Distance Distribution Descriptors.

    3D descriptors based on radial distribution functions
    and inspired by SpookyNet (https://github.com/OUnke/SpookyNet).
    A summation over all atomic pairs of same type (same elements) is done
    to ensure symmetry invariances (translations, rotations, inversion, permutations).

    Parameters
    ----------
    cutoff : float, optional
        Distance cutoff used for considering an atomic pair (in Angstrom).
        Default value: 10 Bohr.
    nb_basis_functions : int, default=20
        Number of basis functions to use per atomic pair type.
    centers : array_like, optional
        Custom centers for the basis function.
        Default: uniformly distributed centers (linear scale).
        Size: (`nb_basis_functions`,)
    widths : array_like, optional
        Custom widths for the basis function.
        Default: i-th width is taken as the distance between
        the i-th center and its closest adjacent center.
        Size: (`nb_basis_functions`,)
    basis_function : callable, optional
        Basis function to use.
        Signature: func(centers, widths) -> basis_func_values.
        Default: gaussian basis function.
    supported_Z : array_like, optional
        Atomic numbers to support (defines the size of the descriptors).
        Default: On-the-fly determination.

    Returns
    -------
    None
    """
    def __init__(
        self,
        min_cutoff: float = 0,
        cutoff: float = 10*Bohr,
        nb_basis_functions: int = 20,
        centers: Sequence[float] = None,
        widths: Sequence[float] = None,
        basis_function: Callable = None,
        supported_Z: Sequence[int] = None,
    ) -> None:
        self.min_cutoff = min_cutoff
        self.cutoff = cutoff
        self.nb_basis_functions = nb_basis_functions
        
        if centers:
            self.nb_basis_functions = len(centers)
            self.centers = torch.tensor(centers)
        else:
            self.centers = torch.linspace(self.min_cutoff, self.cutoff, self.nb_basis_functions)
        
        if widths:
            self.widths = torch.ones_like(self.centers) * torch.tensor(widths)
        else:
            dist_to_next = torch.abs(self.centers - torch.roll(self.centers, -1))
            dist_to_prev = torch.abs(self.centers - torch.roll(self.centers, +1))
            self.widths = torch.min(dist_to_next, dist_to_prev)
        
        self.basis_function = basis_function or gaussian_basic
        
        self.sorted_Z = sorted_Z
        if self.sorted_Z:
            self.sorted_Z = torch.sort(torch.tensor(self.sorted_Z))

    def compute_descriptors(
        self,
        atomic_numbers: Sequence[int],
        distance_matrix: Sequence[Sequence[float]],
    ) -> torch.Tensor:
        r"""Compute descriptors of a specific input.

        Distance smearing with summation over atomic pairs.
        Example: if input has 3 atomic types A, B and C (sorted by Z),
        and the basis set size chosen is 20, then the descripting vector is:
        [[A-A, A-B, A-C]
         [ ∅ , B-B, B-C]
         [ ∅ ,  ∅ , C-C]] where A-B is a 20-size vector corresponding the smearing
        of all A-B distances within cutoff(s) summed together.
        This matrix is flatten into a single vector without redeundancies:
        [A-A]+[A-B]+[A-C]+[B-B]+[B-C]+[C-C] (where + is the concatenation operator)

        Parameters
        ----------
        atomic_numbers : array_like, optional
            Atomic numbers, one per atom.
            Size: (`nb_atoms`,)
        distance_matrix : array_like, optional
            Distance matrix, same order as `atomic_numbers`.
            Note: only upper triangle is read.
            Size: (`nb_atoms`, `nb_atoms`)

        Returns
        -------
        descriptors : array_like
        """
        atomic_numbers = torch.tensor(atomic_numbers)
        distance_matrix = torch.tensor(distance_matrix)
        
        sorted_Z = self.sorted_Z
        if not sorted_Z:
            sorted_Z = torch.sort(torch.tensor(set(atomic_numbers)))
        
        ZZ_ddd = self.centers.new_zeros((len(sorted_Z), len(sorted_Z), len(self.centers)))
        for i, Z1 in enumerate(sorted_Z):
            for j, Z2 in enumerate(sorted_Z):
                if j < i:
                    continue
                pair_distances = distance_matrix[atomic_numbers == Z1][:, atomic_numbers == Z2]
                pair_distances = pair_distances[torch.triu(pair_distances) > 0]
                ZZ_ddd[i,j] = self.basis_function(pair_distances, self.centers, self.widths).sum()
        
        return(ZZ_ddd.flatten())

if __name__ == '__main__':
    # Processing arguments
    print('Processing arguments')
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=str, help='Input file path (any format recognized by ASE)', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output file path (any format recognized by ASE)', default=None)
    parser.add_argument('--min-cutoff', type=float, help='Minimum distance cutoff (Angstroms)', default=0)
    parser.add_argument('--cutoff', type=float, help='Maximum distance cutoff (Angstroms). Default: 10 Bohr', default=5.291772105638412)
    parser.add_argument('--nb-basis-functions', '--size', type=int, help='Number of basis functions for describing each atomic pair', default=20)
    parser.add_argument('--centers', type=str, help='Custom centers to use (JSON format, list of Angstroms)', default=None)
    parser.add_argument('--widths', type=str, help='Custom widths to use (JSON format, list of Angstroms)', default=None)
    parser.add_argument('--basis-function', type=str, help='Basis function ("gaussian_basic" or "gaussian_normalized")', default='gaussian_basic')
    parser.add_argument('--atomic-types', type=str, help='Atomic types to include in the descriptors (JSON format, list of atomic numbers). Default: the atomic types of the input', default=None)
    args = parser.parse_args()
    
    # Load JSON data if requested
    if args.centers is not None:
        args.centers = json.loads(args.centers)
    if args.widths is not None:
        args.widths = json.loads(args.widths)
    if args.atomic_types is not None:
        args.atomic_types = json.loads(args.atomic_types)

    # Define basis function
    functions = {
        "gaussian_basic": gaussian_basic,
        "gaussian_normalized": gaussian_normalized,
    }
    try:
        args.basis_function = functions[args.basis_function]
    except IndexError as err:
        print(f"""Basis function not supported: {args.basis_function}.
                  Run {sys.argv[0]} -h for help""", file=sys.stderr)
        raise(err)

    # Read input file
    print('Reading input file')
    atoms = ase.io.read(args.input)
    
    # Compute descriptors
    print('Computing descriptors')
    ddd = DDD(
        min_cutoff=args.min_cutoff,
        cutoff=args.cutoff,
        nb_basis_functions=args.nb_basis_functions,
        centers=args.centers,
        widths=args.widths,
        basis_function=args.basis_function,
        supported_Z=args.atomic_types,
    )
    descriptors = ddd.compute_descriptors(
        atomic_numbers=atoms.get_atomic_numbers(),
        distance_matrix=atoms.get_all_distances(),
    )
    
    # Write descriptors
    print('Writing descriptors')
    with open(args.output, 'w') as out_file:
        for i, desc in enumerate(descriptors):
            if desc != 0 or i+1 == len(descriptors):
                out_file.write(f"{i+1}:{desc} ")
    print(f"Descriptors successfully written in: {args.output}")
    
