#!/usr/bin/env python3
"""Exact integer Gram via modular Casimir projection and Chinese remainders.

All arithmetic before the final display matrix is integer. A priori integer
bounds from the projector contraction determine how many independent primes
are needed; no rational fitting of Monte Carlo or floating numbers is used.
"""
from pathlib import Path
import argparse
import math
import os
import sys
import time

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import source_manifest, write_json
from fk4.cartan_sparse import integer_casimir_defect, replica_frame, possible_defect_roots, CAYLEY_MASKS, CAYLEY_SIGNS, EVEN_MASKS


def prime_stream():
    sieve = np.ones(46341, bool)
    sieve[:2] = False
    for j in range(2, 216):
        if sieve[j]:
            sieve[j*j::j] = False
    divisors = np.flatnonzero(sieve)
    candidate = 2147483647
    while True:
        if all(candidate%int(p) for p in divisors if p*p <= candidate):
            yield candidate
        candidate -= 2


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--k", type=int, required=True, choices=[2, 3, 4])
    p.add_argument("--compare", type=Path)
    args = p.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    write_json(out/"manifest.json", source_manifest(out/"source_snapshot"))
    write_json(out/"config.json", {"k": args.k, "compare": str(args.compare) if args.compare else None})
    started = time.monotonic()
    A, codes, tuples, multiplicities, construction = integer_casimir_defect(args.k)
    V, support_rows, powers, _ = replica_frame(args.k, tuples)
    roots = possible_defect_roots(args.k)
    denominator = math.prod(roots)
    norm_squares = [math.factorial(args.k)//math.prod(math.factorial(int(x)) for x in power)*14**int(power[1]) for power in powers]
    numerator_bound = denominator*max(norm_squares)
    categories = np.full(128, -1, int)
    signs = np.zeros(128, int)
    categories[0], signs[0] = 0, 1
    categories[-1], signs[-1] = 2, -1
    ids = np.searchsorted(EVEN_MASKS, CAYLEY_MASKS)
    categories[ids], signs[ids] = 1, CAYLEY_SIGNS
    lookup = {tuple(power): i for i, power in enumerate(powers)}
    initial_columns, initial_signs = [], []
    for row in support_rows:
        counts = np.bincount(categories[tuples[row]], minlength=3)
        initial_columns.append(lookup[tuple(counts)])
        initial_signs.append(int(np.prod(signs[tuples[row]])))
    initial_columns = np.array(initial_columns)
    initial_signs = np.array(initial_signs, np.int64)
    sign_gram = np.zeros((len(powers), len(support_rows)), np.int64)
    sign_gram[initial_columns, np.arange(len(support_rows))] = initial_signs*multiplicities[support_rows]
    maximum_signed_sum = (construction["max_absolute_row_sum"]+max(roots))*2147483647
    if maximum_signed_sum >= 2**63:
        raise OverflowError("modular sparse multiplication lacks an int64 safety bound")
    metadata = {"status": "running", "k": args.k, "construction": construction,
                "defect_roots": roots, "projector_denominator": str(denominator),
                "gram_numerator_absolute_bound": str(numerator_bound),
                "signed_sparse_sum_bound": maximum_signed_sum,
                "uses_float_for_projection": False, "uses_rational_reconstruction_guess": False,
                "primes": []}
    write_json(out/"proof.json", metadata)
    print('construction', construction, 'roots', roots, 'denominator digits',len(str(denominator)),flush=True)
    modulus = 1
    crt = [[0 for _ in powers] for _ in powers]
    for prime in prime_stream():
        values = np.zeros((len(tuples), len(powers)), np.int64)
        values[support_rows, initial_columns] = initial_signs % prime
        for root in roots:
            values = (root*values-A@values) % prime
        residue = (sign_gram@values[support_rows]) % prime
        if not np.array_equal(residue, residue.T):
            raise ArithmeticError("modular Gram lost exact symmetry")
        np.save(out/f"gram_mod_{prime}.npy", residue)
        inverse = pow(modulus % prime, -1, prime)
        for i in range(len(powers)):
            for j in range(len(powers)):
                correction = ((int(residue[i,j])-crt[i][j]) % prime)*inverse % prime
                crt[i][j] += modulus*correction
        modulus *= prime
        metadata["primes"].append(prime)
        metadata["current_modulus"] = str(modulus)
        metadata["elapsed_seconds"] = time.monotonic()-started
        write_json(out/"proof.json", metadata)
        print('prime', prime, 'CRT digits', len(str(modulus)), 'required',len(str(2*numerator_bound)), 'seconds',time.monotonic()-started,flush=True)
        if modulus > 2*numerator_bound:
            break
    numerator = [[value if value <= modulus//2 else value-modulus for value in row] for row in crt]
    if any(abs(value) > numerator_bound for row in numerator for value in row):
        raise ArithmeticError("recovered Gram exceeds its a priori projector norm bound")
    matrix = np.empty((len(powers), len(powers)))
    scales = [math.sqrt(math.factorial(args.k)/math.prod(math.factorial(int(x)) for x in power))*14**(int(power[1])/2) for power in powers]
    for i in range(len(powers)):
        for j in range(len(powers)):
            matrix[i,j] = numerator[i][j]/denominator/scales[i]/scales[j]
    if abs(matrix[0,0]-1) > 1e-14 or abs(matrix[-1,-1]-1) > 1e-14:
        raise ArithmeticError("pure Gaussian diagonal is not exactly one")
    metadata.update(status="complete", gram_numerator=[[str(v) for v in row] for row in numerator],
                    powers=powers.tolist(), exact_integer_gram_recovered=True,
                    final_modulus_exceeds_twice_absolute_bound=True,
                    elapsed_seconds=time.monotonic()-started)
    if args.compare:
        previous = np.load(args.compare/"moment_matrix.npz")["matrix"]
        metadata["independent_chebyshev_matrix_error"] = float(np.linalg.norm(matrix-previous))
        if metadata["independent_chebyshev_matrix_error"] > 1e-9:
            write_json(out/"proof.json", metadata)
            raise ArithmeticError("exact modular projector disagrees with independent spectral filter")
    np.savez_compressed(out/"moment_matrix.npz", matrix=matrix, powers=powers)
    write_json(out/"proof.json", metadata)


if __name__ == "__main__":
    main()
