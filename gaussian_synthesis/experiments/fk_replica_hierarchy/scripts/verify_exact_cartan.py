#!/usr/bin/env python3
"""Replay every modular residue and additionally test A P V=0 modulo p."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import sys
import time
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','1')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.cartan_sparse import integer_casimir_defect,replica_frame,possible_defect_roots,EVEN_MASKS,CAYLEY_MASKS,CAYLEY_SIGNS
from fk4.io import write_json


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    started=time.monotonic()
    proof=json.loads((args.run/'proof.json').read_text())
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    assert proof['status']=='complete'
    k=proof['k']
    A,codes,tuples,multiplicities,report=integer_casimir_defect(k)
    _,rows,powers,_=replica_frame(k,tuples)
    roots=possible_defect_roots(k)
    assert roots==proof['defect_roots']
    denominator=math.prod(roots)
    assert denominator==int(proof['projector_denominator'])
    bound=denominator*max(math.factorial(k)//math.prod(math.factorial(int(v)) for v in power)*14**int(power[1]) for power in powers)
    assert bound==int(proof['gram_numerator_absolute_bound'])
    assert math.prod(proof['primes'])>2*bound
    categories=np.full(128,-1,int)
    signs=np.zeros(128,int)
    categories[0],signs[0]=0,1
    categories[-1],signs[-1]=2,-1
    ids=np.searchsorted(EVEN_MASKS,CAYLEY_MASKS)
    categories[ids],signs[ids]=1,CAYLEY_SIGNS
    lookup={tuple(power):i for i,power in enumerate(powers)}
    columns=np.array([lookup[tuple(np.bincount(categories[tuples[r]],minlength=3))] for r in rows])
    frame_signs=np.array([np.prod(signs[tuples[r]]) for r in rows],dtype=np.int64)
    sign_gram=np.zeros((len(powers),len(rows)),np.int64)
    sign_gram[columns,np.arange(len(rows))]=frame_signs*multiplicities[rows]
    numerator=[[int(v) for v in row] for row in proof['gram_numerator']]
    assert max(abs(v) for row in numerator for v in row)<=bound
    for prime in proof['primes']:
        assert all(prime%j for j in range(2,math.isqrt(prime)+1))
        assert (report['max_absolute_row_sum']+max(roots))*prime<2**63
        assert int(np.max(np.sum(abs(sign_gram),axis=1)))*prime<2**63
        values=np.zeros((len(tuples),len(powers)),np.int64)
        values[rows,columns]=frame_signs%prime
        for root in roots:
            values=(root*values-A@values)%prime
        assert not np.any((A@values)%prime),'omitted Casimir eigenspace or wrong action'
        residue=(sign_gram@values[rows])%prime
        np.testing.assert_array_equal(residue,np.load(args.run/f'gram_mod_{prime}.npy'))
        np.testing.assert_array_equal(residue,np.array([[v%prime for v in row] for row in numerator],np.int64))
        print('verified prime',prime,flush=True)
    result={'artifact_verification_passed':True,'all_primes_replayed':proof['primes'],
            'all_modular_projected_defect_residuals_zero':True,'CRT_unique_by_a_priori_bound':True,
            'integer_overflow_bounds_checked':True,'source_files_checked':len(manifest['source_sha256']),
            'elapsed_seconds':time.monotonic()-started,
            'representation_assumption':'Casimir weight bounds and coherent Cartan projector as derived in CARTAN_THEORY.md'}
    write_json(args.run/'verification.json',result)
    print(json.dumps(result))


if __name__=='__main__':
    main()
