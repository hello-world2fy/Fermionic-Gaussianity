#!/usr/bin/env python3
"""Independent high-precision projection, spectrum and covering-cell audit."""
from pathlib import Path
import argparse,hashlib,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from mpmath import mp
from fk6.io import write_json
from fk6 import paths
from fk5.two_orbit import scaled_basis


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);args=p.parse_args()
    path=args.input/'summary.json';data=json.loads(path.read_text());mp.dps=130;cells=0
    masks=[i for i in range(256) if i.bit_count()%2==0];D=scaled_basis()[masks]
    norms=[mp.sqrt(sum(int(abs(v)**2) for v in D[:,j])) for j in range(3)]
    basis=mp.matrix([[mp.mpc(float(D[i,j].real),float(D[i,j].imag))/norms[j] for j in range(3)] for i in range(128)])
    T=mp.matrix([[mp.mpf(1)/4,-1/mp.sqrt(2),mp.sqrt(7)/4],
                 [mp.sqrt(14)/4,0,-mp.sqrt(2)/4],[mp.mpf(1)/4,1/mp.sqrt(2),mp.sqrt(7)/4]])
    for r in data['rows']:
        source=Path(r['frozen_source'])
        assert hashlib.sha256(source.read_bytes()).hexdigest()==r['frozen_sha256']
        frozen=np.load(source)['witness'];pairs=np.array(r['witness_binary64_real_imag'])
        assert np.array_equal(frozen,pairs[:,0]+1j*pairs[:,1])
        ell=mp.matrix([mp.mpc(float(a),float(b)) for a,b in pairs]);norm=mp.norm(ell)
        z=basis.H*ell;residual=mp.norm(ell-basis*z);w=T*z
        assert mp.mpf(r['outside_singlet_norm_upper'])>=residual
        assert mp.mpf(r['witness_norm_interval'][0])<=norm<=mp.mpf(r['witness_norm_interval'][1])
        t=mp.mpf(r['t']);off=3*mp.sqrt(14)
        energies,V=mp.eigsy(mp.matrix([[-14-4*t,-off,0],[-off,-50,-off],[0,-off,-14+4*t]]))
        q=V[:,0]*mp.sign(V[0,0]);Q=abs((w.H*q)[0])/norm
        X=2*mp.mpf(r['delta'])/(mp.sqrt(16*(t-9)**2+72*mp.mpf(r['delta']))+4*(t-9))
        assert mp.mpf(r['X_cover_upper'])>=min(2,X)
        N=r['partition_steps'];assert len(r['partition_support_uppers'])==N
        for j,bound in enumerate(r['partition_support_uppers']):
            left=mp.mpf(r['X_cover_upper'])*j/N;right=mp.mpf(r['X_cover_upper'])*(j+1)/N
            h4=min(1,mp.sqrt(mp.mpf(3)/2)*right)
            if right<mp.mpf('.5'):h4=min(h4,3*right/mp.sqrt(14*(1-2*right)))
            value=min(norm,abs(w[0])*(1-left/2)**2+abs(w[1])*h4+abs(w[2])*right**2/4+residual)
            assert mp.mpf(bound)>=value,(r['key'],j)
            cells+=1
        assert r['restricted_support_upper']==max(r['partition_support_uppers'])
        h=min(1,mp.mpf(r['restricted_support_upper'])/norm);B=mp.mpf(r['coefficient_budget'])
        assert mp.mpf(r['unit_witness_restricted_support_upper'])>=h
        assert mp.mpf(r['unit_witness_overlap_lower'])<=Q
        for eps,value in r['high_plain_coefficient_mass_lower'].items():
            predicted=0 if h>=1 else max(0,((Q-mp.mpf(float(eps)))/B-h)/(1-h))
            assert mp.mpf(value)<=predicted
        assert mp.mpf(r['low_only_hilbert_error_lower'])<=max(0,Q-B*h)
        b=B*h;F=1 if b>=Q else (Q*b+mp.sqrt(1-Q*Q)*mp.sqrt(1-b*b))**2
        assert mp.mpf(r['low_only_normalized_fidelity_upper'])>=F
    write_json(args.input/'verification.json',dict(passed=True,witnesses=len(data['rows']),
        checked_covering_cells=cells,independent_decimal_digits=130,
        source_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        scope='Every frozen complex ket, singlet remainder, full target spectrum coordinates, all support cells, mass/error/fidelity consequences',
        proof_scope='Previously proved FK full-Gaussian coercivity and amplitude envelopes; exact even ket plus bounded remainder; odd overlap zero'))
    print('verified',len(data['rows']),'witnesses and',cells,'global support cells',flush=True)


if __name__=='__main__':main()
