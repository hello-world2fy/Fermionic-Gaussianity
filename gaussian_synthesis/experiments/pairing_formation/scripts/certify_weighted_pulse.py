#!/usr/bin/env python3
"""Exact rational enclosure of the post-selected late-pulse PG certificate."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import sys
from pathlib import Path
import argparse,json,time
from fractions import Fraction as F
from decimal import Decimal,localcontext
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from pairing_formation.model import picket_fence
from pairing_formation.rational_evolution import integer_taylor_action


def decimal(q):
    with localcontext() as context:
        context.prec=65
        return str(Decimal(q.numerator)/Decimal(q.denominator))


def exact(q):
    return {'numerator':str(q.numerator),'denominator':str(q.denominator),'decimal_display':decimal(q)}


def main(out):
    out.mkdir(parents=True,exist_ok=False);start=time.monotonic()
    model=picket_fence(12,2.);weights=np.array([-12,-18,-36,-110,-367,-727,120,118,147,216,299,370],dtype=np.int64)
    scale=1000;degree=140
    protocol={'classification':'Retrospective certification of a fixed rational weighted observable, not a forecast.',
      'parameters':'L=12, M=6, epsilon_j=(2j+1-12)/12, G=2, t=43/20; H=A/6 with integer A.',
      'weights_numerators':weights.tolist(),'weights_denominator':scale,'Taylor_degree':degree,
      'arithmetic':'Python unbounded integers and fractions.Fraction. Decimal strings are display only.',
      'remainder':'For Hermitian A, ||exp(-i alpha A)F - sum_{k=0}^d(-i alpha A)^k F/k!|| <= (|alpha| max_i sum_j |A_ij|)^(d+1)/(d+1)!.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
    rows=[]
    for i,mask in enumerate(model.masks):
        mask=int(mask);occupied=[j for j in range(12) if mask&(1<<j)];empty=[j for j in range(12) if not mask&(1<<j)]
        row=[(i,sum(2*j+1-12 for j in occupied)-6)]
        for h in occupied:
            for p in empty:row.append((model.index[mask^(1<<h)^(1<<p)],-1))
        rows.append(row)
    # A is exactly symmetric by pair-transfer construction. Check explicitly.
    lookup=[dict(row) for row in rows]
    assert all(lookup[j].get(i,0)==v for i,row in enumerate(rows) for j,v in row)
    real,imag,den,error=integer_taylor_action(rows,model.fermi_index,F(43,120),degree)
    probabilities=[r*r+i*i for r,i in zip(real,imag)];den2=den*den
    norm=F(sum(probabilities),den2);norm_enclosure=2*error+error*error
    assert abs(norm-1)<=norm_enclosure
    occ=model.occ.astype(np.int64);x=occ@weights;y=occ@(weights*weights)
    mean=F(sum(p*int(z) for p,z in zip(probabilities,x)),den2*scale)
    quadratic=F(sum(p*int(xx*xx-yy) for p,xx,yy in zip(probabilities,x,y)),den2*scale*scale)
    a=F(5,6);center=quadratic-a*mean*mean
    # Full 12-electron sector, including singly occupied levels, for operator norms.
    numbers=np.arange(3**12,dtype=np.int64);counts=np.empty((len(numbers),12),dtype=np.int8)
    for j in range(12):counts[:,j]=numbers%3;numbers//=3
    counts=counts[counts.sum(axis=1)==12]
    X=counts@weights;Y=counts@(weights*weights);B=X*X-2*Y
    Xnorm=F(int(np.max(abs(X))),2*scale);Anorm=F(int(np.max(abs(B))),4*scale*scale)
    Xrange=F(int(np.ptp(X)),2*scale);Arange=F(int(np.ptp(B)),4*scale*scale)
    mean_error=Xnorm*norm_enclosure
    excess_error=Anorm*norm_enclosure+a*mean_error*(2*abs(mean)+mean_error)
    lower,upper=center-excess_error,center+excess_error
    assert lower>0
    # Bound range(X^2-Y-2 a mu X) using the rigorous interval for mu.
    operator_range_upper=Arange+2*a*(abs(mean)+mean_error)*Xrange
    low,high=F(0),F(1)
    for _ in range(100):
        middle=(low+high)/2
        if operator_range_upper*middle+a*Xrange*Xrange*middle*middle<lower:low=middle
        else:high=middle
    assert operator_range_upper*low+a*Xrange*Xrange*low*low<lower
    infidelity=low*low
    target=np.load(out.parent/'replica2_L12_G2_01/t2.15.npz')['target']
    q=np.array([float(F(r,den))+1j*float(F(i,den)) for r,i in zip(real,imag)])
    phase=np.vdot(q,target);difference=float(np.linalg.norm(q*phase/abs(phase)-target))
    result={'classification':protocol['classification'],'Hilbert_dimension':model.dim,
      'integer_H_row_norm_bound':max(sum(abs(v) for _,v in row) for row in rows),
      'Taylor_remainder_norm_upper':exact(error),'Taylor_norm_squared':exact(norm),
      'excess_lower':exact(lower),'excess_upper':exact(upper),'mean_center':exact(mean),'mean_error_upper':exact(mean_error),
      'full_number_X_range':exact(Xrange),'full_number_operator_range_upper':exact(operator_range_upper),
      'trace_distance_lower':exact(low),'infidelity_lower':exact(infidelity),'fidelity_upper':exact(1-infidelity),
      'float64_reference_state_difference':difference,'seconds':time.monotonic()-start,
      'exact_rational_certificate':True,'novelty_priority_established':False,'canonical_specific_prediction':False}
    (out/'certificate.json').write_text(json.dumps(result,indent=2))
    (out/'taylor_vector.json').write_text(json.dumps({'denominator':str(den),'real':[str(x) for x in real],'imaginary':[str(x) for x in imag]}))
    print(json.dumps({'excess_interval':[decimal(lower),decimal(upper)],'infidelity_lower':decimal(infidelity),
      'remainder_norm_upper':decimal(error),'float64_state_difference':difference,'seconds':result['seconds']},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args();main(args.output)
