#!/usr/bin/env python3
"""Certify the exact16-term construction; locally challenge its arc dual."""
from pathlib import Path
import argparse
import json
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk5.io import source_manifest,write_json
from fk5.two_orbit import two_orbit_parameters,two_orbit_atoms,simplex_projector_identity,rotation_simplex_identity,arc_dual
from fk5.joint_dual_certificates import exact_schmidt_polar
from fk.model import build_model
from fk.gaussian import sample_gaussian
from fk3.discovery import optimize_gaussian_overlap,covariance_from_kets
from fk4.distributions import full_distribution
from fk4.geometry import contact_hessian


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--certificates',type=Path,default=Path('results/initial_certificates_02'))
    p.add_argument('--starts',type=int,default=128);args=p.parse_args()
    assert json.loads((args.certificates/'verification.json').read_text())['artifact_verification_passed']
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    base=json.loads((args.certificates/'summary.json').read_text());model=build_model()
    proof=simplex_projector_identity(model);write_json(args.output/'simplex_exact_identity.json',proof)
    write_json(args.output/'rotation_simplex_exact_identity.json',rotation_simplex_identity(model))
    spectrum=next(r for r in base['spectra'] if r['t']==20)
    construction,a,positive,negative=two_orbit_parameters(20,spectrum['ground_energy_interval'])
    atoms,c=two_orbit_atoms(model,a,positive,negative);target=model.target(20)
    construction['saved_float_reconstruction_error']=float(np.linalg.norm(c@atoms-target))
    construction['saved_float_norm_error']=float(np.max(abs(np.linalg.norm(atoms,axis=1)-1)))
    assert construction['saved_float_reconstruction_error']<1e-12
    features,quality=full_distribution(atoms,model.j_matrices)
    construction['saved_float_Gaussian_quality']=quality
    construction['features']=features['features'].tolist()
    construction['certified_gap_using_A1_lower']=float(1-next(r['joint_cost_lower'] for r in base['A1_rows'] if r['t']==20)/construction['exact_construction_cost_interval'][1])
    np.savez_compressed(args.output/'construction_t20.npz',atoms=atoms,coefficients=c,target=target,features=features['features'])
    row=next(r for r in base['A1_rows'] if r['t']==20);lam=row['lambda_numerator']/row['lambda_denominator']
    joint=np.load(args.certificates/'t20_witness.npz')['joint_witness_display']
    candidate=model.S@arc_dual(a)
    rng=np.random.default_rng(42001);random=sample_gaussian(8,args.starts,rng).kets
    starts=np.concatenate([atoms,random])
    contacts=[]
    for name,w in [('joint',joint),('arc_dual',candidate)]:
        states,search=optimize_gaussian_overlap(w,starts,max_sweeps=160,gradient_tolerance=1e-11)
        values=abs(states.conj()@w);best=int(np.argmax(values))
        hessian,H=contact_hessian(w,states[best]);f,q=full_distribution(states,model.j_matrices)
        record={'name':name,'local_maximum':float(values.max()),'local_minimum':float(values.min()),
            'full_Gaussian_support_upper':row['support_upper'] if name=='joint' else None,
            'global_support_certified':name=='joint','local_search_is_not_global_certificate':True,
            'search_summary':{k:v for k,v in search.items() if not isinstance(v,np.ndarray)},
            'best_hessian':hessian,'best_features':f['features'][best].tolist(),
            'construction_atom_overlaps':abs(atoms.conj()@w).tolist(),
            'signed_target_overlap':float(np.vdot(w,target).real)}
        contacts.append(record)
        np.savez_compressed(args.output/f'{name}_contacts.npz',witness=w,starts=starts,contacts=states,overlaps=values,features=f['features'],**H)
        print(name,'local_max',values.max(),'target_overlap',record['signed_target_overlap'],'best_eta',record['best_features'][3],flush=True)
    write_json(args.output/'summary.json',{'status':'complete','construction':construction,'contacts':contacts,
        'initial_certificate_source':str(args.certificates.resolve()),'random_seed':42001,'starts':args.starts,
        'independent_verification_required':True,'primal_dual_matching_proved':False,'elapsed_seconds':time.monotonic()-started})
    print('16-term exact construction cost',construction['exact_construction_cost_interval'],flush=True)


if __name__=='__main__':main()
