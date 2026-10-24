#!/usr/bin/env python3
from pathlib import Path
import argparse
import hashlib
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk5.paths import ROOT,PREVIOUS
from fk5.io import source_manifest,write_json
from fk5.direction_certification import certify_direction
from fk5.two_orbit import arc_projection
from fk4.exact_cartan import ExactCartan


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);write_json(args.output/'manifest.json',source_manifest(args.output/'source_snapshot'))
    initial=json.loads((ROOT/'results/initial_certificates_02/summary.json').read_text())
    exact=ExactCartan(PREVIOUS/'results/cartan_exact_k4_01/proof.json')
    construction=json.loads((ROOT/'results/two_orbit_02/summary.json').read_text())['construction']
    a=sum(construction['main_arc_parameter_interval'])/2
    A=arc_projection(a);u=-a;B=np.array([(2*u*u-1)/4,-u/np.sqrt(2),(5+2*u*u)/(4*np.sqrt(7))])
    rows=[]
    for t in [11,20]:
        for k in [8,16,32,64]:
            directory=(ROOT/'results/canonical_k64_t11_n131072_01' if (t,k)==(11,64)
                else PREVIOUS/f'results/fk_k{k}_radial_t{t}_n524288_{"02" if k==64 else "01"}')
            source=directory/f't{t}_k{k}_frozen.npz';frozen=np.load(source)
            summary=json.loads((directory/'summary.json').read_text());assert summary['status']=='complete'
            previous=summary['rows'][0]
            assert previous['all_seeds_reconstruction_pass'] and previous['mass_repeat_pass']
            spectrum=next(r for r in initial['spectra'] if r['t']==t)
            row=certify_direction(frozen['z'],t,k,spectrum,exact)
            baseline=next(r for r in initial['A1_rows'] if r['t']==t)['joint_cost_lower']
            row.update(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                analytic_joint_baseline=baseline,
                any_global_support_certificate_cannot_beat_joint=row['best_possible_direction_cost_upper']<=baseline,
                current_global_certificate_beats_joint=row['certified_direction_cost_lower']>baseline,
                independent_validation_residual=previous['rho_RB_pooled'],canonical_coefficient_cost_point=previous['B_mean'],
                canonical_L_point=previous['L_point_mean'])
            if t==20:
                main=float(np.vdot(frozen['z'],A).real);secondary=float(np.vdot(frozen['z'],B).real)
                row['two_orbit_heights']=dict(main_signed=main,secondary_signed=secondary,
                    target_alignment_signs_recovered=main>0 and secondary<0,
                    scaled_log_height_ratio=float((2*k-1)*np.log(abs(main/secondary))),
                    asymptotic_required_log_mass_ratio=float(np.log(construction['positive_orbit_mass_interval'][0]/construction['negative_orbit_mass_interval'][0])),
                    matching_candidate_global_completeness_assumed_not_proved=True)
            sampling_source=Path(summary.get('source_run',str(directory)))
            measures=[]
            for seed in [24101,24102,24103,24104,24105]:
                path=sampling_source/f't{t}_k{k}_validation_seed{seed}.npz'
                data=np.load(path);features=data['P_all_features']
                assert features.shape[1]==4 and np.isfinite(features).all()
                measures.append(dict(seed=seed,count=len(features),source=str(path),
                    features_sha256=hashlib.sha256(features.tobytes()).hexdigest(),
                    feature_means=features.mean(axis=0).tolist(),
                    mass_eta_gt={str(c):float(np.mean(features[:,3]>c)) for c in [.7,.8,.9]},
                    measure='coefficient measure proportional to overlap^(2k-1), sampled at beta=k-1/2',
                    eta_partition_is_heuristic=True))
            row['canonical_component_diagnostics']=measures
            rows.append(row);write_json(args.output/f't{t}_k{k}.json',row)
            print(t,k,'certified',row['certified_direction_cost_lower'],'absolute ceiling',row['best_possible_direction_cost_upper'],'joint',baseline,flush=True)
    write_json(args.output/'summary.json',dict(status='complete',rows=rows,
        stopping_rule='No unconditional order increase: all t20 k8..64 directions provably cannot beat the joint baseline; t11 k32/64 remain globally uncertified despite possible local improvement.',
        FK_high_order_integrals_point_estimates_only=True))


if __name__=='__main__':main()
