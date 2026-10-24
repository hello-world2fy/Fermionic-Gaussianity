#!/usr/bin/env python3
"""Recompute saved exact-witness mass estimates and physical P diagnostics."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import sys
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('NUMBA_NUM_THREADS','2')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from fk4 import paths
from fk4.io import write_json
from fk4.cartan_sparse import CartanMoments
from fk4.distributions import full_distribution
from fk4.symmetry import time_reverse
from fk.model import build_model


def main():
    p=argparse.ArgumentParser()
    p.add_argument('run',type=Path)
    args=p.parse_args()
    config=json.loads((args.run/'config.json').read_text())
    summary=json.loads((args.run/'summary.json').read_text())
    assert summary['status']=='complete'
    manifest=json.loads((args.run/'manifest.json').read_text())
    for name,digest in manifest['source_sha256'].items():
        assert hashlib.sha256((args.run/'source_snapshot'/name).read_bytes()).hexdigest()==digest
    model=build_model()
    checked=0
    lines=['# 精确 canonical witness 的独立成本积分','',
           'L 使用精确整数矩阵与区间归一化；完整态重构残差来自独立区间计算。B 的 Student-t 半宽反映五个独立 SMC 重复的经验误差，不是严格概率覆盖或严格上界。',
           '', '| t | k | L（保存目标态） | B 均值 | 95% 重复半宽 | 完整态残差上界 | 成本重复门槛 |',
           '|---:|---:|---:|---:|---:|---:|---|']
    for row in summary['rows']:
        t,k=row['t'],row['k']
        frozen=np.load(Path(config['canonical'])/f't{t:g}_k{k}_frozen.npz')
        z,u,target=frozen['z'],frozen['witness'],frozen['target']
        if k==1:
            q=float(np.vdot(z,z).real)
        elif 'exact' in config:
            matrix_directory=Path(config['exact'])
        else:
            matrix_directory=next(Path(d) for d in config['moments'] if json.loads((Path(d)/'proof.json').read_text())['k']==k)
        if k!=1:
            backend=CartanMoments(k,np.load(matrix_directory/'moment_matrix.npz')['matrix'])
            q=backend.evaluate(z,hessian=False).q
        for o in row['observations']:
            a=np.load(args.run/f"t{t:g}_k{k}_seed{o['seed']}.npz")
            if 'source_kets_file' in o:
                source=np.load(o['source_kets_file'])
                g=source[o['source_kets_key']]
                weights=a['weights']
                old=source['witness']
                ratio=(abs(g.conj()@u)/abs(g.conj()@old))**(2*k-1)
                np.testing.assert_allclose(weights,ratio/ratio.sum(),atol=1e-14)
            else:
                g=a['P_kets']
                weights=np.full(len(g),1/len(g))
            overlaps=g.conj()@u
            phases=overlaps/abs(overlaps)
            B=q/(weights@abs(overlaps))
            np.testing.assert_allclose(B,o['B_ratio'],atol=1e-12)
            np.testing.assert_allclose(phases,a['aligned_phases'],atol=1e-13)
            C=B*((weights*phases)@g)
            CRB=model.S@(model.S.conj().T@C)
            CRB=.5*(CRB+time_reverse(CRB))
            np.testing.assert_allclose(a['C_raw'],C,atol=2e-12)
            np.testing.assert_allclose(a['C_RB'],CRB,atol=2e-12)
            features,diagnostics=full_distribution(g,model.j_matrices)
            if 'source_kets_file' in o:
                np.testing.assert_allclose(weights@features['features'],o['P_feature_means'],atol=2e-11)
            else:
                np.testing.assert_allclose(features['features'],a['features'],atol=2e-11)
            assert diagnostics['max_purity_error_all_particles']<1e-9
            checked+=1
        lines.append(f"| {t:g} | {k} | {row['L_certified_saved_target']:.9g} | {row['B_mean']:.9g} | {row['B_95_percent_halfwidth']:.3g} | {row['independent_full_rho_upper']:.3g} | {row['mass_repeat_pass']} |")
    lines+=['','精确物理 Hamiltonian 基态的误差转移另见 `cartan_k2_physical_certificates_01` 与 `cartan_k4_physical_certificates_02`；所有状态距离上界低于 1.4e-14。',
            '', 'Monte Carlo 重构的噪声残差保存在每个种子的记录中；本报告的重构门槛使用与采样独立的精确矩区间验证。']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(args.run/'verification.json',{'artifact_verification_passed':True,'observations_checked':checked,
               'all_particle_purity_checked':True,'sampling_error_certified':False,
               'independent_reconstruction_certified':True,'empirical_mass_gate_passed':summary['next_order_gate_pass']})
    print('verified',checked,'observations')


if __name__=='__main__':
    main()
