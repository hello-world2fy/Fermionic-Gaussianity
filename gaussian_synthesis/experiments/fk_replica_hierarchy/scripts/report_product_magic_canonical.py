#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import os
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text())
    v=json.loads((args.run/'verification.json').read_text())
    assert s['status']=='complete' and v['artifact_verification_passed']
    os.environ['MPLCONFIGDIR']=str(args.run/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows=s['rows']
    fig,axes=plt.subplots(1,2,figsize=(11,4.4))
    k=[r['k'] for r in rows]
    axes[0].errorbar(k,[r['B_mean'] for r in rows],yerr=[r['B_95_percent_halfwidth'] for r in rows],fmt='o-',capsize=3,label='Canonical B estimate / finite orbit cost')
    axes[0].plot(k,[r['L_canonical_point_mean'] for r in rows],'o--',fillstyle='none',label='Canonical L estimate (uncertified at high k)')
    axes[0].axhline(2,color='black',label='Exact support lower = best upper = 2')
    axes[0].set(xscale='log',xlabel='Replica order k',ylabel='Gaussian coefficient cost')
    axes[0].legend(fontsize=8)
    eta=[np.mean([o['full_distribution']['feature_means'][3] for o in r['observations']]) for r in rows]
    upper=[2*(1-2/r['B_mean']) for r in rows]
    axes[1].plot(k,eta,'o-',label='Canonical P mean eta')
    axes[1].plot(k,upper,'s--',label='Sharp all-decomposition necessity upper')
    axes[1].set(xscale='log',yscale='log',xlabel='Replica order k',ylabel='Cross-cluster covariance weight eta')
    axes[1].legend(fontsize=8)
    for ax in axes:ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.run/f'endpoint_hierarchy.{ext}',dpi=180)
    plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4.3))
    high=[r for r in rows if r['k']>=8]
    ax.errorbar([1/r['k'] for r in high],[r['k']*(r['B_mean']-2) for r in high],
                yerr=[r['k']*r['B_95_percent_halfwidth'] for r in high],fmt='o-',capsize=3,label='Independent canonical P sampling')
    ax.axhline(22,color='black',linestyle='--',label='Predicted codimension / 2 = 22')
    ax.set(xlabel='1/k',ylabel='k (B_k - 2)',title='Fixed t=0 direction: B_k = 2 + 22/k + O(1/k²)')
    ax.grid(alpha=.2);ax.legend(fontsize=9);fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(args.run/f'endpoint_asymptotics.{ext}',dpi=180)
    lines=['# t=0：精确方向的 canonical hierarchy','',
      '有限 Gaussian stabilizer 投影给出严格恒等式 C_k(s0)/q_k(s0)=s0，因此无需用噪声优化估计端点方向。P_k 使用 s0 的确定方向，B_k=1/E_P|<g|s0>|。五个独立采样重复提供成本与分布的经验误差。','',
      '每个有限粒子集还给出一个实际256N分支的完整 Gaussian 表示，成本就是表中每次重复的B估计。该构造的重构独立于积分归一化误差。逐粒子Gaussian纯度、全部256N项之和、系数成本和物理必要性包络均已重新核验。','',
      '| k | canonical L点估计 | B均值 | B重复95%半宽 | 最差完整分支残差 | 实际Gaussian项数/重复 |',
      '|---:|---:|---:|---:|---:|---:|']
    for r in rows:
        keys={o['key'] for o in r['observations']}
        residual=max(o['all_branch_reconstruction_residual'] for o in v['observations'] if o['key'] in keys)
        lines.append(f"| {r['k']} | {r['L_canonical_point_mean']:.8g} | {r['B_mean']:.8g} | {r['B_95_percent_halfwidth']:.4g} | {residual:.3g} | {r['actual_branches_per_empirical_decomposition']} |")
    lines+=['','![端点 hierarchy](endpoint_hierarchy.png)','',
      '![由接触余维预言的收敛系数](endpoint_asymptotics.png)','',
      '固定端点方向的完整Gaussian切空间Hessian有12个零方向、44个正法向方向，Morse–Bott展开给出 B_k=2+22/k+O(k^−2)。22由几何推导，未从图中拟合；证明见PRODUCT_MAGIC_ASYMPTOTICS.md。尚无有限k余项包围，因此此式不代替积分认证。','',
      '精确支持归一化给出的L=2与独立最优U=2是另一条认证路线，图中和canonical L点估计分开显示。k>4的q积分、估计的canonical径向尺度和B积分误差仍未做区间认证。有限表示的代数重构恒等式不等于宣称保存的近似数值u具有精确C_k(u)=s0。','',
      '256N项按“父Gaussian ket + 有限酉变换 + 系数”因式化保存。没有把一个投影向量计成一个Gaussian项，也没有用这项数代替E5预算。旧的未通过k8随机重构运行保持原样。']
    (args.run/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':main()
