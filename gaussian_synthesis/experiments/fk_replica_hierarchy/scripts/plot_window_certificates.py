#!/usr/bin/env python3
"""Visualize strict window bounds separately from sampled distributions."""
from pathlib import Path
import argparse
import json
import os
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    s=json.loads((args.run/'summary.json').read_text())
    assert json.loads((args.run/'verification.json').read_text())['artifact_verification_passed']
    rows=[r for r in s['rows'] if r.get('cost_threshold_kind')=='certified_constructive_upper']
    assert rows
    os.environ['MPLCONFIGDIR']=str(args.run/'.mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ts=sorted({r['t'] for r in rows});orders=sorted({r['source_direction_order'] for r in rows})
    colors={k:plt.get_cmap('tab10')(i) for i,k in enumerate(orders)}
    fig,axes=plt.subplots(2,len(ts),figsize=(4*len(ts),7),sharex=True,squeeze=False)
    for col,t in enumerate(ts):
        for k in orders:
            rs=sorted([r for r in rows if r['t']==t and r['source_direction_order']==k],key=lambda r:r['delta'])
            if not rs:continue
            name='Filled reference' if k==0 else f'Direction k={k}'
            for row,field in enumerate(['restricted_support_upper','fraction_upper']):
                axes[row,col].plot([r['delta'] for r in rs],[r[field] for r in rs],color=colors[k],linestyle='--' if k==0 else '-',label=name)
        axes[0,col].set(title=f't={t:g}',ylabel='Window overlap upper bound',ylim=(-.02,1.04))
        lowest=min(r['fraction_upper'] for r in rows if r['t']==t)
        axes[1,col].set(ylabel='Window coefficient fraction upper',ylim=(max(0,lowest-.025),1.005),xlabel='Energy window Delta')
        for ax in axes[:,col]:
            ax.set_xscale('log');ax.axhline(1,color='gray',linewidth=.6);ax.grid(alpha=.2)
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([],[],color=colors[k],linestyle='--' if k==0 else '-',label='Filled reference' if k==0 else f'Direction k={k}') for k in orders],
               loc='upper center',ncol=4,fontsize=9)
    fig.tight_layout(rect=(0,0,1,.9))
    for ext in ['png','pdf']:fig.savefig(args.run/f'window_necessity.{ext}',dpi=180)
    report=args.run/'REPORT.md';addition='\n![严格能窗组成界](window_necessity.png)\n\n图中全部witness采用同一精确q4归一化；U为独立Gaussian修补构造的已认证成本上界。比例上界1表示无排除信息。每个下方面板的纵轴范围已标出，不能把小幅改善解释成强接触恢复。\n'
    if 'window_necessity.png' not in report.read_text():report.write_text(report.read_text()+addition)


if __name__=='__main__':main()
