#!/usr/bin/env python3
"""Attach bounded follow-ups to the eighth-round audit without rewriting data."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parents[1]/'results'
out=root/'assessment_nonlinear_channels_01'
audit=json.loads((out/'audit.json').read_text())
canonical=json.loads((root/'angular_canonical_pilot_01/summary.json').read_text())
spaces=json.loads((root/'angular_canonical_pilot_01/span_comparison.json').read_text())
aliases=json.loads((root/'angular_aliases_01/evaluation.json').read_text())
projective=json.loads((root/'projective_channels_refinement_01/validation.json').read_text())
span_files=sorted((root/'nonlinear_channel_spans_01').glob('L*.json'))
spans=[json.loads(p.read_text()) for p in span_files]
audit.update({'canonical_pilot':{'points':3,'max_residual':max(x['residual'] for x in canonical),
             'max_direct_witness_span_projector_difference':spaces['max_projector_difference'],
             'measure_samples_used':False,'classification':'Posthoc training-geometry pilot; no unique selected-space information at these points.'},
              'projective_refinements':projective['checks'],
              'generated_span_diagnostics':[{k:r[k] for k in ('L','eta','q','dimension','full_dimension','initial_state_error','relative_H_closure_residual')}|{'time2':r['rows'][40]} for r in spans],
              'angular_alias_first_orders':[{k:r[k] for k in ('L','r','first_orders')} for r in aliases['results']],
              'tests':{'passed':57,'seconds':4.901,'log':'tests.log'},
              'remaining_gate':'Identify a physically robust nonlinear coherent observable and show an actual canonical-derived prediction beyond conventional nonlinear/Gaussian-span controls; not achieved here.'})
(out/'audit.json').write_text(json.dumps(audit,indent=2))
(out/'tests.log').write_text((root/'nonlinear_channels_tests_01.log').read_text())
spectra=json.loads((out/'fourier_diagnostics.json').read_text())
cases=json.loads((out/'evaluation.json').read_text())
audit['max_exact_norm_error']=max(row['norm_error'] for case in cases for row in case['exact'])
audit['max_exact_energy_drift']=max(abs(row['energy']-case['exact'][0]['energy']) for case in cases for row in case['exact'])
audit['max_DF_normalized_energy_drift']=max(abs(row['energy']-case['predictions'][method][0]['energy']) for case in cases for method in ('full_cc','source_only_cc') for row in case['predictions'][method])
initial_errors=[]
for case in cases:
    tag=f"L{case['L']}_eta{case['eta']:g}"
    exact=np.load(root/'nonlinear_channels_01'/f'{tag}_exact.npz')['states'][0]
    for method in ('full_cc','source_only_cc','riccati'):
        s=np.load(root/'nonlinear_channels_01'/f'{tag}_{method}.npz')['states'][0]
        initial_errors.append(float(np.linalg.norm(s-exact)))
    s=np.load(root/'projective_channels_01'/f'{tag}_projective.npz')['states'][0]
    initial_errors.append(float(np.linalg.norm(s-exact)))
audit['max_shared_initial_state_difference']=max(initial_errors)
(out/'audit.json').write_text(json.dumps(audit,indent=2))
case=next(x for x in cases if x['L']==14 and x['eta']==1)
selected=[next(x for x in spans if x['L']==14 and x['eta']==1 and x['q']==q) for q in (2,4)]
t=np.array([x['time'] for x in case['exact']])
fig,axes=plt.subplots(2,2,figsize=(12,8))
ax=axes[0,0];rvalues=[5,7,9];components=[]
for r in rvalues:
    row=next(x for x in spectra if x['L']==2*r and x['eta']==1)['rows'][40]
    w=np.array(row['diagonal_probabilities'])
    bright=w[1]+w[-1];third=w[3%r]+w[(-3)%r];uniform=w[0]
    components.append([bright,third,uniform,float(w.sum()-bright-third-uniform)])
bottom=np.zeros(3)
for i,(label,c) in enumerate(zip(('Source indices ±1','Indices ±3','Grid index 0','Other grid indices'),('#0072b2','#d55e00','#009e73','#cc79a7'))):
    values=np.array(components)[:,i];ax.bar(np.arange(3),values,bottom=bottom,label=label,color=c);bottom+=values
ax.set_xticks(range(3),['L=10','L=14','L=18']);ax.set_title('Exact m=1 weights at t=2');ax.set_ylabel('probability in the full state');ax.legend(fontsize=8)
ax.set_ylim(0,1.12*max(bottom))
for ax,key,title in [(axes[0,1],'variance','L=14: counting variance'),(axes[1,0],'count_TV','L=14: counting distribution error'),(axes[1,1],'m1_dark_probability','L=14: probability outside source channels')]:
    if key!='count_TV':ax.plot(t,[x[key] for x in case['exact']],color='black',lw=2,label='Exact')
    ax.plot(t,[x[key] for x in case['predictions']['full_cc']],color='#d55e00',ls='--',label='Variational pair-CC')
    for r,c in zip(selected,('#56b4e9','#009e73')):
        ax.plot(t,[x[key] for x in r['rows']],color=c,label=f"Gaussian span q={r['q']} (dim {r['dimension']})")
    if key=='variance':
        ax.plot(t,[x[key] for x in case['stochastic']['discrete']],color='#cc79a7',ls=':',label='DTWA')
    ax.set_title(title);ax.set_xlabel('time');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Post-baseline controls: extra coherent coefficients and finite-grid aliases',fontsize=13)
fig.tight_layout()
for ext in ('png','pdf'):fig.savefig(out/f'channel_coherence_and_aliases.{ext}',dpi=170)
print(json.dumps({'canonical_residual':audit['canonical_pilot']['max_residual'],
                  'span_difference':spaces['max_projector_difference'],
                  'projective_refinement_max':max(x['max_normalized_ray_difference'] for x in projective['checks'])}))
