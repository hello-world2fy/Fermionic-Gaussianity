"""Physical spectral measurements of circulating pair occupation differences.

The lower band is represented by holes. Its physical electron chirality is
minus the identically defined hole chirality. No Binder sign is a cat test.
"""
import numpy as np
from scipy.linalg import eigh


def fixed_pair_operator(subsets,r,kernel):
    masks=[sum(1<<j for j in s) for s in subsets];lookup={mask:i for i,mask in enumerate(masks)}
    matrix=np.zeros((len(masks),len(masks)),complex)
    for col,mask in enumerate(masks):
        for j in range(r):
            if not mask&(1<<j):continue
            matrix[col,col]+=kernel[j,j]
            for i in range(r):
                if mask&(1<<i):continue
                matrix[lookup[mask^(1<<j)^(1<<i)],col]+=kernel[i,j]
    return matrix


class ChiralitySpectrum:
    def __init__(self,model):
        self.model=model;r=model.r;angle=2*np.pi*np.arange(r)/r
        x=np.sqrt(2/r)*np.cos(angle);y=np.sqrt(2/r)*np.sin(angle)
        self.kernel=1j*(np.outer(x,y)-np.outer(y,x));self.blocks=[]
        reflection=(-np.arange(r))%r
        for m,(subsets,indices) in enumerate(model.blocks):
            C=fixed_pair_operator(subsets,r,self.kernel);values,U=eigh(C)
            mask_lookup={sum(1<<j for j in s):a for a,s in enumerate(subsets)}
            permutation=np.array([mask_lookup[sum(1<<int(reflection[j]) for j in s)] for s in subsets])
            phase=np.array([np.exp(1j*sum(angle[list(s)])) for s in subsets])/np.sqrt(len(subsets))
            plus=np.outer(phase,phase.conj());minus=plus.conj()
            V=np.column_stack((plus.ravel(),minus.ravel()));gram=V.conj().T@V
            self.blocks.append({'indices':indices,'operator':C,'values':values,'vectors':U,
                'reflection':permutation,'positive_branch':plus,'negative_branch':minus,
                'branch_vectors':V,'branch_gram_inverse':np.linalg.pinv(gram,rcond=1e-12)})

    def transform(self,state):
        state=np.asarray(state)/np.linalg.norm(state)
        return [b['vectors'].conj().T@state[b['indices']]@b['vectors'].conj() for b in self.blocks]

    def evaluate(self,state,threshold=.15):
        state=np.asarray(state)/np.linalg.norm(state);r=self.model.r
        rows=[];second=0.;fourth=0.;tail=0.;reflection=0.;cat_projection=0.;hist=np.zeros(81)
        edges=np.linspace(-1,1,82)
        for m,b in enumerate(self.blocks):
            C=state[b['indices']];U=b['vectors'];D=U.conj().T@C@U.conj();p=abs(D)**2
            values=b['values'];physical=values[:,None]-values[None,:];probability=float(p.sum())
            s2=float(np.sum(p*physical**2));s4=float(np.sum(p*physical**4))
            cutoff=threshold*r;positive=physical>cutoff;negative=physical < -cutoff
            pos=float(p[positive].sum());neg=float(p[negative].sum())
            amplitudes=b['branch_vectors'].conj().T@C.ravel()
            projection=float(np.vdot(amplitudes,b['branch_gram_inverse']@amplitudes).real)
            R=b['reflection'];parity=np.vdot(C,C[np.ix_(R,R)])
            reference=b['positive_branch'];ref_D=U.conj().T@reference@U.conj();ref_p=abs(ref_D)**2
            ref_mean=float(np.sum(ref_p*physical));ref_var=float(np.sum(ref_p*physical**2)-ref_mean**2)
            branch_overlap=abs(np.vdot(b['positive_branch'],b['negative_branch']))
            rows.append({'m':m,'probability':probability,'chi_second_full_weight':s2,
              'chi_fourth_full_weight':s4,'positive_tail_probability':pos,'negative_tail_probability':neg,
              'two_circular_orbit_projection_probability':projection,
              'two_circular_orbit_conditional_projection':projection/probability if probability>1e-12 else None,
              'circular_branch_coherence':float(2*np.real(amplitudes[0].conj()*amplitudes[1])) if branch_overlap<1e-10 else None,
              'circular_branch_overlap':float(branch_overlap),'positive_branch_chi_mean':ref_mean,
              'positive_branch_chi_variance':ref_var,'max_abs_chi_eigenvalue':float(np.max(abs(physical)))})
            hist+=np.histogram((physical/r).ravel(),bins=edges,weights=p.ravel())[0]
            second+=s2;fourth+=s4;tail+=pos+neg;reflection+=parity.real
            if 0<m<r:cat_projection+=projection
        return {'chi_mean':float(sum(np.sum(abs(D)**2*(b['values'][:,None]-b['values'][None,:])) for D,b in zip(self.transform(state),self.blocks))),
          'chi_second':second,'chi_fourth':fourth,'chi_second_per_r_squared':second/r**2,
          'chi_binder_diagnostic_only':1-fourth/(3*second**2) if second>1e-12 else None,
          'threshold_per_r':threshold,'macroscopic_tail_probability':tail,
          'reflection_expectation':float(reflection),'two_circular_orbit_projection_excluding_endpoints':cat_projection,
          'histogram_edges':edges.tolist(),'histogram_probabilities':hist.tolist(),'sectors':rows}

    def dephase_sign_components(self,state,threshold=.15):
        """Three orthogonal components: chi<-cr, central, chi>cr; no postselection."""
        state=np.asarray(state)/np.linalg.norm(state);parts=np.zeros((3,self.model.dim),complex)
        for b in self.blocks:
            U=b['vectors'];D=U.conj().T@state[b['indices']]@U.conj()
            chi=b['values'][:,None]-b['values'][None,:];c=threshold*self.model.r
            labels=np.where(chi < -c,0,np.where(chi > c,2,1))
            for group in range(3):parts[group,b['indices']]=U@(D*(labels==group))@U.T
        return parts
