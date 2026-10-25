"""Full-Gaussian pair-correlator symbols with all quadratic intermediate sectors.

Computes M_ij=d2 integral <g|s_i^+ s_j^-|g> |<g|u>|^2 <g|u> g.
The witness and final output are paired and number-fixed. Intermediate
two-copy amplitudes include pair numbers r-1/r+1 and broken-pair sectors.
Dense pair-index scaling is exponential and restricted to L<=14.
"""
from itertools import combinations
from math import comb
import numpy as np
from scipy.special import beta
from .projected_gaussian import torch_module


def masks_of(L,number,excluded=()):
    return np.array(sorted(sum(1<<j for j in group) for group in combinations([j for j in range(L) if j not in excluded],number)),dtype=np.int64)


def layout(left,right,L):
    a=np.repeat(np.arange(len(left),dtype=np.int32),len(right))
    b=np.tile(np.arange(len(right),dtype=np.int32),len(left))
    ma,mb=left[a],right[b];difference=ma^mb
    keys=(ma&mb)|(difference<<L);size=np.bitwise_count(difference)
    return a,b,ma,keys,size


class PairSymbol2:
    def __init__(self,model,device='cuda:0'):
        if model.L>14:raise ValueError('Dense pair-index implementation supports L<=14')
        self.model=model;self.L=model.L;self.r=model.r;self.n=2*model.L
        self.torch=torch_module();self.torch.set_num_threads(1);self.device=device
        self.source=[]
        a,b,ma,keys,size=layout(model.masks,model.masks,self.L)
        for s in range(0,self.L+1,2):
            chosen=np.flatnonzero(size==s);chosen=chosen[np.lexsort((ma[chosen],keys[chosen]))]
            width=comb(s,s//2);aa=a[chosen].reshape(-1,width);bb=b[chosen].reshape(-1,width)
            unique=keys[chosen].reshape(-1,width)[:,0];abstract=masks_of(s,s//2)
            distance=np.bitwise_count(abstract[:,None]^abstract[None,:])
            K=beta(distance+.5,s-distance+.5)/np.pi
            self.source.append({'s':s,'a':aa,'b':bb,'keys':unique,'K':K})
        self.lower=masks_of(self.L,self.r-1);self.upper=masks_of(self.L,self.r+1)
        self.lower_index={int(m):i for i,m in enumerate(self.lower)};self.upper_index={int(m):i for i,m in enumerate(self.upper)}
        self.shift=[];a,b,ma,keys,size=layout(self.lower,self.upper,self.L)
        for source in self.source[1:]:
            s=source['s'];chosen=np.flatnonzero(size==s);chosen=chosen[np.lexsort((ma[chosen],keys[chosen]))]
            width=comb(s,s//2-1);aa=a[chosen].reshape(-1,width);bb=b[chosen].reshape(-1,width)
            unique=keys[chosen].reshape(-1,width)[:,0];rows=np.searchsorted(source['keys'],unique)
            assert np.array_equal(source['keys'][rows],unique)
            output=masks_of(s,s//2-1);input_masks=masks_of(s,s//2)
            distance=np.bitwise_count(output[:,None]^input_masks[None,:]);K=beta(distance+.5,s-distance+.5)/np.pi
            self.shift.append({'a':aa,'b':bb,'input_a':source['a'][rows],'input_b':source['b'][rows],'K':K})

    def tensor(self,array):return self.torch.as_tensor(array,device=self.device)

    def pair_matrices(self,u):
        torch=self.torch;v=self.tensor(np.asarray(u,complex));S=torch.zeros((len(u),len(u)),dtype=torch.complex128,device=self.device)
        for block in self.source:
            a,b=self.tensor(block['a'].astype(np.int64)),self.tensor(block['b'].astype(np.int64))
            K=self.tensor(block['K']).to(torch.complex128);S[a,b]=(v[a]*v[b])@K.T
        shift=torch.zeros((len(self.lower),len(self.upper)),dtype=torch.complex128,device=self.device)
        for block in self.shift:
            a,b=self.tensor(block['input_a'].astype(np.int64)),self.tensor(block['input_b'].astype(np.int64))
            aa,bb=self.tensor(block['a'].astype(np.int64)),self.tensor(block['b'].astype(np.int64))
            K=self.tensor(block['K']).to(torch.complex128);shift[aa,bb]=(v[a]*v[b])@K.T
        return S.cpu().numpy(),shift.cpu().numpy()

    def broken_matrix(self,u,i,j):
        """S on complementary single-occupancy patterns (up,down)/(down,up).

        Each copy has r-1 intact pairs plus one electron at i and j. The
        same-spin pattern has the negative of this matrix. This includes
        the full Cartan projector, not a projector restricted to these states.
        """
        remaining=masks_of(self.L,self.r-1,excluded=(i,j));D=len(remaining)
        a,b,ma,old_keys,size=layout(remaining,remaining,self.L)
        keys=old_keys|(((1<<i)|(1<<j))<<self.L)
        torch=self.torch;v=self.tensor(np.asarray(u,complex));S=torch.zeros((D,D),device=self.device,dtype=torch.complex128)
        for source in self.source[1:]:
            s=source['s'];chosen=np.flatnonzero(size==s-2);chosen=chosen[np.lexsort((ma[chosen],keys[chosen]))]
            if not len(chosen):continue
            width=comb(s-2,(s-2)//2);aa=a[chosen].reshape(-1,width);bb=b[chosen].reshape(-1,width)
            unique=keys[chosen].reshape(-1,width)[:,0];rows=np.searchsorted(source['keys'],unique)
            assert np.array_equal(source['keys'][rows],unique)
            difference=unique>>self.L
            pi=np.bitwise_count(difference&((1<<i)-1)).astype(int)
            pj=np.bitwise_count(difference&((1<<j)-1)).astype(int)
            for pos in np.unique(pi*self.L+pj):
                selected=np.flatnonzero(pi*self.L+pj==pos);x,y=divmod(int(pos),self.L)
                inputs=masks_of(s,s//2);outputs=masks_of(s,s//2-1,excluded=(x,y))
                active=(1<<x)|(1<<y);others=((1<<s)-1)^active
                distance=np.bitwise_count((outputs[:,None]^inputs[None,:])&others)+1
                sign=(-1.)**np.bitwise_count(inputs&active)
                K=beta(distance+.5,s-distance+.5)/np.pi*sign[None,:]
                ia=self.tensor(source['a'][rows[selected]].astype(np.int64));ib=self.tensor(source['b'][rows[selected]].astype(np.int64))
                oa=self.tensor(aa[selected].astype(np.int64));ob=self.tensor(bb[selected].astype(np.int64))
                S[oa,ob]=(v[ia]*v[ib])@self.tensor(K).to(torch.complex128).T
        return remaining,S.cpu().numpy()

    def prepare(self,u):
        self.u=np.asarray(u,complex);self.S,self.shifted=self.pair_matrices(self.u);self.C=self.S@self.u.conj()

    def ladder_vector(self,site,up):
        masks=self.model.masks
        selected=np.flatnonzero(((masks>>site)&1)==(0 if up else 1))
        lookup=self.upper_index if up else self.lower_index
        destination=np.array([lookup[int(masks[k])^(1<<site)] for k in selected])
        out=np.zeros(len(self.upper) if up else len(self.lower),complex);out[destination]=self.u[selected]
        return out,selected,destination

    def pair_cross(self,left,right):
        # s_left^+ on the r-1 output, with s_right^+ u on the contracted r+1 leg.
        upper,_,_=self.ladder_vector(right,True)
        _,selected,destination=self.ladder_vector(left,False)
        result=np.zeros(self.model.dim,complex);result[selected]=(self.shifted@upper.conj())[destination]
        # s_right^- on r+1, with s_left^- u on the r-1 leg.
        lower,_,_=self.ladder_vector(left,False)
        _,selected,destination=self.ladder_vector(right,True)
        result[selected]+=(self.shifted.T@lower.conj())[destination]
        return result

    def symbol_parts(self,i,j):
        denominator=2*self.n*(2*self.n-2)
        plus_denominator=2*self.n*(2*self.n+1)
        if i==j:
            q=2*self.model.occ[:,i]-1
            Mq=(q*self.C+self.S@(q*self.u).conj())/(2*self.n)
            cross=self.pair_cross(i,i)
            M4=(6*self.C+2*q*(self.S@(q*self.u).conj())+4*cross)/denominator
            full=(self.C+2*Mq+M4)/4
            anomalous=((4+2*self.n)*self.C+4*cross+2*self.n*M4)/(4*plus_denominator)
            return {'full':full,'anomalous':anomalous,'normal_exchange':full-anomalous}
        mask=self.model.masks;source=np.flatnonzero(((mask>>j)&1)&(1-((mask>>i)&1)))
        destination=np.array([self.model.index[int(mask[k])^(1<<i)^(1<<j)] for k in source])
        OC=np.zeros(self.model.dim,complex);OC[destination]=self.C[source]
        Odagu=np.zeros(self.model.dim,complex);Odagu[source]=self.u[destination]
        same=OC+self.S@Odagu.conj();cross=self.pair_cross(i,j)
        result=3*same+cross
        remaining,broken=self.broken_matrix(self.u,i,j)
        selected=np.array([self.model.index[int(m)|(1<<i)] for m in remaining])
        result[selected]-=4*(broken@self.u[selected].conj())
        full=result/denominator
        anomalous=(same+cross+2*self.n*full)/plus_denominator
        return {'full':full,'anomalous':anomalous,'normal_exchange':full-anomalous}

    def symbol(self,i,j):return self.symbol_parts(i,j)['full']
