"""DF propagation of pair singles plus frozen connected-transfer clusters.

All excitation monomials commute and are nilpotent. Euler's homogeneous
polynomial identity computes exp(K)|F> exactly in the fixed-pair basis.
This is an extension of conventional variational coupled-cluster theory,
not a novel propagation principle or a polynomial-cost implementation.
"""
import numpy as np
from scipy.sparse import coo_matrix
from .pair_cc_dynamics import PairCCDynamics
from .linear import coupled_cluster_state


def transfer_cluster_operator(model,order,amplitudes):
    """Operator whose action on |F> is the supplied order-transfer vector."""
    subsets,indices=model.blocks[order]
    amplitudes=np.asarray(amplitudes).reshape(len(subsets),len(subsets))
    rows=[];columns=[];values=[]
    for a,particle in enumerate(subsets):
        pbits=sum(1<<(model.r+j) for j in particle)
        for b,hole in enumerate(subsets):
            value=amplitudes[a,b]
            if abs(value)<1e-15:continue
            hbits=sum(1<<j for j in hole)
            target=np.flatnonzero(((model.masks&pbits)==pbits)&((model.masks&hbits)==0))
            parent=np.array([model.index[int(model.masks[x])^pbits^hbits] for x in target])
            rows.extend(target);columns.extend(parent);values.extend([value]*len(target))
    return coo_matrix((values,(rows,columns)),shape=(model.dim,model.dim)).tocsr()


class ClusterDynamics(PairCCDynamics):
    def __init__(self,model,clusters,metric_cutoff=1e-11,radial=False,metric_solver='gram'):
        super().__init__(model,metric_cutoff=metric_cutoff,metric_solver=metric_solver)
        self.radial=radial
        self.orders=[int(order) for order,_ in clusters]
        self.operators=[transfer_cluster_operator(model,order,amplitudes) for order,amplitudes in clusters]
        self.groups=[np.flatnonzero(model.m==m) for m in range(model.r+1)]
        self.single_groups=[np.flatnonzero(model.m[self.destinations]==m) for m in range(model.r+1)]
        self.cluster_rows=[[operator[rows] for operator in self.operators] for rows in self.groups]

    def initial(self):
        return np.r_[1.+0j,np.zeros(self.model.r**2+len(self.operators)+(self.model.r if self.radial else 0),complex)]

    def raw_and_jacobian(self,coordinates):
        model=self.model;r=model.r
        singles=coordinates[:r*r];cluster=coordinates[r*r:r*r+len(self.operators)]
        raw=np.zeros(model.dim,complex);raw[model.fermi_index]=1.
        for m in range(1,r+1):
            rows=self.groups[m];selected=self.single_groups[m]
            values=singles[self.columns[selected]]*raw[self.parents[selected]]
            summed=np.bincount(self.destinations[selected],weights=values.real,minlength=model.dim)+1j*np.bincount(self.destinations[selected],weights=values.imag,minlength=model.dim)
            answer=summed[rows]
            for order,weight,operator in zip(self.orders,cluster,self.cluster_rows[m]):
                if order<=m and weight!=0:answer+=order*weight*(operator@raw)
            raw[rows]=answer/m
        D=np.zeros((model.dim,r*r+len(cluster)),complex)
        D[self.destinations,self.columns]=raw[self.parents]
        for j,operator in enumerate(self.operators):D[:,r*r+j]=operator@raw
        if self.radial:
            beta=np.r_[0.+0j,coordinates[-r:]]
            # A common real rescaling cancels from the normalized physical
            # tangent. Its derivative need not be included in the horizontal
            # Jacobian, and this prevents overflow from radial gauge motion.
            factors=np.exp(beta-beta.real.max())[model.m]
            raw*=factors;D*=factors[:,None]
            radial_D=raw[:,None]*(model.m[:,None]==np.arange(1,r+1))
            D=np.column_stack((D,radial_D))
        return raw,D

    def evolve(self,*args,**kwargs):
        states,coordinates,report=super().evolve(*args,**kwargs)
        report['method']='Hermitian DF pair singles plus frozen connected transfer clusters'
        report['complex_T_coordinates']=self.model.r**2
        report['frozen_cluster_orders']=self.orders
        report['frozen_cluster_coordinates']=len(self.orders)
        report['free_complex_radial_coordinates']=self.model.r if self.radial else 0
        return states,coordinates,report


def connected_two_transfer(model,state):
    """Connected order-two coefficient in intermediate normalization."""
    reference=state[model.fermi_index]
    if abs(reference)<1e-8:raise ValueError('Reference overlap too small for this chart')
    T=state[model.blocks[1][1]]/reference
    independent=coupled_cluster_state(model,T)
    independent=independent/independent[model.fermi_index]
    return state[model.blocks[2][1]]/reference-independent[model.blocks[2][1]]


def connected_source(model,state):
    """Hamiltonian time derivative of the connected two-transfer coefficient."""
    ref=state[model.fermi_index];rate=-1j*(model.H@state);refdot=rate[model.fermi_index]
    T=state[model.blocks[1][1]]/ref
    dT=(rate[model.blocks[1][1]]-T*refdot)/ref
    subsets,indices=model.blocks[2]
    a=np.array(subsets)
    p,q=a[:,0,None],a[:,1,None];h,k=a[None,:,0],a[None,:,1]
    derivative=dT[p,h]*T[q,k]+T[p,h]*dT[q,k]+dT[p,k]*T[q,h]+T[p,k]*dT[q,h]
    return (rate[indices]-state[indices]*refdot/ref)/ref-derivative


def orthogonal_cluster_modes(matrix,common,count):
    """SVD modes after removing the shared radial order-two direction."""
    flat=np.asarray(matrix).ravel().copy();common=np.asarray(common).ravel()
    common=common/np.linalg.norm(common)
    flat-=common*np.vdot(common,flat)
    residual=flat.reshape(np.asarray(matrix).shape)
    u,s,vh=np.linalg.svd(residual,full_matrices=False)
    modes=[]
    for a in range(len(s)):
        v=np.outer(u[:,a],vh[a]).ravel()
        v-=common*np.vdot(common,v)
        for previous in modes:v-=previous*np.vdot(previous,v)
        norm=np.linalg.norm(v)
        if norm>1e-10:modes.append(v/norm)
        if len(modes)==count:break
    return [x.reshape(matrix.shape) for x in modes],s
