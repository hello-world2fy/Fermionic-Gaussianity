"""Full Gaussian-manifold tangent metric and fixed-witness contact Hessian."""
import numpy as np
from .tempering import actions


def contact_hessian(witness,ket):
    permutations,phases=actions(8)
    G=phases*ket[permutations]
    W=phases*witness[permutations]
    a=np.vdot(witness,ket)
    if abs(a)<1e-12:
        raise ValueError('overlap too small for logarithmic contact Hessian')
    b=G@witness.conj()
    c=-W.conj()@G.T
    c=.5*(c+c.T)
    H=-2*(c/a-np.outer(b,b)/(a*a)).real
    horizontal=G-(G@ket.conj())[:,None]*ket[None,:]
    metric=(horizontal.conj()@horizontal.T).real
    eig,V=np.linalg.eigh(.5*(metric+metric.T))
    keep=eig>1e-9
    if keep.sum()!=56:
        raise ArithmeticError('pure Gaussian tangent space must have real dimension 56')
    basis=V[:,keep]/np.sqrt(eig[keep])[None,:]
    reduced=basis.T@H@basis
    eigenvalues=np.linalg.eigvalsh(.5*(reduced+reduced.T))
    gradient=-2*(b/a).real
    report={'overlap':float(abs(a)), 'horizontal_tangent_dimension':56,
            'horizontal_gradient_norm':float(np.linalg.norm(basis.T@gradient)),
            'negative_hessian_count_1e_6':int(np.sum(eigenvalues< -1e-6)),
            'zero_hessian_count_1e_6':int(np.sum(abs(eigenvalues)<=1e-6)),
            'positive_hessian_count_1e_6':int(np.sum(eigenvalues>1e-6)),
            'hessian_minimum':float(eigenvalues[0]),'hessian_maximum':float(eigenvalues[-1]),
            'definition':'Hessian of -log|<w|g>|² in the horizontal Fubini-Study tangent metric',
            'local_stationarity_only':True,'global_support_certified':False}
    return report,{'hessian_eigenvalues':eigenvalues,'horizontal_basis_in_generators':basis,
                   'generator_hessian':H,'generator_metric':metric}
