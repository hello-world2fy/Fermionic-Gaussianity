"""High-precision stationary representatives, independent of ascent residuals.

All Fock operators are exact Gaussian integers/halves. The selected 56
generator indices merely specify coordinates; the metric and Hessian are
recomputed in multiprecision. This is precision convergence, not intervals.
"""
import numpy as np
from scipy.linalg import qr
from mpmath import mp
from . import paths
from .two_orbit import scaled_basis
from .joint_dual_certificates import exact_schmidt_polar
from fk4.tempering import actions


def exact_complex(x):
    return mp.mpc(mp.mpf(float(np.real(x))), mp.mpf(float(np.imag(x))))


def sparse_operator(matrix):
    assert np.array_equal(2*matrix.real, np.rint(2*matrix.real))
    assert np.array_equal(2*matrix.imag, np.rint(2*matrix.imag))
    return [(int(i), int(j), exact_complex(matrix[i,j]))
            for i,j in np.argwhere(matrix != 0)]


def act_sparse(op, v):
    result = [mp.mpc(0) for _ in v]
    for i,j,x in op:
        result[i] += x*v[j]
    return result


def stationary_data(model, name, digits):
    mp.dps = digits
    ids = model.even_masks
    D = scaled_basis()[ids]
    basis = [[exact_complex(x)/s for x in D[:,j]]
             for j,s in enumerate([2,mp.sqrt(8),2*mp.sqrt(7)])]
    if name in ['t9', 't11']:
        saved = np.load(paths.PREVIOUS/f'results/fixed_geometry_01_qualified/{name}_fixed.npz')
        y = [mp.mpf(float(x)) for x in saved['z'].real]
        a = mp.sqrt(2)*y[1]/(-y[0]+5*y[2]/mp.sqrt(7))
    elif name == 't0':
        y = [mp.mpf(2),mp.mpf(0),mp.mpf(0)]
        a = mp.mpf(0)
    elif name == 't20_joint':
        lam = mp.mpf(4713)/1000
        y = [1+2*lam,-mp.sqrt(8),mp.sqrt(7)]
        a = 2/(lam-2)
    else:
        h = mp.matrix([[-56,20*mp.sqrt(2),0],
                       [20*mp.sqrt(2),-14,20*mp.sqrt(14)],
                       [0,20*mp.sqrt(14),-8]])
        eig,V = mp.eigsy(h)
        z = [V[j,0] for j in range(3)]
        if z[0]<0: z=[-v for v in z]
        M0=(5*z[0]+mp.sqrt(7)*z[2])/3
        M1=-mp.sqrt(2)*z[1]
        M2=2*M0-4*z[0]
        d=M0-M2
        cost=(d+mp.sqrt(d*d+4*M1*M1))/2
        a=M1/cost
        alpha=1/(1+a*a)
        y=[7*alpha/3,-mp.sqrt(2)*2*a*alpha,-mp.sqrt(7)*alpha/3]
    w=[sum(y[j]*basis[j][i] for j in range(3)) for i in range(128)]
    plus=[exact_complex(x)/4 for x in exact_schmidt_polar()[ids]]
    if name.endswith('_secondary'):
        u=-a;v=mp.sqrt(1-u*u)
        coeff=[1,1j*v*(4-u)/3,(u-1)*(7-u)/6,1j*v*(u-1)/3,(u-1)**2/6]
        q=sparse_operator(model.q_ops[0,0][np.ix_(ids,ids)])
        g=plus.copy();acted=plus.copy()
        for k in range(1,5):
            acted=act_sparse(q,acted)
            g=[x+coeff[k]*b for x,b in zip(g,acted)]
    else:
        hop=sparse_operator(model.h_hop[np.ix_(ids,ids)])
        mass=sparse_operator((model.q_ops[1,0]-model.q_ops[0,0])[np.ix_(ids,ids)])
        b=mp.sqrt(1-a*a)
        g=plus.copy()
        for shift in [-4,-2,0,2]:
            hv=act_sparse(hop,g);mv=act_sparse(mass,g)
            g=[a*x+b*y+shift*z for x,y,z in zip(hv,mv,g)]
        g=[v/(96*(1+a)**2) for v in g]
    return w,g,a


def mp_contact_hessian(w, g):
    perms,phases=actions(8)
    G=[[exact_complex(f)*g[int(p)] for f,p in zip(ff,pp)] for ff,pp in zip(phases,perms)]
    def inner(x,y):
        return mp.fsum(mp.conj(a)*b for a,b in zip(x,y))
    gf=np.array([complex(v) for v in g])
    Gf=np.array([[complex(v) for v in row] for row in G])
    horizontal=Gf-(Gf@gf.conj())[:,None]*gf
    real=np.concatenate([horizontal.real,horizontal.imag],axis=1).T
    _,R,pivot=qr(real,pivoting=True,mode='economic')
    assert np.sum(abs(np.diag(R))>1e-9)==56
    selected=pivot[:56]
    a=inner(w,g)
    b=[inner(w,G[j]) for j in selected]
    vg=[inner(g,G[j]) for j in selected]
    H=mp.matrix(56);metric=mp.matrix(56)
    support=[i for i,v in enumerate(w) if v!=0]
    for row,i in enumerate(selected):
        for col in range(row,56):
            j=selected[col]
            cij=mp.fsum(mp.conj(w[q])*exact_complex(phases[i,q])*G[j][perms[i,q]] for q in support)
            cji=mp.fsum(mp.conj(w[q])*exact_complex(phases[j,q])*G[i][perms[j,q]] for q in support)
            H[row,col]=H[col,row]=-2*mp.re((cij+cji)/(2*a)-b[row]*b[col]/a**2)
            metric[row,col]=metric[col,row]=mp.re(inner(G[i],G[j])-mp.conj(vg[row])*vg[col])
    invL=mp.inverse(mp.cholesky(metric))
    reduced=invL*H*invL.T
    reduced=(reduced+reduced.T)/2
    eigenvalues=mp.eigsy(reduced,eigvals_only=True)
    gradient=invL*mp.matrix([-2*mp.re(x/a) for x in b])
    return dict(overlap=mp.nstr(abs(a),mp.dps),
                norm_error=mp.nstr(abs(inner(g,g)-1),mp.dps),
                horizontal_gradient_norm=mp.nstr(mp.norm(gradient),mp.dps),
                eigenvalues=[mp.nstr(x,mp.dps) for x in eigenvalues],
                selected_generator_indices=selected.tolist(),
                precision_digits=mp.dps, interval_certified=False,
                definition='Hessian of -log|<w,g>|^2 in the Fubini-Study metric; algebraically stationary orbit representative.')
