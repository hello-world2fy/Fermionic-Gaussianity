"""Fail-closed validation gate for C. Run after owner enhancement.
Checks audit_ket / audit_projector / solve_signed_lp RAISE ValueError (before
linprog) on illegal inputs; legal pure Gaussian smoke solves R=1.
linprog monkeypatched with sentinel (illegal must call it 0 times).
Exit nonzero on any failure.
"""
import numpy as np, sys, os
HERE=os.path.dirname(os.path.abspath(__file__))
SG=os.path.normpath(os.path.join(HERE,"..","signed_gaussian"))
sys.path.insert(0,SG)

def vac_ket():
    g=np.zeros(8,dtype=complex); g[0]=1; return g
def vac_rho():
    g=vac_ket(); return np.outer(g,g.conj())

def main():
    from sggen.audit import audit_ket, audit_projector
    import sggen.synthesis as syn
    calls={"n":0}
    real_lp=syn.linprog
    def sentinel(*a,**k):
        calls["n"]+=1; return real_lp(*a,**k)
    syn.linprog=sentinel

    fail=0
    def expect_raise(fn,name):
        nonlocal fail
        before=calls["n"]
        try:
            fn(); print("FAIL no-raise",name); fail+=1
        except ValueError:
            if calls["n"]!=before: print("FAIL linprog-called-before-raise",name); fail+=1
            else: print("PASS raises",name)
        except Exception as e:
            print("WRONG-TYPE",name,type(e).__name__); fail+=1

    # audit_ket rejects
    bad_ket=vac_ket().copy(); bad_ket[0,1] if False else None
    expect_raise(lambda: audit_ket(np.array([1,2,3],dtype=complex)), "ket_wrongshape")
    expect_raise(lambda: audit_ket(np.array([np.nan]*8,dtype=complex)), "ket_nan")
    badq=np.zeros(8,dtype=complex); badq[0]=1/np.sqrt(2); badq[7]=1/np.sqrt(2)
    expect_raise(lambda: audit_ket(badq), "ket_nonGaussian_quadric")
    # audit_projector rejects
    bad_proj=np.zeros((8,8),complex); bad_proj[0,1]=0.2+0.1j; bad_proj[1,0]=0  # non-Hermitian
    expect_raise(lambda: audit_projector(bad_proj), "proj_nonHermitian")
    expect_raise(lambda: audit_projector(np.eye(8)/8), "proj_mixed")
    expect_raise(lambda: audit_projector(2*vac_rho()), "proj_scaled2")
    expect_raise(lambda: audit_projector(np.full((8,8),np.inf,dtype=complex)), "proj_inf")

    # solver rejects illegal atom BEFORE linprog, target legal
    expect_raise(lambda: syn.solve_signed_lp(vac_rho(), np.array([bad_proj])), "solver_atom_nonHerm")
    expect_raise(lambda: syn.solve_signed_lp(vac_rho(), np.array([np.eye(8)/8])), "solver_atom_mixed")

    # legal smoke
    calls["n"]=0
    r=syn.solve_signed_lp(vac_rho(), np.array([vac_rho()]))
    print("legal smoke value=",r["value"],"n_lp_calls",calls["n"])
    if not (abs(r["value"]-1.0)<1e-6): print("FAIL legal R!=1"); fail+=1
    syn.linprog=real_lp
    print("RESULT","PASS" if fail==0 else "FAIL")
    sys.exit(0 if fail==0 else 1)

if __name__=="__main__": main()
