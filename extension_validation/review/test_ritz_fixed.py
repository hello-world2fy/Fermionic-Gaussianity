"""STRICT acceptance tests for CORRECTED noci.two_dim_ritz_drop.
Exits nonzero on ANY failure; skips never => PASS. Read-only on producer.
"""
import numpy as np, sys, re, os, glob
HERE=os.path.dirname(os.path.abspath(__file__)); SLATER=os.path.normpath(os.path.join(HERE,"..","slater"))
sys.path.insert(0,SLATER)
FAIL=[]
def check(cond,msg):
    print(("PASS " if cond else "FAIL ")+msg)
    if not cond: FAIL.append(msg)

def ref_ritz(H,psi,atom,eps_eff=1e-8):
    W=np.column_stack([psi,atom]); s=np.linalg.svd(W,compute_uv=False); E0=float(np.vdot(psi,H@psi).real)
    if s[-1]/s[0]<eps_eff: return E0
    Q,_=np.linalg.qr(W); return float(np.linalg.eigvalsh(Q.conj().T@H@Q)[0])

def main():
    import noci
    z=np.load(os.path.join(HERE,"ritz_fixtures.npz"),allow_pickle=True)
    bad=0
    for H,psi,atom,E0,eref,tag in zip(z["H"],z["psi"],z["atom"],z["E0"],z["E_ref"],z["tag"]):
        e=noci.two_dim_ritz_drop(H,psi,E0,atom)
        if abs(e-eref)>1e-8: bad+=1; print("  mismatch",tag,e,eref)
    check(bad==0, "[1] all %d fixtures match QR/rank-aware ref (bad=%d)"%(len(z["tag"]),bad))

    # [1b] passed wrong E=-999 must NOT change result (uses actual Hpsi Rayleigh)
    H,psi,atom=z["H"][0],z["psi"][0],z["atom"][0]
    e_real=noci.two_dim_ritz_drop(H,psi,float(np.vdot(psi,H@psi).real),atom)
    e_wrong=noci.two_dim_ritz_drop(H,psi,-999.0,atom)
    check(abs(e_real-e_wrong)<1e-8, "[1b] result independent of wrong passed E (%.4f vs %.4f)"%(e_real,e_wrong))

    # [2] H2 real Thouless bracket E_exact <= E_2D <= E_psi
    try:
        import molecule
        ham,_,_,_=molecule.h2_active(2.0); E,_=ham.exact_ground()
        mask=ham.neel_determinant(); Uup,Udn=noci.reference_U(ham,mask)
        psi=noci.slater_fock_vector(ham,Uup,Udn); Epsi=float(np.vdot(psi,ham.H@psi).real)
        ok=True
        for th in [0.1,0.3,0.6,1.0]:
            atom=noci.rotated_slater(ham,Uup,Udn,np.array([[th]]),np.array([[th]]))
            e2=noci.two_dim_ritz_drop(ham.H,psi,Epsi,atom)
            if not (E-1e-9<=e2+1e-9 and e2<=Epsi+1e-9): ok=False; print("  bracket fail th",th,e2,Epsi,E)
        check(ok,"[2] H2 variational bracket E_exact<=E2D<=Epsi (all theta)")
    except Exception as ex:
        check(False,"[2] H2 bracket raised (NOT skipped): %r"%ex)

    # [4] grep ALL slater *.py for residual non-Hermitian generalized eig
    calls=[]
    for f in glob.glob(os.path.join(SLATER,"*.py")):
        src=open(f).read()
        for m in re.findall(r"eigvalsh?\s*\([^)]*solve\s*\(",src): calls.append((os.path.basename(f),m))
        for m in re.findall(r"eigvalsh?\s*\(\s*np\.linalg\.inv",src): calls.append((os.path.basename(f),m))
    check(len(calls)==0,"[4] no residual eigvalsh(solve(S,H))/eigvalsh(inv(S)H) in any slater/*.py: %s"%calls)

    # [5] deadline actually wraps whole loop in timed_bench.py (source assertion)
    tb=open(os.path.join(SLATER,"timed_bench.py")).read()
    check("setitimer" in tb and "_Deadline" in tb and "setitimer(signal.ITIMER_REAL, 0)" in tb,
          "[5] SIGALRM deadline + disarm in finally present in timed_bench.py")

    # [3] step trajectory replay if fresh timed_bench.csv exists
    # [3] per-run E_hist monotonic/variational/deadline — done by verify_ritz_run.py
    vlog=os.path.join(HERE,"ritz_run_verification.log")
    if os.path.exists(vlog):
        txt=open(vlog).read()
        ok="RESULT PASS" in txt and "'missing': 0" in txt
        check(ok, "[3] verify_ritz_run.py full 456 replay PASS (E_hist mono>=FCI, last E/N match row, t<=deadline)")
    else:
        check(False,"[3] ritz_run_verification.log missing — run verify_ritz_run.py")

    print("\n=== %s ==="%("ALL PASS" if not FAIL else "FAILURES: %d"%len(FAIL)))
    sys.exit(0 if not FAIL else 1)

if __name__=="__main__": main()
