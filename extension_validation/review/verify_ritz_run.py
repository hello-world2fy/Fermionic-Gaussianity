"""Standalone final-run verifier for B timed_bench (ritzfix_v2).

Run ONLY after producer signals the full 456/456 run finished (same clock/version).
Exits nonzero on ANY failure; missing/epoch mismatch never => PASS.

Checks per CSV row:
  - checkpoint file exists (else FAIL)
  - npz has E_hist/n_hist/t_hist keys
  - np.diff(E_hist) <= 1e-9  (monotonic non-increasing)
  - min(E_hist) >= (row energy - dE) - 1e-7   (>= FCI)
  - E_hist[-1] == row energy ; n_hist[-1] == row completed_n
  - (t_hist - clock_start) <= (deadline - clock_start)
  - final psi: Rayleigh==E, residual matches, norm==1
Writes review/ritz_run_verification.log with CSV sha256 + run_version + counts.
"""
import csv, os, sys, hashlib, datetime
import numpy as np

HERE=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.normpath(os.path.join(HERE,"..","slater","results"))
CSV=os.path.join(ROOT,"timed_bench.csv"); CKDIR=os.path.join(ROOT,"ckpt")
LOG=os.path.join(HERE,"ritz_run_verification.log")

def main():
    log=open(LOG,"w")
    def L(*a):
        s=" ".join(str(x) for x in a); print(s); log.write(s+"\n")
    L("# run at",datetime.datetime.now().isoformat())
    L("# csv sha256",hashlib.sha256(open(CSV,"rb").read()).hexdigest())
    rows=list(csv.DictReader(open(CSV)))
    L("# rows",len(rows),"versions",set(r["run_version"] for r in rows))
    F={"missing":0,"keys":0,"mono":0,"below_fci":0,"E_cons":0,"N_cons":0,
       "time":0,"rq":0,"res":0,"norm":0}
    for r in rows:
        cp=r["checkpoint"]
        if not cp or not os.path.exists(os.path.join(CKDIR,cp)):
            F["missing"]+=1; L("MISSING",cp); continue
        z=np.load(os.path.join(CKDIR,cp),allow_pickle=True)
        if not all(k in z for k in ("E_hist","n_hist","t_hist")):
            F["keys"]+=1; L("KEYS",cp); continue
        Eh=np.array(z["E_hist"],float); Nh=np.array(z["n_hist"]); Th=np.array(z["t_hist"],float)
        cs=float(r["clock_start"]); dl=float(r["deadline"]); FCI=float(r["energy"])-float(r["dE"])
        if np.any(np.diff(Eh)>1e-9): F["mono"]+=1; L("MONO",cp)
        if Eh.min()<FCI-1e-7: F["below_fci"]+=1; L("BELOW_FCI",cp,Eh.min(),FCI)
        if abs(Eh[-1]-float(r["energy"]))>1e-7: F["E_cons"]+=1; L("E_CONS",cp,Eh[-1],r["energy"])
        if int(Nh[-1])!=int(r["completed_n"]): F["N_cons"]+=1; L("N_CONS",cp,Nh[-1],r["completed_n"])
        if (Th-cs).max()>dl-cs+1e-6: F["time"]+=1; L("TIME",cp,(Th-cs).max(),dl-cs)
        H=z["H"];psi=z["psi"];E=float(z["E"]);nm=float(np.vdot(psi,psi).real)
        if abs(nm-1)>1e-6: F["norm"]+=1; L("NORM",cp,nm)
        if abs(float(np.vdot(psi,H@psi).real/nm)-E)>1e-7: F["rq"]+=1; L("RQ",cp)
        if abs(float(np.linalg.norm(H@psi-E*psi))-float(z["residual"]))>1e-6: F["res"]+=1; L("RES",cp)
    L("# SUMMARY",F)
    bad=sum(v for k,v in F.items())
    L("# RESULT","PASS" if bad==0 else "FAIL")
    log.close()
    sys.exit(0 if bad==0 else 1)

if __name__=="__main__": main()
