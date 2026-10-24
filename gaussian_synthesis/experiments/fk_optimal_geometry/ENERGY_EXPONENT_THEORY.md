# Energy curvature, excitation order and sharp cost divergence

Let the normalized Gaussian dictionary be phase closed, and let E_G be its
minimum energy. Assume a small energy window is contained in finitely many
coordinate neighborhoods of all minimizing rays (or in a uniform tubular
neighborhood of a compact minimum manifold). In each chart, use a transverse
pairing coordinate Z and assume uniformly

    E(g_Z)-E_G >= c ||Z||^q,
    |ell(g_Z)| <= C ||Z||^r.

Here ell is a bounded complex linear functional, including the relevant
coordinate conventions and normalization. The second inequality requires
its lower-order Taylor jets to vanish at EVERY allowed minimum. A distant
minimum with ell nonzero would invalidate this theorem. The window-containment
and uniformity conditions must therefore not be dropped.

For any approximation phi with ||psi-phi||<=epsilon,

    tau_Delta^(epsilon)(psi) >=
       c^(r/q) (|ell(psi)|-||ell|| epsilon)_+ / (C Delta^(r/q)).

Proof: coercivity bounds ||Z|| by (Delta/c)^(1/q); the functional support is
therefore <=C(Delta/c)^(r/q). Triangle and Cauchy inequalities give the result
for arbitrary finite or convergent continuous coefficient measures. The norm
of ell multiplying epsilon is essential. This elementary implication is not
claimed as a new abstract inequality; the model-specific work is establishing
its hypotheses, leading excitation order and matching constructions.

## FK realizes two different exponents

The global FK bound is stronger than a local chart argument:

    E-E_G >=18x²+4(t-9)x, <N>=4x,
    |<e8,g>| <= x²/4.

Near the hopping Gaussian, x is quadratic in transverse pairing amplitude.
Thus the required eight-particle component has r=4. For t>9, q=2 and the
cost exponent is 2. At t=9, the mass/hopping arc has quartic energy curvature
(q=4), giving exponent 1. These are not inferred from fitting a log-log plot.

For the physical targets considered here q8=<e8,psi> is positive. At any
epsilon<q8, the full dictionary lower bound is 4(q8-epsilon)/X_Delta².
Choose x=.99 X_Delta and the three arc points 0,x/2,x. Their singlet projections
in hopping coordinates are

    p(y)=((1-y/2)², 3(2y-y²)/(2sqrt14), y²/4).

For a target (q0,q4,q8-epsilon), define

    M2=4(q8-epsilon),
    M1=sqrt14 q4/3+2(q8-epsilon),
    M0=q0+sqrt14 q4/3+q8-epsilon.

Then coefficients

    d0=M0-3M1/x+2M2/x²,
    d1=4M1/x-4M2/x²,
    d2=2M2/x²-M1/x

reconstruct that target identically. Each nonzero-y orbit average is realized
by the previously proved eight-point finite Gaussian identity. The y=0 orbit
is just the hopping vacuum, giving 17 actual Gaussian terms in total. Every
term is strictly in the window; no out-of-window repair is present. Their cost
is O(X_Delta^-2), matching the global lower-bound power. The leading upper/lower
constant ratio from these particular interpolation nodes tends to 8/.99²;
constant-factor optimality is not claimed.

The epsilon>0 constructed vector has exact Hilbert error epsilon and is
generally unnormalized. This is the approximation convention of this cost
theorem; it must not be substituted into D1/D2's normalized physical-state
bounds without renormalization and a new error/cost calculation. The epsilon=0
construction is the exactly normalized physical target.

## A separate pairing example attains the leading constant

Consider the eight-mode number operator N, Gaussian minimum vacuum, and target
psi=(3|vac>+|full>)/sqrt10. An energy window <N><=Delta implies pair probability
sum<=Delta/2. AM–GM gives full amplitude <=(Delta/8)², hence

    tau_Delta(psi) >= 1/(sqrt10 p²), p=Delta/8.

For 0<p<1 take five actual four-pair BCS Gaussians

    g_j=product_(l=1)^4 [sqrt(1-p)+sqrt(p) zeta^j a_(2l-2)^dag a_(2l-1)^dag]|vac>,
    zeta=exp(2 pi i/5), j=0,...,4.

Their energy is exactly 8p=Delta. Assign

    c_j=[3/(sqrt10 (1-p)²)+zeta^(-4j)/(sqrt10 p²)]/5.

The fifth-root identity cancels excitation degrees 1,2,3 and matches degrees
0,4 exactly. The resulting cost is at most

    1/(sqrt10 p²)+3/(sqrt10 (1-p)²).

The upper/lower ratio tends to one as Delta tends to zero. This is a finite,
five-Gaussian construction proving that the general r/q=2 power and leading
constant can be sharp. The target need not be a ground state of N: this is a
representation-complexity theorem, not a claim that this toy Hamiltonian solves
the second physical model requested in experiment F.
