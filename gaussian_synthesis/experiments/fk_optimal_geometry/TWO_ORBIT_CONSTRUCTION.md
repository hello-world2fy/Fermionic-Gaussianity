# An exact16-Gaussian construction at t=20

Write the exact physical ground state in the radical singlet basis as z.
Its energy is the isolated smallest root of

`(-56-E)[(-14-E)(-8-E)-14t²]-2t²(-8-E)=0`.

The signs are z0>0,z1<0,z2>0. Ratios
`z1/z0=(E+56)/(sqrt2 t)` and `z2/z1=sqrt14 t/(E+8)`, followed by exact
normalization, define the state algebraically. We bisect the root with exact
rational arithmetic inside the independently verified full-Fock ground
interval, then evaluate all expressions with directed90-digit intervals.

## Two Gaussian orbit families

Let n run over the eight Hadamard simplex directions in R7, each coordinate
being +1/sqrt7 or -1/sqrt7. The mass/hopping Gaussian ground state of

`H_quad(a,n)=sqrt(1-a²) sum n_j(-Q_Aj+Q_Bj)+a H_hop`

has exact energy -4 and is a unique normalized even Gaussian for |a|<1.
The corresponding one-particle generator squares to minus identity by the
Clifford relations. In the even sector its spectral projector is
`P_-(H)=(H+2)H(H-2)(H-4)/384`. With the normalized hopping ground ket
`gplus=omega_S/4`, its phase is fixed by positive overlap
`<gplus,g(a,n)>=(1+a)²/4`.

The exact8-point average is

`A(a)=S((2-a²)/4,-a/sqrt2,(2+5a²)/(4sqrt7))`.

This is an unnormalized average of eight actual Gaussian kets. It is not
counted as a Gaussian. An integer polynomial identity verifies this finite
quadrature in every Fock coordinate, using b²=(1-a²)/7.

The second family is `exp(i theta Q_A(n)) gplus`, where u=cos(theta).
It is maximally entangled, has eta=1, and has exact8-point average

`B(u)=S((2u²-1)/4,-u/sqrt2,(5+2u²)/(4sqrt7))`.

Local Q_A(n) has eigenvalues -2,-1,0,1,2, so its exponential is a degree4
polynomial. Another integer identity proves its average. Independently,
the verifier works with the16-dimensional local spectral projectors and
the Schmidt purification rather than the128-dimensional ground projector.

## Coefficients and full reconstruction

Define moment coordinates

`M0=(5z0+sqrt7 z2)/3`, `M1=-sqrt2 z1`, `M2=2M0-4z0`.

In these coordinates A(a) becomes `(1,a,a²)` and B(u) becomes `(u²,u,1)`.
Set

`d=M0-M2`, `T=(d+sqrt(d²+4M1²))/2`, `a=M1/T`,
`c_minus=(T-M0)/(1+a²)`, `c_plus=T-c_minus`, `u=-a`.

Whenever c_plus,c_minus>0 and0<a<1 (checked with intervals at t20),

`z = c_plus A(a) - c_minus B(-a)`

holds identically, by `T²-dT-M1²=0`. Each orbit coefficient is divided
equally among its eight actual Gaussian branches. This gives16 distinct
Gaussian rays, coefficient cost exactly T, and no Fock repair. The saved
binary64 arrays are approximations to this algebraically specified exact
construction; small numerical residuals alone are not the proof.

## What this does and does not prove

The certified cost is approximately1.29557042066. Combining this exact
construction with A1's globally feasible witness improves the strict cost
interval. A candidate matching witness has singlet coordinates

`y=(7alpha/3,-sqrt2 beta,-sqrt7 alpha/3)`,
`alpha=1/(1+a²)`, `beta=2a/(1+a²)`.

It saturates the two chosen orbit families and its target overlap equals T.
This establishes duality only for the restricted two-family dictionary.
Full-Gaussian local searches have not found a violation in the saved run,
but that does not certify its global support. Exact full-dictionary
optimality remains open until a global bound h_G(Sy)<=1 is proved.

The earlier endpoint-only secondary family in two_orbit_01 has higher cost
1.29584819 and its candidate witness is violated by another maximally
entangled Gaussian (overlap approximately1.0362). That unsuccessful
candidate and its evidence are preserved.
