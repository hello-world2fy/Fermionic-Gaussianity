# Joint Schmidt–magic certificate and composition stability

Throughout, Gaussian atoms are normalized full eight-mode even pure Gaussian
states. A occupies the low four bits and B the high four bits, so the Fock
ket reshapes as `M[B,A]`. The cost is `tau=sum|c|`; extent is `xi=tau²`.

## Exact polar witness for the physical FK curve

In the predecessor's explicit radical singlet basis, define
`omega=S(1,-sqrt(8),sqrt(7))`, `w0=2s0`. The full16x16 reshape of omega is
an exactly signed permutation with entries in `{0,+1,-1,+i,-i}`. Its
operator norm is exactly1, Hilbert norm4 and `<omega,w0>=2`.
Independent verification builds it from scaled dyadic basis matrices,
checks the monomial identity and all three orthogonal block projectors.

More precisely, multiplying the reshaped target by omega† gives
`z0 P1 - z1 P8/sqrt(8) + z2 P7/sqrt(7)`, where the projectors have
ranks1,8,7 and sum to identity. The exact reduced Hamiltonian is

`diag(-56,-14,-8) + t [[0,sqrt2,0],[sqrt2,0,sqrt14],[0,sqrt14,0]]`.

For t>0, conjugating by diag(1,-1,1) makes both off-diagonal couplings
strictly negative. Perron–Frobenius on a sufficiently shifted matrix gives
the unique reduced ground vector signs `z0>0,z1<0,z2>0`. The independently
certified full-Fock ground-state enclosure identifies that reduced state
with the physical ground state. Thus omega is its exact nuclear-norm polar
witness and `S_psi=z0-sqrt8*z1+sqrt7*z2=Re<omega,psi>`.
For any other target, the same monomial omega still defines a valid
nuclear-norm functional, but equality with its nuclear norm is not presumed.

## Universal Gaussian support

For a pure Gaussian with four bipartite canonical pairs let
`s_j=sin(2theta_j) in[0,1]` and `eta=(sum s_j²)/4`. Local Gaussian unitaries
and the Fock-order parity signs preserve Schmidt singular values. Therefore

`S_g=product sqrt(1+s_j) <= (1+mean s_j)² <= (1+sqrt(eta))²`.

The first inequality follows from concavity of log(1+s), the second from
Cauchy–Schwarz. The predecessor's sharp magic theorem gives
`|<w0,g>|<=1-eta/2`. Consequently, for lambda>=4 and x=sqrt(eta),

`|<omega+lambda w0,g>| <= f_lambda(x)`
`f_lambda(x)=1+lambda+2x+(1-lambda/2)x²`
`=H_lambda-(lambda-2)/2*(x-x_star)²`,

where `x_star=2/(lambda-2)` and `H_lambda=1+lambda+2/(lambda-2)`.
This is a complete-dictionary support upper bound. Equality of this scalar
envelope does not by itself prove that the two atom inequalities and their
complex phases can all saturate simultaneously. Global contact saturation
and classification are separate tasks in B1.

The normalized witness W/H therefore gives the physical cost lower bound
`Re<W,psi>/H`. Our saved target is an exact binary64 vector and a separate
Hamiltonian certificate gives its distance epsilon from the physical state.
Directed evaluation subtracts `||W|| epsilon`, using
`||W||²=4lambda²+4lambda+16`. Lambda itself is an exact rational; in
particular t20 uses4713/1000. Floating optimization only chooses the
rational and is not assumed optimal.

## All low-cost decompositions, including arbitrary term counts

For any exact coefficient measure of mass B<=U, write `m=E eta`.
The Schmidt triangle inequality and Jensen give
`S_psi<=B(1+sqrt(m))²`; the magic theorem gives
`L0<=B(1-m/2)`. Hence

`max(0,sqrt(S_psi/U)-1)² <= m <= min(1,2(1-L0/U))`.

Completing the square in the universal joint envelope further gives

`E(sqrt(eta)-x_star)² <= [H_lambda-Re<W,psi>/U]/((lambda-2)/2)`.

Markov's inequality supplies a tail bound by dividing this right hand side
by delta² and clipping to1. This proves concentration in a scalar physical
coordinate. It is not a bound on distance to a complete contact manifold,
nor an assertion that this variance bound is sharp. Quotient geometry and
possible degeneracies remain B1/B2 research tasks.
