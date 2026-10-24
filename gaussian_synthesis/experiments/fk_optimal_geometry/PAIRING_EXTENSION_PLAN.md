# Second-model selection and a global energy-window mechanism

Two actual sparse candidates are prepared: a nearest-neighbor chain of three
four-mode FK clusters (12 complex modes, full dimension 4096), and four
time-reversed orbital pairs (8 complex modes) with number-conserving pair
transfer. The latter is selected before its new ground states are computed,
because its coercivity and excitation structure can test D3 directly. The
three-cluster candidate is retained with an explicit no-go for an ED-only
scale-up; no claim about cheap Gaussian rank at larger cluster number is made.

The selected Hamiltonian is

    H = sum_i epsilon_i (n_(2i)+n_(2i+1))
        - g sum_(i != j) P_i^dagger P_j,
    epsilon=(-3,-1,1,3), P_i=a_(2i+1) a_(2i), g=0.05,0.1,0.2.

The reference Slater determinant fills the first two orbital pairs and has
energy -8. Write beta for particles above this determinant and holes below it,
and delta=<N_beta>. Its kinetic excess is between delta and 3 delta.

For every Gaussian, Wick's rule expresses a four-distinct-mode pair-transfer
expectation as three products of normal or anomalous contractions. With
n_a=<beta_a^dagger beta_a>, Cauchy gives |rho_ab|<=sqrt(n_a n_b) and
|kappa_ab|<=sqrt(min(n_a,n_b)). Each product on four distinct modes is bounded
by one quarter of their total occupation: for anomalous products use AM-GM;
for normal products use product(n_a)^(1/2)<=sum(n_a)/4 since sum(n_a)<=4.
Thus one unordered pair-transfer plus its adjoint is bounded by 3/2 times
the occupation on those four modes. Each mode appears in three unordered
orbital-pair transfers, giving the global bounds

    (1-9g/2) delta <= E(g_Gaussian)+8 <= (3+9g/2) delta.

All chosen g are below 2/9, so the Slater reference is the unique Gaussian
minimum. This applies to all Gaussian states, including odd parity, rather
than only a selected mean-field ansatz. An independent verification of the
Wick contraction cases and finite constructions is required before using
this as a delivered certificate.

The pair-transfer operator has absolute row sum at most 4 in full Fock space.
The unperturbed kinetic ground level is -8 and the next is -7. Therefore the
full interacting ground state is unique with gap at least 1-4g>0. Its energy
is <=-8. Every state outside the six-dimensional paired N=4 block has energy
>-8, so the ground state belongs to that block. The block is a connected
stoquastic Johnson graph with four negative off-diagonal entries per row;
its ground coefficients are strictly positive. Relative to the Slater
reference these six components have quasiparticle occupations 0,4,4,4,4,8.

Let c=1-9g/2 and C=3+9g/2. Any window E<=-8+Delta has delta<=Delta/c.
The eighth-excitation amplitude of a Gaussian is at most (delta/8)^2.
For the physical target coefficient q8>0 this gives

    tau_Delta^(epsilon) >= 64 c^2 (|q8|-epsilon)_+ / Delta^2.

For a matching finite construction, any single four-quasiparticle determinant
is synthesized by three phase-rotated two-pair BCS states; an eight-particle
determinant uses five phase-rotated four-pair BCS states. For m=2 or 4 pairs,
take p=Delta/(2m C) and Fourier coefficients proportional to
exp(-2 pi i m j/(m+1))/((m+1) p^(m/2)). Every atom has delta=2m p and hence lies
in the energy window. The root-of-unity sum cancels all degrees except m.
Four three-term expansions, one five-term expansion, and the reference
determinant give 18 actual Gaussians and cost

    |q0| + (4C/Delta) sum_four |q4| + (8C/Delta)^2 |q8|.

This matches the exact Delta^-2 power, without claiming an optimal constant.
Dropping the eighth-excitation target amplitude yields Hilbert error |q8|
and an O(Delta^-1) construction. Whether a useful finite-error crossover is
visible will be determined from the independently enclosed coefficients;
the approximate vector is not silently assumed normalized.

## Executed result

The subsequent independent certificate and finite constructions are in
`results/pairing_extension_02/REPORT.md` and `verification.json`. They verify
all three rational couplings and all four windows. At epsilon=q8, the general
four-excitation amplitude bound sqrt(3/2) delta/4 additionally gives

    tau_Delta^(q8) >= 4 c (q4-q8)/(sqrt(3/2) Delta),

where q4 is the first four-excitation target coefficient and q4>q8 is enclosed
at each coupling. This matches the 13-term truncated construction's Delta^-1
power. The eigensystem enclosures use exact rational principal-minor ground
isolation and an independent 110-digit diagonalization. The new model's
Gaussian minimum and window coercivity are global, not merely local HFB tests.
