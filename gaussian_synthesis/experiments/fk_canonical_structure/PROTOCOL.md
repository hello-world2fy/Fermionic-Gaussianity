# Canonical Gaussian structure: frozen next-round protocol

This experiment implements the approved priority: trustworthy canonical reconstruction, blind structure discovery, coherent physical interpretation, and cost necessity. Finite branch compression is auxiliary and is not rerun here. Prior FK packages and results are read-only dependencies.

## Scope and resources

The full eight-mode even Gaussian dictionary is the primary dictionary. Deterministic k=2 covers t/V=0,5,9,11,20. Empirical k=4 covers t/V=0,11,20. Three seeds are 20261023,20261024,20261025. New SeedSequence namespace910 separates this experiment from both older rounds; purposes1,2,3,4 denote training, development, discovery and final validation. Development sample sizes are4096,16384,65536, batch256; finalN is fixed65536. Candidate selection is never based on final validation. All numerical stages run sequentially with numerical libraries limited to one thread. Per-stage wall cap180s, numerical-run cap1800s, memory512MiB. There is no automatic scan expansion or additional k.

## A: canonical reconstruction

Audit the algebraic k=2 moment, its signed-integer construction, derivatives and full tensor reconstruction independently of the reduced stationarity check. Require structural errors<=1e-10 and deterministic full reconstruction<=1e-8. Prepare all five exact-formula witnesses. Optimize k=4 on independent nested training caches, keeping every grid point; freeze the largest predeclared sample size, not the best observed residual. Paired raw and symmetry-averaged reconstructions use the SAME frozen witness and SAME samples. A projected atom remains an unnormalized unit-cost Gaussian orbit average.

The exact integer moment does not determine B_2. Scalar mass, invariant distributions and coherent bins still need their own integration checks. Gaussian probes test known q/B moments and the expected raw vector Monte Carlo error; probe consistency is not precision or joint confidence certification.

## B1: invariant distributions

No saturation-arc input is used in canonical training, invariant bin selection or discovery starts. Fixed4x4 bins use eta and local mass_A squared, with a matching Haar reference histogram. Coherent groups are eta<=.25, .25<eta<=.75, eta>.75. Record mass_A,mass_B,mass_dot,eta, complementary-bound slack, B,q,ESS,max weight and all checkpoints. Total ESS>=1000; highlighted histogram bins require ESS>=100. The last two development sizes must have total variation distance<=.05. Across three independently sampled final seeds, mean eta range must be<=.02. All seeds must pass; missing rows are not successful rows. These numerical stability gates do not constitute confidence intervals.

## B2: blind candidate discovery

For each t and seed, compare128 common Fock atoms plus32 new representatives: independent Haar, canonical-k2-witness-guided Gaussian optimization, and a hand-designed saturation arc. Guided optimization starts from the SAME32 Haar atoms as the random control, uses only the frozen witness, and maximizes squared overlap using Gaussian unitary rotations. The physics arc is used only in its own comparison arm. Candidate counts match; optimization work and wall time do not, so actual costs are reported. Max30 local sweeps, no unbounded restarts. Local overlap searches are not global separation oracles.

Fit coefficients by the same complex SOCP on unnormalized orbit projections and repair full Fock residuals. Save genuine original Gaussian kets, phases, coefficients, projections and residuals. Call these costs 'canonical-guided orbit upper bounds', never canonical B_k or the globally optimal extent. The reference target for discovery is <=5% above the matched-size structured baseline; assess improvement over random with all three seeds and report failures. Existing449-atom first-round cost is background, not the matched comparison.

## C: coherent interpretation

Retain complex group vectors. Estimate the Hermitian correlation kernel from independent halves so the quadratic estimate does not square the same noisy group mean. Also retain the plug-in normalized-ket CQ/W diagnostics; label those nonlinear finite-sample estimates accordingly. Require group closure<=.01, CQ/W errors<=.05, all reconstruction/distribution gates, estimated per-group vector RMSE<=.01 and independent-half vector disagreement<=.02 before interpretation. Raw estimates remain available even when interpretations are withheld. A correct total can hide cancellation of inaccurate group contributions. Five t points do not determine a new critical point or a precise peak position.

## D: necessity and evidence

Compare each algorithm-guided repaired upper U with L/(1+sqrt(eta0))^2 at eta0=0,.25,.5,.75. Use the actual independent target eigenstate error estimate. Propagate the inequality to necessary average eta and high-eta absolute-coefficient-mass fractions. Label all such comparisons floating-point numerical evidence; no interval enclosure or global atomic norm optimality is claimed. A local maximum overlap gives a lower bound on the overlap supremum, not the upper bound required by a dual certificate.

Final validation reserves a union-bound failure budget .05 across120 predeclared events:24 canonical candidates times mass, reconstruction and three coherent-group vector estimates. Conservative second-moment bounds are recorded even when the required radius exceeds the sample budget. This budget does not certify histogram stability, covariance-based nonlinear quantities or numerical rounding. No stopped or incomplete run is silently relabeled certified.

## Output and stopping

Freeze source, dependency and input hashes before the formal run. Save config, full dependency versions, witnesses, moment matrices, development and final accumulators, discovery atoms/coefficient arrays, stage timing/RSS, all failure statuses, JSON data and figures. Do not overwrite an existing run. If a stage fails calibration, precision or resources, stop dependent interpretation; independent cost work continues. The final report separates exact-formula reconstruction, canonical distributions, guided orbit upper bounds, and physical necessity. The primary minimum outcome is verified k=2 plus an honest diagnosis of structure precision, not a favorable shape or predetermined cost.
