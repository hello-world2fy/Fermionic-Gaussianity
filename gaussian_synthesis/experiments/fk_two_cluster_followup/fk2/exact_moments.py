"""Deterministic k=2 Haar moment on the three-dimensional FK singlet space.

No Gaussian sampling, tensor-square dense operator, or numerical integration is
used. The exact representation-theory projector is applied to six 128 x 128
tensor vectors. Floating-point evaluation of this algebraic formula is not an
interval-arithmetic certificate. See THEORY.md for the projector derivation.
"""
from dataclasses import dataclass
from functools import lru_cache
import math
import time

import numpy as np


N_MODES = 8
EVEN_DIMENSION = 128
D_8_2 = 6435
PAIRS = tuple((i,j) for i in range(3) for j in range(i,3))


@lru_cache(maxsize=2)
def _majorana_actions(source_parity):
    """Signed permutations gamma: source parity -> opposite parity.

    Returned permutations map output indices to input indices. The associated
    phase is the matrix element in that row. This implements the same
    little-endian convention as the original FK experiment, independently.
    """
    if source_parity not in (0,1):
        raise ValueError("source_parity must be 0 or 1")
    source = np.array([m for m in range(256) if m.bit_count()%2 == source_parity])
    destination = [m for m in range(256) if m.bit_count()%2 != source_parity]
    destination_index = {m:i for i,m in enumerate(destination)}
    inverse = np.empty((16,128), dtype=np.int64)
    phase_at_output = np.empty((16,128), dtype=complex)
    for a in range(16):
        mode, imaginary = divmod(a,2)
        lower = (1 << mode)-1
        for input_index, raw_mask in enumerate(source):
            mask = int(raw_mask)
            sign = -1 if (mask & lower).bit_count()%2 else 1
            phase = sign * ((-1j if (mask >> mode)&1 else 1j) if imaginary else 1)
            output_index = destination_index[mask ^ (1 << mode)]
            inverse[a,output_index] = input_index
            phase_at_output[a,output_index] = phase
    inverse.setflags(write=False)
    phase_at_output.setflags(write=False)
    return inverse, phase_at_output


def apply_lambda(tensors, source_parity=0):
    """Apply Lambda=sum_a gamma_a tensor gamma_a to pair tensor(s).

    Input and output shapes are (...,128,128). Both copies have source_parity;
    the output has the opposite parity in both copies. Tensor products are
    ordinary Hilbert-space tensor copies, not spatial graded tensor products.
    """
    tensors = np.asarray(tensors, dtype=complex)
    if tensors.shape[-2:] != (128,128):
        raise ValueError("pair tensors must have final shape (128,128)")
    permutations, phases = _majorana_actions(source_parity)
    result = np.zeros_like(tensors)
    for permutation, phase in zip(permutations,phases):
        result += (tensors[...,permutation[:,None],permutation[None,:]]
                   * phase[:,None] * phase[None,:])
    return result


def apply_lambda_squared(tensors):
    return apply_lambda(apply_lambda(tensors,0),1)


def project_gaussian_pair(tensors):
    """Project symmetric even-even tensors onto the Gaussian Cartan component.

    On Sym^2(S_+), Lambda^2 has eigenvalues 256,64,0. Hence
    P_G=(I-Lambda^2/64)(I-Lambda^2/256).
    This formula is not a projector on the antisymmetric tensor subspace.
    """
    tensors = np.asarray(tensors, dtype=complex)
    squared = apply_lambda_squared(tensors)
    fourth = apply_lambda_squared(squared)
    return tensors-(5/256)*squared+fourth/16384


def full_tensor_reconstruction(witness):
    """Compute C_2(u) directly in all 128 coordinates, without singlet projection."""
    witness = np.asarray(witness,dtype=complex)
    if witness.shape != (128,):
        raise ValueError("expected a 128-dimensional even witness")
    projected_pair = project_gaussian_pair(np.outer(witness,witness))
    return np.einsum("ij,j->i",projected_pair,witness.conj(),optimize=False)


def closed_form_fk_matrix():
    """Algebraic FK matrix in the fixed S phase convention; see integer proof."""
    a = np.array([[99,0,0,1,0,3], [0,100,0,0,4*math.sqrt(7),0],
                  [0,0,96,4*math.sqrt(14),0,0],
                  [1,0,4*math.sqrt(14),99,0,25],
                  [0,4*math.sqrt(7),0,0,124,0], [3,0,0,25,0,99]],dtype=float)
    return a/128


def integer_fk_matrix_proof():
    """Verify closed-form entries using exact signed-integer tensor arithmetic.

    The Fock support and radicals of S are specified algebraically. Each of its
    symmetric-square columns is an integer vector times one scalar radical.
    Lambda on the pair space has only 0,+1,-1 entries. The numerator of the
    polynomial projector and its Gram matrix are therefore exact int64 values.
    A conservative magnitude bound is below 2**40, far below int64 overflow.
    Only the final comparison to the floating model uses square roots.
    """
    masks = [m for m in range(256) if m.bit_count()%2 == 0]
    index = {m:i for i,m in enumerate(masks)}
    coefficients = np.zeros((3,128),dtype=np.int64)
    for mask,sign in ((119,1),(120,-1),(135,-1),(136,1)):
        coefficients[0,index[mask]] = sign
    for mask,sign in ((15,-1),(60,1),(90,-1),(105,1),(150,1),(165,-1),(195,1),(240,-1)):
        coefficients[1,index[mask]] = sign
    for mask,value in ((30,2),(45,-2),(75,2),(180,2),(210,-2),(225,2),
                       (119,-1),(120,-1),(135,-1),(136,-1)):
        coefficients[2,index[mask]] = value
    tensors = []
    for i,j in PAIRS:
        tensor = np.outer(coefficients[i],coefficients[j])
        tensors.append(tensor if i == j else tensor+tensor.T)
    tensors = np.array(tensors,dtype=np.int64)

    def integer_lambda(values, parity):
        permutations,phases = _majorana_actions(parity)
        result = np.zeros_like(values)
        for permutation,phase in zip(permutations,phases):
            signs = phase[:,None]*phase[None,:]
            if np.any(signs.imag != 0) or not np.all(np.abs(signs.real) == 1):
                raise ArithmeticError("pair Majorana action is not signed integer")
            result += values[:,permutation[:,None],permutation[None,:]]*signs.real.astype(np.int64)
        return result

    squared = integer_lambda(integer_lambda(tensors,0),1)
    fourth = integer_lambda(integer_lambda(squared,0),1)
    numerator = 16384*tensors-320*squared+fourth
    gram = np.einsum("aij,bij->ab",tensors,numerator,optimize=False)
    expected = np.array([
        [202752,0,0,-4096,0,43008],
        [0,819200,0,0,229376,0],
        [0,0,2752512,-229376,0,0],
        [-4096,0,-229376,811008,0,-716800],
        [0,229376,0,0,7110656,0],
        [43008,0,0,-716800,0,9934848]],dtype=np.int64)
    if not np.array_equal(gram,expected):
        raise ArithmeticError("exact integer Gram does not match the closed-form derivation")
    factors = np.array([1/4,1j/8,1/(4*math.sqrt(14)),-1/8,
                        1j/(8*math.sqrt(7)),1/28])
    matrix = gram/16384*factors.conj()[:,None]*factors[None,:]
    exact_basis = (coefficients*np.array([.5,1j/math.sqrt(8),1/(2*math.sqrt(7))])[:,None]).T
    return {"verified":True,"integer_gram":gram,"integer_gram_expected":expected,
            "matrix":matrix,"basis":exact_basis,
            "integer_arithmetic":"int64; conservative intermediate bound < 2**40",
            "radical_factors":factors}


def symmetric_square_basis(basis):
    basis = np.asarray(basis, dtype=complex)
    if basis.shape != (128,3):
        raise ValueError("expected a (128,3) singlet basis")
    result = []
    for i,j in PAIRS:
        tensor = np.outer(basis[:,i],basis[:,j])
        if i != j:
            tensor = (tensor+tensor.T)/math.sqrt(2)
        result.append(tensor)
    return np.array(result)


def veronese(z):
    """Degree-two normalized monomials and first/second real derivatives."""
    z = np.asarray(z, dtype=complex)
    if z.shape != (3,) or not np.all(np.isfinite(z)):
        raise ValueError("z must contain three finite complex coordinates")
    dz = np.concatenate((np.eye(3),1j*np.eye(3)),axis=1)
    values, jacobian, second = [], [], []
    for i,j in PAIRS:
        scale = 1. if i == j else math.sqrt(2)
        values.append(scale*z[i]*z[j])
        jacobian.append(scale*(dz[i]*z[j]+z[i]*dz[j]))
        second.append(scale*(np.outer(dz[i],dz[j])+np.outer(dz[j],dz[i])))
    return np.array(values), np.array(jacobian), np.array(second)


@dataclass
class ExactMomentResult:
    q: float
    gradient: np.ndarray
    hessian: np.ndarray | None
    # The integer k=2 projector does not determine the |overlap|^3 mass.
    mass: None = None
    ess: None = None


class ExactK2Moments:
    """Solver-compatible exact-formula q_2, gradient and Hessian backend.

    ``matrix`` is the scaled moment D_(8,2) * Haar moment in Sym^2(C^3).
    ``haar_moment_matrix`` includes the inverse D_(8,2) normalization.
    The basis can be supplied directly or through a model with attribute S.
    """
    k = 2
    d = D_8_2
    dictionary = "full"

    def __init__(self, model_or_basis, *, verify=True):
        started = time.monotonic()
        self.basis = np.array(getattr(model_or_basis,"S",model_or_basis),dtype=complex,copy=True)
        vectors = symmetric_square_basis(self.basis)
        gram = np.einsum("aij,bij->ab",vectors.conj(),vectors,optimize=False)
        orthogonality_error = float(np.linalg.norm(gram-np.eye(6)))
        if orthogonality_error > 1e-9:
            raise ValueError("singlet basis must be orthonormal")
        projected = project_gaussian_pair(vectors)
        raw_matrix = np.einsum("aij,bij->ab",vectors.conj(),projected,optimize=False)
        self.matrix = (raw_matrix+raw_matrix.conj().T)/2
        self.haar_moment_matrix = self.matrix/self.d
        self.diagnostics = {
            "method":"Spin(16) symmetric-square Cartan projector via Lambda polynomial",
            "dictionary":"full","k":2,"matrix_dimension":6,
            "frame_constant":D_8_2,"pair_vector_shape":[128,128],
            "orthogonality_error":orthogonality_error,
            "moment_hermiticity_error":float(np.linalg.norm(raw_matrix-raw_matrix.conj().T)),
            "minimum_moment_eigenvalue":float(np.linalg.eigvalsh(self.matrix)[0]),
            "maximum_moment_eigenvalue":float(np.linalg.eigvalsh(self.matrix)[-1]),
            "uses_monte_carlo":False,"uses_numerical_quadrature":False,
            "coefficient_mass_available":False,"strict_interval_certified":False,
        }
        if verify:
            annihilation = float(np.linalg.norm(apply_lambda(projected,0)))
            idempotence = float(np.linalg.norm(project_gaussian_pair(projected)-projected))
            self.diagnostics.update({"lambda_kernel_residual":annihilation,
                                     "projector_idempotence_residual":idempotence})
            if max(annihilation,idempotence) > 1e-9:
                raise ArithmeticError("Gaussian pair projector failed its structural checks")
        self.diagnostics["construction_seconds"] = time.monotonic()-started

    def evaluate(self, z, hessian=True):
        values, jacobian, second = veronese(z)
        action = np.einsum("ij,j->i",self.matrix,values,optimize=False)
        q = float(np.vdot(values,action).real)
        # Derivatives of q/(2k)=q/4, matching the original Newton solver.
        gradient = .5*np.einsum("ar,a->r",jacobian.conj(),action,optimize=False).real
        hess = None
        if hessian:
            curvature = np.einsum("ar,ab,bs->rs",jacobian.conj(),self.matrix,jacobian,optimize=False)
            curvature += np.einsum("ars,a->rs",second.conj(),action,optimize=False)
            hess = .5*curvature.real
            hess = (hess+hess.T)/2
        return ExactMomentResult(q,gradient,hess)

    def objective(self, z, target):
        return self.evaluate(z,hessian=False).q/4-float(np.vdot(target,z).real)

    def reduced_reconstruction(self, z):
        gradient = self.evaluate(z,hessian=False).gradient
        return gradient[:3]+1j*gradient[3:]

    def reconstruction(self, z):
        """Exact-formula full even ket C_2(Sz), using proven singlet symmetry."""
        return np.einsum("ij,j->i",self.basis,self.reduced_reconstruction(z),optimize=False)


def build_exact_k2(model_or_basis, *, verify=True):
    return ExactK2Moments(model_or_basis,verify=verify)


def run_exact_reference(model, config, budget=None):
    """Build/validate the deterministic matrix and solve the three bounded targets."""
    from fk.solver import solve
    if budget:
        budget.check()
    started = time.monotonic()
    backend = ExactK2Moments(model)
    proof = integer_fk_matrix_proof()
    basis_error = float(np.linalg.norm(proof["basis"]-model.S))
    matrix_error = float(np.linalg.norm(backend.matrix-proof["matrix"]))
    closed_form_error = float(np.linalg.norm(proof["matrix"]-closed_form_fk_matrix()))
    tolerance = config.get("structural_tolerance",1e-10)
    arrays = {"moment_matrix":backend.matrix,"haar_moment_matrix":backend.haar_moment_matrix,
              "closed_form_matrix":closed_form_fk_matrix(),"integer_gram":proof["integer_gram"],
              "exact_radical_basis":proof["basis"],"S":model.S}
    rows = []
    for t in config.get("t_values",[0,11,20]):
        if budget:
            budget.check()
        target = model.target(t)
        reduced = np.einsum("ij,i->j",model.S.conj(),target,optimize=False)
        solution = solve(backend,reduced,tolerance=config.get("gradient_tolerance",1e-10),
                         max_steps=config.get("max_newton_steps",200),budget=budget)
        z = solution["z"]
        witness = np.einsum("ij,j->i",model.S,z,optimize=False)
        reconstruction = backend.reconstruction(z)
        tensor_reconstruction = full_tensor_reconstruction(witness)
        target_residual = float(np.linalg.norm(tensor_reconstruction-target))
        closure = float(np.linalg.norm(tensor_reconstruction-reconstruction))
        q = backend.evaluate(z,hessian=False).q
        key = f"t{float(t):g}"
        arrays["witness_"+key] = witness
        arrays["z_"+key] = z
        arrays["target_"+key] = target
        arrays["reconstruction_"+key] = reconstruction
        arrays["full_tensor_reconstruction_"+key] = tensor_reconstruction
        arrays["history_"+key] = np.array([[r["iteration"],r["objective"],r["gradient_norm"]]
                                          for r in solution["history"]]).reshape(-1,3)
        rows.append({"t":float(t),"k":2,"dictionary":"full","q":q,
                     "witness_norm":float(np.linalg.norm(witness)),
                     "optimizer_converged":bool(solution["converged"]),
                     "optimizer_status":solution["status"],"iterations":solution["iterations"],
                     "gradient_norm":solution["gradient_norm"],
                     "full_reconstruction_residual":target_residual,
                     "reduced_reconstruction_residual":float(np.linalg.norm(reconstruction-target)),
                     "full_tensor_singlet_closure_residual":closure,
                     "euler_error":abs(q-float(np.vdot(witness,tensor_reconstruction).real)),
                     "coefficient_mass":None,"coefficient_mass_status":"not_determined_by_integer_k2_moment",
                     "finite_k_objective_only":True,"strict_interval_certified":False,
                     "verified":bool(solution["converged"] and max(target_residual,closure) <= tolerance)})
    verified = bool(proof["verified"] and max(basis_error,matrix_error,closed_form_error) <= tolerance
                    and all(row["verified"] for row in rows))
    report = {"status":"complete" if verified else "verification_failed","verified":verified,
              "rows":rows,"diagnostics":backend.diagnostics,
              "integer_closed_form_proof_verified":bool(proof["verified"]),
              "integer_arithmetic":proof["integer_arithmetic"],"exact_basis_match_error":basis_error,
              "integer_vs_tensor_matrix_error":matrix_error,"closed_form_matrix_error":closed_form_error,
              "elapsed_seconds":time.monotonic()-started,
              "interpretation":"Exact algebraic Haar k=2 formula with signed-integer coefficient verification; floating Newton and ket arithmetic, not interval certification. No claim about B_2 or true extent."}
    return report,arrays
