package fe

/*
 Solution algorithm for the eigenvalue problem (K - lambda^2 M)x = 0
*/

import "core:math"
import "core:math/rand"
import "core:slice"

Lanczos_Status :: enum {
	Ok,
	Preconditioner_Failed,
	Linear_Solve_Failed,
}

Lanczos_Result :: struct {
	status:    Lanczos_Status,
	converged: int, // number of Lanczos steps actually completed
}

// Finds the lowest natural frequencies and mode shapes of the
// generalized eigenproblem K v = lambda M v.
// `subspace_dim` size of the Krylov basis to build, increase for more eigen pairs but at higher cost.
// `shift` shifts the eigensolve to a particualr region, choose closest to your frequencies of interest.
// returned eigen values & vectors are in order lowest to highest. Frequency is divided out by 2pi.
inexact_shift_lanczos :: proc(
	K, M: Sparse_Matrix,
	subspace_dim: int,
	eigen_values: []f64,
	eigen_vectors: []Vector,
	shift: f64 = 0.0,
	precond_params: Precond_Params = SA_DEFAULT,
	solver_params: Solver_Params = CG_DEFAULT,
) -> Lanczos_Result {
	LANCZOS_BREAKDOWN_TOL :: 1e-25
	ORTHO_PASSES :: 2
	BISECTION_MAX_ITER :: 64
	BISECTION_TOL :: 1e-15
	BISECTION_BOUND_PAD :: 1.0
	INVERSE_ITER_PASSES :: 2

	TRIDIAGONAL_PIVOT_EPSILON :: 1e-14
	EIGENVECTOR_NORM_FLOOR :: 1e-30

	scratch_guard()
	context.allocator = scratch()

	N := sp_rows(K.sp)
	subspace_dim := subspace_dim
	if subspace_dim > N { subspace_dim = N }

	V := make([][]f64, subspace_dim)
	for i in 0 ..< subspace_dim { V[i] = make([]f64, N) }

	alpha := make([]f64, subspace_dim)
	beta := make([]f64, subspace_dim + 1)
	z := make([]f64, N)
	q := make([]f64, N)
	w := make([]f64, N)

	K_shifted := sp_from_sparsity(K.sp)

	for idx in 0 ..< len(K.values) {
		K_shifted.values[idx] = K.values[idx] - shift * M.values[idx]
	}

	reference_scale := 0.0
	for i in 0 ..< N {
		if diag := sp_get(K, i, i); diag != nil {
			if math.abs(diag^) > reference_scale { reference_scale = math.abs(diag^) }
		}
	}

	precond, ok := amgcl_precond_create(K_shifted, precond_params)
	if !ok { return Lanczos_Result{status = .Preconditioner_Failed} }
	defer amgcl_precond_destroy(precond)

	for i in 0 ..< N { V[0][i] = rand.float64_range(0.0, 1.0) }
	sp_normalize(M, V[0])

	actual_m := 0
	status := Lanczos_Status.Ok

	for j in 0 ..< subspace_dim {
		actual_m = j + 1

		sp_gemv(M, V[j], q)

		if amgcl_result := amgcl_solve(precond, q, z, solver_params); amgcl_result.status != .Converged {
			status = .Linear_Solve_Failed; actual_m -= 1
			break
		}

		alpha[j] = vec_dot(z, q)

		for i in 0 ..< N {
			w[i] = z[i] - alpha[j] * V[j][i]
			if j > 0 {
				w[i] -= beta[j - 1] * V[j - 1][i]
			}
		}

		for pass in 0 ..< ORTHO_PASSES {
			for p in 0 ..< j {
				sp_gemv(M, V[p], q)
				dot := vec_dot(w, q)
				vec_axpy(V[p], w, -dot)
			}
		}

		w_nrm := math.sqrt(sp_inner(M, w, w))

		if w_nrm <= LANCZOS_BREAKDOWN_TOL * reference_scale {
			beta[j] = 0.0
			break
		}

		if j != subspace_dim - 1 {
			copy(V[j + 1], w)
			beta[j] = sp_normalize(M, V[j + 1])
		} else {
			beta[j] = w_nrm
		}
	}

	valid_alpha := alpha[0:actual_m]
	valid_beta := beta[0:actual_m - 1]

	Y := make([][]f64, actual_m)
	for i in 0 ..< actual_m { Y[i] = make([]f64, actual_m) }

	symmetric_tridiagonal_qr(valid_alpha, valid_beta, Y)

	for k in 0 ..< actual_m {
		if k >= len(eigen_values) { break }
		true_eigenvalue := shift + (1.0 / valid_alpha[k])
		omega := math.sqrt(true_eigenvalue)
		eigen_values[k] = omega / (2.0 * math.PI)
	}

	for i in 0 ..< N {
		for k in 0 ..< actual_m {
			sum := 0.0
			for j in 0 ..< actual_m { sum += V[j][i] * Y[k][j] }
			if k < len(eigen_vectors) { eigen_vectors[k][i] = sum }
		}
	}

	perm := slice.sort_with_indices(eigen_values)
	slice.sort_from_permutation_indices(eigen_vectors, perm)

	return Lanczos_Result{status = status, converged = actual_m}

	symmetric_tridiagonal_qr :: proc(d: []f64, e_input: []f64, Y: [][]f64) {
		m := len(d)
		if m <= 1 { return }

		e := make([]f64, m)
		for i in 0 ..< m - 1 {
			if i < len(e_input) { e[i] = e_input[i] }
		}
		e[m - 1] = 0.0

		temp_eigenvalues := make([]f64, m)

		max_bound := 0.0
		for i in 0 ..< m {
			r := 0.0
			if i > 0 { r += math.abs(e[i - 1]) }
			if i < m - 1 { r += math.abs(e[i]) }
			bound := math.abs(d[i]) + r
			if bound > max_bound { max_bound = bound }
		}

		global_low := -max_bound - BISECTION_BOUND_PAD
		global_high := max_bound + BISECTION_BOUND_PAD

		for k in 0 ..< m {
			low := global_low
			high := global_high

			for iter in 0 ..< BISECTION_MAX_ITER {
				if math.abs(high - low) < BISECTION_TOL { break }
				mid := low + (high - low) * 0.5
				if sturm_count(d, e, mid) > (m - 1 - k) {
					high = mid
				} else {
					low = mid
				}
			}
			temp_eigenvalues[k] = low + (high - low) * 0.5
		}

		c := make([]f64, m)

		for k in 0 ..< m {
			mu := temp_eigenvalues[k]

			for i in 0 ..< m { Y[k][i] = rand.float64_range(-1.0, 1.0) }

			for pass in 0 ..< INVERSE_ITER_PASSES {
				for i in 0 ..< m { c[i] = d[i] - mu }

				for i in 1 ..< m {
					if c[i - 1] == 0.0 { c[i - 1] = TRIDIAGONAL_PIVOT_EPSILON }
					factor := e[i - 1] / c[i - 1]
					c[i] -= factor * e[i - 1]
					Y[k][i] -= factor * Y[k][i - 1]
				}

				if c[m - 1] == 0.0 { c[m - 1] = TRIDIAGONAL_PIVOT_EPSILON }
				Y[k][m - 1] /= c[m - 1]
				for i := m - 2; i >= 0; i -= 1 {
					if c[i] == 0.0 { c[i] = TRIDIAGONAL_PIVOT_EPSILON }
					Y[k][i] = (Y[k][i] - e[i] * Y[k][i + 1]) / c[i]
				}

				norm_sq := 0.0
				for i in 0 ..< m { norm_sq += Y[k][i] * Y[k][i] }
				norm := math.sqrt(norm_sq)
				if norm > EIGENVECTOR_NORM_FLOOR {
					for i in 0 ..< m { Y[k][i] /= norm }
				}
			}
		}

		copy(d, temp_eigenvalues)
	}

	sturm_count :: proc(d, e: []f64, x: f64) -> int {
		m := len(d)
		count := 0
		ratio := d[0] - x
		if ratio < 0.0 { count += 1 }

		for i in 1 ..< m {
			if ratio == 0.0 { ratio = TRIDIAGONAL_PIVOT_EPSILON }
			ratio = (d[i] - x) - (e[i - 1] * e[i - 1]) / ratio
			if ratio < 0.0 { count += 1 }
		}
		return count
	}
}
