package fe

/*
 General linear algebra containers for, possibly sparse, systems of linear equations.
 For sparse matrices, CSR format is used
*/

import "core:math"
import "core:slice"

// Arbitrary length vector
Vector :: []f64

// Basic sparse matrix
Sparse_Matrix :: struct {
	using sp: Sparsity,
	values:   []f64,
}

// CSR sparsity pattern
Sparsity :: struct {
	row_ptrs: []i32,
	columns:  []i32,
}

// Column-major
Dense_Matrix :: struct {
	values:     []f64,
	rows, cols: i32,
}

//== Access & Creation

dense_create :: proc(#any_int rows, cols: i32, alloc := context.allocator) -> Dense_Matrix {
	assert(rows > 0 && cols > 0)
	return {values = make([]f64, rows * cols, alloc), rows = rows, cols = cols}
}

// Will panic (if bounds checking is enabled) if is out of bounds.
dn_get :: proc(d: Dense_Matrix, #any_int row, col: i32) -> ^f64 {
	return &d.values[col * d.rows + row]
}

// Column j of d as a Vector view.
dn_col :: proc(d: Dense_Matrix, #any_int j: int) -> Vector {
	return d.values[j * int(d.rows):(j + 1) * int(d.rows)]
}

// Pivot columns for factorizations
dn_pivots :: proc(a: Dense_Matrix, alloc := context.allocator) -> []i32 {
	return make([]i32, a.rows, alloc)
}

// Does not copy sparsity
sparse_from_sparsity :: proc(sp: Sparsity, alloc := context.allocator) -> Sparse_Matrix {
	return {sp = sp, values = make([]f64, len(sp.columns), alloc)}
}

// Returns nil if entry does not exist.
sp_get :: proc(sm: Sparse_Matrix, #any_int row, col: i32) -> ^f64 {
	row_slice := sm.columns[sm.row_ptrs[row]:sm.row_ptrs[row + 1]]
	idx, found := slice.binary_search(row_slice, col)
	if !found { return nil }
	return &sm.values[int(sm.row_ptrs[row]) + idx]
}

sp_n_rows :: proc(sp: Sparsity) -> int {
	return len(sp.row_ptrs) - 1
}

//== Vector operations

// a . b
vec_dot :: proc(a, b: Vector) -> f64 {
	s: f64 = 0
	for i in 0 ..< len(a) { s += a[i] * b[i] }
	return s
}

vec_dot_rank :: proc(a, b: Vector) -> f64 {
	return rank_sum(vec_dot(rank_slice(a), rank_slice(b)))
}

// |a|
vec_norm :: proc(a: Vector) -> f64 {
	return math.sqrt(vec_dot(a, a))
}

vec_norm_rank :: proc(a: Vector) -> f64 {
	return math.sqrt(vec_dot_rank(a, a))
}

// a *= alpha
vec_scale :: proc(a: Vector, alpha: f64) {
	for &v in a { v *= alpha }
}

vec_scale_rank :: proc(a: Vector, alpha: f64) {
	vec_scale(rank_slice(a), alpha)
}

// y += a * x
vec_axpy :: proc(x, y: Vector, a: f64 = 1) {
	assert(len(x) == len(y))
	#no_bounds_check for i in 0 ..< len(x) { y[i] += a * x[i] }
}

vec_axpy_rank :: proc(x, y: Vector, a: f64 = 1) {
	vec_axpy(rank_slice(x), rank_slice(y), a)
}

//== Sparse ops

// y = alpha * A * x + beta * y (x, y must not alias)
sp_gemv :: proc(a: Sparse_Matrix, x, y: Vector, alpha := 1.0, beta := 0.0) {
	for row in 0 ..< sp_n_rows(a) {
		sum: f64 = 0
		for idx in a.row_ptrs[row] ..< a.row_ptrs[row + 1] {
			sum += a.values[idx] * x[a.columns[idx]]
		}
		y[row] = beta == 0 ? alpha * sum : alpha * sum + beta * y[row] // beta == 0 ignores y (may be NaN)
	}
}

sp_gemv_rank :: proc(a: Sparse_Matrix, x, y: Vector, alpha := 1.0, beta := 0.0) {
	rank_sync() // x may have been written by other ranks
	r := rank_range(sp_n_rows(a))
	mine := Sparse_Matrix {
		sp = {row_ptrs = a.row_ptrs[r.min:r.max + 1], columns = a.columns},
		values = a.values,
	}
	sp_gemv(mine, x, y[r.min:r.max], alpha, beta)
	rank_sync() // nobody may write x until every rank is done reading it
}

// x^T * A * y
sp_inner :: proc(a: Sparse_Matrix, x, y: Vector) -> f64 {
	local: f64 = 0
	for row in 0 ..< sp_n_rows(a) {
		row_sum: f64 = 0
		for idx in a.row_ptrs[row] ..< a.row_ptrs[row + 1] {
			row_sum += a.values[idx] * y[a.columns[idx]]
		}
		local += x[row] * row_sum
	}
	return local
}

sp_inner_rank :: proc(a: Sparse_Matrix, x, y: Vector) -> f64 {
	rank_sync() // y may have been written by other ranks
	r := rank_range(sp_n_rows(a))
	mine := Sparse_Matrix {
		sp = {row_ptrs = a.row_ptrs[r.min:r.max + 1], columns = a.columns},
		values = a.values,
	}
	return rank_sum(sp_inner(mine, x[r.min:r.max], y)) // rank_sum's barrier covers the exit
}

// scale v to unit M-norm, returns the original norm
sp_normalize :: proc(M: Sparse_Matrix, v: Vector) -> f64 {
	nrm := math.sqrt(sp_inner(M, v, v))
	if nrm > 1e-30 { vec_scale(v, 1.0 / nrm) }
	return nrm
}

sp_normalize_rank :: proc(M: Sparse_Matrix, v: Vector) -> f64 {
	nrm := math.sqrt(sp_inner_rank(M, v, v)) // identical on all ranks, so all take the same branch
	if nrm > 1e-30 { vec_scale_rank(v, 1.0 / nrm) }
	return nrm
}

//== Dense ops

dn_copy :: proc(dst, src: Dense_Matrix) {
	assert(dst.rows == src.rows && dst.cols == src.cols)
	copy(dst.values, src.values)
}

// d *= alpha
dn_scale :: proc(d: Dense_Matrix, alpha: f64) {
	vec_scale(d.values, alpha)
}

// d = 0
dn_zero :: proc(d: Dense_Matrix) {
	slice.zero(d.values)
}

// y += alpha * x
dn_axpy :: proc(x, y: Dense_Matrix, alpha := 1.0) {
	assert(x.rows == y.rows && x.cols == y.cols)
	vec_axpy(x.values, y.values, alpha)
}

// y = alpha * A * x + beta * y
dn_gemv :: proc(a: Dense_Matrix, x, y: Vector, alpha := 1.0, beta := 0.0) {
	assert(len(x) == int(a.cols) && len(y) == int(a.rows))
	assert(len(x) == 0 || raw_data(x) != raw_data(y))
	if beta == 0 { slice.zero(y) } else if beta != 1 { vec_scale(y, beta) } 	// beta == 0 ignores y (may be NaN)
	for k in 0 ..< a.cols {
		vec_axpy(dn_col(a, k), y, alpha * x[k])
	}
}

// C = alpha * A * B + beta * C
dn_gemm :: proc(a, b, c: Dense_Matrix, alpha := 1.0, beta := 0.0) {
	assert(a.cols == b.rows && a.rows == c.rows && b.cols == c.cols)
	assert(raw_data(c.values) != raw_data(a.values) && raw_data(c.values) != raw_data(b.values))
	if beta == 0 { dn_zero(c) } else if beta != 1 { dn_scale(c, beta) }
	for j in 0 ..< c.cols {
		for k in 0 ..< a.cols {
			vec_axpy(dn_col(a, k), dn_col(c, j), alpha * dn_get(b, k, j)^)
		}
	}
}

// in-place LU with partial pivoting.
dn_lu_factor :: proc(a: Dense_Matrix, pivots: []i32) -> (ok: bool) {
	n := int(a.rows)
	assert(a.rows == a.cols && len(pivots) >= n)
	for k in 0 ..< n {
		colk := dn_col(a, k)

		p := k
		best := math.abs(colk[k])
		for i in k + 1 ..< n {
			if v := math.abs(colk[i]); v > best {
				best = v
				p = i
			}
		}
		if !(best > 0) { return false }

		pivots[k] = i32(p)
		if p != k {
			for j in 0 ..< n {
				cj := dn_col(a, j)
				cj[k], cj[p] = cj[p], cj[k]
			}
		}

		pivot := colk[k]
		for i in k + 1 ..< n { colk[i] /= pivot }
		for j in k + 1 ..< n {
			cj := dn_col(a, j)
			vec_axpy(colk[k + 1:], cj[k + 1:], -cj[k])
		}
	}
	return true
}

// solve A x = b in place using dn_lu_factor's result
dn_lu_solve_vec :: proc(lu: Dense_Matrix, pivots: []i32, b: Vector) {
	n := int(lu.rows)
	assert(lu.rows == lu.cols && len(pivots) >= n && len(b) == n)
	for k in 0 ..< n {
		p := int(pivots[k])
		if p != k { b[k], b[p] = b[p], b[k] }
	}
	for k in 0 ..< n { 	// L y = P b
		vec_axpy(dn_col(lu, k)[k + 1:], b[k + 1:], -b[k])
	}
	for k := n - 1; k >= 0; k -= 1 { 	// U x = y
		colk := dn_col(lu, k)
		b[k] /= colk[k]
		vec_axpy(colk[:k], b[:k], -b[k])
	}
}

// solve A X = B in place, one right-hand side per column of B
dn_lu_solve :: proc(lu: Dense_Matrix, pivots: []i32, b: Dense_Matrix) {
	assert(lu.rows == b.rows)
	for j in 0 ..< b.cols {
		dn_lu_solve_vec(lu, pivots, dn_col(b, j))
	}
}
