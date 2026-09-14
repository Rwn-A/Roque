package fe

/*
 Large, runtime-sized vectors and matrices. Including CSR formatted sparse matrices.
*/

import "core:math"
import "core:slice"

// TODO: Dense matrix ops for HDG

Vector :: []f64

// Basic CSR matrix.
Sparse_Matrix :: struct {
	using sp: Sparsity,
	values:   []f64,
}

Sparsity :: struct {
	row_ptrs: []i32,
	columns:  []i32,
}

// Column-major
Dense_Matrix :: struct {
	values:     []f64,
	rows, cols: i16,
}

dense_get :: proc(d: Dense_Matrix, #any_int row, col: i16) -> ^f64 {
	return &d.values[row * d.cols + col]
}

// Returns nil if entry does not exist.
sp_get :: proc(sm: Sparse_Matrix, #any_int row, col: i32) -> ^f64 {
	row_slice := sm.columns[sm.row_ptrs[row]:sm.row_ptrs[row + 1]]
	idx, found := slice.binary_search(row_slice, col)
	if !found { return nil }
	return &sm.values[int(sm.row_ptrs[row]) + idx]
}

sp_from_sparsity :: proc(sp: Sparsity, alloc := context.allocator) -> Sparse_Matrix {
	return {sp = sp, values = make([]f64, len(sp.columns), alloc)}
}

sp_rows :: proc(sp: Sparsity) -> int {
	return len(sp.row_ptrs) - 1
}


//== basic vector / matrix ops.

vec_dot :: proc(a, b: Vector) -> f64 {
	r := rank_range(len(a))
	local: f64 = 0
	for i in r.min ..< r.max { local += a[i] * b[i] }
	return rank_sum(local)
}

vec_norm :: proc(a: Vector) -> f64 {
	return math.sqrt(vec_dot(a, a))
}

vec_scale :: proc(a: Vector, alpha: f64) {
	r := rank_range(len(a))
	for i in r.min ..< r.max { a[i] *= alpha }
}

vec_axpy :: proc(x, y: Vector, a: f64 = 1) {
	r := rank_range(len(x))
	for i in r.min ..< r.max { y[i] += a * x[i] }
}

sp_gemv :: proc(a: Sparse_Matrix, x: Vector, y: Vector, alpha := 1.0, beta := 0.0) {
	r := rank_range(sp_rows(a))
	for row in r.min ..< r.max {
		sum: f64 = 0
		for idx in a.row_ptrs[row] ..< a.row_ptrs[row + 1] {
			sum += a.values[idx] * x[a.columns[idx]]
		}
		y[row] = alpha * sum + beta * y[row]
	}
}

sp_inner :: proc(a: Sparse_Matrix, x, y: Vector) -> f64 {
	r := rank_range(sp_rows(a))
	local: f64 = 0
	for row in r.min ..< r.max {
		row_sum: f64 = 0
		for idx in a.row_ptrs[row] ..< a.row_ptrs[row + 1] {
			row_sum += a.values[idx] * y[a.columns[idx]]
		}
		local += x[row] * row_sum
	}
	return rank_sum(local)
}

sp_normalize :: proc(M: Sparse_Matrix, v: Vector) -> f64 {
	nrm := math.sqrt(sp_inner(M, v, v))
	if nrm > 1e-30 { vec_scale(v, 1.0 / nrm) }
	return nrm
}
