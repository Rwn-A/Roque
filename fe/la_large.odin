package fe

/*
 Containers and operations for large, possibly sparse, systems of linear equations. Sparse matrices are CSR.

 `_rank` variants are collective: each rank works on its rank_slice, syncing before it reads and after it writes, so
 the data is consistent on every rank before and after the call whatever partition wrote it.
*/

import "core:math"
import "core:slice"

Vector :: []f64

// CSR sparsity pattern.
Sparsity :: struct {
	row_ptrs: []i32,
	columns:  []i32, // sorted per row
}

Sparse_Matrix :: struct {
	using sp: Sparsity,
	values:   []f64,
}

// Square block matrix over a partition of a flat vector. Block (i, j) is blocks[i * n + j], block i's rows and
// columns are [offsets[i], offsets[i + 1]).
Block_Sparse :: struct {
	n:       int,
	offsets: []int,
	blocks:  []Sparse_Matrix,
}

// Column-major.
Dense_Matrix :: struct {
	values:     []f64,
	rows, cols: i32,
}

//== Vector

// a . b
vec_dot :: proc(a, b: Vector) -> f64 {
	assert(len(a) == len(b))
	s: f64 = 0
	for i in 0 ..< len(a) { s += a[i] * b[i] }
	return s
}

vec_dot_rank :: proc(a, b: Vector) -> f64 {
	rank_sync()
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
	rank_sync()
	vec_scale(rank_slice(a), alpha)
	rank_sync()
}

vec_zero_rank :: proc(a: Vector) {
	rank_sync()
	slice.zero(rank_slice(a))
	rank_sync()
}

// y += a * x
vec_axpy :: proc(x, y: Vector, a: f64 = 1) {
	assert(len(x) == len(y))
	#no_bounds_check for i in 0 ..< len(x) { y[i] += a * x[i] }
}

vec_axpy_rank :: proc(x, y: Vector, a: f64 = 1) {
	rank_sync()
	vec_axpy(rank_slice(x), rank_slice(y), a)
	rank_sync()
}

//== Sparse

// Number of rows, the size of a compatible vector.
sp_n_rows :: proc(sp: Sparsity) -> int {
	return len(sp.row_ptrs) - 1
}

// Entry (row, col), nil if it isn't in the pattern.
sp_get :: proc(sm: Sparse_Matrix, #any_int row, col: i32) -> ^f64 {
	row_slice := sm.columns[sm.row_ptrs[row]:sm.row_ptrs[row + 1]]
	idx, found := slice.binary_search(row_slice, col)
	if !found { return nil }
	return &sm.values[int(sm.row_ptrs[row]) + idx]
}

// Zero matrix over an existing pattern, which is shared, not copied.
sp_from_sparsity :: proc(sp: Sparsity, alloc := context.allocator) -> Sparse_Matrix {
	return {sp = sp, values = make([]f64, len(sp.columns), alloc)}
}

//== Sparse ops

// Values to zero, the pattern is kept.
sp_zero :: proc(m: Sparse_Matrix) {
	slice.zero(m.values)
}

// alpha a + beta b on the union of their patterns.
sp_sum :: proc(a, b: Sparse_Matrix, alpha, beta: f64, alloc := context.allocator) -> Sparse_Matrix {
	n := sp_n_rows(a)
	assert(n == sp_n_rows(b))
	row_ptrs := make([]i32, n + 1, alloc)
	columns := make([dynamic]i32, 0, len(a.columns) + len(b.columns), alloc)
	values := make([dynamic]f64, 0, len(a.columns) + len(b.columns), alloc)

	// merge each row's sorted columns
	for row in 0 ..< n {
		ia, ib := a.row_ptrs[row], b.row_ptrs[row]
		for ia < a.row_ptrs[row + 1] || ib < b.row_ptrs[row + 1] {
			ca := a.columns[ia] if ia < a.row_ptrs[row + 1] else max(i32)
			cb := b.columns[ib] if ib < b.row_ptrs[row + 1] else max(i32)
			col, v := min(ca, cb), 0.0
			if ca == col {
				v += alpha * a.values[ia]
				ia += 1
			}
			if cb == col {
				v += beta * b.values[ib]
				ib += 1
			}
			append(&columns, col)
			append(&values, v)
		}
		row_ptrs[row + 1] = i32(len(columns))
	}
	return {sp = {row_ptrs = row_ptrs, columns = columns[:]}, values = values[:]}
}

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

// Scales v to unit M-norm, returns the original norm.
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

//== Block sparse

// Values to zero, the patterns are kept.
bsp_zero :: proc(a: Block_Sparse) {
	for b in a.blocks { sp_zero(b) }
}

// y = alpha * A * x + beta * y on the flat vectors (x, y must not alias)
bsp_gemv :: proc(a: Block_Sparse, x, y: Vector, alpha := 1.0, beta := 0.0) {
	for i in 0 ..< a.n {
		yi := y[a.offsets[i]:a.offsets[i + 1]]
		for j in 0 ..< a.n {
			sp_gemv(a.blocks[i * a.n + j], x[a.offsets[j]:a.offsets[j + 1]], yi, alpha, beta if j == 0 else 1)
		}
	}
}

// Every block in one CSR over the flat vector. A single block is returned as is, shared rather than copied.
bsp_to_sp :: proc(a: Block_Sparse, alloc := context.allocator) -> Sparse_Matrix {
	if a.n == 1 { return a.blocks[0] }

	nnz := 0
	for blk in a.blocks { nnz += len(blk.columns) }
	row_ptrs := make([]i32, a.offsets[a.n] + 1, alloc)
	columns := make([]i32, nnz, alloc)
	values := make([]f64, nnz, alloc)

	// a flat row is its blocks' rows side by side, block columns are disjoint increasing ranges so it stays sorted
	k := 0
	for i in 0 ..< a.n {
		for row in 0 ..< a.offsets[i + 1] - a.offsets[i] {
			for j in 0 ..< a.n {
				blk := a.blocks[i * a.n + j]
				for idx in blk.row_ptrs[row] ..< blk.row_ptrs[row + 1] {
					columns[k] = i32(a.offsets[j]) + blk.columns[idx]
					values[k] = blk.values[idx]
					k += 1
				}
			}
			row_ptrs[a.offsets[i] + row + 1] = i32(k)
		}
	}
	return {sp = {row_ptrs = row_ptrs, columns = columns}, values = values}
}

//== Dense

dn_create :: proc(#any_int rows, cols: i32, alloc := context.allocator) -> Dense_Matrix {
	assert(rows > 0 && cols > 0)
	return {values = make([]f64, rows * cols, alloc), rows = rows, cols = cols}
}

// Entry (row, col), bounds checked.
dn_get :: proc(d: Dense_Matrix, #any_int row, col: i32) -> ^f64 {
	return &d.values[col * d.rows + row]
}

// Column j of d as a Vector view.
dn_col :: proc(d: Dense_Matrix, #any_int j: int) -> Vector {
	return d.values[j * int(d.rows):(j + 1) * int(d.rows)]
}

// Pivot storage for dn_lu_factor.
dn_pivots :: proc(a: Dense_Matrix, alloc := context.allocator) -> []i32 {
	return make([]i32, a.rows, alloc)
}

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

// In-place LU with partial pivoting.
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

// Solves A x = b in place with dn_lu_factor's result.
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

// Solves A X = B in place, one right-hand side per column of B.
dn_lu_solve :: proc(lu: Dense_Matrix, pivots: []i32, b: Dense_Matrix) {
	assert(lu.rows == b.rows)
	for j in 0 ..< b.cols {
		dn_lu_solve_vec(lu, pivots, dn_col(b, j))
	}
}
