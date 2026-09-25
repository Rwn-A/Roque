package fe

/*
 Perform tensor contractions over basis (B), coefficient (C) and point (P) space data.

 Ex. standard stiffness matrix -> K = integral(B^T D B).
 In the above, K is considered coefficient space, B basis space, and D point space.

 For performance, the innermost dimension of a contraction (basis components per dof, fields per component)
 are taken to be compile time known. These two quantities are generally fixed by the PDE, or for more
 dimensionally agnostic pdes, a function of the PDE and the dimension. With only these two dimensions compile time
 the element type, basis family, basis order & quadrature rule are flexible at runtime.
*/

// Basis values for a specific basis quantity each quantity is `cmpnts` long.
Bvec :: struct(T: typeid) {
	points, dofs, cmpnts: int,
	data:                 []T,
}

Bvec_Point :: struct(T: typeid) {
	dofs, cmpnts: int,
	data:         []T,
}

bvec_create :: proc($T: typeid, points, dofs, cmpnts: int, alloc := context.allocator) -> Bvec(T) {
	return {points = points, dofs = dofs, cmpnts = cmpnts, data = make([]T, points * dofs * cmpnts, alloc)}
}

bvec_at_point :: proc(bvec: Bvec($T), point: int) -> Bvec_Point(T) {
	point_size := bvec.cmpnts * bvec.dofs
	return {bvec.dofs, bvec.cmpnts, bvec.data[point * point_size:][:point_size]}
}

// All component values for the dof
bvec_dof_block :: proc(bvp: Bvec_Point($T), dof: int) -> []T {
	return bvp.data[dof * bvp.cmpnts:][:bvp.cmpnts]
}

// All component values as a fixed-size vector
bvec_dof_vec :: proc(bvp: Bvec_Point($T), dof: int, $C: int) -> ^Small_Vec(C, T) {
	assert(C == bvp.cmpnts)
	return small_vec_view_from_slice(bvec_dof_block(bvp, dof), C)
}

// Point space data, `cmpnts` should match basis quantity fields is independent direction for stacking.
Pvec :: struct(T: typeid) {
	points, cmpnts, fields: int,
	data:                   []T,
}

Pvec_Point :: struct(T: typeid) {
	cmpnts, fields: int,
	data:           []T,
}

pvec_create :: proc($T: typeid, np, cmpnts, fields: int, alloc := context.allocator) -> Pvec(T) {
	return {np, cmpnts, fields, make([]T, np * fields * cmpnts, alloc)}
}

pvec_at_point :: proc(pvec: Pvec($T), point: int) -> Pvec_Point(T) {
	point_size := pvec.cmpnts * pvec.fields
	return {pvec.cmpnts, pvec.fields, pvec.data[point * point_size:][:point_size]}
}

// Represent the point data as a matrix with component columns and field rows.
pvec_point_matrix :: proc(pvec: Pvec($T), point: int, $C, $F: int) -> ^Small_Mat(F, C, T) {
	return small_mat_view_from_slice(pvec.data[point * pvec.cmpnts * pvec.fields:], F, C)
}

// Coefficient vector, result of a local element weak form
Cvec :: struct(T: typeid) {
	dofs, fields: int,
	data:         []T,
}

cvec_create :: proc($T: typeid, dofs, fields: int, alloc := context.allocator) -> Cvec(T) {
	return {dofs = dofs, fields = fields, data = make([]T, dofs * fields, alloc)}
}

// A cvec sized for contracting basis `b` with point data `p` (one field vector per basis dof).
cvec_create_for :: proc(b: Bvec($BT), p: Pvec($T), alloc := context.allocator) -> Cvec(T) {
	assert(b.cmpnts == p.cmpnts && b.points == p.points)
	return cvec_create(T, b.dofs, p.fields, alloc)
}

// All field values for the dof
cvec_dof_block :: proc(cvec: Cvec($T), dof: int) -> []T {
	return cvec.data[dof * cvec.fields:][:cvec.fields]
}

// All field values for the dof as a fixed-size vector
cvec_dof_vec :: proc(cvec: Cvec($T), dof: int, $F: int) -> ^Small_Vec(F, T) {
	assert(F == cvec.fields)
	return small_vec_view_from_slice(cvec_dof_block(cvec, dof), F)
}

// c_dof = frames[dof] * c_dof. A length-1 slice applies one matrix to every dof.
cvec_push_inplace :: proc(c: Cvec($T), frames: []Small_Mat($F, F, T)) {
	assert(c.fields == F && (len(frames) == 1 || len(frames) == c.dofs))
	for dof in 0 ..< c.dofs {
		v := cvec_dof_vec(c, dof, F)
		v^ = small_mat_vec_mul(frames[0 if len(frames) == 1 else dof], v^)
	}
}

// c_dof = frames[dof]^T * c_dof. Scatter for a dual (residual, load) that was gathered with `frames`.
cvec_push_inplace_t :: proc($F: int, c: Cvec($T), frames: []Small_Mat(F, F, T)) {
	assert(c.fields == F && (len(frames) == 1 || len(frames) == c.dofs))
	for dof in 0 ..< c.dofs {
		v := cvec_dof_vec(c, dof, F)
		v^ = small_mat_vec_mul_t(frames[0 if len(frames) == 1 else dof], v^)
	}
}

// Coefficient space matrix, result of a local element weak form
Cmat :: struct(T: typeid) {
	row_dofs, col_dofs:     int,
	row_fields, col_fields: int,
	data:                   []T,
}

cmat_create :: proc($T: typeid, r_dofs, c_dofs: int, r_fields, c_fields: int, alloc := context.allocator) -> Cmat(T) {
	return {
		row_dofs = r_dofs,
		col_dofs = c_dofs,
		row_fields = r_fields,
		col_fields = c_fields,
		data = make([]T, r_dofs * c_dofs * r_fields * c_fields, alloc),
	}
}

// A cmat sized for contracting row basis `rb` and column basis `cb` with point matrix `p`.
// For a single shared basis pass it twice.
cmat_create_for :: proc(rb: Bvec($BT), p: Pmat($T), cb: Bvec(BT), alloc := context.allocator) -> Cmat(T) {
	assert(rb.cmpnts == p.layout.row_cmpnts && cb.cmpnts == p.layout.col_cmpnts)
	assert(rb.points == p.points && cb.points == p.points)
	return cmat_create(T, rb.dofs, cb.dofs, p.layout.row_fields, p.layout.col_fields, alloc)
}

// Returns the field block (column-field major) at the dof pair
cmat_dof_block :: proc(cmat: Cmat($T), rdof, cdof: int) -> []T {
	block_size := cmat.col_fields * cmat.row_fields
	offset := (rdof * cmat.col_dofs + cdof) * block_size
	return cmat.data[offset:]
}


// Returns a view of the (row_fields x col_fields) block coupling `rdof` and `cdof`.
cmat_dof_matrix :: proc(cmat: Cmat($T), rdof, cdof: int, $RF, $CF: int) -> ^Small_Mat(RF, CF, T) {
	return small_mat_view_from_slice(cmat_dof_block(cmat, rdof, cdof), RF, CF)
}

// K(rdof, cdof) = R_rdof * K * C_cdof^T.
cmat_push_inplace :: proc(c: Cmat($T), rframes: []Small_Mat($RF, RF, T), cframes: []Small_Mat($CF, CF, T)) {
	assert(c.row_fields == RF && c.col_fields == CF)
	assert(len(rframes) == 1 || len(rframes) == c.row_dofs)
	assert(len(cframes) == 1 || len(cframes) == c.col_dofs)
	for rdof in 0 ..< c.row_dofs {
		r := rframes[0 if len(rframes) == 1 else rdof]
		for cdof in 0 ..< c.col_dofs {
			blk := cmat_dof_matrix(c, rdof, cdof, RF, CF)
			blk^ = small_mat_mul(
				small_mat_mul(r, blk^),
				small_mat_transpose(cframes[0 if len(cframes) == 1 else cdof]),
			)
		}
	}
}

// K(rdof, cdof) = R_rdof^T * K * C_cdof.
cmat_push_inplace_t :: proc(c: Cmat($T), rframes: []Small_Mat($RF, RF, T), cframes: []Small_Mat($CF, CF, T)) {
	assert(c.row_fields == RF && c.col_fields == CF)
	assert(len(rframes) == 1 || len(rframes) == c.row_dofs)
	assert(len(cframes) == 1 || len(cframes) == c.col_dofs)
	for rdof in 0 ..< c.row_dofs {
		rt := small_mat_transpose(rframes[0 if len(rframes) == 1 else rdof])
		for cdof in 0 ..< c.col_dofs {
			blk := cmat_dof_matrix(c, rdof, cdof, RF, CF)
			blk^ = small_mat_mul(small_mat_mul(rt, blk^), cframes[0 if len(cframes) == 1 else cdof])
		}
	}
}

// Matrix of point space data, commonly a 4th order tensor such as the constitutive tensor in elasticty.
Pmat :: struct(T: typeid) {
	points:                 int,
	layout:                 Pmat_Layout,
	field_blocks_per_point: int,
	data:                   []T,
}

Pmat_Layout :: struct {
	row_cmpnts, row_fields: int,
	col_cmpnts, col_fields: int,
	shape:                  Pmat_Shape,
}

// Shape describes the block-shape of the pmat (componext-aixs), each field block is dense.
// DENSE and DIAGONAL field blocks may be rectangular. SYMMETRIC requires equal field counts.
Pmat_Shape :: enum {
	DENSE,
	SYMMETRIC,
	DIAGONAL,
}

Pmat_Point :: struct($T: typeid) {
	using layout: Pmat_Layout,
	data:         []T,
}

Pmat_Block_Status :: enum {
	MISSING,
	FOUND,
	TRANSPOSED,
}

pmat_create :: proc(
	$T: typeid,
	np: int,
	s: Pmat_Shape,
	row_cmpnts, row_fields, col_cmpnts, col_fields: int,
	alloc := context.allocator,
) -> Pmat(T) {
	layout := Pmat_Layout{row_cmpnts, row_fields, col_cmpnts, col_fields, s}

	assert(s == .DENSE || row_cmpnts == col_cmpnts, "Non-dense shape requires equal component counts.")

	num_blocks: int
	switch s {
	case .DENSE: num_blocks = row_cmpnts * col_cmpnts
	case .DIAGONAL: num_blocks = row_cmpnts
	case .SYMMETRIC:
		assert(row_fields == col_fields, "Symmetry requires equal field counts.")
		num_blocks = row_cmpnts * (row_cmpnts + 1) / 2
	}
	data := make([]T, np * num_blocks * row_fields * col_fields, alloc)
	return {points = np, layout = layout, field_blocks_per_point = num_blocks, data = data}
}

// Create a symmetric point matrix
pmat_create_symmetric :: proc($T: typeid, np, cmpnts, fields: int, alloc := context.allocator) -> Pmat(T) {
	return pmat_create(T, np, .SYMMETRIC, cmpnts, fields, cmpnts, fields, alloc)
}

pmat_at_point :: proc(pmat: Pmat($T), point: int) -> Pmat_Point(T) {
	point_size := pmat.field_blocks_per_point * pmat.layout.row_fields * pmat.layout.col_fields
	return {pmat.layout, pmat.data[point * point_size:][:point_size]}
}

// Matrix of field values at the given component pair.
pmat_cmpnt_matrix :: proc(
	pp: Pmat_Point($T),
	rc, cc: int,
	$RF, $CF: int,
) -> (
	^Small_Mat(RF, CF, T),
	Pmat_Block_Status,
) {
	block: int
	switch pp.shape {
	case .DENSE: block = rc * pp.col_cmpnts + cc
	case .DIAGONAL:
		if rc != cc { return {}, .MISSING }
		block = rc
	case .SYMMETRIC:
		if cc > rc {
			block = cc * (cc + 1) / 2 + rc
			return small_mat_view_from_slice(pp.data[block * RF * CF:], RF, CF), .TRANSPOSED
		}
		block = rc * (rc + 1) / 2 + cc
	}
	return small_mat_view_from_slice(pp.data[block * RF * CF:], RF, CF), .FOUND
}

//== contractions

// p = c * b
contract_eval :: proc($CM, $FD: int, p: Pvec($T), c: Cvec(T), b: Bvec($BT)) {
	assert(p.cmpnts == CM && p.fields == FD && c.fields == FD && b.cmpnts == CM)
	assert(b.dofs == c.dofs && p.points == b.points)

	for point in 0 ..< p.points {
		bp, pm := bvec_at_point(b, point), pvec_point_matrix(p, point, CM, FD)
		pm^ = {} // eval overwrites
		for dof in 0 ..< b.dofs {
			bpd := bvec_dof_block(bp, dof)
			small_vec_outer(pm, cvec_dof_vec(c, dof, FD)^, small_vec_from_slice(T, bpd, CM))
		}
	}
}

// c += b * p
contract_linear :: proc($CM, $FD: int, c: Cvec($T), b: Bvec(T), p: Pvec(T)) {
	assert(p.cmpnts == CM && p.fields == FD && c.fields == FD && b.cmpnts == CM)
	assert(b.dofs == c.dofs && p.points == b.points)

	for point in 0 ..< p.points {
		bp, pm := bvec_at_point(b, point), pvec_point_matrix(p, point, CM, FD)
		for dof in 0 ..< b.dofs {
			bv, cv := bvec_dof_vec(bp, dof, CM), cvec_dof_vec(c, dof, FD)
			cv^ = small_vec_add(cv^, small_mat_vec_mul(pm^, bv^))
		}
	}
}

// c += rb * p * cb
contract_bilinear :: proc($RC, $RF, $CC, $CF: int, c: Cmat($T), rb: Bvec(T), p: Pmat(T), cb: Bvec(T)) {
	assert(rb.cmpnts == RC && cb.cmpnts == CC && c.row_fields == RF && c.col_fields == CF)
	assert(p.layout.row_cmpnts == RC && p.layout.row_fields == RF)
	assert(p.layout.col_cmpnts == CC && p.layout.col_fields == CF)
	assert(c.row_dofs == rb.dofs && c.col_dofs == cb.dofs)
	assert(p.points == rb.points && p.points == cb.points)
	when RF != CF { assert(p.layout.shape != .SYMMETRIC, "Symmetric pmat requires equal field counts.") }

	for point in 0 ..< p.points {
		rbp, cbp, pp := bvec_at_point(rb, point), bvec_at_point(cb, point), pmat_at_point(p, point)
		for cdof in 0 ..< cb.dofs {
			cv := bvec_dof_vec(cbp, cdof, CC)
			@(thread_local)
			pc: [RC]Small_Mat(RF, CF, T) // With SIMD small mat maybe not something we want on the stack.

			for rc in 0 ..< RC {
				pc[rc] = {} //bc thread local
				for cc in 0 ..< CC {
					block, status := pmat_cmpnt_matrix(pp, rc, cc, RF, CF)
					switch status {
					case .MISSING: continue
					case .FOUND: small_mat_add_inplace(&pc[rc], block^, cv.data[cc])
					case .TRANSPOSED: when RF == CF {
								small_mat_add_inplace_t(&pc[rc], block^, cv.data[cc])} else {unreachable()
							}
					}
				}
			}
			for rdof in 0 ..< rb.dofs {
				rv := bvec_dof_vec(rbp, rdof, RC)
				dst := cmat_dof_matrix(c, rdof, cdof, RF, CF)
				for rc in 0 ..< RC { small_mat_add_inplace(dst, pc[rc], rv.data[rc]) }
			}
		}
	}
}

// Same as contract bilinear but for cases where `p` is symmetric (.Symmetric or .Diagonal with matching field counts)
// and the left and right basis are the same.
contract_bilinear_same :: proc($CM, $FD: int, c: Cmat($T), b: Bvec(T), p: Pmat(T)) {
	assert(b.cmpnts == CM && c.row_fields == FD && c.col_fields == FD)
	assert(p.layout.row_cmpnts == CM && p.layout.row_fields == FD)
	assert(p.layout.col_cmpnts == CM && p.layout.col_fields == FD)
	assert(c.row_dofs == b.dofs && c.col_dofs == b.dofs)
	assert(p.points == b.points)

	for point in 0 ..< p.points {
		bp, pp := bvec_at_point(b, point), pmat_at_point(p, point)
		for cdof in 0 ..< b.dofs {
			cv := bvec_dof_vec(bp, cdof, CM)
			@(thread_local)
			pc: [CM]Small_Mat(FD, FD, T)

			for rc in 0 ..< CM {
				pc[rc] = {} //bc thread local
				for cc in 0 ..< CM {
					block, status := pmat_cmpnt_matrix(pp, rc, cc, FD, FD)
					switch status {
					case .MISSING: continue
					case .FOUND: small_mat_add_inplace(&pc[rc], block^, cv.data[cc])
					case .TRANSPOSED: small_mat_add_inplace_t(&pc[rc], block^, cv.data[cc])
					}
				}
			}

			for rdof in cdof ..< b.dofs {
				rv := bvec_dof_vec(bp, rdof, CM)

				delta: Small_Mat(FD, FD, T)
				for rc in 0 ..< CM { small_mat_add_inplace(&delta, pc[rc], rv.data[rc]) }

				dst := cmat_dof_matrix(c, rdof, cdof, FD, FD)
				small_mat_add_inplace(dst, delta, T(1))
				if rdof != cdof {
					dst_t := cmat_dof_matrix(c, cdof, rdof, FD, FD)
					small_mat_add_inplace_t(dst_t, delta, T(1))
				}
			}
		}
	}
}
