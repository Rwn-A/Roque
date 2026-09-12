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

CONTRACTION_AXIS :: enum {
	CMPNTS,
	FIELDS,
}

Contraction_Dims :: [CONTRACTION_AXIS]int

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

bvec_dof_block :: proc(bvp: Bvec_Point($T), dof: int) -> []T {
	return bvp.data[dof * bvp.cmpnts:][:bvp.cmpnts]
}

bvec_dof_vec :: proc(bvp: Bvec_Point($T), dof: int, $C: int) -> ^Small_Vec(C, T) {
	assert(C == bvp.cmpnts)
	return small_vec_view_from_slice(bvec_dof_block(bvp, dof), C)
}


Pvec :: struct(T: typeid) {
	points, cmpnts, fields: int,
	data:                   []T,
}

Pvec_Point :: struct(T: typeid) {
	cmpnts, fields: int,
	data:           []T,
}


pvec_create :: proc($T: typeid, np: int, dims: Contraction_Dims, alloc := context.allocator) -> Pvec(T) {
	return {np, dims[.CMPNTS], dims[.FIELDS], make([]T, np * dims[.FIELDS] * dims[.CMPNTS], alloc)}
}

pvec_at_point :: proc(pvec: Pvec($T), point: int) -> Pvec_Point(T) {
	point_size := pvec.cmpnts * pvec.fields
	return {pvec.cmpnts, pvec.fields, pvec.data[point * point_size:][:point_size]}
}

pvec_point_matrix :: proc(pvec: Pvec($T), point: int, $C, $F: int) -> ^Small_Mat(F, C, T) {
	return small_mat_view_from_slice(pvec.data[point * pvec.cmpnts * pvec.fields:], F, C)
}

Cvec :: struct(T: typeid) {
	dofs, fields: int,
	data:         []T,
}

cvec_create :: proc($T: typeid, dofs, fields: int, alloc := context.allocator) -> Cvec(T) {
	return {dofs = dofs, fields = fields, data = make([]T, dofs * fields, alloc)}
}

cvec_dof_block :: proc(cvec: Cvec($T), dof: int) -> []T {
	return cvec.data[dof * cvec.fields:][:cvec.fields]
}

cvec_dof_vec :: proc(cvec: Cvec($T), dof: int, $F: int) -> ^Small_Vec(F, T) {
	assert(F == cvec.fields)
	return small_vec_view_from_slice(cvec_dof_block(cvec, dof), F)
}

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

// Returns a view of the (row_fields x col_fields) block coupling `rdof` and `cdof`.
cmat_dof_matrix :: proc(cmat: Cmat($T), rdof, cdof: int, $RF, $CF: int) -> ^Small_Mat(RF, CF, T) {
	block_size := RF * CF
	offset := (rdof * cmat.col_dofs + cdof) * block_size
	return small_mat_view_from_slice(cmat.data[offset:], RF, CF)
}

cmat_dof_block :: proc(cmat: Cmat($T), rdof, cdof: int) -> []T {
	block_size := cmat.col_fields * cmat.row_fields
	offset := (rdof * cmat.col_dofs + cdof) * block_size
	return cmat.data[offset:]
}

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

// Shape describes the block-shape of the pmat, each field block is dense.
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
	rd, cd: Contraction_Dims,
	alloc := context.allocator,
) -> Pmat(T) {
	layout := Pmat_Layout{rd[.CMPNTS], rd[.FIELDS], cd[.CMPNTS], cd[.FIELDS], s}

	assert(s == .DENSE || layout.row_cmpnts == layout.col_cmpnts, "Non-dense shape requires equal component counts.")

	num_blocks: int
	switch s {
	case .DENSE: num_blocks = layout.row_cmpnts * layout.col_cmpnts
	case .DIAGONAL: num_blocks = layout.row_cmpnts
	case .SYMMETRIC:
		assert(rd[.FIELDS] == cd[.FIELDS], "Symmetry requires equal field counts.")
		num_blocks = layout.row_cmpnts * (layout.row_cmpnts + 1) / 2
	}
	data := make([]T, np * num_blocks * layout.row_fields * layout.col_fields, alloc)
	return {points = np, layout = layout, field_blocks_per_point = num_blocks, data = data}
}

pmat_at_point :: proc(pmat: Pmat($T), point: int) -> Pmat_Point(T) {
	point_size := pmat.field_blocks_per_point * pmat.layout.row_fields * pmat.layout.col_fields
	return {pmat.layout, pmat.data[point * point_size:][:point_size]}
}

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
contract_eval :: proc($dims: Contraction_Dims, p: Pvec($T), c: Cvec(T), b: Bvec($BT)) {
	CM, FD :: dims[.CMPNTS], dims[.FIELDS]
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
contract_linear :: proc($dims: Contraction_Dims, c: Cvec($T), b: Bvec(T), p: Pvec(T)) {
	CM, FD :: dims[.CMPNTS], dims[.FIELDS]
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
contract_bilinear :: proc($rd, $cd: Contraction_Dims, c: Cmat($T), rb: Bvec(T), p: Pmat(T), cb: Bvec(T)) {
	RC, RF :: rd[.CMPNTS], rd[.FIELDS]
	CC, CF :: cd[.CMPNTS], cd[.FIELDS]
	assert(rb.cmpnts == RC && cb.cmpnts == CC && c.row_fields == RF && c.col_fields == CF)
	assert(c.row_dofs == rb.dofs && c.col_dofs == cb.dofs)
	assert(p.points == rb.points && p.points == cb.points)

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
					case .TRANSPOSED: small_mat_add_inplace_t(&pc[rc], block^, cv.data[cc])
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
