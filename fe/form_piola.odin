package fe

/*
 Geometric piola transforms, built from the elements ref->physical jacobian.
*/

import "core:slice"

// Per-point tangent map, A x I. Single entry when affine.
Tangent :: struct(A, I: int, T: typeid) {
	maps:     []Small_Mat(A, I, T),
	constant: bool,
}

// Construct the tangent "jacobian" from nodal coefficients, A is ambient dimension, I intrinsic.
tangent_from_nodes :: proc(
	$A, $I: int,
	basis: Bvec($BT),
	nodes: Cvec($T),
	affine: bool,
	alloc := context.allocator,
) -> Tangent(A, I, T) {
	b_eval := basis
	if affine { b_eval = {1, basis.dofs, basis.cmpnts, basis.data[:basis.cmpnts * basis.dofs]} }

	p := pvec_create(T, b_eval.points, I, A, alloc)
	contract_eval(I, A, p, nodes, b_eval)

	return {slice.reinterpret([]Small_Mat(A, I, T), p.data), affine}
}

tangent_at :: proc(t: Tangent($A, $I, $T), point: int) -> ^Small_Mat(A, I, T) {
	return &t.maps[0 if t.constant else point]
}


// Scaling of the tangent.
tangent_measure :: proc(tng: Tangent($A, $I, $T), point: int) -> T {
	return small_mat_measure(tangent_at(tng, point)^)
}


// Covariant transform, for covectors like H1 gradients, H(curl) values.
Piola_Cov :: struct(A, I: int, T: typeid) {
	maps: Tangent(A, I, T),
}

// Contravariant transform, for vector-flux quantities like H(div) values
Piola_Con :: struct(A, I: int, T: typeid) {
	tng: Tangent(A, I, T),
}

// Density transform, used for scalar values when they represent a intensive quantity like H(div) divergence.
Piola_Den :: struct(A, I: int, T: typeid) {
	tng: Tangent(A, I, T),
}

piola_covariant :: proc(tng: Tangent($A, $I, $T), alloc := context.allocator) -> Piola_Cov(A, I, T) {
	maps := make([]Small_Mat(A, I, T), len(tng.maps), alloc)
	for m, i in tng.maps { maps[i] = small_mat_inv_t(m) }
	return {maps = {maps, tng.constant}}
}

piola_contravariant :: proc(tng: Tangent($A, $I, $T)) -> Piola_Con(A, I, T) {
	return {tng}
}

piola_density :: proc(tng: Tangent($A, $I, $T)) -> Piola_Den(A, I, T) {
	return {tng}
}

// Query the piola map at a particular point.
piola_at :: proc {
	piola_at_cov,
	piola_at_con,
	piola_at_density,
}

piola_at_cov :: proc(p: Piola_Cov($A, $I, $T), point: int) -> Small_Mat(A, I, T) {
	return tangent_at(p.maps, point)^
}

piola_at_con :: proc(p: Piola_Con($A, $I, $T), point: int) -> Small_Mat(A, I, T) {
	m := tangent_at(p.tng, point)^
	small_mat_scale_inplace(&m, T(1) / small_mat_measure(m))
	return m
}

piola_at_density :: proc(p: Piola_Den($A, $I, $T), point: int) -> T {
	return T(1) / tangent_measure(p.tng, point)
}

// Apply the cofactor (det J * J^-T) to `v`, e.g. push a reference facet normal to n dA.
piola_apply_cofactor :: proc(covariant: Small_Mat($A, $I, $T), measure: T, v: Small_Vec(I, T)) -> Small_Vec(A, T) {
	return small_vec_scale(small_mat_vec_mul(covariant, v), measure)
}

// Push the basis vector into the new frame, creates a new basis vector as usually the incoming one is reference data.
piola_push_bvec :: proc {
	piola_push_bvec_cov,
	piola_push_bvec_con,
	piola_push_bvec_density,
}

piola_push_bvec_cov :: proc(bvec: Bvec($BT), p: Piola_Cov($A, $I, $T), alloc := context.allocator) -> Bvec(T) {
	return push_bvec_mat(bvec, p, A, I, T, alloc)
}

piola_push_bvec_con :: proc(bvec: Bvec($BT), p: Piola_Con($A, $I, $T), alloc := context.allocator) -> Bvec(T) {
	return push_bvec_mat(bvec, p, A, I, T, alloc)
}

piola_push_bvec_density :: proc(bvec: Bvec($BT), p: Piola_Den($A, $I, $T), alloc := context.allocator) -> Bvec(T) {
	assert(bvec.cmpnts == 1, "density piola expects a scalar basis")

	out := bvec_create(T, bvec.points, bvec.dofs, 1, alloc)
	for point in 0 ..< bvec.points {
		bp := bvec_at_point(bvec, point)
		out_p := bvec_at_point(out, point)
		s := piola_at(p, point)
		for dof in 0 ..< bvec.dofs {
			bv := bvec_dof_vec(bp, dof, 1)
			outv := bvec_dof_vec(out_p, dof, 1)
			outv^ = small_vec_scale(bv^, s)
		}
	}
	return out
}

@(private = "file")
push_bvec_mat :: proc(bvec: Bvec($BT), p: $P, $A, $I: int, $T: typeid, alloc := context.allocator) -> Bvec(T) {
	assert(bvec.cmpnts == I, "basis component count must match the intrinsic dimension")

	out := bvec_create(T, bvec.points, bvec.dofs, A, alloc)
	for point in 0 ..< bvec.points {
		bp := bvec_at_point(bvec, point)
		out_p := bvec_at_point(out, point)
		m := piola_at(p, point)
		for dof in 0 ..< bvec.dofs {
			bv := bvec_dof_vec(bp, dof, I)
			outv := bvec_dof_vec(out_p, dof, A)
			outv^ = small_mat_vec_mul(m, bv^)
		}
	}
	return out
}

// Evaluation contraction with piola map applied inline.
piola_contract_eval :: proc {
	piola_contract_cov,
	piola_contract_con,
	piola_contract_den,
}

piola_contract_cov :: proc(piola: Piola_Cov($A, $I, $T), $FD: int, p: Pvec(T), c: Cvec(T), b: Bvec($BT)) {
	contract_eval_mat(piola, A, I, FD, p, c, b)
}

piola_contract_con :: proc(piola: Piola_Con($A, $I, $T), $FD: int, p: Pvec(T), c: Cvec(T), b: Bvec($BT)) {
	contract_eval_mat(piola, A, I, FD, p, c, b)
}

piola_contract_den :: proc(piola: Piola_Den($A, $I, $T), $FD: int, p: Pvec(T), c: Cvec(T), b: Bvec($BT)) {
	contract_eval(1, FD, p, c, b)
	for point in 0 ..< p.points {
		small_mat_scale_inplace(pvec_point_matrix(p, point, 1, FD), piola_at(piola, point))
	}
}

@(private = "file")
contract_eval_mat :: proc(piola: $P, $A, $I, $FD: int, p: Pvec($T), c: Cvec(T), b: Bvec($BT)) {
	assert(p.cmpnts == A && p.fields == FD && c.fields == FD && b.cmpnts == I)
	assert(b.dofs == c.dofs && p.points == b.points)

	for point in 0 ..< p.points {
		bp, pm := bvec_at_point(b, point), pvec_point_matrix(p, point, A, FD)
		m := piola_at(piola, point)
		pm^ = {} // eval overwrites
		for dof in 0 ..< b.dofs {
			bv := small_vec_from_slice(T, bvec_dof_block(bp, dof), I)
			small_vec_outer(pm, cvec_dof_vec(c, dof, FD)^, small_mat_vec_mul(m, bv))
		}
	}
}
