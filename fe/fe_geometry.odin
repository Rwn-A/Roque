package fe

/*
 Wrapper of frame code for the specific case of a geometric jacobian frame.
*/

import "core:slice"

// Construct the tangent "jacobian" from nodal coefficients, A is ambient dimesnion, I intrinsic.
geom_tng_from_nodes :: proc(
	$A, $I: int,
	basis: Bvec($BT),
	nodes: Cvec($T),
	affine: bool,
	alloc := context.allocator,
) -> Frame(A, I, T) {
	b_eval := basis
	dims :: Contraction_Dims {
		.CMPNTS = I,
		.FIELDS = A,
	}

	if affine { b_eval = {1, basis.dofs, basis.cmpnts, basis.data[:basis.cmpnts * basis.dofs]} }

	p := pvec_create(T, b_eval.points, dims, alloc)
	contract_eval(dims, p, nodes, b_eval)

	return {slice.reinterpret([]Small_Mat(A, I, T), p.data), affine}
}

// Cotangent (jacobian inverse or pseudo inverse transpose) from tng frame.
geom_ctng :: proc(tng: Frame($R, $C, $T), alloc := context.allocator) -> Frame(R, C, T) {
	out := Frame(R, C, T) {
		constant   = tng.constant,
		transforms = make([]Small_Mat(R, C, T), len(tng.transforms), alloc),
	}

	for m, i in tng.transforms { out.transforms[i] = small_mat_inv_t(m) }

	return out
}

// Frame scaling
geom_measure :: proc(tng: Frame($R, $C, $T), point: int) -> T {
	return small_mat_measure(frame_at(tng, point)^)
}

// Outward facet normal from the reference normal. Requires both the cotangent, and tangent mappings.
geom_facet_normal :: proc(tng, ctng: Small_Mat($R, $C, $T), ref_normal: Small_Vec(C, T)) -> Small_Vec(R, T) {
	return small_vec_scale(small_mat_vec_mul(ctng, ref_normal), small_mat_measure(tng))
}

// Normal pointing out of the local tangent plane of the surface, normalized, not oriented.
geom_surface_normal :: proc(tng: Small_Mat(3, 2, $T)) -> Vec(3, T) {
	return vec_normalize(vec3_cross(mat_col(tng, 0), mat_col(tng, 1)))
}
