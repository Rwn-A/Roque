package fe

/*
 Coordinate frame transformations for contraction data.

 2 purposes:
 	- Per-point transforms for basis quantities & geometry.
  - Per-dof transforms for change of frame on the coefficients attached to each dof.
*/

import "core:slice"

Frame :: struct(R, C: int, T: typeid) {
	transforms: []Small_Mat(R, C, T),
	affine:     bool,
}

frame_from_transform :: proc(transform: ^Small_Mat($R, $C, $T)) -> Frame(R, C, T) {
	return {slice.from_ptr(transform, 1), true}
}

frame_at :: proc(fm: Frame($R, $C, $T), idx: int) -> ^Small_Mat(R, C, T) {
	return &fm.transforms[0 if fm.affine else idx]
}


frame_tangent_from_nodes :: proc(
	$dims: Contraction_Dims,
	basis: Bvec($BT),
	nodes: Cvec($T),
	affine: bool,
	alloc := context.allocator,
) -> Frame(dims[.FIELDS], dims[.CMPNTS], T) {
	b_eval := basis

	if affine { b_eval = {1, basis.dofs, basis.cmpnts, basis.data[:basis.cmpnts * basis.dofs]} }

	p := pvec_create(T, b_eval.points, dims, alloc)
	contract_eval(dims, p, nodes, b_eval)

	return {slice.reinterpret([]Small_Mat(dims[.FIELDS], dims[.CMPNTS], T), p.data), affine}
}


frame_cotangent :: proc(tangent: Frame($R, $C, $T), alloc := context.allocator) -> Frame(R, C, T) {
	out := Frame(R, C, T){ affine = tangent.affine, transforms = make([]Small_Mat(R, C, T), len(tangent.transforms), alloc) }

	for m, i in tangent.transforms { out.transforms[i] = small_mat_inv_t(m) }

	return out
}

frame_push_bvec :: proc(bvec: Bvec($BT), frame: Frame($R, $C, $T), alloc := context.allocator) -> Bvec(T){
	out := bvec_create(T, bvec.points, bvec.dofs, R, alloc)
	for point in 0..<bvec.points{
		bp := bvec_at_point(bvec, point)
		out_p := bvec_at_point(out, point)
		t := frame_at(frame, point)
		for dof in 0..<bvec.dofs{
			bv := bvec_dof_vec(bp, dof, C)
			outv := bvec_dof_vec(out_p, dof, R)
			outv^ = small_mat_vec_mul(t^, bv^)
		}
	}
	return out
}
