package fe

/*
 Coordinate frame transformations for contraction data.

 2 purposes:
 	- Per-point transforms for physical quantities
  - Per-dof transforms for change of frame on the field coefficients attached to each dof.
*/

import "core:slice"

Frame :: struct(R, C: int, T: typeid) {
	transforms: []Small_Mat(R, C, T),
	constant:   bool,
}

// Create a constant frame from a single transform
frame_from_transform :: proc(transform: ^Small_Mat($R, $C, $T)) -> Frame(R, C, T) {
	return {slice.from_ptr(transform, 1), true}
}

// Retrieve the transform at the index (point or dof)
frame_at :: proc(fm: Frame($R, $C, $T), idx: int) -> ^Small_Mat(R, C, T) {
	return &fm.transforms[0 if fm.constant else idx]
}

// Push the basis vector into the new frame, creates a new basis vector as usually the incoming one is reference data.
frame_push_bvec :: proc(bvec: Bvec($BT), frame: Frame($R, $C, $T), alloc := context.allocator) -> Bvec(T) {
	out := bvec_create(T, bvec.points, bvec.dofs, R, alloc)
	for point in 0 ..< bvec.points {
		bp := bvec_at_point(bvec, point)
		out_p := bvec_at_point(out, point)
		t := frame_at(frame, point)
		for dof in 0 ..< bvec.dofs {
			bv := bvec_dof_vec(bp, dof, C)
			outv := bvec_dof_vec(out_p, dof, R)
			outv^ = small_mat_vec_mul(t^, bv^)
		}
	}
	return out
}

// bvcec /= measurem, inplace transform
frame_scale_bvec_by_measure :: proc(bvec: Bvec($BT), frame: Frame($R, $C, $T)) {
	for point in 0 ..< bvec.points {
		bp := bvec_at_point(bvec, point)
		t := frame_at(frame, point)
		m := 1 / small_mat_measure(t^)
		for dof in 0 ..< bvec.dofs {
			bv := bvec_dof_vec(bp, dof, C)
			bv^ = small_vec_scale(bv^, m)
		}
	}
}



// TODO: might need cvec, cmat, pvec, pmat pushing as well.
