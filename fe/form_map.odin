package fe

/*
 Geometry mapping between frames.
 The map stores its tangent component (jacobian), cotangent component and measure.
 If the map is constant, only one is computed & stored.

 Works for reference -> physical mapping, or mapping between two physical frames
*/

import "core:slice"

Element_Map :: struct($A, $I: int, $T: typeid) {
	tng:      []Small_Mat(A, I, T), // J, columns are the tangent frame
	ctng:     []Small_Mat(A, I, T), // J G^-1 (J^-T when square), columns are the dual (cotangent) frame
	measure:  []T, // sqrt(det G) (|det J| when square)
	constant: bool, // only 1 entry is stored
}

// From the geometry's reference gradient table (`grad`) and its nodes. A is the ambient dimension, I the element's.
element_map :: proc($A, $I: int, grad: Bvec($BT), nodes: Cvec($T), affine: bool, alloc := context.allocator) -> (m: Element_Map(A, I, T)) {
	g := grad
	if affine { g = {1, grad.dofs, grad.cmpnts, grad.data[:grad.cmpnts * grad.dofs]} }

	p := pvec_create(T, g.points, I, A, alloc)
	contract_eval(I, A, p, nodes, g)

	m.tng = slice.reinterpret([]Small_Mat(A, I, T), p.data)
	m.ctng = make([]Small_Mat(A, I, T), len(m.tng), alloc)
	m.measure = make([]T, len(m.tng), alloc)
	for t, i in m.tng {
		m.ctng[i] = small_mat_inv_t(t)
		m.measure[i] = small_mat_measure(t)
	}
	m.constant = affine
	return
}

// Jacobian at a point
element_map_tng :: proc(m: Element_Map($A, $I, $T), point: int) -> Small_Mat(A, I, T) {
	return m.tng[0 if m.constant else point]
}

// Jacobian inverse, or pseudo inverse, transpose at a point
element_map_ctng :: proc(m: Element_Map($A, $I, $T), point: int) -> Small_Mat(A, I, T) {
	return m.ctng[0 if m.constant else point]
}

// Measure of the mapping, determinant or equivalent quantity of the jacobian.
element_map_measure :: proc(m: Element_Map($A, $I, $T), point: int) -> T {
	return m.measure[0 if m.constant else point]
}

// Unit outward normal at a point of the facet with reference normal `n_ref` (element_facet_normal), and the facet's
// measure there per unit of its own reference. From Nanson: n dA = measure * ctng * n_ref.
element_map_facet_normal :: proc(m: Element_Map($A, $I, $T), n_ref: Small_Vec(I, T), point: int) -> (n: Small_Vec(A, T), measure: T) {
	nda := small_vec_scale(small_mat_vec_mul(element_map_ctng(m, point), n_ref), element_map_measure(m, point))
	measure = small_vec_norm(nda)
	return small_vec_scale(nda, 1 / measure), measure
}

// Quadrature weights times the map's measure, the dx of a cell site's rule. Plain per point weights, for
// pmat_create_identity or scaling point data.
element_map_weights :: proc(m: Element_Map($A, $I, f64), w: []f64, alloc := context.allocator) -> []f64 {
	dx := make([]f64, len(w), alloc)
	for &d, p in dx { d = element_map_measure(m, p) * w[p] }
	return dx
}

// Unit outward normals and quadrature weights times the facet measure, the da of local facet `lf`'s rule, with `m` the
// cell's map at the facet points.
element_map_facet_weights :: proc(
	m: Element_Map($A, $I, f64),
	et: Element_Type,
	lf: int,
	w: []f64,
	alloc := context.allocator,
) -> (
	normals: []Small_Vec(A, f64),
	da: []f64,
) {
	normals, da = make([]Small_Vec(A, f64), len(w), alloc), make([]f64, len(w), alloc)
	n_ref := element_facet_normal(I, et, lf)
	for p in 0 ..< len(w) {
		measure: f64
		normals[p], measure = element_map_facet_normal(m, n_ref, p)
		da[p] = measure * w[p]
	}
	return
}
