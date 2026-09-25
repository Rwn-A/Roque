/*

*/
package fe

import "core:slice"

//== Types

Element_Type :: enum {
	Point,
	Line,
	Tri,
	Quad,
	Hex,
	Tet,
}

Dimension :: enum {
	D0,
	D1,
	D2,
	D3,
}

Ref_Vec :: [3]f64

MAX_FACES       :: 6
MAX_FACETS      :: 6
MAX_EDGES       :: 12
MAX_VERTICES    :: 8

// Qn is exact for degree n (per direction on quads/hexes, total on simplices).
Quadrature_Set :: enum {
	Q1,
	Q3,
	Q5,
	Q7,
}

@(rodata)
QUAD_SET_DEGREE := [Quadrature_Set]int {
	.Q1 = 1,
	.Q3 = 3,
	.Q5 = 5,
	.Q7 = 7,
}

Order :: enum {
	O0,
	O1,
	O2,
	O3,
}

Basis_Family :: enum {
	Lagrange,
	Raviart_Thomas,
	Nedelec,
}

Map_Type :: enum {
	Unity,
	Covariant,
	Contravariant,
	Density,
}

// S_ / V_: scalar or vector valued basis.
Basis_Quantity :: enum {
	S_Val,
	V_Val,
	S_Grd,
	V_Div,
	V_Curl,
}

@(rodata)
BASIS_QUANTITIES := [Basis_Family]bit_set[Basis_Quantity] {
	.Lagrange       = {.S_Val, .S_Grd},
	.Raviart_Thomas = {.V_Val, .V_Div},
	.Nedelec        = {.V_Val, .V_Curl},
}

// Components per dof, by element dimension.
@(rodata, private = "file")
QUANTITY_CMPNTS := [Basis_Quantity][Dimension]int {
	.S_Val = {.D0 = 1, .D1 = 1, .D2 = 1, .D3 = 1},
	.V_Val = {.D0 = 0, .D1 = 1, .D2 = 2, .D3 = 3},
	.S_Grd = {.D0 = 0, .D1 = 1, .D2 = 2, .D3 = 3},
	.V_Div = {.D0 = 1, .D1 = 1, .D2 = 1, .D3 = 1},
	.V_Curl = {.D0 = 0, .D1 = 0, .D2 = 1, .D3 = 3},
}

Sub_Entity :: struct {
	type:    Element_Type,
	closure: [Dimension][]int, // Sub-entities of each dimension, as indices into the parent's sub_entities.
}

Reference_Topology :: struct {
	dim:               Dimension,
	vertices:          []Ref_Vec,
	sub_entities:      [Dimension][]Sub_Entity, // [dim][index] the cell itself is sub_entities[dim][0]
	facet_normals:     []Ref_Vec, // Outward, length = facet size in the parent / facet size in its own reference
	orientation_perms: [][]int, // [key][local vertex] -> position in canonical order. Line, Tri, Quad only.
}

Reference_Quadrature :: struct {
	points:  []Ref_Vec,
	weights: []f64,
}

// Fills every dof of one quantity at one point
Basis_Eval :: #type proc(r: Ref_Vec, out: Bvec_Point(f64))

Orientation_Kind :: enum u8 {
	Identity, // nothing to apply
	Signed_Perm, // (M v)[i] = sign[i] * v[src[i]], and M^-T == M
	Dense, // m and m_inv
}

// M, where the cell's basis on the entity is phi_cell = M phi_ref.
// Scatter uses M, gather M^T, interpolation M^-T. Only the fields for `kind` are set.
Orientation_Transform :: struct {
	kind:  Orientation_Kind,
	src:   []int,
	sign:  []f64,
	m:     Dense_Matrix,
	m_inv: Dense_Matrix,
}

Rule :: struct {
	element:    Element_Type, // in reference space of this element.
	points: []Ref_Vec,
}

// Functionals for all dofs on one entity, sharing one rule:
//   l_j(v) = sum_p rule.weights[p] * sum_c weights[j][p][c] * v_c(x_p)
//   x_p    = element_lift_to_parent_reference(et, dim, index, rule.ref_points[p])
// Only the rule is in the entity's space. weights are in the parent's reference space.
Entity_Functionals :: struct {
	rule:    Rule,
	weights: Bvec(f64),
}

Reference_Basis :: struct {
	n_dofs:          int,
	dofs_per_entity: [Dimension]int,
	evals:           [Basis_Quantity]Basis_Eval, // nil if the family lacks the quantity
	orientations:    [Element_Type][]Orientation_Transform, // [entity type][key]; nil = never transformed
	functionals:     [Dimension][]Entity_Functionals, // [dim][entity]
}

Reference_Element :: struct {
	topology:   Reference_Topology,
	quadrature: [Quadrature_Set]Reference_Quadrature,
	bases:      [Basis_Family][Order]Reference_Basis,
}

//== Element

Reference_Facet :: struct {
	using entity: Sub_Entity,
	dim:          Dimension,
	normal:       []f64, // len = element dimension
}

element_dim :: proc(et: Element_Type) -> Dimension {
	return REFERENCE_ELEMENTS[et].topology.dim
}

element_num_sub_entities :: proc(et: Element_Type, dim: Dimension) -> int {
	return len(REFERENCE_ELEMENTS[et].topology.sub_entities[dim])
}

element_num_facets :: proc(et: Element_Type) -> int {
	return len(REFERENCE_ELEMENTS[et].topology.facet_normals)
}

// dim == element_dim(et), index 0 is the cell itself.
element_sub_entity :: proc(et: Element_Type, dim: Dimension, index: int) -> Sub_Entity {
	assert(dim <= element_dim(et))
	return REFERENCE_ELEMENTS[et].topology.sub_entities[dim][index]
}

element_facet :: proc(et: Element_Type, facet: int) -> Reference_Facet {
	assert(et != .Point)
	fd := element_dim(et) - Dimension(1)
	t := &REFERENCE_ELEMENTS[et].topology
	return {entity = t.sub_entities[fd][facet], dim = fd, normal = t.facet_normals[facet][:int(element_dim(et))]}
}

element_vertex_coord :: proc(et: Element_Type, vert: int) -> Ref_Vec {
	return REFERENCE_ELEMENTS[et].topology.vertices[vert]
}

// Maps a point in a sub-entity's own reference space into the element's reference space.
element_lift_to_parent_reference :: proc(et: Element_Type, dim: Dimension, index: int, point: Ref_Vec) -> Ref_Vec {
	assert(dim <= element_dim(et))
	if element_dim(et) == dim { return point }

	sub := element_sub_entity(et, dim, index)
	buf: [MAX_VERTICES]f64
	phi := p1_values(sub.type, point, buf[:])

	r: Ref_Vec
	for v, k in sub.closure[.D0] { r += phi[k] * element_vertex_coord(et, v) }
	return r
}

// Orientation key of an entity. local_order: its vertices as this cell sees them.
// target_order: the same vertices in the entity's canonical (mesh-wide) order.
element_orientation :: proc(et: Element_Type, local_order: []$T, target_order: []T) -> u8 {
	n := len(local_order)
	assert(n == len(target_order) && n <= MAX_VERTICES)

	perm: [MAX_VERTICES]int
	for v, i in local_order {
		idx, found := slice.linear_search(target_order, v)
		assert(found, "local_order and target_order are not the same vertex set")
		perm[i] = idx
	}

	table := REFERENCE_ELEMENTS[et].topology.orientation_perms
	assert(table != nil, "element type has no orientations")
	for row, i in table {
		if slice.equal(row, perm[:n]) { return u8(i) }
	}
	panic("vertex order is not a symmetry of this element type")
}

// Maps a point in the entity's reference space to where it sits under orientation key `k`.
element_orient_point :: proc(et: Element_Type, k: u8, p: Ref_Vec) -> (r: Ref_Vec) {
	if k == 0 { return p }
	perm := REFERENCE_ELEMENTS[et].topology.orientation_perms[k]
	buf: [MAX_VERTICES]f64
	phi := p1_values(et, p, buf[:])
	for i in 0 ..< len(perm) { r += phi[perm[i]] * element_vertex_coord(et, i) }
	return r
}

element_quadrature_rule :: proc(et: Element_Type, set: Quadrature_Set) -> (Rule, []f64) {
	q := REFERENCE_ELEMENTS[et].quadrature[set]
	assert(len(q.points) > 0 && len(q.points) == len(q.weights), "quadrature set is not tabulated")
	return {element = et, points = q.points}, q.weights
}

// Vertex (P1) shape function values at r, written into buf.
@(private = "file")
p1_values :: proc(et: Element_Type, r: Ref_Vec, buf: []f64) -> []f64 {
	n := len(REFERENCE_ELEMENTS[et].topology.vertices)
	assert(n <= len(buf))
	out := Bvec_Point(f64) {
		dofs   = n,
		cmpnts = 1,
		data   = buf[:n],
	}
	REFERENCE_ELEMENTS[et].bases[.Lagrange][.O1].evals[.S_Val](r, out)
	return buf[:n]
}

//== Basis

Basis_Desc :: struct {
	element: Element_Type,
	family:  Basis_Family,
	order:   Order,
}

basis_is_defined :: proc(bd: Basis_Desc) -> bool {
	return basis_ref(bd).n_dofs > 0
}

basis_num_dofs :: proc(bd: Basis_Desc) -> int {
	return basis_ref(bd).n_dofs
}

basis_dofs_per_entity :: proc(bd: Basis_Desc, dim: Dimension) -> int {
	return basis_ref(bd).dofs_per_entity[dim]
}

// Dofs owned by one sub-entity: local dofs [start, start + count).
basis_entity_dof_range :: proc(bd: Basis_Desc, dim: Dimension, index: int) -> (start, count: int) {
	assert(dim <= element_dim(bd.element) && basis_is_defined(bd))
	b := basis_ref(bd)
	for d in 0 ..< int(dim) {
		start += element_num_sub_entities(bd.element, Dimension(d)) * b.dofs_per_entity[Dimension(d)]
	}
	count = b.dofs_per_entity[dim]
	start += index * count
	return
}

// Dofs of a sub-entity and everything on its boundary.
basis_closure_dofs :: proc(bd: Basis_Desc, entity: Sub_Entity, alloc := context.allocator) -> []int {
	assert(basis_is_defined(bd))
	out := make([dynamic]int, alloc)
	for dim in Dimension {
		for e in entity.closure[dim] {
			start, count := basis_entity_dof_range(bd, dim, e)
			for d in start ..< start + count { append(&out, d) }
		}
	}
	return out[:]
}

// Dofs with a non-zero trace on the facet.
basis_facet_dofs :: proc(bd: Basis_Desc, facet: int, alloc := context.allocator) -> []int {
	return basis_closure_dofs(bd, element_facet(bd.element, facet), alloc)
}


// Components per dof: the cmpnts to tabulate into.
basis_quantity_cmpnts :: proc(bd: Basis_Desc, q: Basis_Quantity) -> int {
	assert(q in BASIS_QUANTITIES[bd.family])
	return QUANTITY_CMPNTS[q][element_dim(bd.element)]
}

basis_eval :: proc(bd: Basis_Desc, q: Basis_Quantity, r: Ref_Vec, out: Bvec_Point(f64)) {
	eval := basis_ref(bd).evals[q]
	assert(eval != nil, "quantity not tabulated for this basis")
	assert(out.dofs == basis_num_dofs(bd) && out.cmpnts == basis_quantity_cmpnts(bd, q))
	eval(r, out)
}

basis_entity_functionals :: proc(bd: Basis_Desc, dim: Dimension, index: int) -> Entity_Functionals {
	assert(dim <= element_dim(bd.element) && basis_is_defined(bd))
	return basis_ref(bd).functionals[dim][index]
}

// Transform for an entity's dof block at orientation `key`.
basis_orientation :: proc(bd: Basis_Desc, entity_type: Element_Type, key: u8) -> Orientation_Transform {
	table := basis_ref(bd).orientations[entity_type]
	if key == 0 || table == nil { return {} }
	return table[key]
}

basis_quantity_map :: proc(bd: Basis_Desc, q: Basis_Quantity) -> Map_Type {
	assert(q in BASIS_QUANTITIES[bd.family])
	switch q {
	case .S_Val: return .Unity
	case .S_Grd: return .Covariant
	case .V_Div: return .Density
	case .V_Val: return bd.family == .Raviart_Thomas ? .Contravariant : .Covariant
	case .V_Curl: return element_dim(bd.element) == .D3 ? .Contravariant : .Density // 2D curl is a scalar
	}
	unreachable()
}

// Highest polynomial degree in the space (per direction on quads/hexes).
@(rodata, private = "file")
FAMILY_DEGREE_OFFSET := [Basis_Family]int {
	.Lagrange       = 0,
	.Raviart_Thomas = 1,
	.Nedelec        = 1,
}

// Smallest quadrature set that integrates a product of two functions of the basis exactly (a mass matrix),
// plus `extra` degrees for coefficients, geometry or the other operand.
basis_quad_rule :: proc(bd: Basis_Desc, extra := 0) -> (Rule, []f64) {
	need := 2 * (int(bd.order) + FAMILY_DEGREE_OFFSET[bd.family]) + extra
	for set in Quadrature_Set {
		if QUAD_SET_DEGREE[set] >= need { return element_quadrature_rule(bd.element, set) }
	}
	panic("no quadrature set is exact enough; raise extra's source or add a higher set")
}

basis_facet_quad_rule :: proc(bd: Basis_Desc, facet: int, extra := 0) -> (Rule, []f64) {
	need := 2 * (int(bd.order) + FAMILY_DEGREE_OFFSET[bd.family]) + extra
	facet := element_facet(bd.element, facet)
	for set in Quadrature_Set {
		if QUAD_SET_DEGREE[set] >= need { return element_quadrature_rule(facet.type, set) }
	}
	panic("no quadrature set is exact enough; raise extra's source or add a higher set")
}

@(private = "file")
basis_ref :: proc(bd: Basis_Desc) -> ^Reference_Basis {
	return &REFERENCE_ELEMENTS[bd.element].bases[bd.family][bd.order]
}
