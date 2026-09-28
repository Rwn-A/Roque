package fe

/*
 Reference element topology, basis and quadrature rules.
 Tables are generated in a separate metaprogram.

 Access functions are present for commonly queried data.
*/

import "core:slice"

// Supported element shapes
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

// Polynomial order of a basis within its family.
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

// S_ / V_: scalar or vector valued.
Basis_Quantity :: enum {
	S_Val,
	V_Val,
	S_Grd,
	V_Div,
	V_Curl,
}

Quantity_Set :: bit_set[Basis_Quantity]

// How a quantity maps from the reference to the physical element.
Map_Type :: enum {
	Unity,
	Covariant,
	Contravariant,
	Density,
}

// Qn is exact for degree n (per direction on quads and hexes, total on simplices).
Quadrature_Set :: enum {
	Q1,
	Q3,
	Q5,
	Q7,
}

Orientation_Kind :: enum u8 {
	Identity, // nothing to apply
	Signed_Perm, // (M v)[i] = sign[i] * v[src[i]], and M^-T == M
	Dense, // m and m_inv
}

// Everything tabulated for one element type.
Reference_Element :: struct {
	topology:   Reference_Topology,
	quadrature: [Quadrature_Set]Reference_Quadrature,
	bases:      [Basis_Family][Order]Reference_Basis,
}

Reference_Topology :: struct {
	dim:               Dimension,
	vertices:          []Ref_Vec,
	sub_entities:      [Dimension][]Sub_Entity, // [dim][index], the element itself is sub_entities[dim][0]
	facet_normals:     []Ref_Vec, // outward, length = facet size in the parent / facet size in its own reference
	orientation_perms: [][]int, // [key][local vertex] -> position in canonical order. Line, Tri, Quad only
	orientation_maps:  []Ref_Map, // [key] reference point -> where it sits under the key
}

// A vertex, edge, face or the element itself, as seen from its parent element.
Sub_Entity :: struct {
	type:    Element_Type,
	closure: [Dimension][]int, // indices into the parent's sub-entities, in this entity's reference order
	lift:    Ref_Map, // this entity's reference -> the parent's reference
}

// Map between reference spaces, used for mapping between sub entities.
Ref_Map :: struct {
	origin: Ref_Vec,
	jac:    [3]Ref_Vec,
}

// A point in reference coordinates, unused trailing components are 0.
Ref_Vec :: [3]f64

Reference_Quadrature :: struct {
	points:  []Ref_Vec,
	weights: []f64,
}

// Dof layout, evaluation, orientation and interpolation data of one basis.
Reference_Basis :: struct {
	n_dofs:           int,
	dofs_per_entity:  [Dimension]int,
	entity_dof_start: [Dimension]int, // first dof of each dimension's block
	closure_dofs:     [Dimension][][]int, // [dim][entity] -> dofs on the entity's closure
	evals:            [Basis_Quantity]Basis_Eval, // nil if the family lacks the quantity
	orientations:     [Element_Type][]Orientation_Transform, // [entity type][key], nil = never transformed
	functionals:      [Dimension][]Entity_Functionals, // [dim][entity]
}

// Fills every dof of one quantity at one point.
Basis_Eval :: #type proc(r: Ref_Vec, out: Bvec_Point(f64))

// M, where the element's basis on the entity is phi_elem = M phi_ref.
// Scatter uses M, gather M^T, interpolation M^-T. Only the fields for `kind` are set.
Orientation_Transform :: struct {
	kind:  Orientation_Kind,
	src:   []int,
	sign:  []f64,
	m:     Dense_Matrix,
	m_inv: Dense_Matrix,
}

// Functionals for all dofs on one entity: l_j(v) = sum_p sum_c weights[p][j][c] * v_c(x_p), x_p the rule points
// lifted into the element. Quadrature weights are folded into `weights`, which is in the element's reference.
Entity_Functionals :: struct {
	rule:    Rule,
	weights: Bvec(f64),
}

// Points in the reference of `element`.
Rule :: struct {
	element: Element_Type,
	points:  []Ref_Vec,
}

MAX_VERTICES :: 8
MAX_EDGES    :: 12
MAX_FACES    :: 6
MAX_FACETS   :: 6

@(rodata)
BASIS_QUANTITIES := [Basis_Family]Quantity_Set {
	.Lagrange       = {.S_Val, .S_Grd},
	.Raviart_Thomas = {.V_Val, .V_Div},
	.Nedelec        = {.V_Val, .V_Curl},
}

@(rodata, private = "file")
QUAD_SET_DEGREE := [Quadrature_Set]int {
	.Q1 = 1,
	.Q3 = 3,
	.Q5 = 5,
	.Q7 = 7,
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

// Highest polynomial degree of the family at order O0 (per direction on quads and hexes).
@(rodata, private = "file")
FAMILY_DEGREE_OFFSET := [Basis_Family]int {
	.Lagrange       = 0,
	.Raviart_Thomas = 1,
	.Nedelec        = 1,
}

// Topology queries

// Dimension of the given element
element_dim :: proc(et: Element_Type) -> Dimension {
	return REFERENCE_ELEMENTS[et].topology.dim
}

// Dimension of the elements facet (element dim - 1 sub entity).
element_facet_dim :: proc(et: Element_Type) -> Dimension {
	assert(et != .Point)
	return element_dim(et) - Dimension(1)
}

// Number of sub entities of dimension `dim`
element_n_sub_entities :: proc(et: Element_Type, dim: Dimension) -> int {
	return len(REFERENCE_ELEMENTS[et].topology.sub_entities[dim])
}

// Sub-entity (dim, index), the element itself is (element_dim, 0).
element_sub_entity :: proc(et: Element_Type, dim: Dimension, index: int) -> Sub_Entity {
	return REFERENCE_ELEMENTS[et].topology.sub_entities[dim][index]
}

// Outward reference normal of a facet, scaled by the facet's size in the element over its size in its own reference.
element_facet_normal :: proc($I: int, et: Element_Type, facet: int) -> Small_Vec(I, f64) {
	assert(I == int(element_dim(et)), "normal components must match the element's dimension")
	return small_vec_from_slice(f64, REFERENCE_ELEMENTS[et].topology.facet_normals[facet][:], I)
}

// Maps a point in a sub-entity's reference into the element's reference.
element_lift_to_parent_reference :: proc(et: Element_Type, dim: Dimension, index: int, p: Ref_Vec) -> Ref_Vec {
	return ref_map_apply(element_sub_entity(et, dim, index).lift, p)
}

// Rule and weights for given quadrature set, quadrature set may not be tabulated, panics if so.
element_quadrature_rule :: proc(et: Element_Type, set: Quadrature_Set) -> (Rule, []f64) {
	q := REFERENCE_ELEMENTS[et].quadrature[set]
	assert(len(q.points) > 0, "quadrature set is not tabulated")
	return {element = et, points = q.points}, q.weights
}

@(private = "file")
ref_map_apply :: proc(m: Ref_Map, p: Ref_Vec) -> Ref_Vec {
	return m.origin + p.x * m.jac[0] + p.y * m.jac[1] + p.z * m.jac[2]
}

//== Orientation

// Vertex ids of an entity in canonical order: sorted for Line and Tri, the smallest then its smaller neighbour for Quad.
element_canonical_order :: proc(et: Element_Type, verts: []$T) -> (canon: [MAX_VERTICES]T) {
	assert(len(verts) == element_n_sub_entities(et, .D0))
	#partial switch et {
	case .Point, .Line, .Tri:
		copy(canon[:], verts)
		slice.sort(canon[:len(verts)])
	case .Quad:
		first, _ := slice.min_index(verts)
		a, b := verts[first ~ 1], verts[first ~ 2] // neighbours in tensor vertex order
		canon[0], canon[1], canon[2], canon[3] = verts[first], min(a, b), max(a, b), verts[first ~ 3]
	case: panic("only Point, Line, Tri and Quad entities orient")
	}
	return
}

// Orientation key of `local_order` against `target_order`, the same vertices in another order.
element_orientation :: proc(et: Element_Type, local_order: []$T, target_order: []T) -> u8 {
	n := len(local_order)
	assert(n == len(target_order))

	perm: [MAX_VERTICES]int
	for v, i in local_order {
		idx, found := slice.linear_search(target_order, v)
		assert(found, "local_order and target_order are not the same vertex set")
		perm[i] = idx
	}

	table := REFERENCE_ELEMENTS[et].topology.orientation_perms
	for row, i in table {
		if slice.equal(row, perm[:n]) { return u8(i) }
	}
	panic("vertex order is not a symmetry of this element type")
}

// Moves a point in the entity's reference to where it sits under orientation key `key`.
element_orient_point :: proc(et: Element_Type, key: u8, p: Ref_Vec) -> Ref_Vec {
	if key == 0 { return p }
	return ref_map_apply(REFERENCE_ELEMENTS[et].topology.orientation_maps[key], p)
}

//== Basis queries

// A basis on one element type.
Basis_Desc :: struct {
	element: Element_Type,
	family:  Basis_Family,
	order:   Order,
}

// Check if the basis descriptor describes a valid basis.
basis_is_defined :: proc(bd: Basis_Desc) -> bool {
	return basis_ref(bd).n_dofs > 0
}

// Total dofs on the element under this basis
basis_n_dofs :: proc(bd: Basis_Desc) -> int {
	return basis_ref(bd).n_dofs
}

// Dofs supported by entity of dimension `dim`, NOT total dofs for all entities of `dim`.
basis_dofs_per_entity :: proc(bd: Basis_Desc, dim: Dimension) -> int {
	return basis_ref(bd).dofs_per_entity[dim]
}

// Dofs owned by sub-entity (dim, index): [start, start + count).
basis_entity_dof_range :: proc(bd: Basis_Desc, dim: Dimension, index: int) -> (start, count: int) {
	assert(dim <= element_dim(bd.element) && index < element_n_sub_entities(bd.element, dim) && basis_is_defined(bd))
	b := basis_ref(bd)
	count = b.dofs_per_entity[dim]
	return b.entity_dof_start[dim] + index * count, count
}

// Dofs of sub-entity (dim, index) and everything on its boundary.
basis_closure_dofs :: proc(bd: Basis_Desc, dim: Dimension, index: int) -> []int {
	return basis_ref(bd).closure_dofs[dim][index]
}

// Components per dof of a quantity.
basis_quantity_cmpnts :: proc(bd: Basis_Desc, q: Basis_Quantity) -> int {
	assert(q in BASIS_QUANTITIES[bd.family])
	return QUANTITY_CMPNTS[q][element_dim(bd.element)]
}

// Components per dof of a quantity pushed to a physical space of dimension `n_ambient`.
basis_quantity_phys_cmpnts :: proc(bd: Basis_Desc, q: Basis_Quantity, n_ambient: int) -> int {
	switch basis_quantity_map(bd, q) {
	case .Unity: return basis_quantity_cmpnts(bd, q)
	case .Covariant, .Contravariant: return n_ambient
	case .Density: return 1
	}
	unreachable()
}

// How a quantity maps from the reference to the physical element.
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

// Evaluate a basis quantity at location `r`, not really meant to be used directly in weak forms.
basis_eval :: proc(bd: Basis_Desc, q: Basis_Quantity, r: Ref_Vec, out: Bvec_Point(f64)) {
	eval := basis_ref(bd).evals[q]
	assert(eval != nil, "quantity not tabulated for this basis")
	assert(out.dofs == basis_n_dofs(bd) && out.cmpnts == basis_quantity_cmpnts(bd, q))
	eval(r, out)
}

// Return the dof functionals for a given sub entity. Often, but not always, a dof supported by an entity
// will also have its functional defined over that entity.
basis_entity_functionals :: proc(bd: Basis_Desc, dim: Dimension, index: int) -> Entity_Functionals {
	return basis_ref(bd).functionals[dim][index]
}

// Reference coordinates of a nodal basis' nodes, in dof order.
basis_nodal_coords :: proc(bd: Basis_Desc, alloc := context.allocator) -> []Ref_Vec {
	assert(bd.family == .Lagrange)
	out := make([]Ref_Vec, basis_n_dofs(bd), alloc)
	n := 0
	for d in Dimension {
		if d > element_dim(bd.element) { break }
		if basis_dofs_per_entity(bd, d) == 0 { continue }
		for e in 0 ..< element_n_sub_entities(bd.element, d) {
			for p in basis_entity_functionals(bd, d, e).rule.points {
				out[n] = element_lift_to_parent_reference(bd.element, d, e, p)
				n += 1
			}
		}
	}
	return out
}

// Transform of an entity's dof block under orientation key `key`.
basis_orientation :: proc(bd: Basis_Desc, entity_type: Element_Type, key: u8) -> Orientation_Transform {
	table := basis_ref(bd).orientations[entity_type]
	if key == 0 || table == nil { return {} }
	return table[key]
}

// Smallest quadrature set on sub-entity (dim, index) exact for a mass matrix of the basis, plus `extra` degrees.
basis_quad_rule :: proc(bd: Basis_Desc, dim: Dimension, index: int, extra := 0) -> (Rule, []f64) {
	need := 2 * (int(bd.order) + FAMILY_DEGREE_OFFSET[bd.family]) + extra
	et := element_sub_entity(bd.element, dim, index).type
	for set in Quadrature_Set {
		if QUAD_SET_DEGREE[set] >= need { return element_quadrature_rule(et, set) }
	}
	panic("no quadrature set is exact enough")
}

@(private = "file")
basis_ref :: proc(bd: Basis_Desc) -> ^Reference_Basis {
	return &REFERENCE_ELEMENTS[bd.element].bases[bd.family][bd.order]
}
