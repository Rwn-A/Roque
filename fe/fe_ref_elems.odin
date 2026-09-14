package fe

/*
 Reference element topology, quadrature & basis. Largely just access functions around tabulated data.

 Tables are updated as needed and may eventually be generated in a meta program.

 TOPOLOGY TERMNINOLOGY:
  - Vertex (0D point)
 	- Edge (1D line segment)
  - Face (2D shape)
  - Facet (Element dim - 1 entity, may be an edge, face, or vertex)
*/

import "core:slice"

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

// Order refers to increasing accuracy of approximation space, depending on what is applied to,
// may not actually be equivalent to polynomial degree.
Order :: enum {
	O1,
	O2,
}

// Basis families may not be defined over all element types.
Basis_Family :: enum {
	Lagrange,
	Raviart_Thomas,
}

// Qn is high enough to inegrate polynomial of degree `n` exactly.
Quadrature_Set :: enum {
	Q1,
	Q3,
	Q5,
}

MAX_FACES    :: 6
MAX_FACETS   :: 6
MAX_EDGES    :: 12
MAX_VERTICES :: 8

Ref_Vec :: [3]f64

Reference_Element :: struct {
	topo:     Reference_Topology,
	quad:     [Quadrature_Set]Reference_Quadrature,
	lagrange: [Order]Reference_Lagrange,
	rt:       [Order]Reference_RT,
}

// compile-time access if needed
REFERENCE_ELEMENTS_CT :: [Element_Type]Reference_Element {
	.Line  = REF_LINE,
	.Point = REF_POINT,
	.Tri   = REF_TRI,
	.Quad  = REF_QUAD,
	.Hex   = REF_HEX,
	.Tet   = REF_TET,
}


@(rodata)
REFERENCE_ELEMENTS := REFERENCE_ELEMENTS_CT

Reference_Topology :: struct {
	dim:               Dimension,
	facet_types:       []Element_Type,
	facet_ref_normals: []Ref_Vec,
	sub_entity_verts:  [Dimension][][]int, //sub-entity dimension
	sub_entity_edges:  [][]int, // only exists for D3 elements.
	orientation_perms: [][]int, // valid only for lines, quads, tris.
}

Reference_Quadrature :: struct {
	points:  []Ref_Vec,
	weights: []f64,
}

Ref_Scalar_Val :: #type proc(idx: int, r: Ref_Vec) -> f64
Ref_Vector_Val :: #type proc(idx: int, r: Ref_Vec) -> Ref_Vec

Reference_Lagrange :: struct {
	using core: Reference_Basis_Core,
	nodes:      []Ref_Vec, // nodal coordinates of each dof (needed explicitly for dof functionals)
	vals:       Ref_Scalar_Val,
	grads:      Ref_Vector_Val,
}

Reference_RT :: struct {
	using core: Reference_Basis_Core,
	vals:       Ref_Vector_Val,
	divs:       Ref_Scalar_Val,
}

Reference_Basis_Core :: struct {
	facet_restrictions: [][]int, // includes dofs that are non zero anywhere on the facet
	support:            []DOF_Support,
	sub_entity_perms:   [Element_Type][]DOF_Perm,
}

// Maps a local entity dof index (from support) to the reference orientations local entity index.
// With an accomponying sign flip if appropiate.
DOF_Perm :: struct {
	perm: []int,
	sign: []f64,
}

// Topological support of a basis dof, used for building local to global maps.
DOF_Support :: struct {
	entity_dim:       Dimension,
	entity_index:     int, // which local entity on the element
	entity_dof_index: int, // which dof on this specific local entity
}

//== Wrapped table accessors & higher level helpers

element_dim :: proc(et: Element_Type) -> Dimension {
	return REFERENCE_ELEMENTS[et].topo.dim
}

element_num_nodes :: proc(et: Element_Type, order: Order) -> int {
	return len(REFERENCE_ELEMENTS[et].lagrange[order].nodes)
}

element_quad_rule :: proc(et: Element_Type, set: Quadrature_Set) -> Rule {
	q := REFERENCE_ELEMENTS[et].quad[set]
	return {et, q.points, q.weights}
}

element_num_facets :: proc(et: Element_Type) -> int {
	assert(et != .Point)
	return len(REFERENCE_ELEMENTS[et].topo.facet_types)
}

element_facet_dim :: proc(et: Element_Type) -> Dimension {
	assert(et != .Point)
	return REFERENCE_ELEMENTS[element_facet_type(et, 0)].topo.dim
}

element_facet_type :: proc(et: Element_Type, facet: int) -> Element_Type {
	assert(et != .Point)
	return REFERENCE_ELEMENTS[et].topo.facet_types[facet]
}

// Returns the local edge indices for the given element that make up the given facet.
element_facet_edges :: proc(et: Element_Type, facet: int) -> []int {
	assert(element_dim(et) == .D3, "Only 3D element facets have edges.")
	return REFERENCE_ELEMENTS[et].topo.sub_entity_edges[facet]
}

// Returns the local node indices for the given element at the given order for the facet.
element_facet_verts :: proc(et: Element_Type, facet: int) -> []int {
	assert(et != .Point)
	return REFERENCE_ELEMENTS[et].topo.sub_entity_verts[element_facet_dim(et)][facet]
}

element_facet_ref_normal :: proc(et: Element_Type, facet: int) -> Ref_Vec {
	assert(et != .Point)
	return REFERENCE_ELEMENTS[et].topo.facet_ref_normals[facet]
}

// Find orientation key from a given vertex order, based on the target order.
element_orientation :: proc(et: Element_Type, local_order: []$T, target_order: []T) -> u8 {
	n := len(local_order)
	assert(n == len(target_order))

	PERM_BUFFER := [128]int{}

	for v, i in local_order {
		idx, found := slice.linear_search(target_order, v)
		assert(found, "local_order and target_order are not the same vertex set")
		PERM_BUFFER[i] = idx
	}

	table := REFERENCE_ELEMENTS[et].topo.orientation_perms
	assert(table != nil, "element type has no orientation symmetry group")
	for row, i in table {
		if slice.equal(row, PERM_BUFFER[:len(local_order)]) { return u8(i) }
	}

	panic("vertex correspondence is not a valid symmetry for this facet type")
}

// Move a point from the reference space of the elements facet to the given elements ref space.
// In the interest of surface quadrature.
lift_to_parent_reference :: proc(et: Element_Type, facet: int, facet_point: Ref_Vec) -> Ref_Vec {
	assert(et != .Point)
	switch et {
	case .Point: unreachable()
	case .Line: return facet == 0 ? {-1, 0, 0} : {1, 0, 0}
	case .Tri: switch facet {
			case 0: return {facet_point.x, 0, 0}
			case 1: return {1 - facet_point.x, facet_point.x, 0}
			case 2: return {0, 1 - facet_point.x, 0}
			case: unreachable()
			}
	case .Quad: switch facet {
			case 0: return {facet_point.x, -1, 0}
			case 1: return {1, facet_point.x, 0}
			case 2: return {facet_point.x, 1, 0}
			case 3: return {-1, facet_point.x, 0}
			case: unreachable()
			}
	case .Tet: switch facet {
			case 0: return {facet_point.x, facet_point.y, 0}
			case 1: return {facet_point.x, 0, facet_point.y}
			case 2: return {0, facet_point.x, facet_point.y}
			case 3: return {1 - facet_point.x - facet_point.y, facet_point.x, facet_point.y}
			case: unreachable()
			}
	case .Hex: switch facet {
			case 0: return {facet_point.x, facet_point.y, -1}
			case 1: return {facet_point.x, facet_point.y, +1}
			case 2: return {facet_point.x, -1, facet_point.y}
			case 3: return {facet_point.x, +1, facet_point.y}
			case 4: return {-1, facet_point.x, facet_point.y}
			case 5: return {+1, facet_point.x, facet_point.y}
			case: unreachable()
			}
	case: unreachable()
	}
}

Basis_Desc :: struct {
	element: Element_Type,
	family:  Basis_Family,
	order:   Order,
}

Basis_Quantity :: enum {
	Scalar,
	Vector,
	Scalar_Gradient,
	Vector_Divergence,
}

@(rodata)
BASIS_QUANTITIES := [Basis_Family]bit_set[Basis_Quantity] {
	.Lagrange       = {.Scalar, .Scalar_Gradient},
	.Raviart_Thomas = {.Vector, .Vector_Divergence},
}

// Generalization of a quadrature rule
Rule :: struct {
	element:    Element_Type, // points are in ref space of this elem.
	ref_points: []Ref_Vec,
	weights:    []f64, // optional
}

basis_quantity_components :: proc(et: Element_Type, qty: Basis_Quantity) -> int {
	switch qty {
	case .Scalar, .Vector_Divergence: return 1
	case .Vector, .Scalar_Gradient: return int(element_dim(et))
	case: unreachable()
	}
}

basis_info_core :: proc(bd: Basis_Desc) -> Reference_Basis_Core {
	switch bd.family {
	case .Lagrange: return REFERENCE_ELEMENTS[bd.element].lagrange[bd.order]
	case .Raviart_Thomas: return REFERENCE_ELEMENTS[bd.element].rt[bd.order]
	case: unreachable()
	}
}

// Returns all local dofs which are non-zero on the given facet
// not just topologically facet-supported dofs.
basis_facet_restriction :: proc(bd: Basis_Desc, facet: int) -> []int {
	assert(bd.element != .Point)
	return basis_info_core(bd).facet_restrictions[facet]
}

basis_count :: proc(bd: Basis_Desc) -> int {
	return len(basis_support(bd))
}

basis_support :: proc(bd: Basis_Desc) -> []DOF_Support {
	return basis_info_core(bd).support
}

// permutation for dofs on the sub entity, only dofs that are supported by the sub entity directly.
basis_sub_entity_perm :: proc(bd: Basis_Desc, sub_entity: Element_Type, orientation: u8) -> DOF_Perm {
	assert(sub_entity == .Line || sub_entity == .Tri || sub_entity == .Quad, "Sub entity does not have a permuation")
	assert(element_dim(sub_entity) < element_dim(bd.element))
	return basis_info_core(bd).sub_entity_perms[sub_entity][orientation]
}

oriented_local_dof :: proc(bd: Basis_Desc, sub_et: Element_Type, orientation: u8, local: int) -> (canonical: int, flip: bool) {
	perm := basis_sub_entity_perm(bd, sub_et, orientation)
	return perm.perm[local], perm.sign[local] == -1
}

// Functional points & points as a "Rule". The coefficient for the dof is given as:
// sum(value_p * weight_p) where the sum is over points (p). What value means depends on the basis.
basis_functional_rule :: proc(bd: Basis_Desc, dof: int) -> Rule {
	core := basis_info_core(bd)
	sup := core.support[dof]

	switch bd.family {
	case .Lagrange:
		lag := REFERENCE_ELEMENTS[bd.element].lagrange[bd.order]
		return Rule{element = bd.element, ref_points = lag.nodes[dof:dof + 1], weights = nil}
	case .Raviart_Thomas:
		assert(element_facet_dim(bd.element) == sup.entity_dim, "Unimplemented: Interior RT basis")

		ft := element_facet_type(bd.element, sup.entity_index)
		switch bd.order {
		case .O1: return element_quad_rule(ft, .Q1)
		case .O2: return element_quad_rule(ft, .Q3)
		case: unreachable()
		}
	case: unreachable()
	}
}


//== Raw tables

ROOT_3     :: 1.73205080757
REC_ROOT_3 :: 1.0 / ROOT_3
ROOT_3_5   :: 0.7745966692414834

REF_POINT :: Reference_Element {
	topo = {
		dim = .D0,
		facet_types = {},
		facet_ref_normals = {},
		sub_entity_verts = #partial{.D0 = {{0}}},
		sub_entity_edges = {},
		orientation_perms = {},
	},
}

REF_LINE :: Reference_Element {
	topo = {
		dim = .D1,
		facet_types = {.Point, .Point},
		facet_ref_normals = {{-1, 0, 0}, {+1, 0, 0}},
		sub_entity_verts = #partial{.D0 = {{0}, {1}}},
		sub_entity_edges = {},
		orientation_perms = {{0, 1}, {1, 0}},
	},
	quad = {
		.Q1 = {points = {{0, 0, 0}}, weights = {2}},
		.Q3 = {points = {{-REC_ROOT_3, 0, 0}, {REC_ROOT_3, 0, 0}}, weights = {1, 1}},
		.Q5 = {points = {{-ROOT_3_5, 0, 0}, {0, 0, 0}, {ROOT_3_5, 0, 0}}, weights = {5.0 / 9, 8.0 / 9, 5.0 / 9}},
	},
	lagrange = {
		.O1 = {
			support = {{.D0, 0, 0}, {.D0, 1, 0}},
			nodes = {{-1, 0, 0}, {1, 0, 0}},
			facet_restrictions = {{0}, {1}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				switch dof {
				case 0: return (1.0 - r.x) / 2.0
				case 1: return (1.0 + r.x) / 2.0
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				switch dof {
				case 0: return {-0.5, 0, 0}
				case 1: return {0.5, 0, 0}
				case: unreachable()
				}
			},
		},
		.O2 = {
			support = {{.D0, 0, 0}, {.D0, 1, 0}, {.D1, 0, 0}},
			nodes = {{-1, 0, 0}, {1, 0, 0}, {0, 0, 0}},
			facet_restrictions = {{0}, {1}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				switch dof {
				case 0: return -r.x * (1.0 - r.x) / 2.0
				case 1: return r.x * (1.0 + r.x) / 2.0
				case 2: return 1.0 - r.x * r.x
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				switch dof {
				case 0: return {-(1.0 - 2.0 * r.x) / 2.0, 0, 0}
				case 1: return {(1.0 + 2.0 * r.x) / 2.0, 0, 0}
				case 2: return {-2.0 * r.x, 0, 0}
				case: unreachable()
				}
			},
		},
	},
}

// Ref tri vertices: v0=(0,0), v1=(1,0), v2=(0,1). Barycentric L0=1-x-y, L1=x, L2=y.
REF_TRI :: Reference_Element {
	topo = {
		dim = .D2,
		facet_types = {.Line, .Line, .Line},
		facet_ref_normals = {{0, -1, 0}, {REC_ROOT_3, REC_ROOT_3, 0}, {-1, 0, 0}},
		sub_entity_verts = #partial{.D0 = {{0}, {1}, {2}}, .D1 = {{0, 1}, {1, 2}, {2, 0}}, .D2 = {{0, 1, 2}}},
		sub_entity_edges = {},
		orientation_perms = {
			{0, 1, 2},
			{0, 2, 1},
			{1, 0, 2},
			{1, 2, 0},
			{2, 0, 1},
			{2, 1, 0},
		},
	},
	quad = [Quadrature_Set]Reference_Quadrature {
		.Q1 = {points = {{1.0 / 3.0, 1.0 / 3.0, 0}}, weights = {0.5}},
		.Q3 = {
			points = {{1.0 / 6.0, 1.0 / 6.0, 0}, {2.0 / 3.0, 1.0 / 6.0, 0}, {1.0 / 6.0, 2.0 / 3.0, 0}},
			weights = {1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0},
		},
		.Q5 = {
			points = {
				{0.091576213509771, 0.091576213509771, 0},
				{0.816847572980459, 0.091576213509771, 0},
				{0.091576213509771, 0.816847572980459, 0},
				{0.445948490915965, 0.108103018168070, 0},
				{0.108103018168070, 0.445948490915965, 0},
				{0.445948490915965, 0.445948490915965, 0},
			},
			weights = {
				0.054975871827661,
				0.054975871827661,
				0.054975871827661,
				0.111690794839005,
				0.111690794839005,
				0.111690794839005,
			},
		},
	},
	lagrange = {
		.O1 = {
			support = {{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}},
			nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}},
			facet_restrictions = {{0, 1}, {1, 2}, {2, 0}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				switch dof {
				case 0: return 1.0 - r.x - r.y
				case 1: return r.x
				case 2: return r.y
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				switch dof {
				case 0: return {-1, -1, 0}
				case 1: return {1, 0, 0}
				case 2: return {0, 1, 0}
				case: unreachable()
				}
			},
		},
		.O2 = {
			support = {{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D1, 0, 0}, {.D1, 1, 0}, {.D1, 2, 0}},
			nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0}},
			facet_restrictions = {{0, 1, 3}, {1, 2, 4}, {2, 0, 5}},
			sub_entity_perms = #partial {.Line = {{perm = {0}, sign = {1}}, {perm = {0}, sign = {1}}}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				x, y := r.x, r.y
				l0 := 1.0 - x - y
				switch dof {
				case 0: return l0 * (2.0 * l0 - 1.0)
				case 1: return x * (2.0 * x - 1.0)
				case 2: return y * (2.0 * y - 1.0)
				case 3: return 4.0 * l0 * x
				case 4: return 4.0 * x * y
				case 5: return 4.0 * y * l0
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				x, y := r.x, r.y
				switch dof {
				case 0: v := 4.0 * x + 4.0 * y - 3.0; return {v, v, 0}
				case 1: return {4.0 * x - 1.0, 0, 0}
				case 2: return {0, 4.0 * y - 1.0, 0}
				case 3: return {4.0 - 8.0 * x - 4.0 * y, -4.0 * x, 0}
				case 4: return {4.0 * y, 4.0 * x, 0}
				case 5: return {-4.0 * y, 4.0 - 4.0 * x - 8.0 * y, 0}
				case: unreachable()
				}
			},
		},
	},
}

// Ref quad domain [-1,1]^2, vertices v0=(-1,-1), v1=(1,-1), v2=(1,1), v3=(-1,1).
REF_QUAD :: Reference_Element {
	topo = {
		dim = .D2,
		facet_types = {.Line, .Line, .Line, .Line},
		facet_ref_normals = {{0, -1, 0}, {+1, 0, 0}, {0, +1, 0}, {-1, 0, 0}},
		sub_entity_verts = #partial{
			.D0 = {{0}, {1}, {2}, {3}},
			.D1 = {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
			.D2 = {{0, 1, 2, 3}},
		},
		sub_entity_edges = {},
		orientation_perms = {
			{0, 1, 2, 3},
			{1, 2, 3, 0},
			{2, 3, 0, 1},
			{3, 0, 1, 2},
			{0, 3, 2, 1},
			{3, 2, 1, 0},
			{2, 1, 0, 3},
			{1, 0, 3, 2},
		},
	},
	lagrange = {
		.O1 = {
			support = {{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D0, 3, 0}},
			nodes = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}},
			facet_restrictions = {{0, 1}, {1, 2}, {2, 3}, {3, 0}},
			sub_entity_perms = #partial {.Line = {{perm = {0}, sign = {1}}, {perm = {0}, sign = {1}}}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				x, y := r.x, r.y
				switch dof {
				case 0: return (1.0 - x) * (1.0 - y) / 4.0
				case 1: return (1.0 + x) * (1.0 - y) / 4.0
				case 2: return (1.0 + x) * (1.0 + y) / 4.0
				case 3: return (1.0 - x) * (1.0 + y) / 4.0
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				x, y := r.x, r.y
				switch dof {
				case 0: return {-(1.0 - y) / 4.0, -(1.0 - x) / 4.0, 0}
				case 1: return {(1.0 - y) / 4.0, -(1.0 + x) / 4.0, 0}
				case 2: return {(1.0 + y) / 4.0, (1.0 + x) / 4.0, 0}
				case 3: return {-(1.0 + y) / 4.0, (1.0 - x) / 4.0, 0}
				case: unreachable()
				}
			},
		},
		.O2 = {
			support = {
				{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D0, 3, 0},
				{.D1, 0, 0}, {.D1, 1, 0}, {.D1, 2, 0}, {.D1, 3, 0},
				{.D2, 0, 0},
			},
			nodes = {
				{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0},
				{0, -1, 0}, {1, 0, 0}, {0, 1, 0}, {-1, 0, 0},
				{0, 0, 0},
			},
			facet_restrictions = {{0, 1, 4}, {1, 2, 5}, {2, 3, 6}, {3, 0, 7}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				x, y := r.x, r.y
				nm1 :: proc(t: f64) -> f64 { return (t * t - t) / 2.0 }
				n0 :: proc(t: f64) -> f64 { return 1.0 - t * t }
				np1 :: proc(t: f64) -> f64 { return (t * t + t) / 2.0 }
				switch dof {
				case 0: return nm1(x) * nm1(y)
				case 1: return np1(x) * nm1(y)
				case 2: return np1(x) * np1(y)
				case 3: return nm1(x) * np1(y)
				case 4: return n0(x) * nm1(y)
				case 5: return np1(x) * n0(y)
				case 6: return n0(x) * np1(y)
				case 7: return nm1(x) * n0(y)
				case 8: return n0(x) * n0(y)
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				x, y := r.x, r.y
				nm1 :: proc(t: f64) -> f64 { return (t * t - t) / 2.0 }
				n0 :: proc(t: f64) -> f64 { return 1.0 - t * t }
				np1 :: proc(t: f64) -> f64 { return (t * t + t) / 2.0 }
				dnm1 :: proc(t: f64) -> f64 { return t - 0.5 }
				dn0 :: proc(t: f64) -> f64 { return -2.0 * t }
				dnp1 :: proc(t: f64) -> f64 { return t + 0.5 }
				switch dof {
				case 0: return {dnm1(x) * nm1(y), nm1(x) * dnm1(y), 0}
				case 1: return {dnp1(x) * nm1(y), np1(x) * dnm1(y), 0}
				case 2: return {dnp1(x) * np1(y), np1(x) * dnp1(y), 0}
				case 3: return {dnm1(x) * np1(y), nm1(x) * dnp1(y), 0}
				case 4: return {dn0(x) * nm1(y), n0(x) * dnm1(y), 0}
				case 5: return {dnp1(x) * n0(y), np1(x) * dn0(y), 0}
				case 6: return {dn0(x) * np1(y), n0(x) * dnp1(y), 0}
				case 7: return {dnm1(x) * n0(y), nm1(x) * dn0(y), 0}
				case 8: return {dn0(x) * n0(y), n0(x) * dn0(y), 0}
				case: unreachable()
				}
			},
		},
	},
}

// Ref tet vertices: v0=(0,0,0), v1=(1,0,0), v2=(0,1,0), v3=(0,0,1). Barycentric L0=1-x-y-z, L1=x, L2=y, L3=z.
REF_TET :: Reference_Element {
	topo = {
		dim = .D3,
		facet_types = {.Tri, .Tri, .Tri, .Tri},
		facet_ref_normals = {{0, 0, -1}, {0, -1, 0}, {-1, 0, 0}, {REC_ROOT_3, REC_ROOT_3, REC_ROOT_3}},
		sub_entity_verts = #partial {
			.D0 = {{0}, {1}, {2}, {3}},
			.D1 = {
				{0, 1},
				{1, 2},
				{2, 0},
				{0, 3},
				{1, 3},
				{2, 3},
			},
			.D2 = {
				{0, 2, 1},
				{0, 1, 3},
				{0, 3, 2},
				{1, 2, 3},
			},
			.D3 = {{0, 1, 2, 3}},
		},
		sub_entity_edges = {
			{2, 1, 0},
			{0, 4, 3},
			{3, 5, 2},
			{1, 5, 4},
		},
		orientation_perms = {},
	},
	lagrange = {
		.O1 = {
			support = {{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D0, 3, 0}},
			nodes = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
			facet_restrictions = {{0, 2, 1}, {0, 1, 3}, {0, 3, 2}, {1, 2, 3}},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				switch dof {
				case 0: return 1.0 - r.x - r.y - r.z
				case 1: return r.x
				case 2: return r.y
				case 3: return r.z
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				switch dof {
				case 0: return {-1, -1, -1}
				case 1: return {1, 0, 0}
				case 2: return {0, 1, 0}
				case 3: return {0, 0, 1}
				case: unreachable()
				}
			},
		},
		.O2 = {
			support = {
				{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D0, 3, 0},
				{.D1, 0, 0}, {.D1, 1, 0}, {.D1, 2, 0}, {.D1, 3, 0}, {.D1, 4, 0}, {.D1, 5, 0},
			},
			nodes = {
				{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1},
				{0.5, 0, 0}, {0.5, 0.5, 0}, {0, 0.5, 0}, {0, 0, 0.5}, {0.5, 0, 0.5}, {0, 0.5, 0.5},
			},
			facet_restrictions = {
				{0, 2, 1, 6, 5, 4},
				{0, 1, 3, 4, 8, 7},
				{0, 3, 2, 7, 9, 6},
				{1, 2, 3, 5, 9, 8},
			},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				x, y, z := r.x, r.y, r.z
				l0 := 1.0 - x - y - z
				switch dof {
				case 0: return l0 * (2.0 * l0 - 1.0)
				case 1: return x * (2.0 * x - 1.0)
				case 2: return y * (2.0 * y - 1.0)
				case 3: return z * (2.0 * z - 1.0)
				case 4: return 4.0 * l0 * x
				case 5: return 4.0 * x * y
				case 6: return 4.0 * y * l0
				case 7: return 4.0 * l0 * z
				case 8: return 4.0 * x * z
				case 9: return 4.0 * y * z
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				x, y, z := r.x, r.y, r.z
				switch dof {
				case 0: v := 4.0 * x + 4.0 * y + 4.0 * z - 3.0; return {v, v, v}
				case 1: return {4.0 * x - 1.0, 0, 0}
				case 2: return {0, 4.0 * y - 1.0, 0}
				case 3: return {0, 0, 4.0 * z - 1.0}
				case 4: return {4.0 - 8.0 * x - 4.0 * y - 4.0 * z, -4.0 * x, -4.0 * x}
				case 5: return {4.0 * y, 4.0 * x, 0}
				case 6: return {-4.0 * y, 4.0 - 4.0 * x - 8.0 * y - 4.0 * z, -4.0 * y}
				case 7: return {-4.0 * z, -4.0 * z, 4.0 - 4.0 * x - 4.0 * y - 8.0 * z}
				case 8: return {4.0 * z, 0, 4.0 * x}
				case 9: return {0, 4.0 * z, 4.0 * y}
				case: unreachable()
				}
			},
		},
	},
}

// Ref hex domain [-1,1]^3, vertices v0..v3 on z=-1 (ccw), v4..v7 above v0..v3 on z=+1.
REF_HEX :: Reference_Element {
	topo = {
		dim = .D3,
		facet_types = {.Quad, .Quad, .Quad, .Quad, .Quad, .Quad},
		facet_ref_normals = {
			{0, 0, -1},
			{0, 0, +1},
			{0, -1, 0},
			{0, +1, 0},
			{-1, 0, 0},
			{+1, 0, 0},
		},
		sub_entity_verts = #partial {
			.D0 = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
			.D1 = {
				{0, 1},
				{1, 2},
				{2, 3},
				{3, 0},
				{4, 5},
				{5, 6},
				{6, 7},
				{7, 4},
				{0, 4},
				{1, 5},
				{2, 6},
				{3, 7},
			},
			.D2 = {
				{0, 3, 2, 1},
				{4, 5, 6, 7},
				{0, 1, 5, 4},
				{3, 7, 6, 2},
				{0, 4, 7, 3},
				{1, 2, 6, 5},
			},
			.D3 = {{0, 1, 2, 3, 4, 5, 6, 7}},
		},
		sub_entity_edges = {
			{2, 1, 0, 3},
			{4, 5, 6, 7},
			{0, 9, 4, 8},
			{3, 7, 6, 11},
			{8, 7, 11, 3},
			{1, 10, 5, 9},
		},
		orientation_perms = {},
	},
	lagrange = {
		.O1 = {
			support = {
				{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D0, 3, 0},
				{.D0, 4, 0}, {.D0, 5, 0}, {.D0, 6, 0}, {.D0, 7, 0},
			},
			nodes = {
				{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
				{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
			},
			facet_restrictions = {
				{0, 3, 2, 1},
				{4, 5, 6, 7},
				{0, 1, 5, 4},
				{3, 7, 6, 2},
				{0, 4, 7, 3},
				{1, 2, 6, 5},
			},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				x, y, z := r.x, r.y, r.z
				switch dof {
				case 0: return (1.0 - x) * (1.0 - y) * (1.0 - z) / 8.0
				case 1: return (1.0 + x) * (1.0 - y) * (1.0 - z) / 8.0
				case 2: return (1.0 + x) * (1.0 + y) * (1.0 - z) / 8.0
				case 3: return (1.0 - x) * (1.0 + y) * (1.0 - z) / 8.0
				case 4: return (1.0 - x) * (1.0 - y) * (1.0 + z) / 8.0
				case 5: return (1.0 + x) * (1.0 - y) * (1.0 + z) / 8.0
				case 6: return (1.0 + x) * (1.0 + y) * (1.0 + z) / 8.0
				case 7: return (1.0 - x) * (1.0 + y) * (1.0 + z) / 8.0
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				x, y, z := r.x, r.y, r.z
				switch dof {
				case 0: return {-(1.0 - y) * (1.0 - z) / 8.0, -(1.0 - x) * (1.0 - z) / 8.0, -(1.0 - x) * (1.0 - y) / 8.0}
				case 1: return {(1.0 - y) * (1.0 - z) / 8.0, -(1.0 + x) * (1.0 - z) / 8.0, -(1.0 + x) * (1.0 - y) / 8.0}
				case 2: return {(1.0 + y) * (1.0 - z) / 8.0, (1.0 + x) * (1.0 - z) / 8.0, -(1.0 + x) * (1.0 + y) / 8.0}
				case 3: return {-(1.0 + y) * (1.0 - z) / 8.0, (1.0 - x) * (1.0 - z) / 8.0, -(1.0 - x) * (1.0 + y) / 8.0}
				case 4: return {-(1.0 - y) * (1.0 + z) / 8.0, -(1.0 - x) * (1.0 + z) / 8.0, (1.0 - x) * (1.0 - y) / 8.0}
				case 5: return {(1.0 - y) * (1.0 + z) / 8.0, -(1.0 + x) * (1.0 + z) / 8.0, (1.0 + x) * (1.0 - y) / 8.0}
				case 6: return {(1.0 + y) * (1.0 + z) / 8.0, (1.0 + x) * (1.0 + z) / 8.0, (1.0 + x) * (1.0 + y) / 8.0}
				case 7: return {-(1.0 + y) * (1.0 + z) / 8.0, (1.0 - x) * (1.0 + z) / 8.0, (1.0 - x) * (1.0 + y) / 8.0}
				case: unreachable()
				}
			},
		},
		.O2 = {
			support = {
				{.D0, 0, 0}, {.D0, 1, 0}, {.D0, 2, 0}, {.D0, 3, 0}, {.D0, 4, 0}, {.D0, 5, 0}, {.D0, 6, 0}, {.D0, 7, 0},
				{.D1, 0, 0}, {.D1, 1, 0}, {.D1, 2, 0}, {.D1, 3, 0}, {.D1, 4, 0}, {.D1, 5, 0},
				{.D1, 6, 0}, {.D1, 7, 0}, {.D1, 8, 0}, {.D1, 9, 0}, {.D1, 10, 0}, {.D1, 11, 0},
				{.D2, 0, 0}, {.D2, 1, 0}, {.D2, 2, 0}, {.D2, 3, 0}, {.D2, 4, 0}, {.D2, 5, 0},
				{.D3, 0, 0},
			},
			nodes = {
				{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1},
				{-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1},
				{0, -1, -1}, {1, 0, -1}, {0, 1, -1}, {-1, 0, -1},
				{0, -1, 1}, {1, 0, 1}, {0, 1, 1}, {-1, 0, 1},
				{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0},
				{0, 0, -1}, {0, 0, 1}, {0, -1, 0}, {0, 1, 0}, {-1, 0, 0}, {1, 0, 0},
				{0, 0, 0},
			},
			facet_restrictions = {
				{0, 3, 2, 1, 10, 9, 8, 11, 20},
				{4, 5, 6, 7, 12, 13, 14, 15, 21},
				{0, 1, 5, 4, 8, 17, 12, 16, 22},
				{3, 7, 6, 2, 11, 15, 14, 19, 23},
				{0, 4, 7, 3, 16, 15, 19, 11, 24},
				{1, 2, 6, 5, 9, 18, 13, 17, 25},
			},
			vals = proc(dof: int, r: Ref_Vec) -> f64 {
				x, y, z := r.x, r.y, r.z
				nm1 :: proc(t: f64) -> f64 { return (t * t - t) / 2.0 }
				n0 :: proc(t: f64) -> f64 { return 1.0 - t * t }
				np1 :: proc(t: f64) -> f64 { return (t * t + t) / 2.0 }
				switch dof {
				case 0: return nm1(x) * nm1(y) * nm1(z)
				case 1: return np1(x) * nm1(y) * nm1(z)
				case 2: return np1(x) * np1(y) * nm1(z)
				case 3: return nm1(x) * np1(y) * nm1(z)
				case 4: return nm1(x) * nm1(y) * np1(z)
				case 5: return np1(x) * nm1(y) * np1(z)
				case 6: return np1(x) * np1(y) * np1(z)
				case 7: return nm1(x) * np1(y) * np1(z)
				case 8: return n0(x) * nm1(y) * nm1(z)
				case 9: return np1(x) * n0(y) * nm1(z)
				case 10: return n0(x) * np1(y) * nm1(z)
				case 11: return nm1(x) * n0(y) * nm1(z)
				case 12: return n0(x) * nm1(y) * np1(z)
				case 13: return np1(x) * n0(y) * np1(z)
				case 14: return n0(x) * np1(y) * np1(z)
				case 15: return nm1(x) * n0(y) * np1(z)
				case 16: return nm1(x) * nm1(y) * n0(z)
				case 17: return np1(x) * nm1(y) * n0(z)
				case 18: return np1(x) * np1(y) * n0(z)
				case 19: return nm1(x) * np1(y) * n0(z)
				case 20: return n0(x) * n0(y) * nm1(z)
				case 21: return n0(x) * n0(y) * np1(z)
				case 22: return n0(x) * nm1(y) * n0(z)
				case 23: return n0(x) * np1(y) * n0(z)
				case 24: return nm1(x) * n0(y) * n0(z)
				case 25: return np1(x) * n0(y) * n0(z)
				case 26: return n0(x) * n0(y) * n0(z)
				case: unreachable()
				}
			},
			grads = proc(dof: int, r: Ref_Vec) -> Ref_Vec {
				x, y, z := r.x, r.y, r.z
				nm1 :: proc(t: f64) -> f64 { return (t * t - t) / 2.0 }
				n0 :: proc(t: f64) -> f64 { return 1.0 - t * t }
				np1 :: proc(t: f64) -> f64 { return (t * t + t) / 2.0 }
				dnm1 :: proc(t: f64) -> f64 { return t - 0.5 }
				dn0 :: proc(t: f64) -> f64 { return -2.0 * t }
				dnp1 :: proc(t: f64) -> f64 { return t + 0.5 }
				switch dof {
				case 0: return {dnm1(x) * nm1(y) * nm1(z), nm1(x) * dnm1(y) * nm1(z), nm1(x) * nm1(y) * dnm1(z)}
				case 1: return {dnp1(x) * nm1(y) * nm1(z), np1(x) * dnm1(y) * nm1(z), np1(x) * nm1(y) * dnm1(z)}
				case 2: return {dnp1(x) * np1(y) * nm1(z), np1(x) * dnp1(y) * nm1(z), np1(x) * np1(y) * dnm1(z)}
				case 3: return {dnm1(x) * np1(y) * nm1(z), nm1(x) * dnp1(y) * nm1(z), nm1(x) * np1(y) * dnm1(z)}
				case 4: return {dnm1(x) * nm1(y) * np1(z), nm1(x) * dnm1(y) * np1(z), nm1(x) * nm1(y) * dnp1(z)}
				case 5: return {dnp1(x) * nm1(y) * np1(z), np1(x) * dnm1(y) * np1(z), np1(x) * nm1(y) * dnp1(z)}
				case 6: return {dnp1(x) * np1(y) * np1(z), np1(x) * dnp1(y) * np1(z), np1(x) * np1(y) * dnp1(z)}
				case 7: return {dnm1(x) * np1(y) * np1(z), nm1(x) * dnp1(y) * np1(z), nm1(x) * np1(y) * dnp1(z)}
				case 8: return {dn0(x) * nm1(y) * nm1(z), n0(x) * dnm1(y) * nm1(z), n0(x) * nm1(y) * dnm1(z)}
				case 9: return {dnp1(x) * n0(y) * nm1(z), np1(x) * dn0(y) * nm1(z), np1(x) * n0(y) * dnm1(z)}
				case 10: return {dn0(x) * np1(y) * nm1(z), n0(x) * dnp1(y) * nm1(z), n0(x) * np1(y) * dnm1(z)}
				case 11: return {dnm1(x) * n0(y) * nm1(z), nm1(x) * dn0(y) * nm1(z), nm1(x) * n0(y) * dnm1(z)}
				case 12: return {dn0(x) * nm1(y) * np1(z), n0(x) * dnm1(y) * np1(z), n0(x) * nm1(y) * dnp1(z)}
				case 13: return {dnp1(x) * n0(y) * np1(z), np1(x) * dn0(y) * np1(z), np1(x) * n0(y) * dnp1(z)}
				case 14: return {dn0(x) * np1(y) * np1(z), n0(x) * dnp1(y) * np1(z), n0(x) * np1(y) * dnp1(z)}
				case 15: return {dnm1(x) * n0(y) * np1(z), nm1(x) * dn0(y) * np1(z), nm1(x) * n0(y) * dnp1(z)}
				case 16: return {dnm1(x) * nm1(y) * n0(z), nm1(x) * dnm1(y) * n0(z), nm1(x) * nm1(y) * dn0(z)}
				case 17: return {dnp1(x) * nm1(y) * n0(z), np1(x) * dnm1(y) * n0(z), np1(x) * nm1(y) * dn0(z)}
				case 18: return {dnp1(x) * np1(y) * n0(z), np1(x) * dnp1(y) * n0(z), np1(x) * np1(y) * dn0(z)}
				case 19: return {dnm1(x) * np1(y) * n0(z), nm1(x) * dnp1(y) * n0(z), nm1(x) * np1(y) * dn0(z)}
				case 20: return {dn0(x) * n0(y) * nm1(z), n0(x) * dn0(y) * nm1(z), n0(x) * n0(y) * dnm1(z)}
				case 21: return {dn0(x) * n0(y) * np1(z), n0(x) * dn0(y) * np1(z), n0(x) * n0(y) * dnp1(z)}
				case 22: return {dn0(x) * nm1(y) * n0(z), n0(x) * dnm1(y) * n0(z), n0(x) * nm1(y) * dn0(z)}
				case 23: return {dn0(x) * np1(y) * n0(z), n0(x) * dnp1(y) * n0(z), n0(x) * np1(y) * dn0(z)}
				case 24: return {dnm1(x) * n0(y) * n0(z), nm1(x) * dn0(y) * n0(z), nm1(x) * n0(y) * dn0(z)}
				case 25: return {dnp1(x) * n0(y) * n0(z), np1(x) * dn0(y) * n0(z), np1(x) * n0(y) * dn0(z)}
				case 26: return {dn0(x) * n0(y) * n0(z), n0(x) * dn0(y) * n0(z), n0(x) * n0(y) * dn0(z)}
				case: unreachable()
				}
			},
		},
	},
}
