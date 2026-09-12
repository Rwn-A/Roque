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
REFERENCE_ELEMENTS_CT :: #partial [Element_Type]Reference_Element {
	.Line = REF_LINE,
}


@(rodata)
REFERENCE_ELEMENTS := REFERENCE_ELEMENTS_CT

Reference_Topology :: struct {
	dim:               Dimension,
	facet_types:       []Element_Type,
	facet_ref_normals: []Ref_Vec,
	sub_entity_verts:  [Dimension][][]int, //sub-entity dimension
	sub_entity_edges:  [][]int, // only exists for D3 elements.
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
	facet_perms:        [Element_Type][]DOF_Perm, // depending on facet type, has varying numbers of permutations.
	edge_perm:          [2]DOF_Perm, // flipped or not flipped, edges are always lines,
	// permutations relate to the local index on that entity, `entity_dof_index` from DOF support.
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

// Permutation for dofs supported by the edge (excludes vertex dofs that are non-zero on edge)
basis_edge_perm :: proc(bd: Basis_Desc, flipped: bool) -> DOF_Perm {
	assert(element_dim(bd.element) >= .D2)
	return basis_info_core(bd).edge_perm[1 if flipped else 0]
}

// Permutation for dofs supported by the face (excludes edge, vertex dofs that are non-zero on face)
basis_face_perm :: proc(bd: Basis_Desc, facet: int, orientation: u8) -> DOF_Perm {
	assert(element_dim(bd.element) >= .D3)
	return basis_info_core(bd).facet_perms[element_facet_type(bd.element, facet)][orientation]
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

REF_LINE :: Reference_Element {
	topo = {dim = .D1, facet_types = {.Point, .Point}, sub_entity_verts = #partial{.D0 = {{0}, {1}}}},
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
