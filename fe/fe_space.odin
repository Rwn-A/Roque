package fe

/*
 Finite element space. Global discrete function space a quantity is discretized over.
*/

import "core:mem"
import "core:mem/virtual"
import "core:slice"

MAX_FIELDS :: 32 // should be plenty

Space_Continuity :: enum {
	Continuous,
	Discontinuous,
}

// .Cell is the usual case
// .Facet is for trace spaces, numbered directly over facets one dimension down.
Numbered_Entity :: enum {
	Cell,
	Facet,
}

Space_Desc :: struct #all_or_none {
	family:     Basis_Family,
	order:      Order,
	continuity: Space_Continuity,
	regions:    Region_Set,
}

DOF_Numbering :: struct {
	l2g:           [][]i32,
	flip_sign:     [][]bool,
	num_dofs:      int,
	numbered_over: Dimension,
}

Space_ID :: distinct int

Space :: struct {
	using sd:       Space_Desc,
	fields:         int,
	id:             Space_ID, // automatically assigned, for identifying spaces in a multi space.
	total_coeffs:   int,
	numbering:      DOF_Numbering,
	owns_numbering: bool,
	allocator:      mem.Allocator,
}

Space_Vector :: struct {
	using space: ^Space,
	coeffs:      []f64, // len space.total_coeffs
}

@(thread_local, private)
next_id: Space_ID

space_new :: proc(
	mesh: Mesh,
	sd: Space_Desc,
	fields: int,
	nent := Numbered_Entity.Cell,
	alloc := context.allocator,
) -> ^Space {
	assert(fields <= MAX_FIELDS)

	s := new(Space, alloc)

	s^ = {
		sd             = sd,
		fields         = fields,
		owns_numbering = true,
		id             = next_id,
		allocator      = alloc,
	}
	next_id += 1

	context.allocator = alloc
	scratch_guard()

	// building dof numbering

	DOF_Key :: struct {
		id:        Entity_ID,
		local_dof: int, // local dof on that entity
	}

	Numbering_Conn :: struct {
		conn:              Cell_Conn,
		edge_orientations: [MAX_EDGES]u8,
		face_orientations: [MAX_FACES]u8,
	}

	n: DOF_Numbering = {
		numbered_over = mesh.intrinsic_dim if nent == .Cell else Dimension(int(mesh.intrinsic_dim) - 1),
	}
	s.numbering = n

	next_global: int
	shared_maps := [Dimension]map[DOF_Key]int{}
	for &m in shared_maps { m = make(map[DOF_Key]int, scratch()) }

	switch nent {
	case .Cell:
		n.l2g = make([][]i32, len(mesh.cells))
		n.flip_sign = make([][]bool, len(mesh.cells))
		for cell in mesh.cells {
			if cell.region not_in sd.regions { continue }
			bd := space_bd(s, cell.type)
			n.l2g[cell.id] = make([]i32, basis_count(bd))
			n.flip_sign[cell.id] = make([]bool, basis_count(bd))
			conn := Numbering_Conn{mesh.cell_conn[cell.id], cell.edge_orientation, cell.face_orientation}
			number_dofs(&shared_maps, &next_global, bd, sd.continuity, n.l2g[cell.id], n.flip_sign[cell.id], conn)
		}
	case .Facet:
		n.l2g = make([][]i32, len(mesh.facets))
		n.flip_sign = make([][]bool, len(mesh.facets))
		for facet in mesh.facets {
			if facet_regions(mesh, facet) & sd.regions == {} { continue }
			bd := space_bd(s, facet.info.type)
			n.l2g[facet.id] = make([]i32, basis_count(bd))
			n.flip_sign[facet.id] = make([]bool, basis_count(bd))

			cell, local_facet := facet_canonical_cell(mesh, facet)
			cell_conn := mesh.cell_conn[cell.id]

			// temporary facet connectivity
			verts: [MAX_VERTICES]Entity_ID
			edges: [MAX_EDGES]Entity_ID
			edge_orientations: [MAX_EDGES]u8
			// faces are always interior for trace space so we can ignore face orientation

			if mesh.intrinsic_dim == .D3 {
				for edge, i in element_facet_edges(cell.type, local_facet) {
					edge_orientations[i] = cell.edge_orientation[edge]
					edges[i] = cell_conn.edges[edge]
				}
			}

			for vert, i in element_facet_verts(cell.type, local_facet) {
				verts[i] = cell_conn.vertices[vert]
			}

			conn := Numbering_Conn {
				conn = {vertices = verts[:], edges = edges[:]},
				edge_orientations = edge_orientations,
			}

			// face is always internal to a trace space, (unless we go to 4d), so orientation doesnt matter.
			number_dofs(&shared_maps, &next_global, bd, sd.continuity, n.l2g[facet.id], n.flip_sign[facet.id], conn)
		}
	case: unreachable()
	}

	s.numbering = n
	s.numbering.num_dofs = next_global
	s.total_coeffs = s.numbering.num_dofs * s.fields

	return s

	number_dofs :: proc(
		shared: ^[Dimension]map[DOF_Key]int,
		next_global: ^int,
		bd: Basis_Desc,
		continuity: Space_Continuity,
		l2g: []i32,
		signs: []bool,
		conn: Numbering_Conn,
	) {
		for sup, dof in basis_support(bd) {
			if continuity == .Discontinuous {
				l2g[dof] = i32(next_global^)
				signs[dof] = false
				next_global^ += 1
				continue
			}
			switch sup.entity_dim {
			case element_dim(bd.element):
				//internal no sharing
				l2g[dof] = i32(next_global^)
				signs[dof] = false
				next_global^ += 1
			case .D0:
				key := DOF_Key{conn.conn.vertices[sup.entity_index], sup.entity_dof_index}
				l2g[dof] = assign_shared_dof(&shared[.D0], key, next_global)
				signs[dof] = false
			case .D1:
				dof_local, flip := basis_orient_dof(
					bd,
					.Line,
					conn.edge_orientations[sup.entity_index],
					sup.entity_dof_index,
				)
				key := DOF_Key{conn.conn.edges[sup.entity_index], dof_local}
				l2g[dof] = assign_shared_dof(&shared[.D1], key, next_global)
				signs[dof] = flip
			case .D2:
				ft := element_facet_type(bd.element, sup.entity_index)
				dof_local, flip := basis_orient_dof(
					bd,
					ft,
					conn.face_orientations[sup.entity_index],
					sup.entity_dof_index,
				)
				key := DOF_Key{conn.conn.faces[sup.entity_index], dof_local}
				l2g[dof] = assign_shared_dof(&shared[.D2], key, next_global)
				signs[dof] = flip
			case .D3: unreachable() // caught by element dim case.
			}
		}
	}

	assign_shared_dof :: proc(m: ^map[DOF_Key]int, key: DOF_Key, next_global: ^int) -> i32 {
		if existing, ok := m[key]; ok { return i32(existing) }
		v := next_global^
		m[key] = v
		next_global^ += 1
		return i32(v)
	}

}

space_from_existing :: proc(existing: Space, fields: int, alloc := context.allocator) -> ^Space {
	assert(fields <= MAX_FIELDS)

	s := new(Space, alloc)

	s^ = {
		numbering      = existing.numbering,
		sd             = existing.sd,
		fields         = fields,
		total_coeffs   = existing.numbering.num_dofs * fields,
		owns_numbering = false,
		id             = next_id,
		allocator      = alloc,
	}
	next_id += 1

	return s

}

space_new_isoparemetric :: proc(mesh: Mesh, fields: int, alloc := context.allocator) -> ^Space {
	assert(fields <= MAX_FIELDS)

	s := new(Space, alloc)
	s^ = {
		numbering = {l2g = mesh.cell_nodes, num_dofs = len(mesh.nodes), numbered_over = mesh.intrinsic_dim},
		fields = fields,
		total_coeffs = len(mesh.nodes) * fields,
		id = next_id,
		allocator = alloc,
		owns_numbering = false,
	}

	next_id += 1

	return s
}

// Frees the space and the DOF numbering if the space owns its numbering.
space_destroy :: proc(spaces: ..^Space) {
	for s in spaces {
		context.allocator = s.allocator
		if s.owns_numbering {
			for entity in 0 ..< len(s.numbering.l2g) {
				delete(s.numbering.l2g[entity])
				if s.numbering.flip_sign != nil { delete(s.numbering.flip_sign[entity]) }
			}
			delete(s.numbering.l2g)
			delete(s.numbering.flip_sign)
		}
		free(s) //frees the ptr
	}
}

space_bd :: proc(s: ^Space, et: Element_Type) -> Basis_Desc {
	assert(s.numbering.numbered_over == element_dim(et), "Space is not defined on the element dim.")
	return {et, s.family, s.order}
}

space_l2g :: proc(s: ^Space, ent: Entity_ID) -> []i32 {
	l2g := s.numbering.l2g[ent]
	assert(l2g != nil, "Space may not be defined on entity")
	return l2g
}

@(private)
space_dof_sign :: proc(s: ^Space, ent: Entity_ID, ldof: int) -> f64 {
	return -1.0 if s.numbering.flip_sign != nil && s.numbering.flip_sign[ent][ldof] else 1.0
}


// Global dof indices of all dofs that are non-zero on the facet.
space_facet_dofs :: proc(mesh: Mesh, space: ^Space, facet_id: Entity_ID, alloc := context.allocator) -> []i32 {
	facet := mesh.facets[facet_id]

	if space.numbering.numbered_over == element_dim(facet.info.type) { return space_l2g(space, facet_id) }

	cell, local_facet := facet_canonical_cell(mesh, facet)
	restriction := basis_facet_restriction(space_bd(space, cell.type), local_facet)
	l2g := space_l2g(space, cell.id)
	out := make([]i32, len(restriction), alloc)
	for ldof, i in restriction { out[i] = l2g[ldof] }

	return out
}

space_cvec :: proc($T: typeid, space: ^Space, et: Element_Type, alloc := context.allocator) -> Cvec(T) {
	return cvec_create(T, basis_count(space_bd(space, et)), space.fields)
}

space_cmat :: proc(
	$T: typeid,
	test, trial: ^Space,
	test_e, trial_e: Element_Type,
	alloc := context.allocator,
) -> Cmat(T) {
	return cmat_create(
		T,
		basis_count(space_bd(test, test_e)),
		basis_count(space_bd(trial, trial_e)),
		test.fields,
		trial.fields,
	)
}

// Gather all dofs at this entity, casting to type T for mixed-percision.
space_gather :: proc($T: typeid, s: Space_Vector, ent: Entity_ID, alloc := context.allocator) -> Cvec(T) {
	l2g := space_l2g(s, ent)
	cvec := cvec_create(T, len(l2g), s.fields, alloc)

	for gdof, ldof in l2g {
		cvb := cvec_dof_block(cvec, ldof)
		sign := space_dof_sign(s, ent, ldof)
		for entry, i in s.coeffs[gdof * i32(s.fields):][:s.fields] { cvb[i] = cast(T)entry * cast(T)sign }
	}

	return cvec
}


// Scatter local into coeffs, casting to type T for mixed-percision.
space_scatter :: proc(s: Space_Vector, ent: Entity_ID, local: Cvec($T)) {
	l2g := space_l2g(s, ent)

	assert(len(l2g) * s.fields == len(local.data))

	for gdof, ldof in l2g {
		cvb := cvec_dof_block(local, ldof)
		sign := space_dof_sign(s, ent, ldof)
		for &entry, i in s.coeffs[gdof * i32(s.fields):][:s.fields] { entry = cast(f64)cvb[i] * sign }
	}
}

// Scatter only the dofs in Cvec that are in the mask. Cvec must still be sized the same as `scatter`.
space_scatter_masked :: proc(s: Space_Vector, ent: Entity_ID, local: Cvec($T), mask: []int) {
	l2g := space_l2g(s, ent)

	assert(len(l2g) * s.fields == len(local.data))

	for ldof, _ in mask {
		gdof := l2g[ldof]
		cvb := cvec_dof_block(local, ldof)
		sign := space_dof_sign(s, ent, ldof)
		for &entry, i in s.coeffs[gdof * i32(s.fields):][:s.fields] { entry = cast(f64)cvb[i] * sign }
	}
}

//== Interpolator

Interpolator :: struct($A, $I: int) {
	geometry, space:            Space_Vector,
	restriction:                []int,
	num_dofs:                   int, // actual dofs to visit, accounts for restriction
	ent:                        Entity_ID,
	et:                         Element_Type,
	vals:                       Cvec(f64),
	current_dof, current_point: int,
	jacobians:                  Pvec(f64),
	phys_points:                Pvec(f64),
	temp:                       Scratch_Temp,
}

// Interpolate a continous function onto the given FE space, with optional restriction for setting boundary dofs.
// For convienence, some geometry is computed including the physical point location and jacobian mapping.
// - A is the ambient dimension this must be equal to the number of coordinates per node in the geo space.
// - I is the intrinsic dimension and must be equal to dimension of the element type.
// These are compile time to allow for the same geometric operations used in weak forms to apply here.
interpolator :: proc(
	$A, $I: int,
	geo, space: Space_Vector,
	et: Element_Type,
	ent: Entity_ID,
	restriction: []int = nil,
) -> Interpolator(A, I) {
	assert(A == geo.fields)
	assert(I == int(element_dim(et)))

	bd := space_bd(space.space, et)
	return Interpolator(A, I) {
		geometry = geo,
		space = space,
		restriction = restriction,
		ent = ent,
		et = et,
		num_dofs = len(restriction) if restriction != nil else basis_count(bd),
		vals = cvec_create(f64, basis_count(bd), space.fields, scratch()),
		current_dof = -1,
		temp = scratch_begin_temp(),
	}
}

// Yields a point to evaluate at write results into `out` which is sized for each field in the space.
interpolator_next :: proc(
	ip: ^Interpolator($A, $I),
) -> (
	jac: Small_Mat(A, I, f64),
	point: Small_Vec(A, f64),
	out: []f64,
	ok: bool,
) {
	POINT_DIMS :: Contraction_Dims {
		.CMPNTS = 1,
		.FIELDS = A,
	}
	JAC_DIMS :: Contraction_Dims {
		.CMPNTS = I,
		.FIELDS = A,
	}

	context.allocator = scratch()

	for ip.current_point >= ip.phys_points.points {
		ip.current_dof += 1
		if ip.current_dof >= ip.num_dofs {
			return {}, {}, nil, false
		}
		dof := ip.restriction[ip.current_dof] if ip.restriction != nil else ip.current_dof

		spce_bd := space_bd(ip.space, ip.et)
		geo_bd := space_bd(ip.geometry, ip.et)
		rule := basis_functional_rule(spce_bd, dof)

		geo_basis: Basis_Entry
		if rule.element != ip.et {
			assert(element_dim(rule.element) == element_facet_dim(ip.et))
			sup := basis_support(spce_bd)[dof]
			assert(
				sup.entity_dim == element_dim(rule.element),
				"bug: DOF functional over facet rule, not facet supported.",
			)
			geo_basis = bstore_get_facet(geo_bd, sup.entity_index, rule)
		} else {
			geo_basis = bstore_get_interior(geo_bd, rule)
		}

		coords := space_gather(f64, ip.geometry, ip.ent)
		ip.phys_points = pvec_create(f64, len(rule.ref_points), POINT_DIMS)
		ip.jacobians = pvec_create(f64, len(rule.ref_points), JAC_DIMS)

		contract_eval(POINT_DIMS, ip.phys_points, coords, geo_basis[.Scalar])
		contract_eval(JAC_DIMS, ip.jacobians, coords, geo_basis[.Scalar_Gradient])

		ip.current_point = 0
	}

	point = small_vec_view_from_slice(pvec_at_point(ip.phys_points, ip.current_point).data, A)^
	jac = small_mat_view_from_slice(pvec_at_point(ip.jacobians, ip.current_point).data, A, I)^

	dof := ip.restriction[ip.current_dof] if ip.restriction != nil else ip.current_dof
	out = cvec_dof_block(ip.vals, dof)

	ip.current_point += 1
	return jac, point, out, true
}

interpolator_flush :: proc(ip: ^Interpolator($A, $I)) {
	defer scratch_end_temp(ip.temp)
	if ip.restriction != nil {
		space_scatter_masked(ip.space, ip.ent, ip.vals, ip.restriction)
	} else {
		space_scatter(ip.space, ip.ent, ip.vals)
	}
}


//== Multi-Space

State_Map :: map[Space_ID][]f64

state_map_for :: proc(spaces: ..^Space, alloc := context.allocator) -> State_Map {
	stmap := make(map[Space_ID][]f64, alloc)
	for s in spaces {
		assert(s.id not_in stmap, "Passed in same space twice.")
		stmap[s.id] = make([]f64, s.total_coeffs, alloc)
	}
	return stmap
}

state_map_destroy :: proc(s: State_Map) {
	for k, &v in s { delete(v, s.allocator) }
	delete(s)
}

space_vec_from_map :: proc(state_map: State_Map, s: ^Space) -> Space_Vector {
	return {s, state_map[s.id]}
}

Constraint_Essential :: struct {
	boundaries: Boundary_Set,
	leave_free: bit_set[0 ..< MAX_FIELDS], // for partial constraints
}

// `field_transform` applies the transform to all field coefficients per dof.
// This is useful if your field coefficients represent some directional quantity, it must be sized for all
// fields even if some fields are later ignored by `leave_free`.
Constraint_Periodic :: struct {
	periodicity:     Periodicity,
	field_transform: Maybe(Dense_Matrix),
	leave_free:      bit_set[0 ..< MAX_FIELDS],
}

Constraint :: union {
	Constraint_Essential,
	Constraint_Periodic,
}

constraint_essential :: proc(boundary: Boundary_ID, leave_free: bit_set[0 ..< MAX_FIELDS] = {}) -> Constraint {
	return Constraint_Essential{{boundary}, leave_free}
}

Constituent_Space :: struct {
	space:       ^Space,
	constraints: []Constraint,
}

Multi_Space :: struct {
	spaces:           map[Space_ID]^Space,
	ranges:           []Space_Range,
	total_state_size: int, // total dofs before elimination
	total_soln_size:  int, // free dofs after elimination
	dof_map:          []DOF_Map_Entry, // len == total_state_size, flat, global-indexed
	soln_to_dof:      []int, // soln idx -> global dof idx
	sparsity:         Sparsity,
	inhomogeneity:    []f64, // len == total_state_size
	arena:            virtual.Arena,
}

Space_Range :: struct {
	base: int,
	end:  int,
	id:   Space_ID,
}

DOF_Role :: enum {
	Free,
	Constrained,
}

MPC_Term :: struct {
	dof:    int,
	weight: f64,
}

DOF_Map_Entry :: struct {
	role:       DOF_Role,
	soln_index: int, // valid when Free
	terms:      []MPC_Term, // valid when Constrained, nil for a plain essential dof
}

Assembly_Mode :: enum {
	Linear,
	Newton,
}

ms_create :: proc(mesh: Mesh, spaces: ..Constituent_Space) -> (ms: Multi_Space) {
	assert(len(spaces) >= 1)

	scratch_guard()

	numbered_over := spaces[0].space.numbering.numbered_over
	for cs in spaces {
		assert(cs.space.numbering.numbered_over == numbered_over, "Spaces must all be trace, or all be interior")
	}

	if err := virtual.arena_init_growing(&ms.arena); err != nil { panic("Failed to create arena") }
	context.allocator = virtual.arena_allocator(&ms.arena)

	ms.spaces = make(map[Space_ID]^Space)
	ms.ranges = make([]Space_Range, len(spaces))

	offset := 0
	for cs, i in spaces {
		ms.spaces[cs.space.id] = cs.space
		ms.ranges[i] = {
			base = offset,
			end  = offset + cs.space.total_coeffs,
			id   = cs.space.id,
		}
		offset += cs.space.total_coeffs
	}
	ms.total_state_size = offset
	ms.dof_map = make([]DOF_Map_Entry, ms.total_state_size)
	ms.inhomogeneity = make([]f64, ms.total_state_size)

	constrained := make([]bool, ms.total_state_size, scratch())
	for cs, i in spaces {
		base := ms.ranges[i].base
		for c in cs.constraints {
			switch v in c {
			case Constraint_Essential: mark_essential(mesh, cs.space, base, v, ms.dof_map, constrained)
			case Constraint_Periodic: mark_periodic(mesh, cs.space, base, v, ms.dof_map, constrained)
			}
		}
	}

	soln_idx := 0
	for gidx in 0 ..< ms.total_state_size {
		if constrained[gidx] { continue }
		ms.dof_map[gidx] = {
			role       = .Free,
			soln_index = soln_idx,
		}
		soln_idx += 1
	}
	ms.total_soln_size = soln_idx

	ms.soln_to_dof = make([]int, ms.total_soln_size)
	for gidx in 0 ..< ms.total_state_size {
		if ms.dof_map[gidx].role == .Free { ms.soln_to_dof[ms.dof_map[gidx].soln_index] = gidx }
	}

	ms_build_sparsity(&ms)

	return ms

	mark_essential :: proc(
		mesh: Mesh,
		space: ^Space,
		base: int,
		c: Constraint_Essential,
		dof_map: []DOF_Map_Entry,
		constrained: []bool,
	) {
		for facet in mesh.facets {
			if facet.info.boundary not_in c.boundaries { continue }
			scratch_guard()
			for dof in space_facet_dofs(mesh, space, facet.id, scratch()) {
				for field in 0 ..< space.fields {
					if field in c.leave_free { continue }
					gidx := base + int(dof) * space.fields + field
					dof_map[gidx] = {
						role = .Constrained,
					}
					constrained[gidx] = true
				}
			}
		}
	}

	// Ties each slave facet dof to its corresponding master facet dof (optionally through field_transform).
	mark_periodic :: proc(
		mesh: Mesh,
		space: ^Space,
		base: int,
		c: Constraint_Periodic,
		dof_map: []DOF_Map_Entry,
		constrained: []bool,
	) {

		Facet_Dof_Key :: struct {
			dim:   Dimension,
			ent:   int,
			local: int, // sup.entity_dof_index
		}

		xform, has_xform := c.field_transform.?

		for pair in c.periodicity.pairs {
			scratch_guard()

			master_dofs := space_facet_dofs(mesh, space, pair.master, scratch())
			slave_dofs := space_facet_dofs(mesh, space, pair.slave, scratch())
			assert(len(master_dofs) == len(slave_dofs), "periodic pair facets have mismatched dof counts")

			et := mesh.facets[pair.master].info.type
			bd_facet := Basis_Desc{et, space.family, space.order} // only used for its basis_support order,
			// which space_facet_dofs' own ordering (via basis_facet_restriction / space_l2g) matches.

			master_cell, _ := facet_canonical_cell(mesh, mesh.facets[pair.master])
			bd_cell := Basis_Desc{master_cell.type, space.family, space.order}

			slave_flat := make(map[Facet_Dof_Key]int, scratch())
			for sup, dof in basis_support(bd_facet) {
				slave_flat[{sup.entity_dim, sup.entity_index, sup.entity_dof_index}] = dof
			}

			for sup, mldof in basis_support(bd_facet) {
				sldof: int
				sign := 1.0

				switch sup.entity_dim {
				case element_dim(et):
					canonical, flip := basis_orient_dof(bd_cell, et, pair.orientation, sup.entity_dof_index)
					sldof = slave_flat[{sup.entity_dim, sup.entity_index, canonical}]
					sign = -1.0 if flip else 1.0
				case .D0: sldof = slave_flat[{.D0, pair.vertex_map[sup.entity_index], sup.entity_dof_index}]
				case .D1:
					canonical, flip := basis_orient_dof(
						bd_cell,
						.Line,
						pair.edge_orientation[sup.entity_index],
						sup.entity_dof_index,
					)
					sldof = slave_flat[{.D1, pair.edge_map[sup.entity_index], canonical}]
					sign = -1.0 if flip else 1.0

				case .D2, .D3: unreachable()
				case: unreachable()
				}

				for field in 0 ..< space.fields {
					if field in c.leave_free { continue }

					sgidx := base + int(slave_dofs[sldof]) * space.fields + field
					if constrained[sgidx] { continue } 	// already tied by an earlier pair, e.g. a shared corner

					terms: []MPC_Term
					if has_xform {
						buf := make([dynamic]MPC_Term, 0, space.fields)
						for mfield in 0 ..< space.fields {
							w := dense_get(xform, field, mfield)^
							if w == 0 { continue }
							append(&buf, MPC_Term{base + int(master_dofs[mldof]) * space.fields + mfield, w * sign})
						}
						terms = buf[:]
					} else {
						terms = make([]MPC_Term, 1)
						terms[0] = {base + int(master_dofs[mldof]) * space.fields + field, sign}
					}

					dof_map[sgidx] = {
						role  = .Constrained,
						terms = terms,
					}
					constrained[sgidx] = true
				}
			}
		}
	}
}

ms_state :: proc(ms: Multi_Space, alloc := context.allocator) -> (stmap: State_Map) {
	stmap = make(State_Map, alloc)
	for key, val in ms.spaces { stmap[key] = make([]f64, val.total_coeffs, alloc) }
	return stmap
}

@(private)
ms_space_range :: proc(ms: Multi_Space, id: Space_ID) -> Space_Range {
	for r in ms.ranges { if r.id == id { return r } }
	panic("space not in multi space")
}

ms_inhomogeneity :: proc(ms: Multi_Space, space: Space_ID) -> []f64 {
	r := ms_space_range(ms, space)
	return ms.inhomogeneity[r.base:r.end]
}

// Ensures coeffs in state for all spaces in the multi space satisfy the set constraints.
ms_enforce_constraints :: proc(ms: Multi_Space, state: State_Map) {
	for r in ms.ranges {
		local_state := state[r.id]
		for gidx in r.base ..< r.end {
			entry := ms.dof_map[gidx]
			if entry.role != .Constrained { continue }

			val := ms.inhomogeneity[gidx]
			for term in entry.terms { val += term.weight * ms_read(ms, state, term.dof) }
			local_state[gidx - r.base] = val
		}
	}

	ms_read :: proc(ms: Multi_Space, state: State_Map, gidx: int) -> f64 {
		for r in ms.ranges {
			if gidx >= r.base && gidx < r.end { return state[r.id][gidx - r.base] }
		}
		panic("gidx out of range")
	}
}

ms_soln_vector :: proc(ms: Multi_Space, alloc := context.allocator) -> Vector {
	return make(Vector, ms.total_soln_size, alloc)
}

ms_soln_matrix :: proc(ms: Multi_Space, alloc := context.allocator) -> Sparse_Matrix {
	return sp_from_sparsity(ms.sparsity, alloc)
}

// Returns two soln vectors and a matrix for convienence
ms_problem_data :: proc(ms: Multi_Space, alloc := context.allocator) -> (Vector, Vector, Sparse_Matrix) {
	return ms_soln_vector(ms, alloc), ms_soln_vector(ms, alloc), ms_soln_matrix(ms, alloc)
}

// Applies state += update, updates free dofs and re-enforces constraints.
ms_apply_update :: proc(ms: Multi_Space, state: State_Map, update: Vector) {
	for r in ms.ranges {
		local_state := state[r.id]
		for gidx in r.base ..< r.end {
			entry := ms.dof_map[gidx]
			if entry.role != .Free { continue }
			local_state[gidx - r.base] += update[entry.soln_index]
		}
	}
	ms_enforce_constraints(ms, state)
}

// Applies state = soln, updates free dofs and re-enforces constraints.
ms_apply_soln :: proc(ms: Multi_Space, state: State_Map, soln: Vector) {
	for r in ms.ranges {
		local_state := state[r.id]
		for gidx in r.base ..< r.end {
			entry := ms.dof_map[gidx]
			if entry.role != .Free { continue }
			local_state[gidx - r.base] = soln[entry.soln_index]
		}
	}
	ms_enforce_constraints(ms, state)
}

ms_destroy :: proc(ms: ^Multi_Space) {
	virtual.arena_destroy(&ms.arena)
}

@(private)
ms_build_sparsity :: proc(ms: ^Multi_Space) {
	Pair :: struct {
		row, col: i32,
	}

	touch :: proc(ms: ^Multi_Space, touched: ^[dynamic]int, gidx: int) {
		entry := ms.dof_map[gidx]
		if entry.role == .Free {
			append(touched, entry.soln_index)
			return
		}
		for term in entry.terms { touch(ms, touched, term.dof) }
	}

	scratch_guard()
	context.allocator = virtual.arena_allocator(&ms.arena)

	pairs := make([dynamic]Pair, scratch())
	touched := make([dynamic]int, scratch())

	entity_count := len(ms.spaces[ms.ranges[0].id].numbering.l2g)

	for e in 0 ..< entity_count {
		clear(&touched)
		for r in ms.ranges {
			space := ms.spaces[r.id]
			l2g := space.numbering.l2g[e]
			if l2g == nil { continue }

			for gdof in l2g {
				for field in 0 ..< space.fields {
					gidx := r.base + int(gdof) * space.fields + field
					touch(ms, &touched, gidx)
				}
			}
		}

		for a in touched {
			for b in touched { append(&pairs, Pair{i32(a), i32(b)}) }
		}
	}

	slice.sort_by(pairs[:], proc(a, b: Pair) -> bool {
		if a.row != b.row { return a.row < b.row }
		return a.col < b.col
	})

	row_ptrs := make([]i32, ms.total_soln_size + 1)
	columns := make([dynamic]i32)

	row := i32(0)
	for p, i in pairs {
		if i > 0 && p == pairs[i - 1] { continue }
		for row < p.row {
			row += 1
			row_ptrs[row] = i32(len(columns))
		}
		append(&columns, p.col)
	}
	for row < i32(ms.total_soln_size) {
		row += 1
		row_ptrs[row] = i32(len(columns))
	}

	ms.sparsity = {row_ptrs = row_ptrs, columns = columns[:]}
}

@(private)
ms_distribute_vec :: proc(ms: Multi_Space, vec: Vector, gidx: int, val: f64) {
	entry := ms.dof_map[gidx]
	if entry.role == .Free {
		vec[entry.soln_index] += val
		return
	}
	for term in entry.terms { ms_distribute_vec(ms, vec, term.dof, val * term.weight) }
}

ms_scatter_vec :: proc(ms: Multi_Space, vec: Vector, local: Cvec($T), test: Space_ID, ent: Entity_ID) {
	space := ms.spaces[test]
	l2g := space_l2g(space, ent)
	assert(len(l2g) == local.dofs)
	assert(local.fields == space.fields)

	base := ms_space_range(ms, test).base

	for gdof, ldof in l2g {
		sign := space_dof_sign(space, ent, ldof)
		lblock := cvec_dof_block(local, ldof)
		for field in 0 ..< local.fields {
			gidx := base + int(gdof) * space.fields + field
			ms_distribute_vec(ms, vec, gidx, cast(f64)lblock[field] * sign)
		}
	}
}


@(private)
ms_distribute_mat :: proc(
	ms: Multi_Space,
	mat: Sparse_Matrix,
	load: Vector,
	mode: Assembly_Mode,
	grow, gcol: int,
	val: f64,
) {
	rentry := ms.dof_map[grow]
	if rentry.role == .Constrained {
		for term in rentry.terms { ms_distribute_mat(ms, mat, load, mode, term.dof, gcol, val * term.weight) }
		return
	}

	centry := ms.dof_map[gcol]
	if centry.role == .Constrained {
		for term in centry.terms { ms_distribute_mat(ms, mat, load, mode, grow, term.dof, val * term.weight) }
		if mode == .Linear { load[rentry.soln_index] -= val * ms.inhomogeneity[gcol] }
		return
	}

	p := sp_get(mat, rentry.soln_index, centry.soln_index)
	assert(p != nil, "column not found in sparsity pattern for row")
	p^ += val
}

ms_scatter_mat :: proc(
	ms: Multi_Space,
	mat: Sparse_Matrix,
	load: Vector,
	mode: Assembly_Mode,
	local: Cmat($T),
	test, trial: Space_ID,
	ent: Entity_ID,
) {
	assert(mode == .Newton || load != nil, "Linear mode needs a load vector to move constrained columns into")

	tspace := ms.spaces[test]
	rspace := ms.spaces[trial]

	rows := space_l2g(tspace, ent)
	cols := space_l2g(rspace, ent)
	assert(len(rows) == local.row_dofs && len(cols) == local.col_dofs)
	assert(local.row_fields == tspace.fields && local.col_fields == rspace.fields)

	tbase := ms_space_range(ms, test).base
	rbase := ms_space_range(ms, trial).base

	for rgdof, rldof in rows {
		rsign := space_dof_sign(tspace, ent, rldof)
		for cgdof, cldof in cols {
			csign := space_dof_sign(rspace, ent, cldof)
			sign := rsign * csign
			block := cmat_dof_block(local, rldof, cldof)

			for rf in 0 ..< local.row_fields {
				grow := tbase + int(rgdof) * tspace.fields + rf
				for cf in 0 ..< local.col_fields {
					gcol := rbase + int(cgdof) * rspace.fields + cf
					val := cast(f64)block[rf * local.col_fields + cf] * sign
					ms_distribute_mat(ms, mat, load, mode, grow, gcol, val)
				}
			}
		}
	}
}
