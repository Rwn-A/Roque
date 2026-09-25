package fe

import "core:mem"
import "core:mem/virtual"
import "core:slice"

MAX_FIELDS :: 32 // should be plenty

Space_Continuity :: enum {
	Continuous,
	Discontinuous,
}

Numbered_Entity :: enum {
	Cell, // cell is usual case
	Facet, // for trace spaces
}

Space_Desc :: struct #all_or_none {
	family:     Basis_Family,
	order:      Order,
	continuity: Space_Continuity,
	regions:    Region_Set,
}

// Local to global DOF numbering for a finite element space, same for all `fields` on the space.
DOF_Numbering :: struct {
	l2g:           [][]i32,
	num_dofs:      int,
	numbered_over: Dimension,
}

// Finite element space
Space :: struct {
	using sd:       Space_Desc,
	mesh:           ^Mesh,
	fields:         int,
	total_coeffs:   int,
	numbering:      DOF_Numbering,
	owns_numbering: bool,
	allocator:      mem.Allocator,
}

Space_Vector :: struct {
	using space: ^Space,
	coeffs:      []f64, // len space.total_coeffs
}

// New FE space over the mesh
space_new :: proc(
	mesh: ^Mesh,
	sd: Space_Desc,
	fields: int,
	nent := Numbered_Entity.Cell,
	alloc := context.allocator,
) -> ^Space {
	assert(fields <= MAX_FIELDS)

	on_cells := nent == .Cell
	n_elems := len(mesh.cells) if on_cells else len(mesh.facets)

	scratch_guard()

	start: [Dimension][]i32
	for d in Dimension {
		start[d] = make([]i32, mesh.n_entities[d], scratch())
		slice.fill(start[d], -1)
	}

	l2g := make([][]i32, n_elems, alloc)
	next: i32

	for &row, i in l2g {
		elem := Entity_ID(i)
		et: Element_Type
		conn: Connectivity

		if on_cells {
			if mesh.cells[elem].region not_in sd.regions { continue }
			et, conn = mesh.cells[elem].type, mesh.cell_conn[elem]
		} else {
			if facet_regions(mesh^, mesh.facets[elem]) & sd.regions == {} { continue }
			et, conn = mesh.facets[elem].info.type, facet_connectivity(mesh^, elem, scratch())
		}

		bd := Basis_Desc{et, sd.family, sd.order}
		row = make([]i32, basis_num_dofs(bd), alloc)
		ls := 0
		for d in Dimension {
			count := basis_dofs_per_entity(bd, d)
			if count == 0 { continue }
			for gid in conn[d] {
				if sd.continuity == .Discontinuous || start[d][gid] < 0 {
					start[d][gid] = next
					next += i32(count)
				}
				for j in 0 ..< count { row[ls + j] = start[d][gid] + i32(j) }
				ls += count
			}
		}
	}

	s := new(Space, alloc)
	s^ = {
		sd = sd,
		mesh = mesh,
		fields = fields,
		total_coeffs = int(next) * fields,
		numbering = {
			l2g = l2g,
			num_dofs = int(next),
			numbered_over = mesh.intrinsic_dim if on_cells else mesh.intrinsic_dim - Dimension(1),
		},
		owns_numbering = true,
		allocator = alloc,
	}
	return s
}

// Create a new space with the same numbering as the existing space
space_from_existing :: proc(existing: Space, fields: int, alloc := context.allocator) -> ^Space {
	assert(fields <= MAX_FIELDS)

	s := new(Space, alloc)
	s^ = existing
	s.fields = fields
	s.total_coeffs = existing.numbering.num_dofs * fields
	s.owns_numbering = false
	s.allocator = alloc

	return s
}

// Create a new space isoparemtric with the mesh geometry.
space_new_isoparemetric :: proc(mesh: ^Mesh, fields: int, alloc := context.allocator) -> ^Space {
	assert(fields <= MAX_FIELDS)

	s := new(Space, alloc)
	s^ = {
		sd = {.Lagrange, mesh.order, .Continuous, ALL_REGIONS},
		mesh = mesh,
		numbering = {l2g = mesh.cell_nodes, num_dofs = len(mesh.nodes), numbered_over = mesh.intrinsic_dim},
		fields = fields,
		total_coeffs = len(mesh.nodes) * fields,
		owns_numbering = false,
		allocator = alloc,
	}

	return s
}

// Frees the space and the DOF numbering if the space owns its numbering.
space_destroy :: proc(spaces: ..^Space) {
	for s in spaces {
		context.allocator = s.allocator
		if s.owns_numbering {
			for row in s.numbering.l2g { delete(row) }
			delete(s.numbering.l2g)
		}
		free(s)
	}
}

// Local basis descriptor for the space
space_bd :: proc(s: ^Space, et: Element_Type) -> Basis_Desc {
	assert(s.numbering.numbered_over == element_dim(et), "Space is not defined on the element dim.")
	return {et, s.family, s.order}
}


// Gather all dofs of the element, casting to T for mixed precision.
space_gather :: proc($T: typeid, s: Space_Vector, elem: Entity_ID, alloc := context.allocator) -> Cvec(T) {
	f := s.fields
	l2g := s.numbering.l2g[elem]
	assert(l2g != nil, "element is not in the space")
	cvec := cvec_create(T, len(l2g), f, alloc)
	for g, ldof in l2g {
		dst := cvec.data[ldof * f:][:f]
		for x, k in s.coeffs[int(g) * f:][:f] { dst[k] = cast(T)x }
	}
	return cvec
}

// Scatter local into coeffs, casting to f64.
space_scatter :: proc(s: Space_Vector, elem: Entity_ID, local: Cvec($T)) {
	f := s.fields
	l2g := s.numbering.l2g[elem]
	assert(l2g != nil, "element is not in the space")
	assert(len(l2g) * f == len(local.data))
	for g, ldof in l2g {
		src := local.data[ldof * f:][:f]
		for &x, k in s.coeffs[int(g) * f:][:f] { x = cast(f64)src[k] }
	}
}

// Like space_scatter, but only the dofs on `entity`'s closure `local` is still sized for the whole element.
space_scatter_closure :: proc(s: Space_Vector, elem: Entity_ID, entity: Sub_Entity, local: Cvec($T)) {
	f := s.fields
	bd := space_bd(s.space, element_type(s.space, elem))
	scratch_guard()
	for ldof in basis_closure_dofs(bd, entity, scratch()) {
		g := int(s.numbering.l2g[elem][ldof])
		src := local.data[ldof * f:][:f]
		for &x, k in s.coeffs[g * f:][:f] { x = cast(f64)src[k] }
	}
}

// Global dof indices of all dofs that are non-zero on the facet.
space_facet_dofs :: proc(s: ^Space, facet_id: Entity_ID, alloc := context.allocator) -> []i32 {
	if s.numbering.numbered_over != s.mesh.intrinsic_dim {
		return slice.clone(s.numbering.l2g[facet_id], alloc) // trace space: the facet is the element, all its dofs
	}
	cell, local_facet := facet_canonical_cell(s.mesh^, s.mesh.facets[facet_id])
	local := basis_closure_dofs(space_bd(s, cell.type), element_facet(cell.type, local_facet).entity, alloc)
	out := make([]i32, len(local), alloc)
	for ldof, i in local { out[i] = s.numbering.l2g[cell.id][ldof] }
	return out
}

//== Interpolation

MAX_INTERP_BLOCKS :: MAX_VERTICES + MAX_EDGES + MAX_FACES + 1

Interp_Block :: struct($A, $I, $F: int) {
	points:  Pvec(f64), // physical points, 1 x A
	tangent: Tangent(A, I, f64),
	out:     Pvec(f64), // filled by the user
}

Interpolator :: struct($A, $I, $F: int) {
	space:     Space_Vector,
	elem:      Entity_ID,
	bd:        Basis_Desc,
	geo_bd:    Basis_Desc,
	vector:    bool, // vector family (RT, Nedelc)
	value_map: Map_Type,
	affine:    bool,
	closure:   Sub_Entity,
	coords:    Cvec(f64), // geometry nodes of the cell
	vals:      Cvec(f64), // the cell's reference dof values, filled block by block
	blocks:    [MAX_INTERP_BLOCKS][2]int, // (dim, local entity) to visit
	n_blocks:  int,
	cur:       int,
	blk:       Interp_Block(A, I, F),
	temp:      Scratch_Temp,
}

// Interpolates a function given at physical points onto an FE space
interpolator :: proc(
	$A, $I, $F: int,
	geo, space: Space_Vector,
	elem: Entity_ID,
	closure: Maybe(Sub_Entity) = nil, // nil: the whole cell
) -> (
	ip: Interpolator(A, I, F),
) {
	assert(A == geo.fields && F == space.fields)
	assert(space.numbering.numbered_over == space.mesh.intrinsic_dim, "interpolation is over cells")
	cell := &space.mesh.cells[elem]
	assert(I == int(element_dim(cell.type)))

	ip.temp = scratch_begin_temp()
	ip.space = space
	ip.elem = elem
	ip.bd = space_bd(space.space, cell.type)
	ip.geo_bd = space_bd(geo.space, cell.type)
	ip.vector = .V_Val in BASIS_QUANTITIES[ip.bd.family]
	ip.value_map = basis_quantity_map(ip.bd, .V_Val if ip.vector else .S_Val)
	ip.affine = cell.affine
	ip.closure = closure.? or_else element_sub_entity(cell.type, element_dim(cell.type), 0)
	ip.coords = space_gather(f64, geo, elem, scratch())
	ip.vals = cvec_create(f64, basis_num_dofs(ip.bd), F, scratch())

	for d in Dimension {
		if basis_dofs_per_entity(ip.bd, d) == 0 { continue }
		for e in ip.closure.closure[d] {
			ip.blocks[ip.n_blocks] = {int(d), e}
			ip.n_blocks += 1
		}
	}
	ip.cur = -1
	return
}

// Yields the next sub-entity block to fill. Finishes the previous one first.
interpolator_next :: proc(ip: ^Interpolator($A, $I, $F)) -> (blk: Interp_Block(A, I, F), ok: bool) {
	context.allocator = scratch()

	if ip.cur >= ip.n_blocks { return }
	if ip.cur >= 0 { finish_block(ip) }
	ip.cur += 1
	if ip.cur >= ip.n_blocks { return }

	d, e := Dimension(ip.blocks[ip.cur][0]), ip.blocks[ip.cur][1]
	rule := basis_entity_functionals(ip.bd, d, e).rule
	geo := bstore_sub_entity(ip.geo_bd, d, e, rule) // the cell itself is its own sub-entity
	np := len(rule.points)

	ip.blk.points = pvec_create(f64, np, 1, A)
	contract_eval(1, A, ip.blk.points, ip.coords, geo[.S_Val])
	ip.blk.tangent = tangent_from_nodes(A, I, geo[.S_Grd], ip.coords, ip.affine)
	ip.blk.out = pvec_create(f64, np, A if ip.vector else 1, F)
	return ip.blk, true

	finish_block :: proc(ip: ^Interpolator($A, $I, $F)) {
		d, e := Dimension(ip.blocks[ip.cur][0]), ip.blocks[ip.cur][1]
		fn := basis_entity_functionals(ip.bd, d, e)
		ls, n := basis_entity_dof_range(ip.bd, d, e)
		dofs := Cvec(f64) {
			dofs   = n,
			fields = F,
			data   = ip.vals.data[ls * F:][:n * F],
		} 	// view into the cell's values

		if !ip.vector {
			contract_linear(1, F, dofs, fn.weights, ip.blk.out)
			return
		}

		cov: Piola_Cov(A, I, f64)
		if ip.value_map == .Contravariant { cov = piola_covariant(ip.blk.tangent) }

		ref := pvec_create(f64, ip.blk.out.points, I, F)
		for pt in 0 ..< ref.points {
			m: Small_Mat(A, I, f64)
			if ip.value_map == .Covariant {
				m = tangent_at(ip.blk.tangent, pt)^
			} else {
				m = piola_at(cov, pt)
				small_mat_scale_inplace(&m, tangent_measure(ip.blk.tangent, pt))
			}
			pvec_point_matrix(ref, pt, I, F)^ = small_mat_mul(pvec_point_matrix(ip.blk.out, pt, A, F)^, m)
		}
		contract_linear(I, F, dofs, fn.weights, ref)
	}
}

// Call once every block has been visited. Does not reset the iterator.
interpolator_flush :: proc(ip: ^Interpolator($A, $I, $F)) {
	defer scratch_end_temp(ip.temp)
	assert(ip.cur >= ip.n_blocks, "interpolator flushed before every block was visited")
	if ip.space.continuity == .Continuous {
		basis_orient_dofs(ip.bd, cell_entity_keys(&ip.space.mesh.cells[ip.elem]), ip.vals)
	}
	space_scatter_closure(ip.space, ip.elem, ip.closure, ip.vals)
}

//== Multi Space

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

constraint_periodic :: proc(
	periodicity: Periodicity,
	field_transform: Maybe(Dense_Matrix) = nil,
	leave_free: bit_set[0 ..< MAX_FIELDS] = {},
) -> Constraint {
	return Constraint_Periodic{periodicity, field_transform, leave_free}
}

Constituent_Space :: struct {
	space:       ^Space,
	constraints: []Constraint,
}

//== Multi-Space

// How constrained dofs appear in the solved system.
//  .Eliminate: removed. The system only has free dofs.
//  .Identity:  kept as rows u_i = rhs_i (1 on the diagonal), columns still lifted, so the matrix stays symmetric.
//              Call sys_constrain_rows after assembling.
Constraint_Mode :: enum {
	Eliminate,
	Identity,
}

Multi_Space :: struct {
	ranges:           []Space_Range,
	mode:             Constraint_Mode,
	total_state_size: int, // total dofs
	total_soln_size:  int, // dofs in the solved system
	dof_map:          []DOF_Map_Entry, // len == total_state_size, flat, global-indexed
	soln_to_dof:      []int, // soln idx -> global dof idx
	arena:            virtual.Arena,
}

Space_Range :: struct {
	base:  int,
	end:   int,
	space: ^Space,
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
	soln_index: int, // row in the solved system, -1 if eliminated
	terms:      []MPC_Term, // valid when Constrained, nil for a plain essential dof
}

Assembly_Mode :: enum {
	Linear,
	Newton,
}

// Flat state vector for all spaces in a multi space
State :: distinct []f64

// Combines constituent spaces into one flat dof numbering, applies constraints, and numbers the solved dofs.
ms_create :: proc(mode: Constraint_Mode, spaces: ..Constituent_Space) -> (ms: Multi_Space) {
	assert(len(spaces) >= 1)

	scratch_guard()

	numbered_over := spaces[0].space.numbering.numbered_over
	for cs in spaces {
		assert(cs.space.numbering.numbered_over == numbered_over, "Spaces must all be trace, or all be interior")
	}

	if err := virtual.arena_init_growing(&ms.arena); err != nil { panic("Failed to create arena") }
	context.allocator = virtual.arena_allocator(&ms.arena)

	ms.mode = mode
	ms.ranges = make([]Space_Range, len(spaces))

	offset := 0
	for cs, i in spaces {
		ms.ranges[i] = {
			base  = offset,
			end   = offset + cs.space.total_coeffs,
			space = cs.space,
		}
		offset += cs.space.total_coeffs
	}
	ms.total_state_size = offset
	ms.dof_map = make([]DOF_Map_Entry, ms.total_state_size)

	constrained := make([]bool, ms.total_state_size, scratch())

	// essential first
	for cs, i in spaces {
		for c in cs.constraints {
			if v, ok := c.(Constraint_Essential);
			   ok { mark_essential(cs.space, ms.ranges[i].base, v, ms.dof_map, constrained) }
		}
	}
	for cs, i in spaces {
		for c in cs.constraints {
			if v, ok := c.(Constraint_Periodic);
			   ok { mark_periodic(cs.space, ms.ranges[i].base, v, ms.dof_map, constrained) }
		}
	}

	flatten_periodic_chains(ms.dof_map)

	soln_idx := 0
	for gidx in 0 ..< ms.total_state_size {
		entry := &ms.dof_map[gidx]
		if !constrained[gidx] { entry.role = .Free }
		entry.soln_index = -1
		if entry.role == .Free || mode == .Identity {
			entry.soln_index = soln_idx
			soln_idx += 1
		}
	}
	ms.total_soln_size = soln_idx

	ms.soln_to_dof = make([]int, ms.total_soln_size)
	for entry, gidx in ms.dof_map {
		if entry.soln_index >= 0 { ms.soln_to_dof[entry.soln_index] = gidx }
	}

	return ms

	mark_essential :: proc(
		space: ^Space,
		base: int,
		c: Constraint_Essential,
		dof_map: []DOF_Map_Entry,
		constrained: []bool,
	) {
		for facet in space.mesh.facets {
			if facet.info.boundary not_in c.boundaries { continue }
			scratch_guard()
			for dof in space_facet_dofs(space, facet.id, scratch()) {
				for field in 0 ..< space.fields {
					if field in c.leave_free { continue }
					gidx := base + int(dof) * space.fields + field
					if constrained[gidx] { continue } 	// first essential write wins
					dof_map[gidx] = {
						role = .Constrained,
					}
					constrained[gidx] = true
				}
			}
		}
	}

	// Ties each slave facet dof to master facet dofs (optionally through field_transform), one sub-entity block
	// at a time. For facet sub-entity k of dimension d on the master, pair.maps[d][k] is the matching slave
	// sub-entity and pair.orientations[d][k] the key relating the two canonical orders: element_orientation(type,
	// master's canonical vertices as slave vertex ids, slave's canonical vertices). Slave block = M^-T master block.
	mark_periodic :: proc(
		space: ^Space,
		base: int,
		c: Constraint_Periodic,
		dof_map: []DOF_Map_Entry,
		constrained: []bool,
	) {
		assert(space.continuity == .Continuous, "periodic constraints need a continuous space")
		xform, has_xform := c.field_transform.?
		nf := space.fields

		for pair in c.periodicity.pairs {
			et := space.mesh.facets[pair.master].info.type
			cell, _ := facet_canonical_cell(space.mesh^, space.mesh.facets[pair.master])
			tables := Basis_Desc{cell.type, space.family, space.order} // orientation tables for the facet's blocks

			for d in Dimension {
				if d > element_dim(et) { break }
				for k in 0 ..< element_num_sub_entities(et, d) {
					m := facet_entity_dofs(space, pair.master, d, k)
					if len(m) == 0 { continue }
					sk := pair.maps[d][k] if d < element_dim(et) else 0
					s := facet_entity_dofs(space, pair.slave, d, sk)
					assert(len(s) == len(m), "periodic pair facets have mismatched dof counts")

					key := pair.orientations[d][k] if k < len(pair.orientations[d]) else 0
					t := basis_orientation(tables, element_sub_entity(et, d, k).type, key)
					n := len(m)

					for i in 0 ..< n {
						for field in 0 ..< nf {
							if field in c.leave_free { continue }
							sgidx := base + int(s[i]) * nf + field
							if constrained[sgidx] { continue } 	// already tied (by essential, or an earlier pair)

							// slave dof i = sum_l A[i, l] master dof l, A = M^-T
							terms := make([dynamic]MPC_Term, 0, n * nf)
							for l in 0 ..< n {
								a: f64
								switch t.kind {
								case .Identity: a = 1 if l == i else 0
								case .Signed_Perm: a = t.sign[i] if l == t.src[i] else 0
								case .Dense: a = dn_get(t.m_inv, l, i)^
								}
								if a == 0 { continue }

								if has_xform {
									for mfield in 0 ..< nf {
										w := dn_get(xform, field, mfield)^
										if w != 0 { append(&terms, MPC_Term{base + int(m[l]) * nf + mfield, a * w}) }
									}
								} else {
									append(&terms, MPC_Term{base + int(m[l]) * nf + field, a})
								}
							}

							dof_map[sgidx] = {
								role  = .Constrained,
								terms = terms[:],
							}
							constrained[sgidx] = true
						}
					}
				}
			}
		}
	}

	// Global dofs of sub-entity k (dimension d, facet-local order) of a facet, in canonical order.
	facet_entity_dofs :: proc(space: ^Space, facet: Entity_ID, d: Dimension, k: int) -> []i32 {
		if space.numbering.numbered_over != space.mesh.intrinsic_dim { 	// trace space: the facet is the element
			ls, n := basis_entity_dof_range(space_bd(space, space.mesh.facets[facet].info.type), d, k)
			return space.numbering.l2g[facet][ls:][:n]
		}
		cell, local_facet := facet_canonical_cell(space.mesh^, space.mesh.facets[facet])
		e := element_facet(cell.type, local_facet).closure[d][k]
		ls, n := basis_entity_dof_range(space_bd(space, cell.type), d, e)
		return space.numbering.l2g[cell.id][ls:][:n]
	}

	flatten_periodic_chains :: proc(dof_map: []DOF_Map_Entry) {
		scratch_guard()
		resolved := make([]bool, len(dof_map), scratch())
		visiting := make([]bool, len(dof_map), scratch())

		flatten_one :: proc(dof_map: []DOF_Map_Entry, resolved, visiting: []bool, gidx: int) {
			entry := &dof_map[gidx]
			if entry.role != .Constrained || entry.terms == nil || resolved[gidx] { return } 	// free / essential leaf / done
			assert(
				!visiting[gidx],
				"cyclic periodic constraint (two dofs periodically defined in terms of each other)",
			)
			visiting[gidx] = true

			flat := make([dynamic]MPC_Term, 0, len(entry.terms))
			for term in entry.terms {
				target := dof_map[term.dof]
				if target.role != .Constrained || target.terms == nil {
					append(&flat, term) // already terminal
				} else {
					flatten_one(dof_map, resolved, visiting, term.dof)
					for inner in dof_map[term.dof].terms {
						append(&flat, MPC_Term{inner.dof, inner.weight * term.weight})
					}
				}
			}
			entry.terms = flat[:]
			visiting[gidx] = false
			resolved[gidx] = true
		}

		for gidx in 0 ..< len(dof_map) { flatten_one(dof_map, resolved, visiting, gidx) }
	}
}

// Allocates fresh state buffer
ms_state_alloc :: proc(ms: Multi_Space, alloc := context.allocator) -> State {
	return make(State, ms.total_state_size, alloc)
}

// Returns the raw coefficient slice for `space` out of a flat State buffer.
ms_state_slice :: proc(ms: Multi_Space, state: State, space: ^Space) -> []f64 {
	r := ms_space_range(ms, space)
	return cast([]f64)state[r.base:r.end]
}

// Returns the (space, coeffs) view for `space` out of state.
ms_space_vec :: proc(ms: Multi_Space, state: State, space: ^Space) -> Space_Vector {
	return {space, ms_state_slice(ms, state, space)}
}

// Re-derives every constrained dof's value in `state` from `inhom` and its MPC terms.
ms_enforce_constraints :: proc(ms: Multi_Space, state: State, inhom: State) {
	rank_sync()
	for r in ms.ranges {
		sub := rank_range(r.end - r.base)
		for gidx in r.base + sub.min ..< r.base + sub.max {
			entry := ms.dof_map[gidx]
			if entry.role != .Constrained { continue }

			val := inhom[gidx]
			for term in entry.terms { val += term.weight * dof_value(ms, state, inhom, term.dof) }
			state[gidx] = val
		}
	}
	rank_sync()

	dof_value :: proc(ms: Multi_Space, state, inhom: State, gidx: int) -> f64 {
		entry := ms.dof_map[gidx]
		if entry.role == .Free { return state[gidx] }
		return inhom[gidx] // essential leaf
	}
}

// Allocates a vector sized for the solved system.
ms_soln_vector :: proc(ms: Multi_Space, alloc := context.allocator) -> Vector {
	return make(Vector, ms.total_soln_size, alloc)
}

// Applies state += update to the free dofs, then re-derives constrained dofs.
ms_apply_update :: proc(ms: Multi_Space, state: State, inhom: State, update: Vector) {
	for r in ms.ranges {
		sub := rank_range(r.end - r.base)
		for gidx in r.base + sub.min ..< r.base + sub.max {
			entry := ms.dof_map[gidx]
			if entry.role != .Free { continue }
			state[gidx] += update[entry.soln_index]
		}
	}
	ms_enforce_constraints(ms, state, inhom)
}

// Applies state = soln to the free dofs, then re-derives constrained dofs.
ms_apply_soln :: proc(ms: Multi_Space, state: State, inhom: State, soln: Vector) {
	for r in ms.ranges {
		sub := rank_range(r.end - r.base)
		for gidx in r.base + sub.min ..< r.base + sub.max {
			entry := ms.dof_map[gidx]
			if entry.role != .Free { continue }
			state[gidx] = soln[entry.soln_index]
		}
	}
	ms_enforce_constraints(ms, state, inhom)
}

// Construct an amgcl-conforming representation of the near nullspace from the given State vectors,
// covering exactly the rows of the solved system.
ms_near_null_space :: proc(ms: Multi_Space, vectors: ..State, alloc := context.allocator) -> (nns: []f64, cols: int) {
	cols = len(vectors)
	nns = make([]f64, ms.total_soln_size * cols, alloc)

	for entry, gidx in ms.dof_map {
		if entry.soln_index < 0 { continue }
		for v, col in vectors { nns[entry.soln_index * cols + col] = v[gidx] }
	}

	return
}

// Frees the multi space's arena.
ms_destroy :: proc(ms: ^Multi_Space) {
	virtual.arena_destroy(&ms.arena)
}

@(private)
ms_space_range :: proc(ms: Multi_Space, space: ^Space) -> Space_Range {
	for r in ms.ranges { if r.space == space { return r } }
	panic("space not in multi space")
}

//== Sys

// `across_facets`: couple the two cells on each side of every interior facet (DG face terms).
// `through_cells`: for HDG trace spaces
Coupling :: struct {
	test, trial:   ^Space,
	across_facets: bool,
	through_cells: bool,
}

Sys :: struct {
	ms:            Multi_Space,
	couplings:     []Coupling,
	couplings_set: map[[2]^Space]bool,
	sparsity:      Sparsity,
	arena:         virtual.Arena,
}

// Builds the sparsity pattern for exactly the declared (test, trial) couplings
// For a fully-coupled 2-variable problem (u, p), pass all four: {u,u}, {u,p}, {p,u}, {p,p}.
// Builds the sparsity pattern for exactly the declared (test, trial) couplings
// For a fully-coupled 2-variable problem (u, p), pass all four: {u,u}, {u,p}, {p,u}, {p,p}.
sys_create :: proc(ms: Multi_Space, couplings: ..Coupling) -> (sys: Sys) {
	assert(len(couplings) >= 1)

	virtual.arena_init_growing(&sys.arena) or_else panic("Failed to create arena")
	context.allocator = virtual.arena_allocator(&sys.arena)

	sys.ms = ms
	sys.couplings = make([]Coupling, len(couplings))
	copy(sys.couplings, couplings)

	sys.couplings_set = make(map[[2]^Space]bool, len(couplings))
	for c in couplings { sys.couplings_set[{c.test, c.trial}] = true }

	build_sparsity(&sys)

	return sys

	build_sparsity :: proc(sys: ^Sys) {
		Pair :: struct {
			row, col: i32,
		}

		scratch_guard()
		context.allocator = virtual.arena_allocator(&sys.arena)

		pairs := make([dynamic]Pair, scratch())
		rows := make([dynamic]int, scratch())
		cols := make([dynamic]int, scratch())

		add_block :: proc(ms: Multi_Space, c: Coupling, row_elem, col_elem: Entity_ID, rows, cols: ^[dynamic]int, pairs: ^[dynamic]Pair) {
			clear(rows)
			clear(cols)
			collect_elem(ms, c.test, row_elem, rows)
			collect_elem(ms, c.trial, col_elem, cols)
			for a in rows { for b in cols { append(pairs, Pair{i32(a), i32(b)}) } }
		}

		for c in sys.couplings {
			for e in 0 ..< len(c.test.numbering.l2g) { add_block(sys.ms, c, Entity_ID(e), Entity_ID(e), &rows, &cols, &pairs) }

			mesh := c.test.mesh
			if c.across_facets {
				assert(c.test.numbering.numbered_over == mesh.intrinsic_dim, "across_facets couples cells")
				for facet in mesh.facets {
					inc := facet_incidences(mesh^, facet)
					if len(inc) != 2 { continue }
					a, b := inc[0].cell, inc[1].cell
					add_block(sys.ms, c, a, b, &rows, &cols, &pairs)
					add_block(sys.ms, c, b, a, &rows, &cols, &pairs)
				}
			}
			if c.through_cells {
				assert(c.test.numbering.numbered_over != mesh.intrinsic_dim, "through_cells couples facets")
				for cell in mesh.cells {
					for a in cell.facets {
						for b in cell.facets { add_block(sys.ms, c, a, b, &rows, &cols, &pairs) }
					}
				}
			}
		}

		// add diagonal entries, this keeps some iterative solvers happy without blowing up sparsity for 0 blocks.
		// In .Identity mode these are also the constrained rows' identity entries.
		for i in 0 ..< sys.ms.total_soln_size { append(&pairs, Pair{i32(i), i32(i)}) }

		slice.sort_by(pairs[:], proc(a, b: Pair) -> bool {
			if a.row != b.row { return a.row < b.row }
			return a.col < b.col
		})

		row_ptrs := make([]i32, sys.ms.total_soln_size + 1)
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
		for row < i32(sys.ms.total_soln_size) {
			row += 1
			row_ptrs[row] = i32(len(columns))
		}

		sys.sparsity = {
			row_ptrs = row_ptrs,
			columns  = columns[:],
		}
	}

	// Solved-system rows an element's dofs of `space` land on (free dofs, or the free masters of constrained ones).
	collect_elem :: proc(ms: Multi_Space, space: ^Space, elem: Entity_ID, touched: ^[dynamic]int) {
		l2g := space.numbering.l2g[elem]
		if l2g == nil { return }
		base := ms_space_range(ms, space).base
		for gdof in l2g {
			for field in 0 ..< space.fields {
				entry := ms.dof_map[base + int(gdof) * space.fields + field]
				if entry.role == .Free {
					append(touched, entry.soln_index)
					continue
				}
				for term in entry.terms {
					if target := ms.dof_map[term.dof]; target.role == .Free { append(touched, target.soln_index) }
				}
			}
		}
	}
}

// Allocates a sparse matrix with the sparsity pattern of system
sys_soln_matrix :: proc(sys: Sys, alloc := context.allocator) -> Sparse_Matrix {
	return sparse_from_sparsity(sys.sparsity, alloc)
}

// Adds a local element vector (built with the oriented basis) for `test` at element `ent` into `vec`.
sys_scatter_vec :: proc(sys: Sys, vec: Vector, local: Cvec($T), test: ^Space, ent: Entity_ID) {
	base := ms_space_range(sys.ms, test).base
	l2g := test.numbering.l2g[ent]
	assert(len(l2g) == local.dofs && local.fields == test.fields)

	for gdof, ldof in l2g {
		lblock := cvec_dof_block(local, ldof)
		for field in 0 ..< local.fields {
			entry := sys.ms.dof_map[base + int(gdof) * test.fields + field]
			val := cast(f64)lblock[field]
			if entry.role == .Free {
				vec[entry.soln_index] += val
			} else {
				for term in entry.terms {
					if target := sys.ms.dof_map[term.dof];
					   target.role == .Free { vec[target.soln_index] += val * term.weight }
				}
			}
		}
	}
}

// Adds a local element matrix (built with the oriented basis) for (test, trial) at element `ent` into `mat`.
// Linear mode lifts constrained columns into `load` using `inhom`, Newton mode ignores both.
sys_scatter_mat :: proc(
	sys: Sys,
	mat: Sparse_Matrix,
	load: Vector,
	inhom: State,
	mode: Assembly_Mode,
	local: Cmat($T),
	test, trial: ^Space,
	ent: Entity_ID,
) {
	sys_scatter_mat_pair(sys, mat, load, inhom, mode, local, test, trial, ent, ent)
}

// Same, with rows from `test_ent` and columns from `trial_ent` (DG face terms between neighbouring cells;
// the coupling must be declared with across_facets).
sys_scatter_mat_pair :: proc(
	sys: Sys,
	mat: Sparse_Matrix,
	load: Vector,
	inhom: State,
	mode: Assembly_Mode,
	local: Cmat($T),
	test, trial: ^Space,
	test_ent, trial_ent: Entity_ID,
) {
	assert(mode == .Newton || load != nil, "Linear mode needs a load vector.")
	assert([2]^Space{test, trial} in sys.couplings_set, "pair was not declared in sys_create")

	test_base := ms_space_range(sys.ms, test).base
	trial_base := ms_space_range(sys.ms, trial).base
	rows := test.numbering.l2g[test_ent]
	cols := trial.numbering.l2g[trial_ent]
	assert(len(rows) == local.row_dofs && len(cols) == local.col_dofs)
	assert(local.row_fields == test.fields && local.col_fields == trial.fields)

	for rgdof, rldof in rows {
		for cgdof, cldof in cols {
			block := cmat_dof_block(local, rldof, cldof)
			for rf in 0 ..< local.row_fields {
				grow := test_base + int(rgdof) * test.fields + rf
				for cf in 0 ..< local.col_fields {
					gcol := trial_base + int(cgdof) * trial.fields + cf
					distribute(sys.ms, mat, load, inhom, mode, grow, gcol, cast(f64)block[cf * local.row_fields + rf])
				}
			}
		}
	}

	// Row/col are each at most one hop from end due to flattening, so no infinite recursion concerns.
	distribute :: proc(
		ms: Multi_Space,
		mat: Sparse_Matrix,
		load: Vector,
		inhom: State,
		mode: Assembly_Mode,
		grow, gcol: int,
		val: f64,
	) {
		rentry := ms.dof_map[grow]
		if rentry.role == .Constrained {
			for term in rentry.terms { distribute(ms, mat, load, inhom, mode, term.dof, gcol, val * term.weight) }
			return
		}

		centry := ms.dof_map[gcol]
		if centry.role == .Constrained {
			for term in centry.terms { distribute(ms, mat, load, inhom, mode, grow, term.dof, val * term.weight) }
			if mode == .Linear { load[rentry.soln_index] -= val * inhom[gcol] }
			return
		}

		p := sp_get(mat, rentry.soln_index, centry.soln_index)
		assert(p != nil, "column not found in sparsity pattern for row")
		p^ += val
	}
}

// .Identity mode: call after assembling. Gives each constrained dof the row u_i = rhs_i (1 on the diagonal).
// Linear: rhs_i = inhom_i. Newton: rhs_i = 0 (ms_apply_update re-derives constrained values anyway).
// No-op in .Eliminate mode.
sys_constrain_rows :: proc(sys: Sys, mat: Sparse_Matrix, rhs: Vector, inhom: State, mode: Assembly_Mode) {
	if sys.ms.mode != .Identity { return }
	for entry, gidx in sys.ms.dof_map {
		if entry.role != .Constrained { continue }
		i := entry.soln_index
		sp_get(mat, i, i)^ = 1
		if rhs != nil { rhs[i] = inhom[gidx] if mode == .Linear else 0 }
	}
}

// Frees the sys's arena. Does not affect the Multi_Space it was built from.
sys_destroy :: proc(sys: ^Sys) {
	virtual.arena_destroy(&sys.arena)
}
