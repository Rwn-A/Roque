package fe

/*
 Finite element space. Global discrete function space a quantity is discretized over.
*/

import "core:log"
import "core:mem"
import "core:mem/virtual"
import "core:slice"

Space_Continuity :: enum {
	Continuous,
	Discontinuous,
}

Numbered_Entity :: enum {
	Cell, // most fe spaces
	Facet, // trace spaces
}

MAX_FIELDS :: 32

Space_Desc :: struct {
	family:     Basis_Family,
	order:      Order,
	fields:     int,
	continuity: Space_Continuity,
	regions:    Region_Set,
}

DOF_Numbering :: struct {
	l2g:           [][]i32,
	flip_sign:     [][]bool, // parallel indexing to l2g, whole thing is optional.
	num_dofs:      int,
	numbered_over: Dimension, // cell dimension for the geenral case, trace spaces would be lower
}

Space_ID :: distinct int

Space :: struct {
	using sd:       Space_Desc,
	numbering:      DOF_Numbering,
	owns_numbering: bool,
	total_coeffs:   int,
	id:             Space_ID, // automatically assigned, for identifying spaces in a multi space.
	allocator:      mem.Allocator,
}

Space_State :: struct {
	using space: ^Space,
	data:        []f64, // len space.total_coeffs
}

State_Map :: map[Space_ID][]f64


@(thread_local, private)
next_id: Space_ID


space_new :: proc(mesh: Mesh, sd: Space_Desc, nent := Numbered_Entity.Cell, alloc := context.allocator) -> ^Space {
	s := new(Space, alloc)

	// TODO.

	return s
}

space_from_existing :: proc(existing: Space, fields: int, alloc := context.allocator) -> ^Space {
	s := new(Space, alloc)

	s.numbering = existing.numbering
	s.sd = existing.sd
	s.fields = fields
	s.total_coeffs = s.numbering.num_dofs * fields
	s.owns_numbering = false
	s.id = next_id
	s.allocator = alloc

	next_id += 1

	return s
}

space_isoparemetric :: proc(mesh: Mesh, fields: int, alloc := context.allocator) -> ^Space {
	s := new(Space, alloc)

	s.numbering.l2g = mesh.cell_nodes
	s.numbering.num_dofs = len(mesh.nodes)
	s.numbering.numbered_over = mesh.intrinsic_dim
	// no arena or anything, isnt owned anyway so wont be attempted to be freed

	s.fields = fields
	s.total_coeffs = len(mesh.nodes) * fields
	s.owns_numbering = false
	s.id = next_id
	s.allocator = alloc

	next_id += 1

	return s

}

// Frees the space and the DOF numbering if the space owns its numbering (created with `space_create`).
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
		free(s)
	}
}


space_bd :: proc(s: ^Space, et: Element_Type) -> Basis_Desc {
	assert(s.numbering.numbered_over == element_dim(et), "Space is not defined on the element dim.")
	return {et, s.family, s.order}
}

space_state_from_map :: proc(s: ^Space, state_map: State_Map) -> Space_State {
	return {s, state_map[s.id]}
}

space_gather :: proc($T: typeid, s: Space_State, ent: Entity_ID, alloc := context.allocator) -> Cvec(T) {
	l2g := s.numbering.l2g[ent]
	assert(l2g != nil, "Space may not be defined on entity")

	cvec := cvec_create(T, len(l2g), s.fields, alloc)

	for gdof, ldof in l2g {
		cvb := cvec_dof_block(cvec, ldof)
		sign: T = -1.0 if s.numbering.flip_sign != nil && s.numbering.flip_sign[ent][ldof] else 1.0
		for entry, i in s.data[gdof * i32(s.fields):][:s.fields] {
			cvb[i] = cast(T)entry * sign
		}
	}

	return cvec
}

space_scatter :: proc(s: Space_State, ent: Entity_ID, local: Cvec($T)) {
	l2g := s.numbering.l2g[ent]
	assert(l2g != nil, "Space may not be defined on entity")
	assert(len(l2g) * s.fields == len(local.data))

	for gdof, ldof in l2g {
		cvb := cvec_dof_block(local, ldof)
		sign: T = -1.0 if s.numbering.flip_sign != nil && s.numbering.flip_sign[ent][ldof] else 1.0
		for &entry, i in s.data[gdof * i32(s.fields):][:s.fields] {
			entry = cast(f64)cvb[i] * sign
		}
	}
}

space_scatter_restricted :: proc(s: Space_State, ent: Entity_ID, local: Cvec($T), restriction: []int) {
	l2g := s.numbering.l2g[ent]
	assert(l2g != nil, "Space may not be defined on entity")
	assert(len(l2g) * s.fields == len(local.data))

	for ldof, rdof in restriction {
		gdof := l2g[ldof]
		cvb := cvec_dof_block(local, ldof)
		sign: T = -1.0 if s.numbering.flip_sign != nil && s.numbering.flip_sign[ent][ldof] else 1.0
		for &entry, i in s.data[gdof * i32(s.fields):][:s.fields] {
			entry = cast(f64)cvb[i] * sign
		}
	}
}

// Deletes the map, and each coefficient buffer
state_map_destroy :: proc(s: State_Map) {
	for k, &v in s { delete(v, s.allocator) }
	delete(s)
}

//== Interpolator

Interpolator :: struct($A, $I: int) {
	geometry:      Space_State,
	space:         Space_State,
	restriction:   []int,
	ent:           Entity_ID,
	bd:            Basis_Desc,
	vals:          Cvec(f64),
	current_dof:   int,
	current_point: int,
	n_dofs:        int,
	jacobians:     Pvec(f64),
	phys_points:   Pvec(f64),
	temp:          Scratch_Temp,
}

interpolator :: proc(
	$A, $I: int,
	geo, space: Space_State,
	ent: Entity_ID,
	et: Element_Type,
	rstrct: []int = nil,
) -> Interpolator(A, I) {
	bd := space_bd(space.space, et)
	num_dofs := len(rstrct) if rstrct != nil else basis_count(bd)


	return Interpolator(A, I) {
		geometry = geo,
		space = space,
		restriction = rstrct,
		ent = ent,
		n_dofs = num_dofs,
		bd = bd,
		vals = cvec_create(f64, basis_count(bd), space.fields, scratch()),
		current_dof = -1,
		temp = scratch_begin_temp(),
	}

}

interpolator_next :: proc(
	ip: ^Interpolator($A, $I),
) -> (
	jac: Small_Mat(A, I, f64),
	point: Small_Vec(A, f64),
	out: []f64,
	ok: bool, // <- whole field block now, not ^f64 into field 0
) {
	POINT_DIMS :: Contraction_Dims {
		.CMPNTS = 1,
		.FIELDS = A,
	}
	JAC_DIMS :: Contraction_Dims {
		.CMPNTS = I,
		.FIELDS = A,
	}

	scratch_end_temp(ip.temp)
	ip.temp = scratch_begin_temp()

	context.allocator = scratch()

	num_dofs := ip.n_dofs


	for ip.current_point >= ip.phys_points.points {
		ip.current_dof += 1
		if ip.current_dof >= num_dofs {
			return {}, {}, nil, false
		}

		dof := ip.restriction[ip.current_dof] if ip.restriction != nil else ip.current_dof

		rule := basis_functional_rule(ip.bd, dof)


		geo_bd := space_bd(ip.geometry, ip.bd.element)

		geo_grads: Ref_Basis_Tbl
		geo_vals: Ref_Basis_Tbl
		if rule.element != ip.bd.element {
			assert(element_dim(rule.element) == element_facet_dim(ip.bd.element))
			sup := basis_support(ip.bd)[dof]
			assert(sup.entity_dim == element_dim(rule.element))
			geo_grads = bstore_get_facet(geo_bd, sup.entity_index, rule, .Scalar_Gradient)
			geo_vals = bstore_get_facet(geo_bd, sup.entity_index, rule, .Scalar)
		} else {
			geo_grads = bstore_get_interior(geo_bd, rule, .Scalar_Gradient)
			geo_vals = bstore_get_interior(geo_bd, rule, .Scalar)
		}

		coords := space_gather(f64, ip.geometry, ip.ent)

		ip.phys_points = pvec_create(f64, len(rule.ref_points), POINT_DIMS)
		ip.jacobians = pvec_create(f64, len(rule.ref_points), JAC_DIMS)

		contract_eval(POINT_DIMS, ip.phys_points, coords, geo_vals)
		contract_eval(JAC_DIMS, ip.jacobians, coords, geo_grads)
		ip.current_point = 0

		block := cvec_dof_block(ip.vals, dof)
		for &v in block { v = 0 }
	}

	point = small_vec_view_from_slice(pvec_at_point(ip.phys_points, ip.current_point).data, A)^
	jac = small_mat_view_from_slice(pvec_at_point(ip.jacobians, ip.current_point).data, A, I)^

	dof := ip.restriction[ip.current_dof] if ip.restriction != nil else ip.current_dof

	out = cvec_dof_block(ip.vals, dof)

	ip.current_point += 1
	return jac, point, out, true
}

interpolator_flush :: proc(ip: ^Interpolator($A, $I)) {
	if ip.restriction != nil {
		space_scatter_restricted(ip.space, ip.ent, ip.vals, ip.restriction)
	} else {
		space_scatter(ip.space, ip.ent, ip.vals)
	}
}

//== Multi Space

Constraint_Essential :: struct {
	boundaries: Boundary_Set,
	leave_free: bit_set[0 ..< MAX_FIELDS], // for partial constraints
}

Constraint :: union {
	Constraint_Essential,
}

Constiuent_Space :: struct {
	space:       ^Space,
	constraints: []Constraint,
}

Multi_Space :: struct {
	spaces:           map[Space_ID]^Space,
	ranges:           []Space_Range, // sorted, contiguous, one entry per space -- only used at the State_Map boundary
	total_state_size: int, // total dofs before elimination
	total_soln_size:  int, // free dofs after elimination
	dof_map:          []DOF_Map_Entry, // len == total_state_size, flat, global-indexed
	soln_to_dof:      []int, // soln idx -> global dof idx
	sparsity:         Sparsity,
	inhomogeneity:    []f64, // len == total_state_size, flat
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
	terms:      []MPC_Term, // valid when Constrained; nil for a plain essential dof
}

ms_create :: proc(mesh: Mesh, spaces: ..Constiuent_Space) -> (ms: Multi_Space) {
	assert(len(spaces) >= 1)

	scratch_guard()

	numbered_over := spaces[0].space.numbering.numbered_over
	for space in spaces {
		assert(space.space.numbering.numbered_over == numbered_over, "Spaces must all be trace, or all be interior")
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
		if ms.dof_map[gidx].role == .Free {
			ms.soln_to_dof[ms.dof_map[gidx].soln_index] = gidx
		}
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
			cell, local_facet := facet_canonical_cell(mesh, facet)

			if space.numbering.numbered_over == element_dim(cell.type) {
				l2g := space.numbering.l2g[cell.id]
				restriction := basis_facet_restriction(space_bd(space, cell.type), local_facet)
				for ldof in restriction {
					gdof := l2g[ldof]
					for field in 0 ..< space.fields {
						if field in c.leave_free { continue }
						gidx := base + int(gdof) * space.fields + field
						dof_map[gidx] = {
							role  = .Constrained,
							terms = nil,
						}
						constrained[gidx] = true
					}
				}
			} else {
				l2g := space.numbering.l2g[facet.id]
				for gdof in l2g {
					for field in 0 ..< space.fields {
						if field in c.leave_free { continue }
						gidx := base + int(gdof) * space.fields + field
						dof_map[gidx] = {
							role  = .Constrained,
							terms = nil,
						}
						constrained[gidx] = true
					}
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

ms_inhomogeneity :: proc(ms: Multi_Space, space: Space_ID) -> []f64 {
	for r in ms.ranges { if r.id == space { return ms.inhomogeneity[r.base:r.end] } }
	panic("space not in multi space")
}

// ensures coeffs in state for all spaces in the multi space satisfy the set constraints.
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


ms_destroy :: proc(ms: ^Multi_Space) {
	virtual.arena_destroy(&ms.arena)
}


@(private)
ms_build_sparsity :: proc(ms: ^Multi_Space) {
	Pair :: struct {
		row, col: i32,
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
					entry := ms.dof_map[gidx]
					if entry.role == .Free {
						append(&touched, entry.soln_index)
					}
				}
			}
		}

		for a in touched {
			for b in touched {
				append(&pairs, Pair{i32(a), i32(b)})
			}
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

	ms.sparsity = {
		row_ptrs = row_ptrs,
		columns  = columns[:],
	}
}

@(private)
ms_range_base :: proc(ms: Multi_Space, id: Space_ID) -> int {
	for r in ms.ranges { if r.id == id { return r.base } }
	panic("space not in multi space")
}

@(private)
ms_distribute_vec :: proc(ms: Multi_Space, vec: Vector, gidx: int, val: f64) {
	entry := ms.dof_map[gidx]
	if entry.role == .Free {
		vec[entry.soln_index] += val
		return
	}
	for term in entry.terms {
		ms_distribute_vec(ms, vec, term.dof, val * term.weight)
	}
}

ms_scatter_vec :: proc(ms: Multi_Space, vec: Vector, local: Cvec($T), test: Space_ID, ent: Entity_ID) {
	space := ms.spaces[test]
	l2g := space.numbering.l2g[ent]
	assert(l2g != nil, "Space may not be defined on entity")
	assert(len(l2g) == local.dofs)
	assert(local.fields == space.fields)

	base := ms_range_base(ms, test)

	for gdof, ldof in l2g {
		sign: f64 = -1.0 if space.numbering.flip_sign != nil && space.numbering.flip_sign[ent][ldof] else 1.0
		lblock := cvec_dof_block(local, ldof)
		for field in 0 ..< local.fields {
			gidx := base + int(gdof) * space.fields + field
			ms_distribute_vec(ms, vec, gidx, cast(f64)lblock[field] * sign)
		}
	}
}

@(private)
ms_distribute_mat :: proc(ms: Multi_Space, mat: Sparse_Matrix, grow, gcol: int, val: f64) {
	rentry := ms.dof_map[grow]
	if rentry.role == .Constrained {
		for term in rentry.terms {
			ms_distribute_mat(ms, mat, term.dof, gcol, val * term.weight)
		}
		return
	}

	centry := ms.dof_map[gcol]
	if centry.role == .Constrained {
		for term in centry.terms {
			ms_distribute_mat(ms, mat, grow, term.dof, val * term.weight)
		}
		return
	}

	p := sp_get(mat, rentry.soln_index, centry.soln_index)
	assert(p != nil, "column not found in sparsity pattern for row")
	p^ += val
}

ms_scatter_mat :: proc(ms: Multi_Space, mat: Sparse_Matrix, local: Cmat($T), test, trial: Space_ID, ent: Entity_ID) {
	tspace := ms.spaces[test]
	rspace := ms.spaces[trial]

	rows := tspace.numbering.l2g[ent]
	cols := rspace.numbering.l2g[ent]
	assert(rows != nil && cols != nil, "Space may not be defined on entity")
	assert(len(rows) == local.row_dofs && len(cols) == local.col_dofs)
	assert(local.row_fields == tspace.fields && local.col_fields == rspace.fields)

	tbase := ms_range_base(ms, test)
	rbase := ms_range_base(ms, trial)

	for rgdof, rldof in rows {
		rsign: f64 = -1.0 if tspace.numbering.flip_sign != nil && tspace.numbering.flip_sign[ent][rldof] else 1.0
		for cgdof, cldof in cols {
			csign: f64 = -1.0 if rspace.numbering.flip_sign != nil && rspace.numbering.flip_sign[ent][cldof] else 1.0
			sign := rsign * csign
			block := cmat_dof_block(local, rldof, cldof)

			for rf in 0 ..< local.row_fields {
				grow := tbase + int(rgdof) * tspace.fields + rf
				for cf in 0 ..< local.col_fields {
					gcol := rbase + int(cgdof) * rspace.fields + cf
					val := cast(f64)block[rf * local.col_fields + cf] * sign
					ms_distribute_mat(ms, mat, grow, gcol, val)
				}
			}
		}
	}
}
