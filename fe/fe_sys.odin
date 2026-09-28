package fe

/*
 A system: spaces numbered into one state with their constraints applied, and the matrices and vectors solved on it.
*/

import "core:mem"
import "core:mem/virtual"
import "core:slice"

// How constrained dofs appear in the solved system.
Constraint_Mode :: enum {
	Eliminate, // removed, only free dofs are solved
	Identity, // kept as rows u_i = rhs_i (sys_constrain_matrix, sys_constrain_rhs), node blocks stay aligned for AMG block sizes
}

DOF_Role :: enum {
	Free,
	Essential, // fixed to its inhom value
	Tied, // inhom value + a combination of other dofs
}

// Spaces combined into one numbering with their constraints applied. Solved rows are numbered block by block.
Sys :: struct {
	spaces:        []^Space,
	space_offsets: []int, // space i's state is [space_offsets[i], space_offsets[i + 1])
	mode:          Constraint_Mode,
	dof_map:       []DOF_Map_Entry, // per state index
	soln_to_dof:   []int, // solved row -> state index
	n_soln:        int, // rows of the solved system
	block_offsets: []int, // block b's solved rows are [block_offsets[b], block_offsets[b + 1])
	arena:         virtual.Arena,
}

Sys_Space :: struct {
	space:       ^Space,
	constraints: []Constraint,
	block:       int, // solve block
}

Constraint :: union {
	Constraint_Essential,
	Constraint_Tie,
}

// Dofs on the closure of facets tagged `tags` are fixed to their inhom values.
Constraint_Essential :: struct {
	tags:       Tag_Set,
	leave_free: bit_set[0 ..< MAX_FIELDS], // fields left unconstrained
}

// Each linked slave entity's dofs = its master entity's dofs, oriented by the link key and optionally transformed per
// field (directional quantities, sized for all fields).
// Periodic: the mesh periodicity's links. Interface: mesh_interface_links with `master` the other region's space.
Constraint_Tie :: struct {
	links:           [Dimension][]Entity_Link,
	master:          ^Space,
	field_transform: Maybe(Dense_Matrix),
	leave_free:      bit_set[0 ..< MAX_FIELDS],
}

DOF_Map_Entry :: struct {
	role:       DOF_Role,
	soln_index: int, // row in the solved system, -1 if eliminated
	block:      int, // solve block of the dof's space
	terms:      []MPC_Term, // tied dofs: value = inhom + sum of terms, each a free or essential dof
}

MPC_Term :: struct {
	dof:    int,
	weight: f64,
}

// Every field of every space, space after space, dof-major within a space.
State :: []f64

// A matrix being assembled. While `build` is set scatter inserts missing entries, sys_matrix_finalize then
// compresses the rows into `bsp`. Build rows grow inside one arena, so the first assembly holds up to a few times
// the final storage until finalize frees it.
System_Matrix :: struct {
	bsp:   Block_Sparse,
	build: ^Matrix_Build, // nil once finalized
}

// Rows of a system matrix being built, columns in scatter order with repeats, merged when a row grows and by finalize.
Matrix_Build :: struct {
	cols:  [][][dynamic]i32, // [block][row]
	vals:  [][][dynamic]f64,
	alloc: mem.Allocator, // for the finalized blocks
	arena: virtual.Arena, // row storage
}

// Numbers the spaces into one state, applies their constraints and numbers the solved dofs block by block.
sys_create :: proc(spaces: ..Sys_Space, mode := Constraint_Mode.Eliminate) -> (sys: Sys) {
	assert(len(spaces) >= 1)

	virtual.arena_init_growing(&sys.arena) or_else panic("Failed to create arena")
	context.allocator = virtual.arena_allocator(&sys.arena)

	sys.mode = mode
	sys.spaces = make([]^Space, len(spaces))
	sys.space_offsets = make([]int, len(spaces) + 1)
	for ss, i in spaces {
		sys.spaces[i] = ss.space
		sys.space_offsets[i + 1] = sys.space_offsets[i] + ss.space.total_coeffs
	}
	sys.dof_map = make([]DOF_Map_Entry, sys.space_offsets[len(spaces)])

	// essential first, ties skip dofs already fixed
	for ss, i in spaces {
		for c in ss.constraints {
			v := c.(Constraint_Essential) or_continue
			mark_essential(ss.space, sys.space_offsets[i], v, sys.dof_map)
		}
	}

	for ss, i in spaces {
		for c in ss.constraints {
			v := c.(Constraint_Tie) or_continue
			master_base := sys.space_offsets[i] if v.master == nil else sys_space_base(sys, v.master)
			mark_tie(ss.space, sys.space_offsets[i], v, master_base, sys.dof_map)
		}
	}

	flatten_terms(sys.dof_map)

	n_blocks := 0
	for ss in spaces { n_blocks = max(n_blocks, ss.block + 1) }
	sys.block_offsets = make([]int, n_blocks + 1)
	for b in 0 ..< n_blocks {
		sys.block_offsets[b] = sys.n_soln
		used := false
		for ss, i in spaces {
			if ss.block != b { continue }
			used = true
			for gidx in sys.space_offsets[i] ..< sys.space_offsets[i + 1] {
				entry := &sys.dof_map[gidx]
				entry.block = b
				entry.soln_index = -1
				if entry.role == .Free || mode == .Identity {
					entry.soln_index = sys.n_soln
					sys.n_soln += 1
				}
			}
		}
		assert(used, "every block needs at least one space")
	}
	sys.block_offsets[n_blocks] = sys.n_soln

	sys.soln_to_dof = make([]int, sys.n_soln)
	for entry, gidx in sys.dof_map {
		if entry.soln_index >= 0 { sys.soln_to_dof[entry.soln_index] = gidx }
	}
	return

	mark_essential :: proc(space: ^Space, base: int, c: Constraint_Essential, dof_map: []DOF_Map_Entry) {
		scratch_guard()
		fd := mesh_facet_dim(space.mesh^)
		dofs := make([dynamic]int, scratch())
		for facet in space.mesh.facets {
			if facet.tags & c.tags == {} { continue }
			clear(&dofs)
			if space.over == .Facets {
				for dof in space.numbering.l2g[facet.id] { append(&dofs, int(dof)) }
			} else if space.continuity == .Continuous {
				for d in Dimension {
					if d > fd { break }
					for gid in facet.conn[d] {
						start := space.numbering.entity_dofs[d][gid]
						if start < 0 { continue }
						count := space.numbering.entity_counts[mesh_entity_type(space.mesh^, d, gid)]
						for i in 0 ..< count { append(&dofs, int(start) + i) }
					}
				}
			} else {
				for cf in facet.cofaces {
					l2g := space.numbering.l2g[cf.entity]
					if l2g == nil { continue }
					bd := space_bd(space, &space.mesh.cells[cf.entity])
					for ldof in basis_closure_dofs(bd, fd, cf.local) { append(&dofs, int(l2g[ldof])) }
				}
			}

			for dof in dofs {
				for field in 0 ..< space.fields {
					if field in c.leave_free { continue }
					dof_map[base + dof * space.fields + field].role = .Essential
				}
			}
		}
	}

	// Slave entity block = M^-T master entity block (optionally through field_transform), M the link key's transform.
	mark_tie :: proc(space: ^Space, base: int, c: Constraint_Tie, master_base: int, dof_map: []DOF_Map_Entry) {
		master := c.master if c.master != nil else space
		assert(space.continuity == .Continuous && master.continuity == .Continuous, "ties need continuous spaces")
		assert(
			space.family == master.family && space.order == master.order && space.fields == master.fields,
			"tied spaces must match",
		)
		xform, has_xform := c.field_transform.?
		nf := space.fields

		// a cell basis holding each entity type, for its orientation transforms
		tables: [Element_Type]Basis_Desc
		for ct in space.mesh.encountered_cell_types {
			for d in Dimension {
				if d > element_dim(ct) { break }
				for k in 0 ..< element_n_sub_entities(ct, d) {
					tables[element_sub_entity(ct, d, k).type] = {ct, space.family, space.order}
				}
			}
		}

		for d in Dimension {
			for link in c.links[d] {
				m0, s0 := master.numbering.entity_dofs[d][link.master], space.numbering.entity_dofs[d][link.slave]
				if m0 < 0 || s0 < 0 { continue }
				et := mesh_entity_type(space.mesh^, d, link.master)
				n := space.numbering.entity_counts[et]
				t := basis_orientation(tables[et], et, link.key)

				for i in 0 ..< n {
					for field in 0 ..< nf {
						if field in c.leave_free { continue }
						sgidx := base + (int(s0) + i) * nf + field
						if dof_map[sgidx].role != .Free { continue } 	// already fixed or tied

						// slave dof i = sum_l A[i, l] master dof l, A = M^-T
						terms := make([dynamic]MPC_Term, 0, n * nf)
						for l in 0 ..< n {
							a := orientation_entry(t, .M_Inv_T, i, l)
							if a == 0 { continue }
							if has_xform {
								for mfield in 0 ..< nf {
									w := dn_get(xform, field, mfield)^
									if w != 0 { append(&terms, MPC_Term{master_base + (int(m0) + l) * nf + mfield, a * w}) }
								}
							} else {
								append(&terms, MPC_Term{master_base + (int(m0) + l) * nf + field, a})
							}
						}
						dof_map[sgidx] = {
							role  = .Tied,
							terms = terms[:],
						}
					}
				}
			}
		}
	}

	// Terms that point at tied dofs are replaced by those dofs' terms, so every term is a free or essential dof.
	flatten_terms :: proc(dof_map: []DOF_Map_Entry) {
		scratch_guard()
		resolved := make([]bool, len(dof_map), scratch())
		visiting := make([]bool, len(dof_map), scratch())

		flatten_one :: proc(dof_map: []DOF_Map_Entry, resolved, visiting: []bool, gidx: int) {
			entry := &dof_map[gidx]
			if entry.role != .Tied || resolved[gidx] { return }
			assert(!visiting[gidx], "cyclic constraint: dofs defined in terms of each other")
			visiting[gidx] = true

			flat := make([dynamic]MPC_Term, 0, len(entry.terms))
			for term in entry.terms {
				if dof_map[term.dof].role != .Tied {
					append(&flat, term)
					continue
				}
				flatten_one(dof_map, resolved, visiting, term.dof)
				for inner in dof_map[term.dof].terms { append(&flat, MPC_Term{inner.dof, inner.weight * term.weight}) }
			}
			entry.terms = flat[:]
			visiting[gidx] = false
			resolved[gidx] = true
		}

		for gidx in 0 ..< len(dof_map) { flatten_one(dof_map, resolved, visiting, gidx) }
	}
}

sys_destroy :: proc(systems: ..^Sys) {
	for sys in systems { virtual.arena_destroy(&sys.arena) }
}

// Allocates a state, every space's coefficients.
sys_state :: proc(sys: Sys, alloc := context.allocator) -> State {
	return make(State, sys.space_offsets[len(sys.spaces)], alloc)
}

// Allocates a vector sized for the solved system.
sys_vector :: proc(sys: Sys, alloc := context.allocator) -> Vector {
	return make(Vector, sys.n_soln, alloc)
}

// A system matrix to assemble into, its pattern built by the first assembly. The finalized blocks go in `alloc`.
sys_matrix :: proc(sys: Sys, alloc := context.allocator) -> (m: System_Matrix) {
	n := len(sys.block_offsets) - 1
	m.bsp = {
		n       = n,
		offsets = slice.clone(sys.block_offsets, alloc),
		blocks  = make([]Sparse_Matrix, n * n, alloc),
	}

	m.build = new(Matrix_Build, alloc)
	m.build.alloc = alloc
	virtual.arena_init_growing(&m.build.arena) or_else panic("Failed to create arena")
	rows_alloc := virtual.arena_allocator(&m.build.arena)
	m.build.cols = make([][][dynamic]i32, n * n, rows_alloc)
	m.build.vals = make([][][dynamic]f64, n * n, rows_alloc)
	for i in 0 ..< n {
		rows := sys.block_offsets[i + 1] - sys.block_offsets[i]
		for j in 0 ..< n {
			m.build.cols[i * n + j] = make([][dynamic]i32, rows, rows_alloc)
			m.build.vals[i * n + j] = make([][dynamic]f64, rows, rows_alloc)
			for r in 0 ..< rows {
				m.build.cols[i * n + j][r] = make([dynamic]i32, rows_alloc)
				m.build.vals[i * n + j][r] = make([dynamic]f64, rows_alloc)
			}
		}
	}
	return
}

// The space's coefficients within `state`.
sys_space_coeffs :: proc(sys: Sys, state: State, space: ^Space) -> Space_Coeffs {
	base := sys_space_base(sys, space)
	return {space, cast([]f64)state[base:][:space.total_coeffs]}
}

// Adds a local element vector for `test`'s element at `site` into `v`.
sys_scatter_vec :: proc(sys: Sys, v: Vector, local: Cvec($T), test: ^Space, site: Site) {
	base := sys_space_base(sys, test)
	e, _ := space_elem(test, site)
	l2g := test.numbering.l2g[e.id]
	assert(len(l2g) == local.dofs && local.fields == test.fields)

	for gdof, ldof in l2g {
		lblock := cvec_dof_block(local, ldof)
		for field in 0 ..< local.fields {
			entry := sys.dof_map[base + int(gdof) * test.fields + field]
			val := cast(f64)lblock[field]
			if entry.role == .Free {
				v[entry.soln_index] += val
				continue
			}
			for term in entry.terms {
				if target := sys.dof_map[term.dof]; target.role == .Free {
					v[target.soln_index] += val * term.weight
				}
			}
		}
	}
}

// Adds a local element matrix for (test, trial) at `site` into `mat`, each space's rows or columns from its own
// element there. With `rhs` and `inhom`, constrained columns are lifted into the rhs. Without them they are dropped,
// right for Newton updates and eigen matrices, but a linear solve then sees homogeneous constraints.
sys_scatter_mat :: proc(
	sys: Sys,
	mat: System_Matrix,
	local: Cmat($T),
	test, trial: ^Space,
	site: Site,
	rhs: Vector = nil,
	inhom: State = nil,
) {
	sys_scatter_mat_pair(sys, mat, local, test, trial, site, site, rhs, inhom)
}

// Same, with rows from `test_site` and columns from `trial_site` (terms across a facet, between two cells).
sys_scatter_mat_pair :: proc(
	sys: Sys,
	mat: System_Matrix,
	local: Cmat($T),
	test, trial: ^Space,
	test_site, trial_site: Site,
	rhs: Vector = nil,
	inhom: State = nil,
) {
	assert((rhs == nil) == (inhom == nil), "lifting needs both rhs and inhom")

	test_base, trial_base := sys_space_base(sys, test), sys_space_base(sys, trial)
	test_elem, _ := space_elem(test, test_site)
	trial_elem, _ := space_elem(trial, trial_site)
	rows := test.numbering.l2g[test_elem.id]
	cols := trial.numbering.l2g[trial_elem.id]
	assert(len(rows) == local.row_dofs && len(cols) == local.col_dofs)
	assert(local.row_fields == test.fields && local.col_fields == trial.fields)

	for rgdof, rldof in rows {
		for cgdof, cldof in cols {
			block := cmat_dof_block(local, rldof, cldof)
			for rf in 0 ..< local.row_fields {
				grow := test_base + int(rgdof) * test.fields + rf
				for cf in 0 ..< local.col_fields {
					gcol := trial_base + int(cgdof) * trial.fields + cf
					distribute(sys, mat, rhs, inhom, grow, gcol, cast(f64)block[cf * local.row_fields + rf])
				}
			}
		}
	}

	// Tied rows and columns go to their terms, essential ones are dropped (lifted for columns).
	distribute :: proc(sys: Sys, mat: System_Matrix, rhs: Vector, inhom: State, grow, gcol: int, val: f64) {
		rentry := sys.dof_map[grow]
		if rentry.role != .Free {
			for term in rentry.terms { distribute(sys, mat, rhs, inhom, term.dof, gcol, val * term.weight) }
			return
		}

		centry := sys.dof_map[gcol]
		if centry.role != .Free {
			for term in centry.terms { distribute(sys, mat, rhs, inhom, grow, term.dof, val * term.weight) }
			if rhs != nil { rhs[rentry.soln_index] -= val * inhom[gcol] }
			return
		}

		rb, cb := rentry.block, centry.block
		row, col := rentry.soln_index - mat.bsp.offsets[rb], centry.soln_index - mat.bsp.offsets[cb]
		matrix_entry(mat, rb, cb, row, col)^ += val
	}
}

// Compresses the built rows into CSR blocks keeping their values, every row of a diagonal block with a diagonal
// entry (zero blocks, constrained rows). Returns the matrix to solve with, repeat calls return it again. Collective.
sys_matrix_finalize :: proc(m: ^System_Matrix) -> Block_Sparse {
	rank_sync()
	b := m.build
	if b == nil { return m.bsp }
	n := m.bsp.n

	for i in 0 ..< n {
		sub := rank_range(len(b.cols[i * n + i]))
		for r in sub.min ..< sub.max { matrix_entry(m^, i, i, r, r)^ += 0 }
	}
	for k in 0 ..< n * n {
		sub := rank_range(len(b.cols[k]))
		for r in sub.min ..< sub.max { merge_row(&b.cols[k][r], &b.vals[k][r]) }
	}
	rank_sync()

	// row pointers and storage on one rank, then each rank copies its rows in
	if rank_idx() == 0 {
		for k in 0 ..< n * n {
			rows := len(b.cols[k])
			row_ptrs := make([]i32, rows + 1, b.alloc)
			for r in 0 ..< rows { row_ptrs[r + 1] = row_ptrs[r] + i32(len(b.cols[k][r])) }
			nnz := row_ptrs[rows]
			m.bsp.blocks[k] = {sp = {row_ptrs = row_ptrs, columns = make([]i32, nnz, b.alloc)}, values = make([]f64, nnz, b.alloc)}
		}
	}
	rank_sync()

	for k in 0 ..< n * n {
		blk := m.bsp.blocks[k]
		sub := rank_range(len(b.cols[k]))
		for r in sub.min ..< sub.max {
			copy(blk.columns[blk.row_ptrs[r]:], b.cols[k][r][:])
			copy(blk.values[blk.row_ptrs[r]:], b.vals[k][r][:])
		}
	}
	rank_sync()

	if rank_idx() == 0 {
		alloc := b.alloc
		virtual.arena_destroy(&b.arena)
		free(b, alloc)
		m.build = nil
	}
	rank_sync()
	return m.bsp
}

// .Identity mode, once after the first finalize: each constrained row becomes diag * u_i (the rhs is set by
// sys_constrain_rhs). Eigenproblems: diag 1 on K, 0 on M. No-op in .Eliminate mode. Collective.
sys_constrain_matrix :: proc(sys: Sys, A: Block_Sparse, diag := 1.0) {
	if sys.mode != .Identity { return }
	rank_sync()
	for entry in rank_slice(sys.dof_map) {
		if entry.role == .Free { continue }
		local := entry.soln_index - A.offsets[entry.block]
		sp_get(A.blocks[entry.block * A.n + entry.block], local, local)^ = diag
	}
	rank_sync()
}

// .Identity mode, after every assembly of `rhs`: each constrained row's rhs is its inhom value (0 without inhom, e.g.
// Newton updates). No-op in .Eliminate mode. Collective.
sys_constrain_rhs :: proc(sys: Sys, rhs: Vector, inhom: State = nil) {
	if sys.mode != .Identity { return }
	rank_sync()
	sub := rank_range(len(sys.dof_map))
	for gidx in sub.min ..< sub.max {
		entry := sys.dof_map[gidx]
		if entry.role == .Free { continue }
		rhs[entry.soln_index] = inhom[gidx] if inhom != nil else 0
	}
	rank_sync()
}

// Near null space for amgcl from the given states over the solved rows, row-major: mode k at row i is
// nns[i * cols + k], block b's rows are [block_offsets[b] * cols, block_offsets[b + 1] * cols).
sys_near_null_space :: proc(sys: Sys, vectors: ..State, alloc := context.allocator) -> (nns: []f64, cols: int) {
	cols = len(vectors)
	nns = make([]f64, sys.n_soln * cols, alloc)
	for entry, gidx in sys.dof_map {
		if entry.soln_index < 0 { continue }
		for v, col in vectors { nns[entry.soln_index * cols + col] = v[gidx] }
	}
	return
}

// Rigid body modes of `space` as states of the sys, for sys_near_null_space: a translation per geometry component,
// then a rotation per pair of axes (3 modes in 2D, 6 in 3D) about the centroid of the geometry nodes. Assumes a
// Lagrange space whose fields are the displacement components in the geometry's frame, rotated nodal frames are not
// covered. Serial.
sys_rigid_body_modes :: proc(sys: Sys, geo: Geometry, space: ^Space, alloc := context.allocator) -> []State {
	Mode :: struct {
		k:      int,
		centre: [3]f64,
	}
	A := geo.sc.space.fields
	assert(space.family == .Lagrange && space.over == .Cells && space.fields == A, "rigid body modes need a Lagrange cell space with the geometry's components as fields")

	mode: Mode
	n_nodes := len(geo.sc.coeffs) / A
	for n in 0 ..< n_nodes {
		for a in 0 ..< A { mode.centre[a] += geo.sc.coeffs[n * A + a] / f64(n_nodes) }
	}

	modes := make([]State, A + A * (A - 1) / 2 if A > 1 else 1, alloc)
	for &state, k in modes {
		state = sys_state(sys, alloc)
		mode.k = k
		sc := sys_space_coeffs(sys, state, space)
		switch 10 * A + int(space.dim) {
		case 11: interpolate_cells(1, 1, 1, geo, sc, rigid_mode, &mode)
		case 21: interpolate_cells(2, 1, 2, geo, sc, rigid_mode, &mode)
		case 22: interpolate_cells(2, 2, 2, geo, sc, rigid_mode, &mode)
		case 31: interpolate_cells(3, 1, 3, geo, sc, rigid_mode, &mode)
		case 32: interpolate_cells(3, 2, 3, geo, sc, rigid_mode, &mode)
		case 33: interpolate_cells(3, 3, 3, geo, sc, rigid_mode, &mode)
		case: unreachable()
		}
	}
	return modes

	rigid_mode :: proc(x, out: Pvec(f64), data: rawptr) {
		m := (^Mode)(data)
		A := x.fields
		for p in 0 ..< x.points {
			xp, o := pvec_at_point(x, p).data, pvec_at_point(out, p).data
			if m.k < A {
				o[m.k] = 1
				continue
			}
			axis_pairs := [3][2]int{{0, 1}, {1, 2}, {2, 0}} // rotations in the xy, yz and zx planes
			i, j := axis_pairs[m.k - A][0], axis_pairs[m.k - A][1]
			o[i], o[j] = -(xp[j] - m.centre[j]), xp[i] - m.centre[i]
		}
	}
}

// state = soln on the free dofs, then re-derives the constrained ones. Collective.
sys_apply_soln :: proc(sys: Sys, state: State, inhom: State, soln: Vector) {
	sub := rank_range(len(sys.dof_map))
	for gidx in sub.min ..< sub.max {
		entry := sys.dof_map[gidx]
		if entry.role == .Free { state[gidx] = soln[entry.soln_index] }
	}
	sys_enforce_constraints(sys, state, inhom)
}

// state += update on the free dofs, then re-derives the constrained ones. Collective.
sys_apply_update :: proc(sys: Sys, state: State, inhom: State, update: Vector) {
	sub := rank_range(len(sys.dof_map))
	for gidx in sub.min ..< sub.max {
		entry := sys.dof_map[gidx]
		if entry.role == .Free { state[gidx] += update[entry.soln_index] }
	}
	sys_enforce_constraints(sys, state, inhom)
}

// Re-derives every constrained dof's value in `state` from `inhom` and its terms. Call before the first Newton
// residual, and after changing inhom, so the state satisfies the constraints. Collective.
sys_enforce_constraints :: proc(sys: Sys, state: State, inhom: State) {
	rank_sync()
	sub := rank_range(len(sys.dof_map))
	for gidx in sub.min ..< sub.max {
		entry := sys.dof_map[gidx]
		if entry.role == .Free { continue }
		val := inhom[gidx]
		for term in entry.terms {
			val += term.weight * (state[term.dof] if sys.dof_map[term.dof].role == .Free else inhom[term.dof])
		}
		state[gidx] = val
	}
	rank_sync()
}

// Values to zero, the pattern (built or finalized) is kept. Collective.
sys_matrix_zero :: proc(m: System_Matrix) {
	rank_sync()
	for k in 0 ..< len(m.bsp.blocks) {
		if m.build == nil {
			slice.zero(rank_slice(m.bsp.blocks[k].values))
		} else {
			for &row in rank_slice(m.build.vals[k]) { slice.zero(row[:]) }
		}
	}
	rank_sync()
}

// Start of the space's coefficients in a state.
@(private)
sys_space_base :: proc(sys: Sys, space: ^Space) -> int {
	for s, i in sys.spaces { if s == space { return sys.space_offsets[i] } }
	panic("space is not in the sys")
}

// Entry (row, col) of block (i, j): inserted while the matrix is being built, asserted to exist once finalized.
@(private = "file")
matrix_entry :: proc(m: System_Matrix, i, j, row, col: int) -> ^f64 {
	k := i * m.bsp.n + j
	if m.build == nil {
		p := sp_get(m.bsp.blocks[k], row, col)
		assert(p != nil, "entry not in the sparsity pattern")
		return p
	}
	cols, vals := &m.build.cols[k][row], &m.build.vals[k][row]
	if len(cols) == cap(cols) { merge_row(cols, vals) } // before growing, so a row holds at most ~2x its entries
	append(cols, i32(col))
	append(vals, 0)
	return &vals[len(vals) - 1]
}

// Sorts a row being built by column and sums repeated columns. The row's increasing prefix is already merged, so only
// the tail after it is sorted, then merged in.
@(private = "file")
merge_row :: proc(cols: ^[dynamic]i32, vals: ^[dynamic]f64) {
	head := 1
	for head < len(cols) && cols[head] > cols[head - 1] { head += 1 }
	if head >= len(cols) { return }

	scratch_guard()
	tail := make([]u64, len(cols) - head, scratch()) // column << 32 | position, sorts by column
	for &t, i in tail { t = u64(cols[head + i]) << 32 | u64(head + i) }
	slice.sort(tail)

	out_cols := make([]i32, len(cols), scratch())
	out_vals := make([]f64, len(cols), scratch())
	n, h := 0, 0
	for t in tail {
		col, at := i32(t >> 32), int(t & 0xffff_ffff)
		for h < head && cols[h] < col {
			out_cols[n], out_vals[n] = cols[h], vals[h]
			n, h = n + 1, h + 1
		}
		if h < head && cols[h] == col {
			out_cols[n], out_vals[n] = cols[h], vals[h]
			n, h = n + 1, h + 1
		}
		if n > 0 && out_cols[n - 1] == col {
			out_vals[n - 1] += vals[at]
			continue
		}
		out_cols[n], out_vals[n] = col, vals[at]
		n += 1
	}
	for ; h < head; h += 1 {
		out_cols[n], out_vals[n] = cols[h], vals[h]
		n += 1
	}
	copy(cols[:n], out_cols[:n])
	copy(vals[:n], out_vals[:n])
	resize(cols, n)
	resize(vals, n)
}
