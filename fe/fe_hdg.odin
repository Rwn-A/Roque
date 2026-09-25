package fe

/*
 Static condensation, for HDG and other hybridized methods.

 Per cell, the unknowns split into interior (cell-numbered spaces, eliminated) and trace (facet-numbered spaces,
 kept). The cell's local system is

     [A B] [x]   [f]      x: interior dofs of the cell
     [C D] [l] = [g]      l: trace dofs on the cell's facets

 and eliminating x leaves the trace system

     S l = r,   S = D - C A^-1 B,   r = g - C A^-1 f

 which is scattered through the Sys like any other element matrix. Only the trace spaces go into the Multi_Space
 and Sys; declare their couplings with `through_cells` so every pair of facets sharing a cell is in the sparsity.
 Once l is solved, x = A^-1 f - A^-1 B l is recovered cell by cell from what was cached here.

 Usage, per cell:
     cond_begin(&c, &cell)
     cond_add_mat(&c, local, test, trial[, test_facet, trial_facet])   // any mix of interior / trace blocks
     cond_add_vec(&c, local, test[, facet])
     cond_end(&c, sys, K, rhs, inhom, mode)
 then, after solving:
     cond_reconstruct(&c, trace_vectors, interior_vectors)

 Trace blocks are given per facet: `facet` is the cell-local facet index, and the local Cmat/Cvec is sized for the
 trace space's element on that facet.
*/

import "core:mem/virtual"

MAX_CONDENSED_SPACES :: 8

Condenser :: struct {
	mesh:     ^Mesh,
	interior: []^Space, // eliminated
	trace:    []^Space, // kept, the only spaces in the Multi_Space / Sys
	cache:    []Cell_Solve, // [cell]
	arena:    virtual.Arena,
	cur:      Cell_Work,
}

// Kept per cell to recover the interior: x = ainv_f - ainv_b * l.
Cell_Solve :: struct {
	ainv_b: Dense_Matrix,
	ainv_f: Vector,
}

// Where each space's dofs sit in the cell's stacked interior / trace vectors.
Cell_Layout :: struct {
	n_interior, n_trace: int,
	interior_offset:     [MAX_CONDENSED_SPACES]int,
	trace_offset:        [MAX_CONDENSED_SPACES][MAX_FACETS]int, // [trace space][cell-local facet]
}

Cell_Work :: struct {
	using layout: Cell_Layout,
	cell:         ^Cell,
	A, B, C, D:   Dense_Matrix,
	f, g:         Vector,
}

cond_create :: proc(mesh: ^Mesh, interior, trace: []^Space) -> (c: Condenser) {
	assert(len(interior) <= MAX_CONDENSED_SPACES && len(trace) <= MAX_CONDENSED_SPACES)
	for s in interior { assert(s.numbering.numbered_over == mesh.intrinsic_dim, "interior spaces are cell-numbered") }
	for s in trace { assert(s.numbering.numbered_over != mesh.intrinsic_dim, "trace spaces are facet-numbered") }

	virtual.arena_init_growing(&c.arena) or_else panic("Failed to create arena")
	context.allocator = virtual.arena_allocator(&c.arena)

	c.mesh = mesh
	c.interior = make([]^Space, len(interior))
	c.trace = make([]^Space, len(trace))
	copy(c.interior, interior)
	copy(c.trace, trace)
	c.cache = make([]Cell_Solve, len(mesh.cells))
	return
}

cond_destroy :: proc(c: ^Condenser) {
	virtual.arena_destroy(&c.arena)
}

// Start a cell. Work matrices come from context.allocator (a scratch allocator is ideal).
cond_begin :: proc(c: ^Condenser, cell: ^Cell) {
	w := &c.cur
	w^ = {
		layout = cell_layout(c, cell),
		cell   = cell,
	}
	w.A = dense_create(w.n_interior, w.n_interior)
	w.B = dense_create(w.n_interior, w.n_trace)
	w.C = dense_create(w.n_trace, w.n_interior)
	w.D = dense_create(w.n_trace, w.n_trace)
	w.f = make(Vector, w.n_interior)
	w.g = make(Vector, w.n_trace)
}

// Add a local element matrix. Trace spaces need the cell-local facet the block belongs to.
cond_add_mat :: proc(c: ^Condenser, local: Cmat($T), test, trial: ^Space, facet := -1) {
	r0, row_trace := block_offset(c, test, facet)
	c0, col_trace := block_offset(c, trial, facet)
	m := c.cur.A
	if !row_trace && col_trace { m = c.cur.B }
	if row_trace && !col_trace { m = c.cur.C }
	if row_trace && col_trace { m = c.cur.D }

	rf_n, cf_n := local.row_fields, local.col_fields
	for rd in 0 ..< local.row_dofs {
		for cd in 0 ..< local.col_dofs {
			blk := cmat_dof_block(local, rd, cd)
			for rf in 0 ..< rf_n {
				for cf in 0 ..< cf_n {
					dn_get(m, r0 + rd * rf_n + rf, c0 + cd * cf_n + cf)^ += f64(blk[cf * rf_n + rf])
				}
			}
		}
	}
}

// Add a local element vector. Trace spaces need the cell-local facet.
cond_add_vec :: proc(c: ^Condenser, local: Cvec($T), test: ^Space, facet := -1) {
	o, is_trace := block_offset(c, test, facet)
	v := c.cur.g if is_trace else c.cur.f
	for x, k in local.data { v[o + k] += f64(x) }
}

// Eliminate the interior, cache what reconstruction needs, and scatter the trace system into K / rhs.
cond_end :: proc(
	c: ^Condenser,
	sys: Sys,
	K: Sparse_Matrix,
	rhs: Vector,
	inhom: State,
	mode: Assembly_Mode,
) {
	w := &c.cur

	pivots := dn_pivots(w.A)
	ok := dn_lu_factor(w.A, pivots)
	assert(ok, "singular interior block")

	// Cache A^-1 B and A^-1 f, reusing storage across repeated assemblies (e.g. Newton iterations).
	cs := &c.cache[w.cell.id]
	if cs.ainv_f == nil {
		arena := virtual.arena_allocator(&c.arena)
		cs.ainv_b = dense_create(w.n_interior, w.n_trace, arena)
		cs.ainv_f = make(Vector, w.n_interior, arena)
	}
	dn_copy(cs.ainv_b, w.B)
	dn_lu_solve(w.A, pivots, cs.ainv_b)
	copy(cs.ainv_f, w.f)
	dn_lu_solve_vec(w.A, pivots, cs.ainv_f)

	dn_gemm(w.C, cs.ainv_b, w.D, alpha = -1, beta = 1) // D <- S = D - C A^-1 B
	dn_gemv(w.C, cs.ainv_f, w.g, alpha = -1, beta = 1) // g <- r = g - C A^-1 f

	// Scatter per (trace space, facet) blocks, since the Sys works per trace element.
	for rs, ri in c.trace {
		for rfid, rlf in w.cell.facets {
			r0, rn := w.trace_offset[ri][rlf], trace_dofs(c, rs, rfid)

			v := cvec_create(f64, rn, rs.fields)
			copy(v.data, w.g[r0:][:rn * rs.fields])
			sys_scatter_vec(sys, rhs, v, rs, rfid)

			for cs_, ci in c.trace {
				for cfid, clf in w.cell.facets {
					c0, cn := w.trace_offset[ci][clf], trace_dofs(c, cs_, cfid)
					m := cmat_create(f64, rn, cn, rs.fields, cs_.fields)
					for rd in 0 ..< rn {
						for cd in 0 ..< cn {
							blk := cmat_dof_block(m, rd, cd)
							for rf in 0 ..< rs.fields {
								for cf in 0 ..< cs_.fields {
									blk[cf * rs.fields + rf] = dn_get(w.D, r0 + rd * rs.fields + rf, c0 + cd * cs_.fields + cf)^
								}
							}
						}
					}
					sys_scatter_mat_pair(sys, K, rhs, inhom, mode, m, rs, cs_, rfid, cfid)
				}
			}
		}
	}
}

// Recover the interior from a solved trace: x = A^-1 f - A^-1 B l, written into the interior vectors.
// `trace` and `interior` are in the same order as the spaces given to cond_create.
cond_reconstruct :: proc(c: ^Condenser, trace, interior: []Space_Vector) {
	assert(len(trace) == len(c.trace) && len(interior) == len(c.interior))

	for &cell in c.mesh.cells {
		cs := c.cache[cell.id]
		if cs.ainv_f == nil { continue } // not assembled (outside the spaces' regions)

		scratch_guard()
		context.allocator = scratch()
		lay := cell_layout(c, &cell)

		l := make(Vector, lay.n_trace)
		for tv, ti in trace {
			for fid, lf in cell.facets {
				v := space_gather(f64, tv, fid)
				copy(l[lay.trace_offset[ti][lf]:], v.data)
			}
		}

		x := make(Vector, lay.n_interior)
		copy(x, cs.ainv_f)
		dn_gemv(cs.ainv_b, l, x, alpha = -1, beta = 1)

		for iv, ii in interior {
			o := lay.interior_offset[ii]
			n := basis_num_dofs(space_bd(iv.space, cell.type))
			space_scatter(iv, cell.id, Cvec(f64){dofs = n, fields = iv.fields, data = x[o:][:n * iv.fields]})
		}
	}
}

//== Internals

@(private = "file")
cell_layout :: proc(c: ^Condenser, cell: ^Cell) -> (lay: Cell_Layout) {
	for s, k in c.interior {
		lay.interior_offset[k] = lay.n_interior
		lay.n_interior += basis_num_dofs(space_bd(s, cell.type)) * s.fields
	}
	for s, k in c.trace {
		for fid, lf in cell.facets {
			lay.trace_offset[k][lf] = lay.n_trace
			lay.n_trace += trace_dofs(c, s, fid) * s.fields
		}
	}
	return
}

@(private = "file")
trace_dofs :: proc(c: ^Condenser, s: ^Space, facet_id: Entity_ID) -> int {
	return basis_num_dofs(space_bd(s, c.mesh.facets[facet_id].info.type))
}

// Offset of a space's block in the current cell's stacked vectors, and whether it is a trace block.
@(private = "file")
block_offset :: proc(c: ^Condenser, s: ^Space, facet: int) -> (offset: int, is_trace: bool) {
	for x, k in c.interior {
		if x == s { return c.cur.interior_offset[k], false } // facet is irrelevant for interior spaces
	}
	for x, k in c.trace {
		if x == s {
			assert(facet >= 0, "trace blocks need the cell-local facet")
			return c.cur.trace_offset[k][facet], true
		}
	}
	panic("space is not part of the condenser")
}
