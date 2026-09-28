package fe

/*
 Static condensation for hybridized methods (HDG, EDG, hybridized mixed).

 Per cell the unknowns split into interior dofs x (discontinuous cell spaces, eliminated) and trace dofs l (facet
 spaces, kept in the Sys). Eliminating x from the cell's system

  [A B] [x]   [f]
  [C D] [l] = [g]

 which is scattered into the Sys like any element matrix. Once l is solved, x = A^-1 f - A^-1 B l cell by cell.
*/

import "core:mem/virtual"

// Eliminates interior spaces cell by cell, keeping what is needed to recover them from the trace.
Condenser :: struct {
	mesh:     ^Mesh,
	interior: []^Space, // eliminated, not in the Sys
	trace:    []^Space, // kept, in the Sys
	cache:    []Cell_Solve, // [cell]
	arena:    virtual.Arena,
}

// Kept per cell to recover the interior: x = ainv_f - ainv_b l.
Cell_Solve :: struct {
	ainv_b: Dense_Matrix,
	ainv_f: Vector,
}

// One cell being assembled. Owned by the caller, so each thread assembles its own cells.
Cond_Work :: struct {
	using layout: Cell_Layout,
	cond:         ^Condenser,
	cell:         ^Entity,
	A, B, C, D:   Dense_Matrix,
	f, g:         Vector,
}

// Where each space's dofs sit in the cell's stacked interior and trace vectors.
Cell_Layout :: struct {
	n_interior, n_trace: int,
	interior_offset:     [MAX_CONDENSED_SPACES]int,
	trace_offset:        [MAX_CONDENSED_SPACES][MAX_FACETS]int, // [trace space][cell-local facet]
}

MAX_CONDENSED_SPACES :: 8

cond_create :: proc(mesh: ^Mesh, interior, trace: []^Space) -> (c: Condenser) {
	assert(len(interior) <= MAX_CONDENSED_SPACES && len(trace) <= MAX_CONDENSED_SPACES)
	for s in interior { assert(s.over == .Cells && s.continuity == .Discontinuous, "interior spaces are discontinuous cell spaces") }
	for s in trace { assert(s.over == .Facets, "trace spaces are facet spaces") }

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

cond_destroy :: proc(conds: ..^Condenser) {
	for c in conds { virtual.arena_destroy(&c.arena) }
}

// Begin condensation for a cell.
cond_begin :: proc(c: ^Condenser, cell: ^Entity, alloc := context.allocator) -> (w: Cond_Work) {
	context.allocator = alloc

	w.layout = cell_layout(c, cell)
	w.cond = c
	w.cell = cell
	w.A = dn_create(w.n_interior, w.n_interior)
	w.B = dn_create(w.n_interior, w.n_trace)
	w.C = dn_create(w.n_trace, w.n_interior)
	w.D = dn_create(w.n_trace, w.n_trace)
	w.f = make(Vector, w.n_interior)
	w.g = make(Vector, w.n_trace)
	return
}

// Adds a local element matrix for (test, trial) at `site` on the cell, each space's element from the site.
cond_add_mat :: proc(w: ^Cond_Work, local: Cmat($T), test, trial: ^Space, site: Site) {
	r0, row_trace := block_offset(w, test, site)
	c0, col_trace := block_offset(w, trial, site)
	m := w.A
	if !row_trace && col_trace { m = w.B }
	if row_trace && !col_trace { m = w.C }
	if row_trace && col_trace { m = w.D }

	rf, cf := local.row_fields, local.col_fields
	for rd in 0 ..< local.row_dofs {
		for cd in 0 ..< local.col_dofs {
			blk := cmat_dof_block(local, rd, cd)
			for i in 0 ..< rf {
				for j in 0 ..< cf { dn_get(m, r0 + rd * rf + i, c0 + cd * cf + j)^ += f64(blk[j * rf + i]) }
			}
		}
	}
}

// Adds a local element vector for `test` at `site` on the cell.
cond_add_vec :: proc(w: ^Cond_Work, local: Cvec($T), test: ^Space, site: Site) {
	o, is_trace := block_offset(w, test, site)
	v := w.g if is_trace else w.f
	for x, k in local.data { v[o + k] += f64(x) }
}

// Eliminates the interior, caches what recovering it needs, and scatters the trace system into K and rhs. With
// `inhom`, constrained trace columns are lifted into rhs.
cond_end :: proc(w: ^Cond_Work, sys: Sys, K: System_Matrix, rhs: Vector, inhom: State = nil) {
	c := w.cond
	scratch_guard()
	pivots := dn_pivots(w.A, scratch())
	ok := dn_lu_factor(w.A, pivots)
	assert(ok, "singular interior block")

	// reuses the cell's storage across assemblies (Newton, time steps)
	cs := &c.cache[w.cell.id]
	if cs.ainv_f == nil {
		cs.ainv_b = dn_create(w.n_interior, w.n_trace, virtual.arena_allocator(&c.arena))
		cs.ainv_f = make(Vector, w.n_interior, virtual.arena_allocator(&c.arena))
	}
	dn_copy(cs.ainv_b, w.B)
	dn_lu_solve(w.A, pivots, cs.ainv_b)
	copy(cs.ainv_f, w.f)
	dn_lu_solve_vec(w.A, pivots, cs.ainv_f)

	dn_gemm(w.C, cs.ainv_b, w.D, alpha = -1, beta = 1) // D <- D - C A^-1 B
	dn_gemv(w.C, cs.ainv_f, w.g, alpha = -1, beta = 1) // g <- g - C A^-1 f

	// the Sys works per trace element, so S and r go out per facet and facet pair
	n_facets := element_n_sub_entities(w.cell.type, element_facet_dim(w.cell.type))
	for rs, ri in c.trace {
		for rlf in 0 ..< n_facets {
			rsite := cell_facet_site(w.cell^, rlf)
			r0, rn := w.trace_offset[ri][rlf], basis_n_dofs(space_local_basis(rs, rsite).bd)

			v := cvec_create(f64, rn, rs.fields, scratch())
			copy(v.data, w.g[r0:][:rn * rs.fields])
			sys_scatter_vec(sys, rhs, v, rs, rsite)

			for cs_, ci in c.trace {
				for clf in 0 ..< n_facets {
					csite := cell_facet_site(w.cell^, clf)
					c0, cn := w.trace_offset[ci][clf], basis_n_dofs(space_local_basis(cs_, csite).bd)
					m := cmat_create(f64, rn, rs.fields, cn, cs_.fields, scratch())
					for rd in 0 ..< rn {
						for cd in 0 ..< cn {
							blk := cmat_dof_block(m, rd, cd)
							for i in 0 ..< rs.fields {
								for j in 0 ..< cs_.fields {
									blk[j * rs.fields + i] = dn_get(w.D, r0 + rd * rs.fields + i, c0 + cd * cs_.fields + j)^
								}
							}
						}
					}
					sys_scatter_mat_pair(sys, K, m, rs, cs_, rsite, csite, rhs if inhom != nil else nil, inhom)
				}
			}
		}
	}
}

// Interior = A^-1 f - A^-1 B l, l the trace in `state` (after sys_apply_soln). Interior coefficients are matched to the
// condensed spaces by space, any subset in any order. Collective.
cond_apply_soln :: proc(c: ^Condenser, sys: Sys, state: State, interior: ..Space_Coeffs) {
	cond_apply(c, sys, state, nil, interior)
}

// Interior += A^-1 f - A^-1 B dl, dl the trace part of the Newton update `du` (after sys_apply_update). Collective.
cond_apply_update :: proc(c: ^Condenser, sys: Sys, du: Vector, interior: ..Space_Coeffs) {
	cond_apply(c, sys, nil, du, interior)
}

@(private = "file")
cond_apply :: proc(c: ^Condenser, sys: Sys, state: State, du: Vector, interior: []Space_Coeffs) {
	rank_sync()
	sub := rank_range(len(c.mesh.cells))
	for &cell in c.mesh.cells[sub.min:sub.max] {
		cs := c.cache[cell.id]
		if cs.ainv_f == nil { continue } // not assembled, outside the spaces' regions

		scratch_guard()
		context.allocator = scratch()
		lay := cell_layout(c, &cell)

		l := make(Vector, lay.n_trace)
		for ts, ti in c.trace {
			base := sys_space_base(sys, ts)
			for lf in 0 ..< element_n_sub_entities(cell.type, element_facet_dim(cell.type)) {
				e, _ := space_elem(ts, cell_facet_site(cell, lf))
				for g, ld in ts.numbering.l2g[e.id] {
					for k in 0 ..< ts.fields {
						gidx := base + int(g) * ts.fields + k
						v := &l[lay.trace_offset[ti][lf] + ld * ts.fields + k]
						if state != nil {
							v^ = state[gidx]
							continue
						}
						// a free dof's update is solved, a tied one follows its terms, an essential one doesn't move
						entry := sys.dof_map[gidx]
						if entry.role == .Free {
							v^ = du[entry.soln_index]
							continue
						}
						for term in entry.terms {
							if target := sys.dof_map[term.dof]; target.role == .Free { v^ += term.weight * du[target.soln_index] }
						}
					}
				}
			}
		}

		x := make(Vector, lay.n_interior)
		copy(x, cs.ainv_f)
		dn_gemv(cs.ainv_b, l, x, alpha = -1, beta = 1)

		for sc in interior {
			for s, k in c.interior {
				if s != sc.space { continue }
				o := lay.interior_offset[k]
				nf := s.fields
				for g, ld in s.numbering.l2g[cell.id] {
					for f in 0 ..< nf {
						dst := &sc.coeffs[int(g) * nf + f]
						dst^ = x[o + ld * nf + f] + (dst^ if du != nil else 0)
					}
				}
			}
		}
	}
	rank_sync()
}

@(private = "file")
cell_layout :: proc(c: ^Condenser, cell: ^Entity) -> (lay: Cell_Layout) {
	for s, k in c.interior {
		lay.interior_offset[k] = lay.n_interior
		lay.n_interior += basis_n_dofs(space_bd(s, cell)) * s.fields
	}
	for s, k in c.trace {
		for lf in 0 ..< element_n_sub_entities(cell.type, element_facet_dim(cell.type)) {
			lay.trace_offset[k][lf] = lay.n_trace
			lay.n_trace += basis_n_dofs(space_local_basis(s, cell_facet_site(cell^, lf)).bd) * s.fields
		}
	}
	return
}

// Offset of a space's block at `site` in the cell's stacked vectors, and whether it is a trace block.
@(private = "file")
block_offset :: proc(w: ^Cond_Work, s: ^Space, site: Site) -> (offset: int, is_trace: bool) {
	assert(site.cell == w.cell.id, "site is not on the cell being condensed")
	for x, k in w.cond.interior {
		if x == s { return w.interior_offset[k], false }
	}
	for x, k in w.cond.trace {
		if x == s {
			assert(site.dim == element_facet_dim(w.cell.type), "trace blocks need a facet site")
			return w.trace_offset[k][site.index], true
		}
	}
	panic("space is not in the condenser")
}
