package fe

/*
 Finite element spaces: a basis family and order over the cells, or over the facets (trace spaces), of a mesh, with a
 local to global dof numbering shared by every field.
*/

import "core:mem/virtual"
import "core:slice"

Space_Continuity :: enum {
	Continuous,
	Discontinuous,
}

// Which mesh entities a space's elements are. Facets are for trace spaces.
Space_Over :: enum {
	Cells,
	Facets,
}

Space :: struct {
	using sd:     Space_Desc,
	mesh:         ^Mesh,
	over:         Space_Over,
	dim:          Dimension, // dimension of the space's elements
	fields:       int,
	total_coeffs: int,
	numbering:    DOF_Numbering,
	arena:        virtual.Arena, // the space itself and its numbering
}

Space_Desc :: struct #all_or_none {
	family:     Basis_Family,
	order:      Order,
	continuity: Space_Continuity,
	regions:    Tag_Set, // cell tags
}

// Local to global dof numbering, the same for every field of the space.
DOF_Numbering :: struct {
	l2g:           [][]i32,
	n_dofs:        int,
	entity_dofs:   [Dimension][]i32, // first dof on each global entity, -1 if none. Continuous spaces only
	entity_counts: [Element_Type]int, // dofs on one entity of each type. Continuous spaces only
}

// A space and its coefficients, dof-major: field f of dof d is coeffs[d * space.fields + f].
Space_Coeffs :: struct {
	space:  ^Space,
	coeffs: []f64, // len space.total_coeffs
}

// Physical coordinates as a continuous Lagrange field of the mesh's order, over the cells or (geo_trace) the facets.
Geometry :: struct {
	sc:     Space_Coeffs,
	affine: []bool, // per element of the space, constant Jacobian
}

// Fills `out` (F fields, A components for vector families) at the physical points `x`.
Interp_Proc :: #type proc(x, out: Pvec(f64), data: rawptr)

MAX_FIELDS :: 32

// New space over the mesh's cells, or over its facets (trace spaces, scalar valued). `broken`: per facet of a
// continuous trace space, facets that get their own dofs (HDG) while the rest share them (EDG).
space_create :: proc(mesh: ^Mesh, sd: Space_Desc, fields: int, over := Space_Over.Cells, broken: []bool = nil) -> ^Space {
	assert(fields <= MAX_FIELDS)
	assert(over == .Cells || .S_Val in BASIS_QUANTITIES[sd.family], "trace spaces must be scalar valued")
	assert(broken == nil || (over == .Facets && sd.continuity == .Continuous), "broken facets need a continuous trace space")

	s := virtual.arena_growing_bootstrap_new(Space, "arena") or_else panic("Failed to create arena")
	s.sd = sd
	s.mesh = mesh
	s.over = over
	s.dim = mesh.intrinsic_dim if over == .Cells else mesh_facet_dim(mesh^)
	s.fields = fields
	alloc := virtual.arena_allocator(&s.arena)

	scratch_guard()
	start: [Dimension][]i32
	for d in Dimension {
		start[d] = make([]i32, mesh.n_entities[d], scratch())
		slice.fill(start[d], -1)
	}

	l2g := make([][]i32, len(mesh.entities[s.dim]), alloc)
	next: i32
	counts: [Element_Type]int
	for &row, i in l2g {
		e := mesh.entities[s.dim][i]
		tags := e.tags if over == .Cells else facet_cell_tags(mesh^, e)
		if tags & sd.regions == {} { continue }

		own := sd.continuity == .Discontinuous || (broken != nil && broken[i])

		bd := Basis_Desc{e.type, sd.family, sd.order}
		row = make([]i32, basis_n_dofs(bd), alloc)
		ls := 0
		for d in Dimension {
			count := basis_dofs_per_entity(bd, d)
			if count == 0 { continue }
			for gid, k in e.conn[d] {
				counts[element_sub_entity(e.type, d, k).type] = count
				first := start[d][gid]
				if own || first < 0 {
					first = next
					next += i32(count)
					if !own { start[d][gid] = first }
				}
				for j in 0 ..< count { row[ls + j] = first + i32(j) }
				ls += count
			}
		}
	}

	s.numbering.l2g = l2g
	s.numbering.n_dofs = int(next)
	s.total_coeffs = int(next) * fields
	if sd.continuity == .Continuous {
		for d in Dimension { s.numbering.entity_dofs[d] = slice.clone(start[d], alloc) }
		s.numbering.entity_counts = counts
	}
	return s
}

// Scalar space over facets sharing a continuous cell space's dofs, both read the same coefficients.
@(private = "file")
space_create_trace :: proc(parent: ^Space) -> ^Space {
	assert(parent.continuity == .Continuous && parent.over == .Cells, "trace of a continuous cell space")
	assert(.S_Val in BASIS_QUANTITIES[parent.family], "trace spaces must be scalar valued")

	s := virtual.arena_growing_bootstrap_new(Space, "arena") or_else panic("Failed to create arena")
	s.sd = parent.sd
	s.mesh = parent.mesh
	s.over = .Facets
	s.dim = mesh_facet_dim(parent.mesh^)
	s.fields = parent.fields
	s.total_coeffs = parent.total_coeffs
	s.numbering.n_dofs = parent.numbering.n_dofs
	alloc := virtual.arena_allocator(&s.arena)

	l2g := make([][]i32, len(parent.mesh.facets), alloc)
	for facet, f in parent.mesh.facets {
		if facet_cell_tags(parent.mesh^, facet) & parent.regions == {} { continue }
		bd := Basis_Desc{facet.type, parent.family, parent.order}
		l2g[f] = make([]i32, basis_n_dofs(bd), alloc)
		ls := 0
		for d in Dimension {
			count := basis_dofs_per_entity(bd, d)
			if count == 0 { continue }
			for gid in facet.conn[d] {
				assert(parent.numbering.entity_counts[mesh_entity_type(parent.mesh^, d, gid)] == count)
				for j in 0 ..< count { l2g[f][ls + j] = parent.numbering.entity_dofs[d][gid] + i32(j) }
				ls += count
			}
		}
	}
	s.numbering.l2g = l2g
	return s
}

space_destroy :: proc(spaces: ..^Space) {
	for s in spaces {
		arena := s.arena // the space lives in its own arena
		virtual.arena_destroy(&arena)
	}
}

// Zeroed coefficients for a space outside a Sys.
space_coeffs :: proc(s: ^Space, alloc := context.allocator) -> Space_Coeffs {
	return {s, make([]f64, s.total_coeffs, alloc)}
}

//== At a site

// Basis of the space on one of its elements.
space_bd :: proc(s: ^Space, elem: ^Entity) -> Basis_Desc {
	assert(element_dim(elem.type) == s.dim, "element is not of the space's dimension")
	return {elem.type, s.family, s.order}
}

// The space's element at `site`, and the site's index among the element's sub-entities of dimension site.dim.
@(private)
space_elem :: proc(s: ^Space, site: Site) -> (elem: ^Entity, index: int) {
	cell := &s.mesh.cells[site.cell]
	if s.over == .Cells { return cell, site.index }
	assert(site.dim == s.dim, "a trace space only sees sites on its own entities")
	return &s.mesh.entities[s.dim][cell.conn[s.dim][site.index]], 0
}

// Local basis of the space's element at `site`, for basis_tab.
space_local_basis :: proc(s: ^Space, site: Site) -> Local_Basis {
	e, index := space_elem(s, site)
	return {space_bd(s, e), e.keys, s.continuity == .Continuous, site.dim, index}
}

// Smallest quadrature set on `site` exact for a mass matrix of the space's basis, plus `extra` degrees.
space_quad_rule :: proc(s: ^Space, site: Site, extra := 0) -> (Rule, []f64) {
	e, index := space_elem(s, site)
	return basis_quad_rule(space_bd(s, e), site.dim, index, extra)
}

// The dofs of the space's element at `site`, cast to T.
space_gather :: proc($T: typeid, sc: Space_Coeffs, site: Site, alloc := context.allocator) -> Cvec(T) {
	e, _ := space_elem(sc.space, site)
	l2g := sc.space.numbering.l2g[e.id]
	assert(l2g != nil, "element is not in the space")
	f := sc.space.fields
	local := cvec_create(T, len(l2g), f, alloc)
	for g, ldof in l2g {
		dst := local.data[ldof * f:][:f]
		for x, k in sc.coeffs[int(g) * f:][:f] { dst[k] = cast(T)x }
	}
	return local
}

// Writes the dofs of the space's element at `site` from `local`.
space_scatter :: proc(sc: Space_Coeffs, site: Site, local: Cvec($T)) {
	e, _ := space_elem(sc.space, site)
	l2g := sc.space.numbering.l2g[e.id]
	assert(l2g != nil, "element is not in the space")
	f := sc.space.fields
	assert(len(l2g) * f == len(local.data))
	for g, ldof in l2g {
		src := local.data[ldof * f:][:f]
		for &x, k in sc.coeffs[int(g) * f:][:f] { x = cast(f64)src[k] }
	}
}

// Quantity `q` of the field at the rule's points on `site`, pushed to physical space by `m` (built on the same rule).
// Assumes the coefficients are components in the geometry's frame, rotated nodal frames need a gather and push.
space_eval :: proc(
	$F: int,
	sc: Space_Coeffs,
	site: Site,
	rule: Rule,
	m: Element_Map($A, $I, f64),
	q: Basis_Quantity,
	alloc := context.allocator,
) -> Pvec(f64) {
	assert(F == sc.space.fields)
	assert(m.constant || len(m.tng) == len(rule.points), "element map is not at the rule's points")
	lb := space_local_basis(sc.space, site)
	cmpnts := basis_quantity_phys_cmpnts(lb.bd, q, A)
	x := pvec_create(f64, len(rule.points), cmpnts, F, alloc) // before the guard, alloc may be scratch

	scratch_guard()
	tab := basis_tab(lb, rule, m, Quantity_Set{q}, scratch())[q]
	c := space_gather(f64, sc, site, scratch())
	switch cmpnts {
	case 1: contract_eval(1, F, x, c, tab)
	case 2: contract_eval(2, F, x, c, tab)
	case 3: contract_eval(3, F, x, c, tab)
	case: panic("quantity has more than 3 components")
	}
	return x
}

//== Geometry

// The mesh geometry, node coordinates through `frame`. Moving the mesh is writing the coefficients, then
// geo_update_affine.
geo_create :: proc(mesh: ^Mesh, frame: Small_Mat(3, $C, f64)) -> (geo: Geometry) {
	s := space_create(mesh, {.Lagrange, mesh.order, .Continuous, ALL_TAGS}, C)
	alloc := virtual.arena_allocator(&s.arena)
	geo.sc = {s, make([]f64, s.total_coeffs, alloc)}
	geo.affine = make([]bool, len(mesh.cells), alloc)
	inv_f := small_mat_inv(frame)

	// a cell's reference Lagrange dofs are its nodes in reference order, oriented onto the canonical numbering
	scratch_guard()
	for cell in mesh.cells {
		nodes := mesh.cell_nodes[cell.id]
		vals := cvec_create(f64, len(nodes), C, scratch())
		for n, i in nodes {
			x := small_mat_vec_mul(inv_f, Small_Vec(3, f64){mesh.nodes[n]})
			copy(vals.data[i * C:][:C], x.data[:])
		}
		site := cell_site(cell)
		basis_orient_dofs(space_local_basis(s, site), vals)
		space_scatter(geo.sc, site, vals)
	}
	geo_update_affine(geo)
	return
}

// Geometry over the facets, sharing the cell geometry's coefficients so it moves with them. It has its own affine
// flags, update them too after the mesh moves.
geo_trace :: proc(geo: Geometry) -> (trace: Geometry) {
	s := space_create_trace(geo.sc.space)
	trace.sc = {s, geo.sc.coeffs}
	trace.affine = make([]bool, len(s.mesh.facets), virtual.arena_allocator(&s.arena))
	geo_update_affine(trace)
	return
}

// Recomputes which elements have a constant Jacobian from the current coefficients. The Jacobian of a degree k map is
// constant iff it is the same at every point of a positive rule exact to degree 2k, the mass matrix rule.
geo_update_affine :: proc(geo: Geometry) {
	s := geo.sc.space
	for e, i in s.mesh.entities[s.dim] {
		scratch_guard()
		site := cell_site(e) if s.over == .Cells else facet_site(e)
		rule, _ := space_quad_rule(s, site)
		grad := basis_tab(space_local_basis(s, site), rule, scratch())[.S_Grd]
		x := space_gather(f64, geo.sc, site, scratch())

		n := s.fields * grad.cmpnts
		j0, jp := make([]f64, n, scratch()), make([]f64, n, scratch())
		scale := 0.0
		geo.affine[i] = true
		for p in 0 ..< grad.points {
			slice.zero(jp)
			gp := bvec_at_point(grad, p)
			for dof in 0 ..< grad.dofs {
				g := bvec_dof_block(gp, dof)
				for a in 0 ..< s.fields {
					for k in 0 ..< grad.cmpnts { jp[a * grad.cmpnts + k] += x.data[dof * s.fields + a] * g[k] }
				}
			}
			if p == 0 {
				copy(j0, jp)
				for v in j0 { scale = max(scale, abs(v)) }
				continue
			}
			for v, k in jp { if abs(v - j0[k]) > 1e-10 * scale { geo.affine[i] = false } }
		}
	}
}

geo_destroy :: proc(geos: ..Geometry) {
	for geo in geos { space_destroy(geo.sc.space) }
}

// Element map of the geometry's element at the rule's points on `site`.
geo_map :: proc($A, $I: int, geo: Geometry, site: Site, rule: Rule, alloc := context.allocator) -> Element_Map(A, I, f64) {
	assert(A == geo.sc.space.fields)
	e, _ := space_elem(geo.sc.space, site)
	tab := basis_tab(space_local_basis(geo.sc.space, site), rule, scratch())
	return element_map(A, I, tab[.S_Grd], space_gather(f64, geo.sc, site, scratch()), geo.affine[e.id], alloc)
}

// Physical points of the rule on `site`, A per point.
geo_points :: proc($A: int, geo: Geometry, site: Site, rule: Rule, alloc := context.allocator) -> Pvec(f64) {
	assert(A == geo.sc.space.fields)
	x := pvec_create(f64, len(rule.points), 1, A, alloc)
	tab := basis_tab(space_local_basis(geo.sc.space, site), rule, scratch())
	contract_eval(1, A, x, space_gather(f64, geo.sc, site, scratch()), tab[.S_Val])
	return x
}

//== Interpolation

Interp_Block :: struct($A, $I, $F: int) {
	points: Pvec(f64), // physical points, 1 x A
	em:     Element_Map(A, I, f64), // of the space's element
	out:    Pvec(f64), // filled by the user
}

Interpolator :: struct($A, $I, $F: int) {
	sc:       Space_Coeffs,
	site:     Site,
	lb:       Local_Basis, // of the space's element at the site
	geo_bd:   Basis_Desc,
	vector:   bool, // vector family (RT, Nedelec)
	affine:   bool,
	coords:   Cvec(f64), // geometry nodes of the element
	vals:     Cvec(f64), // the element's reference dof values, filled block by block
	blocks:   [MAX_INTERP_BLOCKS]struct {
		dim:   Dimension,
		index: int,
	}, // sub-entities of the element to visit
	n_blocks: int,
	cur:      int,
	blk:      Interp_Block(A, I, F),
	temp:     Scratch_Temp,
}

@(private = "file")
MAX_INTERP_BLOCKS :: MAX_VERTICES + MAX_EDGES + MAX_FACES + 1

// Interpolates a function given at physical points onto the closure of `site` in the space's element there. `geo`
// must be over the same entities as the space (geo_trace for trace spaces). Assumes the coefficients are components
// in the geometry's frame.
interpolator :: proc($A, $I, $F: int, geo: Geometry, sc: Space_Coeffs, site: Site) -> (ip: Interpolator(A, I, F)) {
	assert(A == geo.sc.space.fields && F == sc.space.fields)
	assert(geo.sc.space.over == sc.space.over, "geometry must be over the same entities as the space")
	elem, _ := space_elem(sc.space, site)
	ip.sc = sc
	ip.site = site
	ip.lb = space_local_basis(sc.space, site)
	assert(I == int(element_dim(ip.lb.bd.element)))

	ip.temp = scratch_begin_temp()
	ip.geo_bd = space_bd(geo.sc.space, elem)
	ip.vector = .V_Val in BASIS_QUANTITIES[ip.lb.bd.family]
	ip.affine = geo.affine[elem.id]
	ip.coords = space_gather(f64, geo.sc, site, scratch())
	ip.vals = cvec_create(f64, basis_n_dofs(ip.lb.bd), F, scratch())

	closure := element_sub_entity(ip.lb.bd.element, ip.lb.dim, ip.lb.index).closure
	for d in Dimension {
		if basis_dofs_per_entity(ip.lb.bd, d) == 0 { continue }
		for e in closure[d] {
			ip.blocks[ip.n_blocks] = {d, e}
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
	if ip.cur >= 0 {
		// the previous block's reference dof values from the filled `out`
		b := ip.blocks[ip.cur]
		fn := basis_entity_functionals(ip.lb.bd, b.dim, b.index)
		ls, n := basis_entity_dof_range(ip.lb.bd, b.dim, b.index)
		dofs := Cvec(f64){dofs = n, fields = F, data = ip.vals.data[ls * F:][:n * F]}

		if !ip.vector {
			contract_linear(1, F, dofs, fn.weights, ip.blk.out)
		} else {
			ref := pvec_create(f64, ip.blk.out.points, I, F)
			covariant := basis_quantity_map(ip.lb.bd, .V_Val) == .Covariant
			for pt in 0 ..< ref.points {
				m: Small_Mat(A, I, f64)
				if covariant {
					m = element_map_tng(ip.blk.em, pt)
				} else {
					m = element_map_ctng(ip.blk.em, pt)
					small_mat_scale_inplace(&m, element_map_measure(ip.blk.em, pt))
				}
				pvec_point_matrix(ref, pt, I, F)^ = small_mat_mul(pvec_point_matrix(ip.blk.out, pt, A, F)^, m)
			}
			contract_linear(I, F, dofs, fn.weights, ref)
		}
	}
	ip.cur += 1
	if ip.cur >= ip.n_blocks { return }

	b := ip.blocks[ip.cur]
	rule := basis_entity_functionals(ip.lb.bd, b.dim, b.index).rule
	geo := basis_tab(Local_Basis{bd = ip.geo_bd, dim = b.dim, index = b.index}, rule) // functional rules are in the element's own frame
	np := len(rule.points)

	ip.blk.points = pvec_create(f64, np, 1, A)
	contract_eval(1, A, ip.blk.points, ip.coords, geo[.S_Val])
	ip.blk.em = element_map(A, I, geo[.S_Grd], ip.coords, ip.affine)
	ip.blk.out = pvec_create(f64, np, A if ip.vector else 1, F)
	return ip.blk, true
}

// Call once every block has been visited, writes the dofs on the closure of the site. Does not reset the iterator.
interpolator_flush :: proc(ip: ^Interpolator($A, $I, $F)) {
	defer scratch_end_temp(ip.temp)
	assert(ip.cur >= ip.n_blocks, "interpolator flushed before every block was visited")
	basis_orient_dofs(ip.lb, ip.vals)

	e, _ := space_elem(ip.sc.space, ip.site)
	l2g := ip.sc.space.numbering.l2g[e.id]
	assert(l2g != nil, "element is not in the space")
	for ldof in basis_closure_dofs(ip.lb.bd, ip.lb.dim, ip.lb.index) {
		for &x, k in ip.sc.coeffs[int(l2g[ldof]) * F:][:F] { x = ip.vals.data[ldof * F + k] }
	}
}

// Interpolates `f` onto the closure of every facet tagged `tags`, as interpolator. At a dof shared between elements
// the value comes from the last element visited, so a discontinuous `f` takes one side's value there.
// `cells` from mesh_colour_cells keeps ranks off each other's dofs, the empty colouring runs serially on one rank.
// Collective.
interpolate_facets :: proc(
	$A, $I, $F: int,
	geo: Geometry,
	sc: Space_Coeffs,
	tags: Tag_Set,
	f: Interp_Proc,
	data: rawptr = nil,
	cells := Colouring{},
) {
	scratch_guard()
	mesh := sc.space.mesh
	fd := mesh_facet_dim(mesh^)
	for it := colour_iterator(serial_colouring(mesh, cells)); id in colour_iterator_next(&it) {
		cell := mesh.cells[id]
		for fid, lf in cell.conn[fd] {
			if mesh.facets[fid].tags & tags == {} { continue }
			interpolate_site(A, I, F, geo, sc, cell_facet_site(cell, lf), f, data)
		}
	}
}

// Interpolates `f` onto every element of the space, as interpolate_facets. For initial conditions and fields given as
// functions.
interpolate_cells :: proc(
	$A, $I, $F: int,
	geo: Geometry,
	sc: Space_Coeffs,
	f: Interp_Proc,
	data: rawptr = nil,
	cells := Colouring{},
) {
	scratch_guard()
	mesh := sc.space.mesh
	fd := mesh_facet_dim(mesh^)
	for it := colour_iterator(serial_colouring(mesh, cells)); id in colour_iterator_next(&it) {
		cell := mesh.cells[id]
		if sc.space.over == .Cells {
			interpolate_site(A, I, F, geo, sc, cell_site(cell), f, data)
			continue
		}
		for fid, lf in cell.conn[fd] {
			if mesh.facets[fid].cofaces[0].entity != cell.id { continue } // each facet once
			interpolate_site(A, I, F, geo, sc, cell_facet_site(cell, lf), f, data)
		}
	}
}

@(private = "file")
interpolate_site :: proc($A, $I, $F: int, geo: Geometry, sc: Space_Coeffs, site: Site, f: Interp_Proc, data: rawptr) {
	if e, _ := space_elem(sc.space, site); sc.space.numbering.l2g[e.id] == nil { return } // outside the space
	ip := interpolator(A, I, F, geo, sc, site)
	for blk in interpolator_next(&ip) { f(blk.points, blk.out, data) }
	interpolator_flush(&ip)
}

// `cells`, or every cell as one colour when it is empty. The empty colouring is only safe on one rank.
@(private = "file")
serial_colouring :: proc(mesh: ^Mesh, cells: Colouring) -> Colouring {
	if cells.offsets != nil { return cells }
	assert(rank_count() == 1, "interpolating on several ranks needs a cell colouring")
	offsets := make([]int, 2, scratch())
	offsets[1] = len(mesh.cells)
	items := make([]Entity_ID, len(mesh.cells), scratch())
	for &id, i in items { id = Entity_ID(i) }
	return {offsets, items}
}
