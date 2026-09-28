package fio

import "core:os"
import "core:mem/virtual"
import "core:log"

import fe"../"

//== mesh input

Mesh_Format :: enum {
	Gmsh_V2_Binary,
}

load_mesh :: proc(path: string, format: Mesh_Format) -> (m: fe.Mesh, ok: bool) {
	context.temp_allocator = fe.scratch()
	fe.scratch_guard()

	assert(format == .Gmsh_V2_Binary)

	contents, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != nil {
		log.error(os.error_string(err)); return {}, false
	}

	if err := virtual.arena_init_growing(&m.arena); err != nil { log.panic("Could not create arena.") }

	if !gmsh_load_mesh(contents, &m) {
		fe.mesh_destroy(&m)
		return {}, false
	}
	return m, true
}

//== soln input

// TODO: solution input

//== output

// Only filled for the element types of the mesh cells
// These rules are ALWAYS equivalent to the nodal lagrange points of the requested visualization order.
Output_Rules :: [fe.Element_Type]fe.Rule

Output_Config :: union {
	VTU_Config,
}

// Output points are every cell's rule points, cell after cell.
Output_Writer :: struct {
	rules:       Output_Rules,
	point_start: []int, // [cell] first output point of the cell, [len(cells)] every point
	format:      union {
		^VTU_Writer,
	},
	arena:       virtual.Arena,
}

// Values at every output point, n_cmpnts per point.
Output_Field :: struct {
	name:     string,
	n_cmpnts: int,
	data:     []f64,
}

Write_Request :: struct {
	path:   string, // excluding extension
	fields: []Output_Field,
	step:   Maybe(int), // appends _step to the path.
	time:   Maybe(f64), // optional
}

// Output in the coordinates of the geometry, not the ambient coordinates of the mesh.
// Order allows one to refine the visualization mesh to better visualize higher order solutions over it.
output_create :: proc(mesh: fe.Mesh, geo: fe.Geometry, order: fe.Order, cfg: Output_Config) -> ^Output_Writer {
	w := virtual.arena_growing_bootstrap_new(Output_Writer, "arena") or_else log.panic("Could not create arena.")
	alloc := virtual.arena_allocator(&w.arena)

	for elem in mesh.encountered_cell_types {
		assert(len(SUBCELLS[elem][order].points) > 0, "no output subcells for this element type and order")
		w.rules[elem] = {points = SUBCELLS[elem][order].points, element = elem}
	}
	w.point_start = make([]int, len(mesh.cells) + 1, alloc)
	for cell, i in mesh.cells { w.point_start[i + 1] = w.point_start[i] + len(w.rules[cell.type].points) }

	switch c in cfg {
	case VTU_Config: w.format = vtu_create(mesh, geo, order, w.rules, c, alloc)
	case: log.panic("Output_Config cannot be nil")
	}
	return w
}

output_destroy :: proc(w: ^Output_Writer) {
	switch f in w.format {
	case ^VTU_Writer: vtu_destroy(f)
	case: unreachable()
	}
	arena := w.arena // the writer lives in its own arena
	virtual.arena_destroy(&arena)
}

output_write :: proc(w: ^Output_Writer, req: Write_Request) -> bool {
	switch f in w.format {
	case ^VTU_Writer: return vtu_write(f, req)
	case: unreachable()
	}
}

// A zeroed field over the writer's output points.
output_field_create :: proc(w: ^Output_Writer, name: string, n_cmpnts: int, alloc := context.allocator) -> Output_Field {
	return {name = name, n_cmpnts = n_cmpnts, data = make([]f64, w.point_start[len(w.point_start) - 1] * n_cmpnts, alloc)}
}

// The field's values on one cell's output points.
output_field_cell :: proc(w: ^Output_Writer, f: Output_Field, cell: fe.Entity_ID) -> []f64 {
	return f.data[w.point_start[cell] * f.n_cmpnts:w.point_start[cell + 1] * f.n_cmpnts]
}

// Quantity `q` of a cell space's F fields at the output points, through fe.space_eval and its assumptions. Cells
// outside the space are left zero.
output_field_from_space :: proc(
	$A, $I, $F: int,
	w: ^Output_Writer,
	geo: fe.Geometry,
	sc: fe.Space_Coeffs,
	q: fe.Basis_Quantity,
	name: string,
	alloc := context.allocator,
) -> Output_Field {
	mesh := sc.space.mesh
	bd := fe.space_bd(sc.space, &mesh.cells[0])
	out := output_field_create(w, name, fe.basis_quantity_phys_cmpnts(bd, q, A) * F, alloc)
	output_field_fill_space(A, I, F, w, geo, sc, q, out)
	return out
}

// output_field_from_space into an existing field, reusing its storage (time series).
output_field_fill_space :: proc(
	$A, $I, $F: int,
	w: ^Output_Writer,
	geo: fe.Geometry,
	sc: fe.Space_Coeffs,
	q: fe.Basis_Quantity,
	out: Output_Field,
) {
	assert(sc.space.over == .Cells, "output is over the cells")
	for cell in sc.space.mesh.cells {
		if sc.space.numbering.l2g[cell.id] == nil { continue }
		fe.scratch_guard()
		site := fe.cell_site(cell)
		m := fe.geo_map(A, I, geo, site, w.rules[cell.type], fe.scratch())
		x := fe.space_eval(F, sc, site, w.rules[cell.type], m, q, fe.scratch())
		assert(x.cmpnts * x.fields == out.n_cmpnts)
		copy(output_field_cell(w, out, cell.id), x.data)
	}
}

// `f` at the output points, n_cmpnts values per point in one field.
output_field_from_proc :: proc(
	$A: int,
	w: ^Output_Writer,
	geo: fe.Geometry,
	name: string,
	n_cmpnts: int,
	f: fe.Interp_Proc,
	data: rawptr = nil,
	alloc := context.allocator,
) -> Output_Field {
	out := output_field_create(w, name, n_cmpnts, alloc)
	for cell in geo.sc.space.mesh.cells {
		fe.scratch_guard()
		rule := w.rules[cell.type]
		x := fe.geo_points(A, geo, fe.cell_site(cell), rule, fe.scratch())
		vals := fe.Pvec(f64){x.points, n_cmpnts, 1, output_field_cell(w, out, cell.id)}
		f(x, vals, data)
	}
	return out
}

//== subcell rules

Reference_Subcell :: struct {
	points:       []fe.Ref_Vec,
	connectivity: [][]int,
}

@(rodata)
SUBCELLS := [fe.Element_Type][fe.Order]Reference_Subcell {
	.Point         = {},
	.Line          = LINE_SUBCELL,
	.Quad = QUAD_SUBCELL,
	.Tri      = TRI_SUBCELL,
	.Tet   = TET_SUBCELL,
	.Hex    = HEX_SUBCELL,
}

LINE_SUBCELL :: #partial [fe.Order]Reference_Subcell {
	.O1 = {points = {{-1, 0, 0}, {1, 0, 0}}, connectivity = {{0, 1}}},
	.O2 = {points = {{-1, 0, 0}, {1, 0, 0}, {0, 0, 0}}, connectivity = {{0, 2}, {2, 1}}},
}

QUAD_SUBCELL :: #partial [fe.Order]Reference_Subcell {
	.O1 = {points = {{-1, -1, 0}, {1, -1, 0}, {1, 1, 0}, {-1, 1, 0}}, connectivity = {{0, 1, 2, 3}}},
	.O2 = {
		points = {
			{-1, -1, 0},
			{1, -1, 0},
			{1, 1, 0},
			{-1, 1, 0},
			{0, -1, 0},
			{1, 0, 0},
			{0, 1, 0},
			{-1, 0, 0},
			{0, 0, 0},
		},
		connectivity = {{0, 4, 8, 7}, {4, 1, 5, 8}, {7, 8, 6, 3}, {8, 5, 2, 6}},
	},
}

TRI_SUBCELL :: #partial [fe.Order]Reference_Subcell {
	.O1 = {points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}}, connectivity = {{0, 1, 2}}},
	.O2 = {
		points = {
			{0, 0, 0},
			{1, 0, 0},
			{0, 1, 0},
			{0.5, 0, 0},
			{0.5, 0.5, 0},
			{0, 0.5, 0},
		},
		connectivity = {
			{0, 3, 5},
			{3, 1, 4},
			{5, 4, 3},
			{5, 2, 4},
		},
	},
}

TET_SUBCELL :: #partial [fe.Order]Reference_Subcell {
	.O1 = {points = {{0, 0, 0}, {1, 0, 0}, {0, 1, 0}, {0, 0, 1}}, connectivity = {{0, 1, 2, 3}}},
	.O2 = {
		points = {
			{0, 0, 0},
			{1, 0, 0},
			{0, 1, 0},
			{0, 0, 1},
			{0.5, 0, 0},
			{0.5, 0.5, 0},
			{0, 0.5, 0},
			{0, 0, 0.5},
			{0.5, 0, 0.5},
			{0, 0.5, 0.5},
		},
		connectivity = {
			{0, 4, 6, 7},
			{1, 5, 4, 8},
			{2, 6, 5, 9},
			{3, 7, 9, 8},
			{4, 5, 6, 8},
			{6, 5, 8, 9},
			{4, 6, 7, 8},
			{6, 7, 8, 9},
		},
	},
}

HEX_SUBCELL :: #partial [fe.Order]Reference_Subcell {
	.O1 = {
		points = {{-1, -1, -1}, {1, -1, -1}, {1, 1, -1}, {-1, 1, -1}, {-1, -1, 1}, {1, -1, 1}, {1, 1, 1}, {-1, 1, 1}},
		connectivity = {{0, 1, 2, 3, 4, 5, 6, 7}},
	},
	.O2 = {
		points = {
			{-1, -1, -1},
			{1, -1, -1},
			{1, 1, -1},
			{-1, 1, -1},
			{-1, -1, 1},
			{1, -1, 1},
			{1, 1, 1},
			{-1, 1, 1},
			{0, -1, -1},
			{1, 0, -1},
			{0, 1, -1},
			{-1, 0, -1},
			{0, -1, 1},
			{1, 0, 1},
			{0, 1, 1},
			{-1, 0, 1},
			{-1, -1, 0},
			{1, -1, 0},
			{1, 1, 0},
			{-1, 1, 0},
			{0, 0, -1},
			{0, 0, 1},
			{0, -1, 0},
			{0, 1, 0},
			{-1, 0, 0},
			{1, 0, 0},
			{0, 0, 0},
		},
		connectivity = {
			{0, 8, 20, 11, 16, 22, 26, 24},
			{8, 1, 9, 20, 22, 17, 25, 26},
			{11, 20, 10, 3, 24, 26, 23, 19},
			{20, 9, 2, 10, 26, 25, 18, 23},
			{16, 22, 26, 24, 4, 12, 21, 15},
			{22, 17, 25, 26, 12, 5, 13, 21},
			{24, 26, 23, 19, 15, 21, 14, 7},
			{26, 25, 18, 23, 21, 13, 6, 14},
		},
	},
}
