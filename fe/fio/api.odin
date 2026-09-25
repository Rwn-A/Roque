package fio

import "core:os"
import "core:mem/virtual"
import "core:log"

import fe"../"

//== mesh input

Mesh_Format :: enum {
	GMSH_V2_BINARY,
}

load_mesh :: proc(path: string, format: Mesh_Format) -> (m: fe.Mesh, ok: bool) {
	context.temp_allocator = fe.scratch()
	fe.scratch_guard()

	assert(format == .GMSH_V2_BINARY)

	contents, err := os.read_entire_file_from_path(path, context.temp_allocator)
	if err != nil {
		log.error(os.error_string(err)); return {}, false
	}

	if err := virtual.arena_init_growing(&m.arena); err != nil { log.panic("Could not create arena.") }

	ok =  gmsh_load_mesh(contents, &m)

	return m, ok
}

//== soln input

// TODO: solution input

//== output
Output_Field :: struct{
	name: string,
	value_components: int, 	// amount of individual scalars at a point the output will produce.
	data: [][]f64, //per cell, per rule point
}

Output_Config :: union{
	VTU_Config,
}

Output_Writer :: union{
	^VTU_Writer,
}

// Only filled for the element types of the mesh cells
// These rules are ALWAYS equivalent to the nodal lagrange points of the requested visualization order.
// This means, for FE solutions, using lagrange basis, of the same order, you can simply gather and write
// without needing to reconstruct.
Output_Rules :: [fe.Element_Type]fe.Rule

// Output in the coordinates of the coordinate space, not the ambient coordinates of the mesh.
// Order allows one to refine the visualization mesh to better visualize higher order solutions over it.
output_setup :: proc(
	mesh: fe.Mesh,
	coord_space: ^fe.Space,
	coords: []f64,
	order: fe.Order,
	cfg: Output_Config
) -> (Output_Writer, Output_Rules) {
	rules: Output_Rules
	for elem in mesh.encountered_cell_types{
		rules[elem] = {points = SUBCELLS[elem][order].points, element = elem}
	}

	switch c in cfg{
	case VTU_Config:
		writer := vtu_output_writer(mesh, order, coord_space, coords, c)
		return writer, rules
	case:
		log.panic("Output_Config cannot be nil")
	}

}

Write_Request :: struct {
	path: string, // excluding extension
	fields: []Output_Field,
	step: Maybe(int), // appends _step to the path.
	time: Maybe(f64), // optional
}

output_write :: proc(writer: Output_Writer, req: Write_Request) {
	switch w in writer{
	case ^VTU_Writer: vtu_write(w, req)
	case: unreachable()
	}
}

output_takedown :: proc(writer: Output_Writer) {
	switch w in writer{
	case ^VTU_Writer: vtu_takedown(w)
	case: unreachable()
	}
}

output_field_create :: proc(mesh: fe.Mesh, name: string, cmpnts: int, rules: Output_Rules, alloc := context.allocator) -> Output_Field{
	of: Output_Field
	of.data = make([][]f64, len(mesh.cells))
	for cell in mesh.cells{
		of.data[cell.id] = make([]f64, len(rules[cell.type].points) * cmpnts)
	}
	of.value_components = cmpnts
	of.name = name
	return of
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
