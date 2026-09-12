package fio


import "core:mem/virtual"
import "core:log"
import "core:io"
import "core:os"
import "core:strings"
import  "core:strconv"

import fe"../"

GMSH_EXPECTED_MSH_VERSION :: "2.2 1 8"

@(private = "file")
ALIGNMENT_TOLERANCE :: 1e-12

@(private = "file")
Gmsh_Element_Type :: enum {
	MSH_LINE_2     = 1,
	MSH_LINE_3     = 8,
	MSH_TRIANGLE_3 = 2,
	MSH_TRIANGLE_6 = 9,
	MSH_QUAD_4     = 3,
	MSH_QUAD_9     = 10,
	MSH_TETRA_4    = 4,
	MSH_TETRA_10   = 11,
	MSH_HEX_8      = 5,
	MSH_HEX_27     = 12,
	MSH_POINT      = 15,
}

@(private = "file")
GMSH_ELEMENT_NUM_NODES := #sparse[Gmsh_Element_Type]int {
	.MSH_LINE_2     = 2,
	.MSH_LINE_3     = 3,
	.MSH_TRIANGLE_3 = 3,
	.MSH_TRIANGLE_6 = 6,
	.MSH_QUAD_4     = 4,
	.MSH_QUAD_9     = 9,
	.MSH_TETRA_4    = 4,
	.MSH_TETRA_10   = 10,
	.MSH_HEX_8      = 8,
	.MSH_HEX_27     = 27,
	.MSH_POINT      = 1,
}


@(private = "file")
SUPPORTED_ELEMENTS :: bit_set[Gmsh_Element_Type] {
	.MSH_LINE_2,
	.MSH_LINE_3,
	.MSH_TRIANGLE_3,
	.MSH_TRIANGLE_6,
	.MSH_QUAD_4,
	.MSH_QUAD_9,
	.MSH_TETRA_4,
	.MSH_TETRA_10,
	.MSH_HEX_8,
	.MSH_HEX_27,
	.MSH_POINT,
}

@(private = "file")
Raw_Element :: struct {
	type:         Gmsh_Element_Type,
	tags:         []i32,
	node_indices: []int,
}

@(private="file")
Gmsh_Validation_Error :: enum {
	Parse_Error,
	Unsupported_Element,
	Incorrect_Dimension,
}

@(private="file")
Gmsh_Error :: union {
	io.Error,
	os.Error,
	Gmsh_Validation_Error,
}


gmsh_load_mesh :: proc(mesh_data: []u8, mesh: ^fe.Mesh) -> bool {
	context.temp_allocator = fe.scratch()
	fe.scratch_guard()

	context.allocator = virtual.arena_allocator(&mesh.arena)


	return true
}

@(private = "file")
gmsh_type_to_element_info :: proc(type: Gmsh_Element_Type) -> (fe.Element_Type, fe.Order) {
	switch type {
	case .MSH_LINE_2:
		return .Line, .O1
	case .MSH_LINE_3:
		return .Line, .O2
	case .MSH_TRIANGLE_3:
		return .Tri, .O1
	case .MSH_TRIANGLE_6:
		return .Tri, .O2
	case .MSH_QUAD_4:
		return .Quad, .O1
	case .MSH_QUAD_9:
		return .Quad, .O2
	case .MSH_TETRA_4:
		return .Tet, .O1
	case .MSH_TETRA_10:
		return .Tet, .O2
	case .MSH_HEX_8:
		return .Hex, .O1
	case .MSH_HEX_27:
		return .Hex, .O2
	case .MSH_POINT:
		return .Point, .O1
	case:
		unreachable()
	}
}
