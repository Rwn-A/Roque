package fio

import "core:fmt"
import "core:io"
import "core:log"
import "core:mem"
import "core:mem/virtual"
import "core:os"
import "core:path/filepath"
import "core:slice"
import "core:strconv"

import fe "../"

VTU_Config :: struct {
	pvd_path: Maybe(string), // pvd will be written on takedown.
}

VTU_Writer :: struct {
	viz_mesh: VTU_Mesh,
	cfg:      VTU_Config,
	arena:    virtual.Arena,
	pvd_fd:     ^os.File,
	pvd_writer: XML_Writer,
}

VTU_Mesh :: struct {
	vertices:     [][3]f64,
	connectivity: []i32,
	offsets:      []i32,
	types:        []u8,
	output_order: fe.Order,
}

VTK_Element_Type :: enum u8 {
	Point         = 1,
	Line          = 3,
	Triangle      = 5,
	Quadrilateral = 9,
	Tetrahedron   = 10,
	Hexahedron    = 12,
}

@(rodata)
VTK_ELEMENT_TYPE_FROM_NATIVE := [fe.Element_Type]VTK_Element_Type {
	.Point = .Point,
	.Line  = .Line,
	.Tri   = .Triangle,
	.Quad  = .Quadrilateral,
	.Tet   = .Tetrahedron,
	.Hex   = .Hexahedron,
}

vtu_output_writer :: proc(
	mesh: fe.Mesh,
	order: fe.Order,
	coord_space: ^fe.Space,
	coords: []f64,
	cfg: VTU_Config,
) -> ^VTU_Writer {
	w, err := virtual.arena_growing_bootstrap_new_by_name(VTU_Writer, "arena")

	if err != nil { log.panic("Could not create arena.") }

	context.allocator = virtual.arena_allocator(&w.arena)
	context.temp_allocator = fe.scratch()
	fe.scratch_guard()

	w.cfg = cfg

	if pvd_path, has := cfg.pvd_path.?; has {
		fd, err := os.open(pvd_path, {.Read, .Write, .Create, .Trunc})
		if err != nil { log.panic("Could not create pvd file.") }

		w.pvd_fd = fd
		w.pvd_writer.w = io.to_writer(os.to_writer(fd))

		xml_writer_write_string(&w.pvd_writer, "<?xml version=\"1.0\"?>\n")
		xml_open_tag(
			&w.pvd_writer,
			.VTKFile,
			{{"type", "Collection"}, {"version", "0.1"}, {"byte_order", "LittleEndian"}},
		)
		xml_open_tag(&w.pvd_writer, .Collection)
	}

	vertices := make([dynamic][3]f64)
	connectivity := make([dynamic]i32)
	offsets := make([dynamic]i32)
	types := make([dynamic]u8)

	for cell, i in mesh.cells {
		fe.scratch_guard()

		rule := fe.Rule {
			ref_points = SUBCELLS[cell.type][order].points,
			weights    = nil,
			element    = cell.type,
		}

		basis := fe.bstore_get_interior(fe.space_bd(coord_space, cell.type), rule)
		elem_coords := fe.space_gather(f64, {coord_space, coords}, cell.id, context.temp_allocator)

		phys_points := fe.pvec_create(
			f64,
			len(rule.ref_points),
			{.CMPNTS = 1, .FIELDS = coord_space.fields},
			context.temp_allocator,
		)

		// Because contractions need compile-time dims
		switch coord_space.fields {
		case 1: fe.contract_eval({.CMPNTS = 1, .FIELDS = 1}, phys_points, elem_coords, basis[.Scalar])
		case 2: fe.contract_eval({.CMPNTS = 1, .FIELDS = 2}, phys_points, elem_coords, basis[.Scalar])
		case 3: fe.contract_eval({.CMPNTS = 1, .FIELDS = 3}, phys_points, elem_coords, basis[.Scalar])
		case: unreachable()
		}

		base_vertex_idx := i32(len(vertices))

		for i in 0 ..< len(rule.ref_points) {
			pvp := fe.pvec_at_point(phys_points, i)
			padded_point: [3]f64
			copy(padded_point[:], pvp.data)
			append(&vertices, padded_point)
		}


		for subcell in SUBCELLS[cell.type][order].connectivity {
			for node_index in subcell { append(&connectivity, base_vertex_idx + i32(node_index)) }
			append(&offsets, i32(len(connectivity)))
			append(&types, u8(VTK_ELEMENT_TYPE_FROM_NATIVE[cell.type]))
		}
	}

	w.viz_mesh = {
		vertices     = vertices[:],
		connectivity = connectivity[:],
		offsets      = offsets[:],
		types        = types[:],
		output_order = order,
	}

	return w
}

vtu_write :: proc(w: ^VTU_Writer, req: Write_Request) -> bool {
	if len(req.fields) == 0 {
		log.warnf("Nothing to output: Not writing to %s", req.path); return false
	}

	context.allocator = fe.scratch()
	fe.scratch_guard()

	number_conversion_buffer: [32]u8
	step_str: string
	if step, has := req.step.?; has {
		step_str = strconv.write_int(number_conversion_buffer[:], i64(step), 10)
	}

	path := fmt.aprintf("%s_%s.vtu", req.path, step_str)

	fd, err := os.open(path, {.Read, .Write, .Create, .Trunc})

	if err != nil {
		log.errorf(os.error_string(err))
		return false
	}

	defer os.close(fd)

	xml_w := XML_Writer {
		w = io.to_writer(os.to_writer(fd)),
	}

	vtu_mesh := w.viz_mesh

	xml_writer_write_string(&xml_w, "<?xml version=\"1.0\"?>\n")

	xml_open_tag(
		&xml_w,
		.VTKFile,
		{{"type", "UnstructuredGrid"}, {"version", "0.1"}, {"byte_order", "LittleEndian"}, {"header_type", "UInt64"}},
	)
	defer xml_close_tag(&xml_w)

	b0, b1: [32]u8
	num_points := strconv.write_int(b0[:], cast(i64)len(vtu_mesh.vertices), 10)
	num_cells := strconv.write_int(b1[:], cast(i64)len(vtu_mesh.offsets), 10)

	xml_open_tag(&xml_w, .UnstructuredGrid)
	xml_open_tag(&xml_w, .Piece, {{"NumberOfPoints", num_points}, {"NumberOfCells", num_cells}})

	appended := make([dynamic]u8)
	defer delete(appended)

	{
		xml_open_tag(&xml_w, .Points)
		defer xml_close_tag(&xml_w)

		offset := appended_write(&appended, mem.slice_to_bytes(vtu_mesh.vertices))

		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "Float64"}, {"NumberOfComponents", "3"}, {"format", "appended"}, {"offset", offset}},
		)
		defer xml_close_tag(&xml_w)
	}

	{
		xml_open_tag(&xml_w, .Cells)
		defer xml_close_tag(&xml_w)

		offset_connectivity := appended_write(&appended, mem.slice_to_bytes(vtu_mesh.connectivity))

		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "Int32"}, {"Name", "connectivity"}, {"format", "appended"}, {"offset", offset_connectivity}},
		)
		xml_close_tag(&xml_w)

		offset_offsets := appended_write(&appended, mem.slice_to_bytes(vtu_mesh.offsets))
		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "Int32"}, {"Name", "offsets"}, {"format", "appended"}, {"offset", offset_offsets}},
		)
		xml_close_tag(&xml_w)

		offset_types := appended_write(&appended, mem.slice_to_bytes(vtu_mesh.types))
		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "UInt8"}, {"Name", "types"}, {"format", "appended"}, {"offset", offset_types}},
		)
		xml_close_tag(&xml_w)
	}

	// actual field data
	{
		xml_open_tag(&xml_w, .PointData)
		defer xml_close_tag(&xml_w)

		for field in req.fields {
			data_offset := appended_write_chunks(&appended, field.data)

			str_components := strconv.write_int(b0[:], cast(i64)field.value_components, 10)
			xml_open_tag(
				&xml_w,
				.DataArray,
				{
					{"type", "Float64"},
					{"Name", field.name},
					{"format", "appended"},
					{"NumberOfComponents", str_components},
					{"offset", data_offset},
				},
			)
			xml_close_tag(&xml_w)
		}
	}

	xml_close_tag(&xml_w) // close piece
	xml_close_tag(&xml_w) // close unstructered grid

	xml_open_tag(&xml_w, .AppendedData, {{"encoding", "raw"}})
	xml_writer_write_string(&xml_w, "_")
	xml_writer_write_bytes(&xml_w, appended[:])
	xml_close_tag(&xml_w)

	if xml_w.err != nil {
		log.errorf("failed writing vtu file %q: %v", path, xml_w.err)
		return false
	}

	if _, has := w.cfg.pvd_path.?; has {
		b: [32]u8
		time_val := req.time.? or_else cast(f64)(req.step.? or_else 0)
		time_str := strconv.write_float(b[:], time_val, 'f', 8, 64)

		xml_open_tag(
			&w.pvd_writer,
			.DataSet,
			{{"timestep", time_str[1:]}, {"group", ""}, {"part", "0"}, {"file", path}}, // [:1] slices off the leading '+'
		)
		xml_close_tag(&w.pvd_writer)
	}

	return true

	appended_write :: proc(appended: ^[dynamic]u8, data: []u8) -> (offset: string) {
		@(static) b: [32]u8
		offset = strconv.write_int(b[:], cast(i64)len(appended), 10)
		len_bytes := transmute([8]u8)uint(len(data))
		append(appended, ..(len_bytes[:]))
		append(appended, ..data)
		return offset
	}

	appended_write_chunks :: proc(appended: ^[dynamic]u8, chunks: [][]f64) -> (offset: string) {
		@(static) b: [32]u8
		offset = strconv.write_int(b[:], cast(i64)len(appended), 10)

		header_pos := len(appended)
		resize(appended, header_pos + 8) // placeholder length, patched below

		data_start := len(appended)
		for chunk in chunks {
			append(appended, ..mem.slice_to_bytes(chunk))
		}
		data_len := len(appended) - data_start

		len_bytes := transmute([8]u8)uint(data_len)
		copy(appended[header_pos:header_pos + 8], len_bytes[:])

		return offset
	}
}

vtu_takedown :: proc(w: ^VTU_Writer) {
	if _, has := w.cfg.pvd_path.?; has {
		xml_close_tag(&w.pvd_writer) // closes Collection
		xml_close_tag(&w.pvd_writer) // closes VTKFile
		os.close(w.pvd_fd)
	}

	arena := w.arena //stops arena from freeing itself
	virtual.arena_destroy(&arena)
}


@(private = "file")
XML_Tag :: enum {
	VTKFile,
	UnstructuredGrid,
	Piece,
	Points,
	DataArray,
	Cells,
	CellData,
	PointData,
	Collection,
	DataSet,
	AppendedData,
}

@(private = "file")
MAX_NESTED_TAGS :: 16


@(private = "file")
XML_Writer :: struct {
	open: [dynamic; MAX_NESTED_TAGS]XML_Tag,
	w:    io.Stream,
	err:  io.Error,
}

@(private = "file")
XML_Attribute :: struct {
	key:   string,
	value: string,
}

@(private = "file")
xml_writer_write_string :: proc(xml_w: ^XML_Writer, s: string) {
	if xml_w.err != nil do return
	_, err := io.write_string(xml_w.w, s)
	if err != nil {
		xml_w.err = err
	}
}

@(private = "file")
xml_writer_write_bytes :: proc(xml_w: ^XML_Writer, data: []u8) {
	if xml_w.err != nil do return
	_, err := io.write(xml_w.w, data)
	if err != nil {
		xml_w.err = err
	}
}


@(private = "file")
xml_writer_printf :: proc(xml_w: ^XML_Writer, format: string, args: ..any) {
	if xml_w.err != nil do return
	s := fmt.tprintf(format, ..args)
	xml_writer_write_string(xml_w, s)
}


@(private = "file")
xml_open_tag :: proc(xml_w: ^XML_Writer, tag: XML_Tag, attrib: []XML_Attribute = {}) {
	append(&xml_w.open, tag)

	xml_writer_printf(xml_w, "<%s", tag)
	for a in attrib {
		xml_writer_printf(xml_w, " %s = \"%s\"", a.key, a.value)
	}
	xml_writer_write_string(xml_w, ">")
}

@(private = "file")
xml_close_tag :: proc(xml_w: ^XML_Writer) {
	tag := pop(&xml_w.open)
	xml_writer_printf(xml_w, "</%s>\n", tag)
}
