package fio

import "core:fmt"
import "core:io"
import "core:log"
import "core:mem"
import "core:os"
import "core:path/filepath"
import "core:strconv"

import fe "../"

VTU_Config :: struct {
	pvd_path: Maybe(string), // pvd will be written on takedown.
}

VTU_Writer :: struct {
	viz_mesh:   VTU_Mesh,
	cfg:        VTU_Config,
	pvd_fd:     ^os.File,
	pvd_writer: XML_Writer,
}

VTU_Mesh :: struct {
	vertices:     [][3]f64,
	connectivity: []i32,
	offsets:      []i32,
	types:        []u8,
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

// Vertices are every cell's rule points, cell after cell, matching the Output_Writer's points.
vtu_create :: proc(mesh: fe.Mesh, geo: fe.Geometry, order: fe.Order, rules: Output_Rules, cfg: VTU_Config, alloc: mem.Allocator) -> ^VTU_Writer {
	context.allocator = alloc
	context.temp_allocator = fe.scratch()
	fe.scratch_guard()

	w := new(VTU_Writer)
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

	for cell in mesh.cells {
		fe.scratch_guard()
		rule := rules[cell.type]

		// geo_points needs the dimension at compile time
		site := fe.cell_site(cell)
		phys_points: fe.Pvec(f64)
		switch geo.sc.space.fields {
		case 1: phys_points = fe.geo_points(1, geo, site, rule, context.temp_allocator)
		case 2: phys_points = fe.geo_points(2, geo, site, rule, context.temp_allocator)
		case 3: phys_points = fe.geo_points(3, geo, site, rule, context.temp_allocator)
		case: unreachable()
		}

		base_vertex_idx := i32(len(vertices))

		for i in 0 ..< len(rule.points) {
			pvp := fe.pvec_at_point(phys_points, i)
			padded_point: [3]f64
			copy(padded_point[:], pvp.data)
			append(&vertices, padded_point)
		}

		for sub in SUBCELLS[cell.type][order].connectivity {
			for node_index in sub { append(&connectivity, base_vertex_idx + i32(node_index)) }
			append(&offsets, i32(len(connectivity)))
			append(&types, u8(VTK_ELEMENT_TYPE_FROM_NATIVE[cell.type]))
		}
	}

	w.viz_mesh = {
		vertices     = vertices[:],
		connectivity = connectivity[:],
		offsets      = offsets[:],
		types        = types[:],
	}
	return w
}

vtu_write :: proc(w: ^VTU_Writer, req: Write_Request) -> bool {
	if fe.rank_count() != 1 { panic("Tried to write file from multiple threads at once.") }

	if len(req.fields) == 0 {
		log.warnf("Nothing to output: Not writing to %s", req.path); return false
	}

	context.allocator = fe.scratch()
	fe.scratch_guard()

	path := fmt.aprintf("%s.vtu", req.path)
	if step, has := req.step.?; has { path = fmt.aprintf("%s_%d.vtu", req.path, step) }

	fd, err := os.open(path, {.Read, .Write, .Create, .Trunc})

	if err != nil {
		log.errorf(os.error_string(err))
		return false
	}

	defer os.close(fd)

	xml_w := XML_Writer {
		w = io.to_writer(os.to_writer(fd)),
	}

	vm := w.viz_mesh

	// appended blocks are a u64 byte count then the bytes, offsets known up front so the data streams to the file
	blocks := make([dynamic][]u8)
	append(&blocks, mem.slice_to_bytes(vm.vertices))
	append(&blocks, mem.slice_to_bytes(vm.connectivity))
	append(&blocks, mem.slice_to_bytes(vm.offsets))
	append(&blocks, mem.slice_to_bytes(vm.types))
	for field in req.fields { append(&blocks, mem.slice_to_bytes(field.data)) }
	block_offsets := make([]string, len(blocks))
	at := 0
	for b, i in blocks {
		block_offsets[i] = fmt.aprintf("%d", at)
		at += 8 + len(b)
	}

	xml_writer_write_string(&xml_w, "<?xml version=\"1.0\"?>\n")

	xml_open_tag(
		&xml_w,
		.VTKFile,
		{{"type", "UnstructuredGrid"}, {"version", "0.1"}, {"byte_order", "LittleEndian"}, {"header_type", "UInt64"}},
	)
	defer xml_close_tag(&xml_w)

	num_points := fmt.aprintf("%d", len(vm.vertices))
	num_cells := fmt.aprintf("%d", len(vm.offsets))

	xml_open_tag(&xml_w, .UnstructuredGrid)
	xml_open_tag(&xml_w, .Piece, {{"NumberOfPoints", num_points}, {"NumberOfCells", num_cells}})

	{
		xml_open_tag(&xml_w, .Points)
		defer xml_close_tag(&xml_w)

		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "Float64"}, {"NumberOfComponents", "3"}, {"format", "appended"}, {"offset", block_offsets[0]}},
		)
		xml_close_tag(&xml_w)
	}

	{
		xml_open_tag(&xml_w, .Cells)
		defer xml_close_tag(&xml_w)

		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "Int32"}, {"Name", "connectivity"}, {"format", "appended"}, {"offset", block_offsets[1]}},
		)
		xml_close_tag(&xml_w)

		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "Int32"}, {"Name", "offsets"}, {"format", "appended"}, {"offset", block_offsets[2]}},
		)
		xml_close_tag(&xml_w)

		xml_open_tag(
			&xml_w,
			.DataArray,
			{{"type", "UInt8"}, {"Name", "types"}, {"format", "appended"}, {"offset", block_offsets[3]}},
		)
		xml_close_tag(&xml_w)
	}

	// actual field data
	{
		xml_open_tag(&xml_w, .PointData)
		defer xml_close_tag(&xml_w)

		for field, i in req.fields {
			xml_open_tag(
				&xml_w,
				.DataArray,
				{
					{"type", "Float64"},
					{"Name", field.name},
					{"format", "appended"},
					{"NumberOfComponents", fmt.aprintf("%d", field.n_cmpnts)},
					{"offset", block_offsets[4 + i]},
				},
			)
			xml_close_tag(&xml_w)
		}
	}

	xml_close_tag(&xml_w) // close piece
	xml_close_tag(&xml_w) // close unstructured grid

	xml_open_tag(&xml_w, .AppendedData, {{"encoding", "raw"}})
	xml_writer_write_string(&xml_w, "_")
	for b in blocks {
		n := transmute([8]u8)u64(len(b))
		xml_writer_write_bytes(&xml_w, n[:])
		xml_writer_write_bytes(&xml_w, b)
	}
	xml_close_tag(&xml_w)

	if xml_w.err != nil {
		log.errorf("failed writing vtu file %q: %v", path, xml_w.err)
		return false
	}

	if pvd_path, has := w.cfg.pvd_path.?; has {
		b: [32]u8
		time_val := req.time.? or_else cast(f64)(req.step.? or_else 0)
		time_str := strconv.write_float(b[:], time_val, 'f', 8, 64)

		// readers resolve the file against the pvd's directory, not the working directory
		file, rel_err := filepath.rel(filepath.dir(pvd_path), path)
		if rel_err != nil { file = path }

		xml_open_tag(
			&w.pvd_writer,
			.DataSet,
			{{"timestep", time_str[1:]}, {"group", ""}, {"part", "0"}, {"file", file}}, // [1:] slices off the leading '+'
		)
		xml_close_tag(&w.pvd_writer)
	}

	return true
}

vtu_destroy :: proc(w: ^VTU_Writer) {
	if _, has := w.cfg.pvd_path.?; has {
		xml_close_tag(&w.pvd_writer) // closes Collection
		xml_close_tag(&w.pvd_writer) // closes VTKFile
		os.close(w.pvd_fd)
	}
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
	if xml_w.err != nil { return }
	_, err := io.write_string(xml_w.w, s)
	if err != nil {
		xml_w.err = err
	}
}

@(private = "file")
xml_writer_write_bytes :: proc(xml_w: ^XML_Writer, data: []u8) {
	if xml_w.err != nil { return }
	_, err := io.write(xml_w.w, data)
	if err != nil {
		xml_w.err = err
	}
}


@(private = "file")
xml_writer_printf :: proc(xml_w: ^XML_Writer, format: string, args: ..any) {
	if xml_w.err != nil { return }
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
