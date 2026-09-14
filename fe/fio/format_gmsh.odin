package fio


import "base:intrinsics"
import "core:log"
import "core:math"
import "core:mem/virtual"
import "core:slice"
import "core:strconv"
import "core:strings"

import fe "../"

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

@(private = "file")
Raw_Periodic_Link :: struct {
	dim:        int,
	slave_tag:  i32,
	master_tag: i32,
	transform:  fe.Small_Mat(4, 4, f64),
	pairs:      [][2]int, // {slave_node_idx, master_node_idx}, mesh-local indices
}

@(private = "file")
gmsh_type_to_element_info :: proc(type: Gmsh_Element_Type) -> (fe.Element_Type, fe.Order) {
	switch type {
	case .MSH_LINE_2: return .Line, .O1
	case .MSH_LINE_3: return .Line, .O2
	case .MSH_TRIANGLE_3: return .Tri, .O1
	case .MSH_TRIANGLE_6: return .Tri, .O2
	case .MSH_QUAD_4: return .Quad, .O1
	case .MSH_QUAD_9: return .Quad, .O2
	case .MSH_TETRA_4: return .Tet, .O1
	case .MSH_TETRA_10: return .Tet, .O2
	case .MSH_HEX_8: return .Hex, .O1
	case .MSH_HEX_27: return .Hex, .O2
	case .MSH_POINT: return .Point, .O1
	case: unreachable()
	}
}

@(private = "file")
Gmsh_Parser :: struct {
	data:           []u8,
	cursor:         int,
	swap_num_bytes: bool,
}

@(private = "file")
parser_consume :: proc(p: ^Gmsh_Parser) -> (u8, bool) {
	more := p.cursor < len(p.data)
	if !more {
		log.error("Gmsh: Unexpected EOF"); return 0, false
	}
	defer p.cursor += 1
	return p.data[p.cursor], true
}

@(private = "file")
ascii_line :: proc(p: ^Gmsh_Parser) -> (text: string, ok: bool) {
	start := p.cursor
	for (parser_consume(p) or_return) != '\n' {  }
	text = cast(string)p.data[start:p.cursor]
	return strings.trim_space(text), true
}


@(private = "file")
expect_ascii_line :: proc(p: ^Gmsh_Parser, text: string) -> bool {
	got := ascii_line(p) or_return
	if got == text { return true }
	log.errorf("Gmsh: Expected %s, got %s", text, got)
	return false
}

@(private = "file")
parse_int :: proc(text: string) -> (n: int, ok: bool) {
	n, ok = strconv.parse_int(text)

	if !ok { log.errorf("Gmsh: Expected integer got %s", text) }

	return n, ok
}

@(private = "file")
read_number :: proc($T: typeid, p: ^Gmsh_Parser) -> (n: T, ok: bool) {
	bytes: [size_of(T)]u8
	for i in 0 ..< size_of(T) { bytes[i] = parser_consume(p) or_return }

	t := (cast(^T)raw_data(bytes[:]))^
	if p.swap_num_bytes { t = intrinsics.byte_swap(t) }

	return t, true
}

gmsh_load_mesh :: proc(mesh_data: []u8, mesh: ^fe.Mesh) -> bool {
	context.temp_allocator = fe.scratch()
	fe.scratch_guard()

	context.allocator = virtual.arena_allocator(&mesh.arena)

	p := Gmsh_Parser {
		data = mesh_data,
	}

	{
		expect_ascii_line(&p, "$MeshFormat") or_return
		expect_ascii_line(&p, GMSH_EXPECTED_MSH_VERSION) or_return
		endianess := read_number(i32, &p) or_return
		p.swap_num_bytes = endianess != 1
		parser_consume(&p) or_return // newline between above number and mesh format text
		expect_ascii_line(&p, "$EndMeshFormat") or_return
	}

	mesh.boundary_names = make(map[string]fe.Boundary_ID)
	mesh.region_names = make(map[string]fe.Region_ID)

	names := make(map[string]struct {
			id:  int,
			dim: fe.Dimension,
		}, context.temp_allocator)

	{
		expect_ascii_line(&p, "$PhysicalNames") or_return
		num_groups := parse_int(ascii_line(&p) or_return) or_return

		max_dimension: int
		for _ in 0 ..< num_groups {
			line := ascii_line(&p) or_return

			components := strings.split(line, " ", context.temp_allocator)
			if len(components) < 3 { return false }

			group_dim := parse_int(components[0]) or_return
			if group_dim > max_dimension { max_dimension = group_dim }

			name := strings.clone(strings.trim(components[2], "\""))
			names[name] = {parse_int(components[1]) or_return, fe.Dimension(group_dim)}
		}
		expect_ascii_line(&p, "$EndPhysicalNames") or_return

		mesh.intrinsic_dim = fe.Dimension(max_dimension)

		for key, value in names {
			if value.dim < mesh.intrinsic_dim {
				mesh.boundary_names[key] = fe.Boundary_ID(value.id)
			} else {
				mesh.region_names[key] = fe.Region_ID(value.id)
			}
		}
	}

	gmsh_id_to_node_id := make(map[i32]int, context.temp_allocator)
	{
		expect_ascii_line(&p, "$Nodes") or_return
		num_nodes := parse_int(ascii_line(&p) or_return) or_return

		mesh.nodes = make([][3]f64, num_nodes)

		for i in 0 ..< num_nodes {
			id := read_number(i32, &p) or_return
			x := read_number(f64, &p) or_return
			y := read_number(f64, &p) or_return
			z := read_number(f64, &p) or_return
			gmsh_id_to_node_id[id] = i
			mesh.nodes[i] = {x, y, z}
		}
		parser_consume(&p) or_return
		expect_ascii_line(&p, "$EndNodes") or_return
	}

	raw_primary_elements := make([dynamic]Raw_Element, context.temp_allocator)
	raw_boundary_elements := make([dynamic]Raw_Element, context.temp_allocator)
	{
		expect_ascii_line(&p, "$Elements") or_return
		num_elements := parse_int(ascii_line(&p) or_return) or_return

		for elements_read: i32 = 0; elements_read < i32(num_elements); {
			type_integer := read_number(i32, &p) or_return
			num_in_group := read_number(i32, &p) or_return
			num_tags := read_number(i32, &p) or_return

			type := Gmsh_Element_Type(type_integer)
			if type not_in SUPPORTED_ELEMENTS {
				log.errorf("Element type %s is not supported.", type)
				return false
			}

			fem_type, order := gmsh_type_to_element_info(type)

			if elements_read == 0 { mesh.order = order }

			if order != mesh.order {
				log.errorf("Mesh must be consistent order, expected %v, got %v", mesh.order, order)
				return false
			}

			el_dim := fe.element_dim(fem_type)

			for _ in 0 ..< num_in_group {
				elements_read += 1
				element: Raw_Element
				element.type = type
				_ = read_number(i32, &p) or_return //id

				if num_tags == 0 {
					if el_dim < mesh.intrinsic_dim { log.warn("Untagged boundary element, defaulting to id 0") }
					element.tags = make([]i32, 1, context.temp_allocator)
					element.tags[0] = 0
				} else {
					element.tags = make([]i32, num_tags, context.temp_allocator)
					for j in 0 ..< len(element.tags) {
						element.tags[j] = read_number(i32, &p) or_return
					}
				}

				node_count := GMSH_ELEMENT_NUM_NODES[type]
				element.node_indices = make([]int, node_count)

				for j in 0 ..< node_count {
					node_idx := read_number(i32, &p) or_return
					node_id := gmsh_id_to_node_id[node_idx]
					element.node_indices[j] = node_id
				}

				#partial switch el_dim {
				case fe.Dimension(int(mesh.intrinsic_dim) - 1): append(&raw_boundary_elements, element)
				case mesh.intrinsic_dim:
					append(&raw_primary_elements, element)
					mesh.encountered_cell_types += {fem_type}
				case:
					log.errorf(
						"Gmsh: Found a mesh element dim %v that was inconsistent with the dimensions of the mesh %v.",
						el_dim,
						mesh.intrinsic_dim,
					)
					return false
				}
			}
		}
	}

	parser_consume(&p) or_return // newline after last binary element data
	expect_ascii_line(&p, "$EndElements") or_return


	raw_periodic_links := make([dynamic]Raw_Periodic_Link, context.temp_allocator)
	if p.cursor < len(p.data) {
		periodic_start := p.cursor
		line, has_line := ascii_line(&p)

		if has_line && line == "$Periodic" {
			num_links := parse_int(ascii_line(&p) or_return) or_return

			for _ in 0 ..< num_links {
				header := strings.split(ascii_line(&p) or_return, " ", context.temp_allocator)
				if len(header) < 3 {
					log.error("Gmsh: malformed $Periodic entity header")
					return false
				}

				link: Raw_Periodic_Link
				link.dim = parse_int(header[0]) or_return
				link.slave_tag = i32(parse_int(header[1]) or_return)
				link.master_tag = i32(parse_int(header[2]) or_return)
				for i in 0 ..< 4 { link.transform.data[i][i] = 1 } 	// identity unless overridden below

				affine_start := p.cursor
				maybe_affine, has_affine_line := ascii_line(&p)
				if has_affine_line && strings.has_prefix(maybe_affine, "Affine") {
					raw_values := strings.split(
						strings.trim_space(maybe_affine[len("Affine"):]),
						" ",
						context.temp_allocator,
					)
					values := make([dynamic]f64, context.temp_allocator)
					for raw_v in raw_values {
						trimmed := strings.trim_space(raw_v)
						if len(trimmed) == 0 { continue }
						v, vok := strconv.parse_f64(trimmed)
						if !vok {
							log.errorf("Gmsh: bad $Periodic affine value %s", trimmed)
							return false
						}
						append(&values, v)
					}
					if len(values) != 12 && len(values) != 16 {
						log.errorf("Gmsh: $Periodic affine transform expected 12 or 16 values, got %d", len(values))
						return false
					}

					rows := 3 if len(values) == 12 else 4
					idx := 0
					for r in 0 ..< rows {
						for c in 0 ..< 4 {
							link.transform.data[c][r] = values[idx]
							idx += 1
						}
					}
					if rows == 3 {
						link.transform.data[0][3] = 0
						link.transform.data[1][3] = 0
						link.transform.data[2][3] = 0
						link.transform.data[3][3] = 1
					}
				} else {
					p.cursor = affine_start
				}

				num_nodes := parse_int(ascii_line(&p) or_return) or_return
				pairs := make([][2]int, num_nodes)
				for i in 0 ..< num_nodes {
					node_line := strings.split(ascii_line(&p) or_return, " ", context.temp_allocator)
					if len(node_line) < 2 {
						log.error("Gmsh: malformed $Periodic node correspondence line")
						return false
					}

					slave_gid := i32(parse_int(node_line[0]) or_return)
					master_gid := i32(parse_int(node_line[1]) or_return)

					slave_idx, slave_found := gmsh_id_to_node_id[slave_gid]
					master_idx, master_found := gmsh_id_to_node_id[master_gid]
					if !slave_found || !master_found {
						log.errorf(
							"Gmsh: $Periodic node reference (%d, %d) not found among mesh nodes",
							slave_gid,
							master_gid,
						)
						return false
					}
					pairs[i] = {slave_idx, master_idx}
				}
				link.pairs = pairs

				append(&raw_periodic_links, link)
			}

			expect_ascii_line(&p, "$EndPeriodic") or_return
		} else {
			p.cursor = periodic_start
		}
	}

	return gmsh_build_topology(mesh, raw_primary_elements[:], raw_boundary_elements[:], raw_periodic_links[:])
}

@(private = "file")
Facet_Key :: [4]int

@(private = "file")
NO_VERT :: max(int)

@(private = "file")
facet_key :: proc(ids: []int) -> Facet_Key {
	key := Facet_Key{NO_VERT, NO_VERT, NO_VERT, NO_VERT}
	copy(key[:], ids)
	slice.sort(key[:len(ids)])
	return key
}

@(private = "file")
node_matches_avg :: proc(node: [3]f64, corners: [][3]f64) -> bool {
	avg: [3]f64
	for c in corners { avg += c }
	avg *= 1.0 / f64(len(corners))
	diff := avg - node
	for k in 0 ..< 3 {
		if math.abs(diff[k]) > ALIGNMENT_TOLERANCE { return false }
	}
	return true
}


@(private = "file")
is_affine :: proc(et: fe.Element_Type, order: fe.Order, node_ids: []int, nodes: [][3]f64) -> bool {
	num_verts := fe.element_num_nodes(et, .O1)
	verts := node_ids[:num_verts]

	#partial switch et {
	case .Point, .Line, .Tri, .Tet:
	case .Quad:
		diff := (nodes[verts[0]] + nodes[verts[2]]) - (nodes[verts[1]] + nodes[verts[3]])
		for k in 0 ..< 3 {
			if math.abs(diff[k]) > ALIGNMENT_TOLERANCE { return false }
		}

	case .Hex:
		v0 := nodes[verts[0]]
		e1 := nodes[verts[1]] - v0
		e3 := nodes[verts[3]] - v0
		e4 := nodes[verts[4]] - v0
		expect := [8][3]f64{v0, v0 + e1, v0 + e1 + e3, v0 + e3, v0 + e4, v0 + e1 + e4, v0 + e1 + e3 + e4, v0 + e3 + e4}
		for i in 0 ..< 8 {
			diff := expect[i] - nodes[verts[i]]
			for k in 0 ..< 3 {
				if math.abs(diff[k]) > ALIGNMENT_TOLERANCE { return false }
			}
		}

	case: return false
	}

	if order == .O1 { return true }

	edges := fe.REFERENCE_ELEMENTS[et].topo.sub_entity_verts[.D1]
	cursor := num_verts
	for edge in edges {
		corners := [][3]f64{nodes[verts[edge[0]]], nodes[verts[edge[1]]]}
		if !node_matches_avg(nodes[node_ids[cursor]], corners) { return false }
		cursor += 1
	}

	#partial switch et {
	case .Hex:
		num_facets := fe.element_num_facets(et)
		for f in 0 ..< num_facets {
			face := fe.element_facet_verts(et, f)
			corners := make([][3]f64, len(face), context.temp_allocator)
			for c, i in face { corners[i] = nodes[verts[c]] }
			if !node_matches_avg(nodes[node_ids[cursor]], corners) { return false }
			cursor += 1
		}
		corners := make([][3]f64, num_verts, context.temp_allocator)
		for c, i in verts { corners[i] = nodes[c] }
		if cursor < len(node_ids) && !node_matches_avg(nodes[node_ids[cursor]], corners) { return false }

	case .Quad:
		corners := make([][3]f64, num_verts, context.temp_allocator)
		for c, i in verts { corners[i] = nodes[c] }
		if cursor < len(node_ids) && !node_matches_avg(nodes[node_ids[cursor]], corners) { return false }
	}

	return true
}

@(private = "file")
Facet_Build :: struct {
	type:            fe.Element_Type,
	canonical_verts: []int, // vertex order as first encountered; orientation reference
	boundary:        fe.Boundary_ID,
	incidences:      [dynamic]fe.Facet_Incidence,
}

@(private = "file")
gmsh_build_topology :: proc(
	mesh: ^fe.Mesh,
	primary: []Raw_Element,
	boundary: []Raw_Element,
	periodic_links: []Raw_Periodic_Link,
) -> bool {
	if mesh.intrinsic_dim < .D1 {
		log.error("Gmsh: mesh must be at least 1D to build topology")
		return false
	}
	facet_dim := fe.Dimension(int(mesh.intrinsic_dim) - 1)

	mesh.cells = make([]fe.Cell, len(primary))
	mesh.cell_conn = make([]fe.Cell_Conn, len(primary))
	mesh.cell_nodes = make([][]fe.Entity_ID, len(primary))

	facet_map := make(map[Facet_Key]int, context.temp_allocator)
	facet_builds := make([dynamic]Facet_Build, context.temp_allocator)

	edge_map := make(map[Facet_Key]int, context.temp_allocator)
	edge_canonical := make([dynamic][2]int, context.temp_allocator)

	for elem, cell_idx in primary {
		et, _ := gmsh_type_to_element_info(elem.type)

		region := fe.Region_ID(0)
		if len(elem.tags) > 0 { region = fe.Region_ID(elem.tags[0]) }

		num_verts := fe.element_num_nodes(et, .O1)

		cell := &mesh.cells[cell_idx]
		cell.id = fe.Entity_ID(cell_idx)
		cell.region = region
		cell.type = et
		cell.affine = is_affine(et, mesh.order, elem.node_indices, mesh.nodes)

		cell_nodes := make([]fe.Entity_ID, len(elem.node_indices))
		for j in 0 ..< len(elem.node_indices) {
			cell_nodes[j] = fe.Entity_ID(elem.node_indices[j])
		}
		mesh.cell_nodes[cell_idx] = cell_nodes

		vertices := make([]fe.Entity_ID, num_verts)
		for j in 0 ..< num_verts { vertices[j] = fe.Entity_ID(elem.node_indices[j]) }

		if et == .Point {
			mesh.cell_conn[cell_idx] = {
				vertices = vertices,
			}
			continue
		}

		num_facets := fe.element_num_facets(et)
		cell_facets := make([]fe.Entity_ID, num_facets)

		for f in 0 ..< num_facets {
			local_verts := fe.element_facet_verts(et, f)
			global_verts := make([]int, len(local_verts), context.temp_allocator)
			for k, kk in local_verts { global_verts[kk] = elem.node_indices[k] }

			facet_type := fe.element_facet_type(et, f)
			key := facet_key(global_verts)

			idx, existing := facet_map[key]
			if !existing {
				idx = len(facet_builds)
				facet_map[key] = idx
				append(
					&facet_builds,
					Facet_Build {
						type = facet_type,
						canonical_verts = slice.clone(global_verts, context.temp_allocator),
						boundary = fe.Boundary_ID(fe.NOT_A_BOUNDARY),
					},
				)
			}

			fb := &facet_builds[idx]
			append(&fb.incidences, fe.Facet_Incidence{local_facet = i8(f), cell = cell.id})
			cell_facets[f] = fe.Entity_ID(idx)

			if len(global_verts) > 1 {
				orient := fe.element_orientation(facet_type, global_verts, fb.canonical_verts)
				#partial switch facet_dim {
				case .D1: cell.edge_orientation[f] = orient
				case .D2: cell.face_orientation[f] = orient
				}
			}
		}
		cell.facets = cell_facets

		#partial switch facet_dim {
		case .D1: // mesh facets ARE this cell's edges
				mesh.cell_conn[cell_idx] = {
					vertices = vertices,
					edges    = cell_facets,
				}

		case .D2:
			// mesh facets are faces; edges here are internal (edge-dof) connectivity
			edge_pairs := fe.REFERENCE_ELEMENTS[et].topo.sub_entity_verts[.D1]
			cell_edges := make([]fe.Entity_ID, len(edge_pairs))

			for pair, ei in edge_pairs {
				a := elem.node_indices[pair[0]]
				b := elem.node_indices[pair[1]]
				gp := []int{a, b}
				key := facet_key(gp)

				idx, existing := edge_map[key]
				if !existing {
					idx = len(edge_canonical)
					edge_map[key] = idx
					append(&edge_canonical, [2]int{a, b})
				}
				cell_edges[ei] = fe.Entity_ID(idx)

				canon := edge_canonical[idx]
				cell.edge_orientation[ei] = fe.element_orientation(.Line, gp, canon[:])
			}
			mesh.cell_conn[cell_idx] = {
				vertices = vertices,
				edges    = cell_edges,
				faces    = cell_facets,
			}

		case: mesh.cell_conn[cell_idx] = {
					vertices = vertices,
				}
		}
	}

	// Tag facets from boundary elements.
	for elem in boundary {
		bt, _ := gmsh_type_to_element_info(elem.type)
		num_verts := fe.element_num_nodes(bt, .O1)
		verts := elem.node_indices[:num_verts]
		key := facet_key(verts)

		idx, found := facet_map[key]
		if !found {
			log.errorf("Gmsh: boundary element (nodes=%v) does not match any facet", elem.node_indices)
			return false
		}
		boundary_id := fe.Boundary_ID(fe.NOT_A_BOUNDARY)
		if len(elem.tags) > 0 { boundary_id = fe.Boundary_ID(elem.tags[0]) }
		facet_builds[idx].boundary = boundary_id
	}


	total_incidences := 0
	for fb in facet_builds { total_incidences += len(fb.incidences) }

	mesh.facets = make([]fe.Facet, len(facet_builds))
	mesh.incidences = make([]fe.Facet_Incidence, total_incidences)

	cursor := 0
	for fb, i in facet_builds {
		facet_affine := true
		for inc in fb.incidences {
			if !mesh.cells[inc.cell].affine { facet_affine = false; break }
		}

		mesh.facets[i] = fe.Facet {
			id = fe.Entity_ID(i),
			incidence_count = i8(len(fb.incidences)),
			incidence_start = cursor,
			info = fe.Facet_Info{affine = facet_affine, boundary = fb.boundary, type = fb.type},
		}
		for inc in fb.incidences {
			mesh.incidences[cursor] = inc
			cursor += 1
		}
	}

	if len(periodic_links) > 0 {
		periodics, pok := gmsh_build_periodicity(boundary, facet_map, facet_builds[:], periodic_links)
		if !pok { return false }
		mesh.periodics = periodics
	}

	return true
}


@(private = "file")
gmsh_build_periodicity :: proc(
	boundary: []Raw_Element,
	facet_map: map[Facet_Key]int,
	facet_builds: []Facet_Build,
	links: []Raw_Periodic_Link,
) -> (
	periodics: []fe.Periodicity,
	ok: bool,
) {
	entity_to_boundary := make(map[i32]fe.Boundary_ID, context.temp_allocator)
	for elem in boundary {
		if len(elem.tags) < 2 { continue }
		entity_to_boundary[elem.tags[1]] = fe.Boundary_ID(elem.tags[0])
	}

	result := make([]fe.Periodicity, len(links))

	for link, li in links {
		node_to_master := make(map[int]int, context.temp_allocator)
		for pair in link.pairs { node_to_master[pair[0]] = pair[1] }

		slave_boundary := fe.Boundary_ID(fe.NOT_A_BOUNDARY)
		if b, found := entity_to_boundary[link.slave_tag]; found { slave_boundary = b }
		master_boundary := fe.Boundary_ID(fe.NOT_A_BOUNDARY)
		if b, found := entity_to_boundary[link.master_tag]; found { master_boundary = b }

		pairs := make([dynamic]fe.Periodic_Pair)

		for elem in boundary {
			if len(elem.tags) < 2 || elem.tags[1] != link.slave_tag { continue }

			et, _ := gmsh_type_to_element_info(elem.type)
			num_verts := fe.element_num_nodes(et, .O1)
			slave_verts := elem.node_indices[:num_verts]

			slave_idx, found_slave := facet_map[facet_key(slave_verts)]
			if !found_slave {
				log.errorf("Gmsh: $Periodic slave element (nodes=%v) does not match any facet", elem.node_indices)
				return nil, false
			}

			scanon := facet_builds[slave_idx].canonical_verts

			master_verts := make([]int, len(scanon), context.temp_allocator)
			for v, i in scanon {
				m, found_map := node_to_master[v]
				if !found_map {
					log.errorf("Gmsh: $Periodic vertex %d has no master correspondence", v)
					return nil, false
				}
				master_verts[i] = m
			}

			master_idx, found_master := facet_map[facet_key(master_verts)]
			if !found_master {
				log.errorf("Gmsh: $Periodic master facet (nodes=%v) does not match any facet", master_verts)
				return nil, false
			}

			mcanon := facet_builds[master_idx].canonical_verts
			assert(facet_builds[master_idx].type == facet_builds[slave_idx].type, "periodic pair facet types differ")


			translated_scanon := make([]int, len(scanon), context.temp_allocator)
			for v, i in scanon { translated_scanon[i] = node_to_master[v] }

			vertex_map := make([]int, len(mcanon))
			for v, i in mcanon {
				j, found := slice.linear_search(translated_scanon, v)
				if !found {
					log.errorf("Gmsh: $Periodic could not resolve vertex correspondence for facet %d", master_idx)
					return nil, false
				}
				vertex_map[i] = j
			}
			orientation := fe.element_orientation(et, mcanon, translated_scanon)

			edge_pairs := fe.REFERENCE_ELEMENTS[et].topo.sub_entity_verts[.D1]
			edge_map: []int
			edge_orientation: []u8

			if len(edge_pairs) > 0 {
				edge_map = make([]int, len(edge_pairs))
				edge_orientation = make([]u8, len(edge_pairs))

				for me, ei in edge_pairs {
					a, b := mcanon[me[0]], mcanon[me[1]]

					se := -1
					for cand, sei in edge_pairs {
						ta, tb := translated_scanon[cand[0]], translated_scanon[cand[1]]
						if (ta == a && tb == b) || (ta == b && tb == a) {
							se = sei
							break
						}
					}
					if se == -1 {
						log.errorf("Gmsh: $Periodic could not match edge %d of facet %d to its slave", ei, master_idx)
						return nil, false
					}

					edge_map[ei] = se
					sv := edge_pairs[se]
					s_global := [2]int{translated_scanon[sv[0]], translated_scanon[sv[1]]}
					edge_orientation[ei] = fe.element_orientation(.Line, []int{a, b}, s_global[:])
				}
			}

			append(
				&pairs,
				fe.Periodic_Pair {
					slave            = fe.Entity_ID(slave_idx),
					master           = fe.Entity_ID(master_idx),
					vertex_map       = vertex_map,
					orientation      = orientation,
					edge_map         = edge_map,
					edge_orientation = edge_orientation,
				},
			)
		}

		result[li] = fe.Periodicity {
			slave     = slave_boundary,
			master    = master_boundary,
			transform = link.transform,
			pairs     = pairs[:],
		}
	}

	return result, true
}
