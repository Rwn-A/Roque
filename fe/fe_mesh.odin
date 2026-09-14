package fe

/*
 Basic conforming mesh for 1D-3D.

 Facets are nearly first-class with cells, this is in the interest of HDG, DG methods.

 A facet may have > 2 incident cells, this is in the interest of 1D truss structures.
*/

import "core:mem/virtual"
import "core:slice"

Entity_ID   :: i32
Boundary_ID :: i8
Region_ID   :: i8

Region_Set   :: bit_set[0 ..< max(Region_ID)]
Boundary_Set :: bit_set[0 ..< max(Boundary_ID)]

NOT_A_BOUNDARY :: -1
ALL_REGIONS: Region_Set : ~{}

Mesh :: struct {
	cells:                  []Cell,
	cell_conn:              []Cell_Conn, // parallel array to cells
	facets:                 []Facet,
	incidences:             []Facet_Incidence,
	cell_nodes:             [][]Entity_ID, // NOTE: must be same structure as an l2g in DOF numbering
	periodics:              []Periodicity,
	nodes:                  [][3]f64,
	order:                  Order,
	intrinsic_dim:          Dimension,
	encountered_cell_types: bit_set[Element_Type],
	boundary_names:         map[string]Boundary_ID,
	region_names:           map[string]Region_ID,
	arena:                  virtual.Arena, // all mesh data in here.
}

Cell :: struct {
	id:               Entity_ID,
	facets:           []Entity_ID, // directly references of the arrays from connectivity depending on dim.
	edge_orientation: [MAX_EDGES]u8,
	face_orientation: [MAX_FACETS]u8,
	using info:       Cell_Info,
}

Cell_Conn :: struct {
	faces, edges, vertices: []Entity_ID,
}

Cell_Info :: struct {
	affine: bool,
	region: Region_ID,
	type:   Element_Type,
}

Facet :: struct {
	id:              Entity_ID,
	incidence_count: i8,
	incidence_start: int,
	info:            Facet_Info,
}

Facet_Info :: struct {
	affine:   bool,
	boundary: Boundary_ID,
	type:     Element_Type,
}

Facet_Incidence :: struct {
	local_facet: i8,
	cell:        Entity_ID,
}

Periodicity :: struct {
	master, slave: Boundary_ID,
	transform:     Small_Mat(4, 4, f64), // kept around for user info, mesh already has what it needs in each pair.
	pairs:         []Periodic_Pair,
}

Periodic_Pair :: struct {
	master, slave:    Entity_ID, // facet ids
	vertex_map:       []int, // et-local vertex correspondence: vertex_map[master_local_v] = slave_local_v.
	orientation:      u8, // Orientation relating master cannonical rotation to slaves

	// for 3d only, edge orientation mappings.
	edge_map:         []int,
	edge_orientation: []u8,
}

//== Accessors & general mesh queries

MESH_FRAME :: Small_Mat(3, 3, f64) {
	data = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
}

X_SEGMENT_FRAME :: Small_Mat(3, 1, f64) {
	data = {{1, 0, 0}},
}

XY_PLANE_FRAME :: Small_Mat(3, 2, f64) {
	data = {{1, 0, 0}, {0, 1, 0}},
}

mesh_destroy :: proc(mesh: ^Mesh) {
	virtual.arena_destroy(&mesh.arena)
}

mesh_boundary_set_from_names :: proc(mesh: Mesh, names: ..string) -> (bs: Boundary_Set, ok: bool) {
	for name in names {
		bs += {mesh.boundary_names[name] or_return}
	}
	return bs, true
}

mesh_region_set_from_names :: proc(mesh: Mesh, names: ..string) -> (rs: Region_Set, ok: bool) {
	for name in names {
		rs += {mesh.region_names[name] or_return}
	}
	return rs, true
}

mesh_periodicity :: proc(mesh: Mesh, master, slave: Boundary_ID) -> (Periodicity, bool) {
	for per in mesh.periodics {
		if per.master == master && per.slave == slave { return per, true }
	}
	return {}, false
}

mesh_periodicity_from_names :: proc(mesh: Mesh, master_name, slave_name: string) -> (p: Periodicity, ok: bool) {
	master := mesh.boundary_names[master_name] or_return
	slave := mesh.boundary_names[slave_name] or_return
	for per in mesh.periodics {
		if per.master == master && per.slave == slave { return per, true }
	}
	return {}, false
}

// Mesh coordinate coefficients. This the "state" vector for the FE geometry space. For convienence,
// a simple frame can be applied, this allows one to reduce the ambient space from 3D, or any other transform.
// NOTE: badly behaved transforms are not detected.
mesh_coord_coeffs :: proc(mesh: Mesh, frame: Small_Mat(3, $C, f64), alloc := context.allocator) -> []f64 {
	out := make([]f64, len(mesh.nodes) * C, alloc)

	when C == 3 {
		if frame == MESH_FRAME {
			copy(out, slice.reinterpret([]f64, mesh.nodes)); return out
		}
	}

	inv_f := small_mat_inv(frame)
	for point, i in mesh.nodes {
		sv := Small_Vec(3, f64){point}
		new_point := small_mat_vec_mul(inv_f, sv)
		copy(out[i * C:], new_point.data[:])
	}

	return out
}

// All local facet indices of the cells boundary facets
cell_boundary_facet_set :: proc(mesh: Mesh, c: Cell) -> (r: bit_set[0 ..< MAX_FACETS]) {
	for local_facet, i in c.facets {
		facet := mesh.facets[local_facet]
		if facet_is_boundary(facet) { r += {int(i)} }
	}
	return r
}

// Local facet indexes of the cell that have a boundary in the set.
cell_boundary_facet_set_of :: proc(mesh: Mesh, c: Cell, bs: Boundary_Set) -> (r: bit_set[0 ..< MAX_FACETS]) {
	for local_facet, i in c.facets {
		facet := mesh.facets[local_facet]
		if facet.info.boundary in bs { r += {int(i)} }
	}
	return r
}

facet_incidences :: proc(mesh: Mesh, facet: Facet) -> []Facet_Incidence {
	return mesh.incidences[facet.incidence_start:][:facet.incidence_count]
}

// Returns the first stored incident cell.
facet_canonical_cell :: proc(mesh: Mesh, facet: Facet) -> (Cell, int) {
	inc := facet_incidences(mesh, facet)[0]
	return mesh.cells[inc.cell], int(inc.local_facet)
}

facet_is_boundary :: proc(facet: Facet) -> bool {
	return facet.info.boundary != NOT_A_BOUNDARY
}

// All regions of incident cells.
facet_regions :: proc(mesh: Mesh, facet: Facet) -> (r: Region_Set) {
	for incidence in facet_incidences(mesh, facet) {
		r += {mesh.cells[incidence.cell].info.region}
	}
	return r
}

//== Parallel & Batching

//TODO:

//== Builtin basic meshes

// 1D mesh along X. boundaries: "left", "right" region: "domain".
segment_mesh :: proc(n_cells: int, start, end: f64) -> Mesh {
	assert(n_cells > 0)

	n_points := n_cells + 1

	mesh: Mesh
	assert(virtual.arena_init_growing(&mesh.arena) == nil)
	context.allocator = virtual.arena_allocator(&mesh.arena)

	mesh.order = .O1
	mesh.intrinsic_dim = .D1
	mesh.encountered_cell_types = {.Line}

	mesh.boundary_names = make(map[string]Boundary_ID)
	mesh.boundary_names["left"] = 0
	mesh.boundary_names["right"] = 1

	mesh.region_names = make(map[string]Region_ID)
	mesh.region_names["domain"] = 0
	domain_id := mesh.region_names["domain"]

	mesh.nodes = make([][3]f64, n_points)
	dx := (end - start) / f64(n_cells)
	for i in 0 ..< n_points { mesh.nodes[i] = {start + f64(i) * dx, 0, 0} }

	mesh.cells = make([]Cell, n_cells)
	mesh.cell_conn = make([]Cell_Conn, n_cells)

	for c in 0 ..< n_cells {
		verts := make([]Entity_ID, 2)
		verts[0], verts[1] = Entity_ID(c), Entity_ID(c + 1)
		mesh.cell_conn[c].vertices = verts

		mesh.cells[c] = Cell {
			id = Entity_ID(c),
			facets = verts,
			info = Cell_Info{affine = true, region = domain_id, type = .Line},
		}
	}

	mesh.facets = make([]Facet, n_points)
	mesh.incidences = make([]Facet_Incidence, 2 * n_cells)

	left_id := mesh.boundary_names["left"]
	right_id := mesh.boundary_names["right"]

	mesh.incidences[0] = Facet_Incidence {
		local_facet = 0,
		cell        = 0,
	}
	mesh.facets[0] = Facet {
		id = 0,
		incidence_count = 1,
		incidence_start = 0,
		info = Facet_Info{affine = true, boundary = left_id, type = .Point},
	}

	for p in 1 ..< n_points - 1 {
		off := 2 * p - 1
		mesh.incidences[off] = Facet_Incidence {
			local_facet = 1,
			cell        = Entity_ID(p - 1),
		}
		mesh.incidences[off + 1] = Facet_Incidence {
			local_facet = 0,
			cell        = Entity_ID(p),
		}
		mesh.facets[p] = Facet {
			id = Entity_ID(p),
			incidence_count = 2,
			incidence_start = off,
			info = Facet_Info{affine = true, boundary = Boundary_ID(NOT_A_BOUNDARY), type = .Point},
		}
	}

	last := n_points - 1
	off := 2 * n_cells - 1
	mesh.incidences[off] = Facet_Incidence {
		local_facet = 1,
		cell        = Entity_ID(n_cells - 1),
	}
	mesh.facets[last] = Facet {
		id = Entity_ID(last),
		incidence_count = 1,
		incidence_start = off,
		info = Facet_Info{affine = true, boundary = right_id, type = .Point},
	}

	l2g := make([][]i32, n_cells)
	for c in 0 ..< n_cells {
		l2g[c] = make([]i32, 2)
		l2g[c][0], l2g[c][1] = i32(c), i32(c + 1)
	}

	mesh.cell_nodes = l2g

	return mesh
}
