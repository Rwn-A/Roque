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

Entity_Orientation :: [Dimension][]u8

Mesh :: struct {
	cells:                  []Cell,
	cell_conn:              []Connectivity, // parallel array to cells
	facets:                 []Facet,
	incidences:             []Facet_Incidence,
	periodics:              []Periodicity,
	n_entities:             [Dimension]int, // global entity counts per dimension, sizes space numbering
	cell_nodes:             [][]Entity_ID, // [cell][local node] -> index into nodes, reference node order (l2g)
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
	facets:           []Entity_ID, // aliases cell_conn[id][facet dim]
	edge_orientation: [MAX_EDGES]u8, // read through cell_orientation
	face_orientation: [MAX_FACES]u8, // read through cell_orientation
	using info:       Cell_Info,
}

// Global ids of an element's sub-entities, in reference-element order. [own dim] = {own id}.
Connectivity :: [Dimension][]Entity_ID

Cell_Info :: struct {
	affine: bool,
	region: Region_ID,
	type:   Element_Type,
}

// A facet's local vertex order (as an element, for trace spaces) is the order its canonical cell sees it in,
// which must also be its canonical order, so only its edges can be oriented.
Facet :: struct {
	id:               Entity_ID,
	incidence_count:  i8,
	incidence_start:  int,
	edge_orientation: [MAX_EDGES]u8,
	info:             Facet_Info,
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

// Matches a master facet to its slave facet.
Periodic_Pair :: struct {
	master, slave: Entity_ID, // facet ids
	maps:          [Dimension][]int, // maps[d][master local] = slave local
	orientations:  [Dimension][]u8, // orientations[d][master local]
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

// Orientation keys of a cell's sub-entities, [dim][local entity], for basis_orient. Views into the cell.
// Entries that never orient (vertices, the cell itself) are left 0 by the loader.
cell_entity_keys :: proc(c: ^Cell) -> (k: Entity_Orientation) {
	k[.D1] = c.edge_orientation[:]
	k[.D2] = c.face_orientation[:]
	return
}

// Same for a facet used as an element (trace spaces): only a 3D facet's edges orient.
facet_entity_keys :: proc(f: ^Facet) -> (k: Entity_Orientation) {
	k[.D1] = f.edge_orientation[:]
	return
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

// Connectivity of a facet as an element, in its own local order.
facet_connectivity :: proc(mesh: Mesh, facet: Entity_ID, alloc := context.allocator) -> (conn: Connectivity) {
	cell, local_facet := facet_canonical_cell(mesh, mesh.facets[facet])
	f := element_facet(cell.type, local_facet)
	for d in Dimension {
		conn[d] = make([]Entity_ID, len(f.closure[d]), alloc)
		for cell_local, k in f.closure[d] { conn[d][k] = mesh.cell_conn[cell.id][d][cell_local] }
	}
	return
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
