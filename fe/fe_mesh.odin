package fe

/*
 Conforming mesh for 1D-3D.

 Cells and facets are both entities with their own connectivity, facets know the cells they bound. A facet may
 have > 2 incident cells, this is in the interest of 1D truss structures. Lower dimensions are counted, not stored.
*/

import "core:mem"
import "core:mem/virtual"

Entity_ID :: i32
Tag_ID    :: u8
Tag_Set   :: bit_set[0 ..< MAX_TAGS]

MAX_TAGS :: 128

ALL_TAGS: Tag_Set : ~{}

// Orientation key of every sub-entity, [dim][local entity]. Vertices and the entity itself are 0.
Entity_Keys :: [Dimension][MAX_EDGES]u8

Mesh :: struct {
	entities:               [Dimension][]Entity, // cells and facets, nil for lower dimensions
	cells:                  []Entity, // entities[intrinsic_dim], aliases entities
	facets:                 []Entity, // entities[intrinsic_dim - 1], aliases entities
	n_entities:             [Dimension]int, // global entity counts per dimension, sizes space numbering
	tag_names:              [Dimension]map[string]Tag_ID,
	periodics:              []Periodicity,
	nodes:                  [][3]f64,
	cell_nodes:             [][]Entity_ID, // [cell][local node] -> index into nodes, reference node order
	order:                  Order,
	intrinsic_dim:          Dimension,
	encountered_cell_types: bit_set[Element_Type],
	arena:                  virtual.Arena, // all mesh data in here
}

// A single mesh entity
Entity :: struct {
	id:      Entity_ID,
	type:    Element_Type,
	tags:    Tag_Set,
	keys:    Entity_Keys,
	conn:    [Dimension][]Entity_ID, // sub-entity ids in reference order, [own dim] = {id}
	cofaces: []Coface, // entities one dimension up that contain this one, set on facets
}

Coface :: struct {
	entity: Entity_ID,
	local:  int, // this entity's index among the coface's sub-entities
}

// A site is some sub entity of the meshes main cell type.
// This is one area where facets are not first class, but simplifies all sub entity handling.
Site :: struct {
	cell:  Entity_ID,
	dim:   Dimension,
	index: int,
}

// Information from periodic boundaries defined at the mesh level.
Periodicity :: struct {
	master, slave: Tag_ID,
	transform:     Small_Mat(4, 4, f64), // kept for user info, the links hold everything the mesh needs
	links:         [Dimension][]Entity_Link, // every entity on the master boundary once, dims 0 ..= facet dim
}

// A master entity and the slave entity it maps onto. key orients the master's canonical order onto the slave's.
Entity_Link :: struct {
	master, slave: Entity_ID,
	key:           u8,
}

MESH_FRAME :: Small_Mat(3, 3, f64) {
	data = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
}

X_SEGMENT_FRAME :: Small_Mat(3, 1, f64) {
	data = {{1, 0, 0}},
}

XY_PLANE_FRAME :: Small_Mat(3, 2, f64) {
	data = {{1, 0, 0}, {0, 1, 0}},
}

//== Mesh

mesh_destroy :: proc(meshes: ..^Mesh) {
	for mesh in meshes {
		virtual.arena_destroy(&mesh.arena)
		mesh^ = {}
	}
}

// Dimension of facet entities.
mesh_facet_dim :: proc(mesh: Mesh) -> Dimension {
	return mesh.intrinsic_dim - Dimension(1)
}

// Tags of entities of dimension `dim` by name.
mesh_tags_from_names :: proc(mesh: Mesh, dim: Dimension, names: ..string) -> (tags: Tag_Set, ok: bool) {
	for name in names {
		tags += {int(mesh.tag_names[dim][name] or_return)}
	}
	return tags, true
}

mesh_periodicity :: proc(mesh: Mesh, master, slave: Tag_ID) -> (Periodicity, bool) {
	for per in mesh.periodics {
		if per.master == master && per.slave == slave { return per, true }
	}
	return {}, false
}

mesh_periodicity_from_names :: proc(mesh: Mesh, master_name, slave_name: string) -> (p: Periodicity, ok: bool) {
	fd := mesh_facet_dim(mesh)
	master := mesh.tag_names[fd][master_name] or_return
	slave := mesh.tag_names[fd][slave_name] or_return
	return mesh_periodicity(mesh, master, slave)
}

// Every entity on the closure of facets tagged `tags` linked to itself. For conforming interfaces conditions.
mesh_interface_links :: proc(mesh: Mesh, tags: Tag_Set, alloc := context.allocator) -> (links: [Dimension][]Entity_Link) {
	scratch_guard()
	fd := mesh_facet_dim(mesh)

	seen: [Dimension][]bool
	out: [Dimension][dynamic]Entity_Link
	for d in Dimension {
		if d > fd { break } // dont link to cells
		seen[d] = make([]bool, mesh.n_entities[d], scratch())
		out[d] = make([dynamic]Entity_Link, alloc)
	}

	for facet in mesh.facets {
		if facet.tags & tags == {} { continue }
		for d in Dimension {
			if d > fd { break }
			for gid in facet.conn[d] {
				if seen[d][gid] { continue }
				seen[d][gid] = true
				append(&out[d], Entity_Link{master = gid, slave = gid})
			}
		}
	}

	for d in Dimension { links[d] = out[d][:] }

	return
}

// Type of any entity, including those without records (edges in 3D).
mesh_entity_type :: proc(mesh: Mesh, dim: Dimension, id: Entity_ID) -> Element_Type {
	if dim == .D0 { return .Point }
	if mesh.entities[dim] != nil { return mesh.entities[dim][id].type }
	return .Line
}

//== Cells and facets

// Local facets of the cell on the mesh boundary.
cell_boundary_facets :: proc(mesh: Mesh, cell: Entity) -> (r: bit_set[0 ..< MAX_FACETS]) {
	for f, i in cell.conn[mesh_facet_dim(mesh)] {
		if facet_is_boundary(mesh.facets[f]) { r += {i} }
	}
	return r
}

// Local facets of the cell with a tag in `tags`.
cell_tagged_facets :: proc(mesh: Mesh, cell: Entity, tags: Tag_Set) -> (r: bit_set[0 ..< MAX_FACETS]) {
	for f, i in cell.conn[mesh_facet_dim(mesh)] {
		if mesh.facets[f].tags & tags != {} { r += {i} }
	}
	return r
}

facet_is_boundary :: proc(facet: Entity) -> bool {
	return len(facet.cofaces) == 1
}

// Tags of the facet's incident cells.
facet_cell_tags :: proc(mesh: Mesh, facet: Entity) -> (r: Tag_Set) {
	for c in facet.cofaces { r += mesh.cells[c.entity].tags }
	return r
}

//== Sites

// The cell as its own sub-entity.
cell_site :: proc(cell: Entity) -> Site {
	return {cell.id, element_dim(cell.type), 0}
}

cell_facet_site :: proc(cell: Entity, local_facet: int) -> Site {
	return {cell.id, element_facet_dim(cell.type), local_facet}
}

// The entity seen from one of its cofaces.
coface_site :: proc(entity: Entity, c: Coface) -> Site {
	return {c.entity, element_dim(entity.type), c.local}
}

// The facet seen from its first incident cell.
facet_site :: proc(facet: Entity) -> Site {
	return coface_site(facet, facet.cofaces[0])
}

//== Parallel iteration

// Entities split into groups with no write conflicts inside a group. Colour c is items[offsets[c]:offsets[c + 1]].
Colouring :: struct {
	offsets: []int,
	items:   []Entity_ID, // sorted by element type within each colour
}

// Parallel iterator for coloured mesh entities.
Colour_Iterator :: struct {
	colouring: Colouring,
	colour:    int, // current colour, -1 before the first
	items:     []Entity_ID, // this rank's share of the current colour
	next:      int,
}

// Visits every entity in the colouring, colour by colour, split across ranks with a barrier between colours.
// Collective, every rank must run it to the end: no break or early return out of the loop.
colour_iterator :: proc(col: Colouring) -> Colour_Iterator {
	return {colouring = col, colour = -1}
}

// for it := colour_iterator(col); entity in colour_iterator_next(it) {
// 	cell := mesh.cells[entity]
// }
colour_iterator_next :: proc(it: ^Colour_Iterator) -> (id: Entity_ID, ok: bool) {
	for it.next >= len(it.items) {
		if it.colour >= 0 { rank_sync() } 	// every rank finishes a colour before the next starts
		if it.colour += 1; it.colour >= len(it.colouring.offsets) - 1 { return }

		col := it.colouring
		it.items = rank_slice(col.items[col.offsets[it.colour]:col.offsets[it.colour + 1]])
		it.next = 0
	}
	id = it.items[it.next]
	it.next += 1
	return id, true
}

//== Mesh based colouring helpers

MAX_MESH_COLOURS :: 128


// Cells sharing a vertex, or periodic partners of one, get different colours. Useful base colouring
// that doesnt rely on sparsity.
mesh_colour_cells :: proc(mesh: Mesh, alloc := context.allocator) -> Colouring {
	scratch_guard()

	// periodic partners collapse onto one representative vertex
	rep := make([]Entity_ID, mesh.n_entities[.D0], scratch())
	for &r, v in rep { r = Entity_ID(v) }
	for per in mesh.periodics {
		for link in per.links[.D0] {
			a, b := find(rep, link.master), find(rep, link.slave)
			rep[max(a, b)] = min(a, b)
		}
	}

	offsets := make([]int, len(mesh.cells) + 1, scratch())
	for cell, i in mesh.cells { offsets[i + 1] = offsets[i] + len(cell.conn[.D0]) }

	keys := make([]Entity_ID, offsets[len(mesh.cells)], scratch())
	for cell, i in mesh.cells {
		for v, k in cell.conn[.D0] { keys[offsets[i] + k] = find(rep, v) }
	}

	return greedy_colouring(mesh.cells, offsets, keys, mesh.n_entities[.D0], alloc)

	find :: proc(rep: []Entity_ID, v: Entity_ID) -> Entity_ID {
		r := v
		for rep[r] != r { r = rep[r] }
		return r
	}
}

// Facets sharing a cell get different colours. Useful default for facet-major loops.
mesh_colour_facets :: proc(mesh: Mesh, alloc := context.allocator) -> Colouring {
	scratch_guard()

	offsets := make([]int, len(mesh.facets) + 1, scratch())
	for facet, i in mesh.facets { offsets[i + 1] = offsets[i] + len(facet.cofaces) }

	keys := make([]Entity_ID, offsets[len(mesh.facets)], scratch())
	for facet, i in mesh.facets {
		for c, k in facet.cofaces { keys[offsets[i] + k] = c.entity }
	}

	return greedy_colouring(mesh.facets, offsets, keys, len(mesh.cells), alloc)
}


// Entity i may share keys[offsets[i]:offsets[i + 1]]. Each takes the lowest colour no earlier entity sharing a key
// has, then entities are grouped by colour and sorted by type within a colour.
@(private = "file")
greedy_colouring :: proc(
	entities: []Entity,
	offsets: []int,
	keys: []Entity_ID,
	n_keys: int,
	alloc: mem.Allocator,
) -> (
	col: Colouring,
) {
	used := make([]bit_set[0 ..< MAX_MESH_COLOURS], n_keys, scratch())
	colour := make([]int, len(entities), scratch())
	n_colours := 0

	for i in 0 ..< len(entities) {
		taken: bit_set[0 ..< MAX_MESH_COLOURS]
		for k in keys[offsets[i]:offsets[i + 1]] { taken += used[k] }

		c := 0
		for c < MAX_MESH_COLOURS && (c in taken) { c += 1 }
		assert(c < MAX_MESH_COLOURS, "more than MAX_MESH_COLOURS colours needed")

		for k in keys[offsets[i]:offsets[i + 1]] { used[k] += {c} }
		colour[i] = c
		n_colours = max(n_colours, c + 1)
	}

	// counting sort on (colour, type)
	n_types := len(Element_Type)
	bucket := make([]int, n_colours * n_types + 1, scratch())
	for c, i in colour { bucket[c * n_types + int(entities[i].type) + 1] += 1 }
	for b in 1 ..< len(bucket) { bucket[b] += bucket[b - 1] }

	col.offsets = make([]int, n_colours + 1, alloc)
	for c in 0 ..= n_colours { col.offsets[c] = bucket[c * n_types] }

	col.items = make([]Entity_ID, len(entities), alloc)
	for c, i in colour {
		b := c * n_types + int(entities[i].type)
		col.items[bucket[b]] = Entity_ID(i)
		bucket[b] += 1
	}

	return col
}
