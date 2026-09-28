package fe

/*
 Local basis quantities.

 Reference basis are cached in a thread local lazy cache, this is simple & safe.

 When querying a basis it can be oriented and/or pushed into physical space. Orientation is required
 for continuous spaces as the FE space assumes it.
*/

import "base:runtime"
import "core:mem/virtual"

// One element's basis at a site: which basis, how it is oriented, and which sub-entity (dim, index) the site is.
Local_Basis :: struct {
	bd:       Basis_Desc,
	keys:     Entity_Keys, // the element's orientation keys for sub entities
	oriented: bool, // do we actually need to orient this basis
	dim:      Dimension,
	index:    int,
}

Basis_Entry :: [Basis_Quantity]Bvec(f64)

ALL_QUANTITIES :: ~Quantity_Set{}

basis_tab :: proc {
	basis_tab_ref,
	basis_tab_phys,
}

// Reference frame, oriented if the local basis is, not pushed to physical space.
basis_tab_ref :: proc(lb: Local_Basis, rule: Rule, alloc := context.allocator) -> (out: Basis_Entry) {
	ref := ref_tables(lb, rule)
	ob := orient_blocks(lb)
	for q in BASIS_QUANTITIES[lb.bd.family] { out[q] = orient_table(&ob, ref[q], f64, alloc) }
	return
}

// Physical frame: oriented if the local basis is, each quantity pushed by its map from `m`.
// only tabulates over given quantities.
basis_tab_phys :: proc(
	lb: Local_Basis,
	rule: Rule,
	m: Element_Map($A, $I, $T),
	qs := ALL_QUANTITIES,
	alloc := context.allocator,
) -> (
	out: [Basis_Quantity]Bvec(T),
) {
	assert(I == int(element_dim(lb.bd.element)), "element map must have intrinsic dimension of the elements dim.")

	ref := ref_tables(lb, rule)
	ob := orient_blocks(lb)
	for q in qs & BASIS_QUANTITIES[lb.bd.family] {
		kind := basis_quantity_map(lb.bd, q)
		if kind == .Unity {
			out[q] = orient_table(&ob, ref[q], T, alloc)
		} else {
			out[q] = orient_push(&ob, ref[q], m, kind, alloc)
		}
	}
	return
}

// Reference dof values (e.g. from interpolation) -> dof values of the local basis.
basis_orient_dofs :: proc(lb: Local_Basis, vals: Cvec($T)) {
	ob := orient_blocks(lb)
	f := vals.fields

	scratch_guard()
	for &b in ob.blocks[:ob.n] {
		dst := vals.data[b.start * f:][:b.n * f]
		src := make([]T, len(dst), scratch())
		copy(src, dst)
		orientation_apply(b.t, .M_Inv_T, b.n, f, src, dst)
	}
}

//== Reference tables

@(private = "file")
Tbl_Key :: struct {
	bd:        Basis_Desc,
	dim:       Dimension, // dimension of the entity the rule lives on (element dim = interior)
	entity:    int,
	point_key: u8, // orientation applied to the rule points before lifting
	points:    rawptr,
}

@(private = "file")
Basis_Store :: struct {
	tables: map[Tbl_Key]Basis_Entry,
	arena:  virtual.Arena,
}

@(thread_local, private = "file")
_bstore: Basis_Store

// Every quantity of the basis at the rule points, cached by the address of the points, so rules must be static.
// The site's key orients the rule points onto the sub-entity before they are lifted into the element.
@(private = "file")
ref_tables :: proc(lb: Local_Basis, rule: Rule) -> Basis_Entry {
	assert(rule.element == element_sub_entity(lb.bd.element, lb.dim, lb.index).type)
	point_key := lb.keys[lb.dim][lb.index]
	key := Tbl_Key{lb.bd, lb.dim, lb.index, point_key, raw_data(rule.points)}

	if _bstore.tables == nil {
		virtual.arena_init_growing(&_bstore.arena) or_else panic("Failed to create arena.")
		_bstore.tables = make(map[Tbl_Key]Basis_Entry, virtual.arena_allocator(&_bstore.arena))
	}

	if tbl, ok := _bstore.tables[key]; ok { return tbl }

	tbl: Basis_Entry
	alloc := virtual.arena_allocator(&_bstore.arena)

	for qty in BASIS_QUANTITIES[lb.bd.family] {
		tbl[qty] = bvec_create(f64, len(rule.points), basis_n_dofs(lb.bd), basis_quantity_cmpnts(lb.bd, qty), alloc)
	}

	for p, i in rule.points {
		r := element_lift_to_parent_reference(
			lb.bd.element,
			lb.dim,
			lb.index,
			element_orient_point(rule.element, point_key, p),
		)

		for qty in BASIS_QUANTITIES[lb.bd.family] {
			if tbl[qty].cmpnts > 0 { basis_eval(lb.bd, qty, r, bvec_at_point(tbl[qty], i)) }
		}
	}

	_bstore.tables[key] = tbl
	return tbl
}

@(init, private = "file")
bstore_register_cleaner :: proc "contextless" () {
	runtime.add_thread_local_cleaner(proc "contextless" () {
		if _bstore.arena == {} { return }
		context = runtime.default_context()
		virtual.arena_destroy(&_bstore.arena)
	})
}

//== Orientation

// Sub-entity blocks of an element that actually transform under its keys. None for an unoriented local basis.
@(private = "file")
Orient_Blocks :: struct {
	blocks: [MAX_EDGES + MAX_FACES]struct {
		start, n: int,
		t:        Orientation_Transform,
	},
	n:      int,
}

@(private)
Orient_Op :: enum {
	M,
	M_Inv_T,
}

@(private = "file")
orient_blocks :: proc(lb: Local_Basis) -> (ob: Orient_Blocks) {
	if !lb.oriented { return }
	for d in Dimension {
		if d == .D0 || d >= element_dim(lb.bd.element) { continue } 	// vertices and the element itself never orient
		n := basis_dofs_per_entity(lb.bd, d)
		if n == 0 { continue }
		for e in 0 ..< element_n_sub_entities(lb.bd.element, d) {
			t := basis_orientation(lb.bd, element_sub_entity(lb.bd.element, d, e).type, lb.keys[d][e])
			if t.kind == .Identity { continue }
			start, _ := basis_entity_dof_range(lb.bd, d, e)
			ob.blocks[ob.n] = {start, n, t}
			ob.n += 1
		}
	}
	return
}

// One point of a table, every dof row through M.
@(private = "file")
orient_point :: proc(ob: ^Orient_Blocks, src: Bvec_Point($T), dst: []T) {
	c := src.cmpnts
	copy(dst, src.data)
	for &b in ob.blocks[:ob.n] {
		orientation_apply(b.t, .M, b.n, c, src.data[b.start * c:][:b.n * c], dst[b.start * c:][:b.n * c])
	}
}

// dst = op(M) src over n rows of `stride` values.
@(private = "file")
orientation_apply :: proc(t: Orientation_Transform, op: Orient_Op, n, stride: int, src, dst: []$T) {
	switch t.kind {
	case .Identity:
		copy(dst, src)
	case .Signed_Perm:
		for i in 0 ..< n {
			sg := T(t.sign[i])
			for k in 0 ..< stride { dst[i * stride + k] = sg * src[t.src[i] * stride + k] }
		}
	case .Dense:
		for i in 0 ..< n {
			for k in 0 ..< stride {
				acc: T
				for l in 0 ..< n { acc += T(orientation_entry(t, op, i, l)) * src[l * stride + k] }
				dst[i * stride + k] = acc
			}
		}
	}
}

// Entry (i, l) of op(M). Signed perms have M^-T == M.
@(private)
orientation_entry :: proc(t: Orientation_Transform, op: Orient_Op, i, l: int) -> f64 {
	switch t.kind {
	case .Identity: return 1 if l == i else 0
	case .Signed_Perm: return t.sign[i] if l == t.src[i] else 0
	case .Dense: return dn_get(t.m, i, l)^ if op == .M else dn_get(t.m_inv, l, i)^
	}
	unreachable()
}

// Oriented copy of a reference table as T. Returned as is when nothing changes.
@(private = "file")
orient_table :: proc(ob: ^Orient_Blocks, ref: Bvec(f64), $T: typeid, alloc := context.allocator) -> Bvec(T) {
	when T == f64 { if ob.n == 0 { return ref } } // same type no orientation


	out := bvec_create(T, ref.points, ref.dofs, ref.cmpnts, alloc)

	scratch_guard()
	tmp := make([]f64, ref.dofs * ref.cmpnts, scratch())
	for point in 0 ..< ref.points {
		src := bvec_at_point(ref, point).data
		if ob.n > 0 {
			orient_point(ob, bvec_at_point(ref, point), tmp)
			src = tmp
		}
		for &x, k in bvec_at_point(out, point).data { x = T(src[k]) }
	}
	return out
}

// Oriented and pushed by `kind`.
@(private = "file")
orient_push :: proc(
	ob: ^Orient_Blocks,
	ref: Bvec(f64),
	em: Element_Map($A, $I, $T),
	kind: Map_Type,
	alloc := context.allocator,
) -> Bvec(T) {
	assert(ref.cmpnts == (1 if kind == .Density else I), "basis components don't match the map")

	out := bvec_create(T, ref.points, ref.dofs, 1 if kind == .Density else A, alloc)

	scratch_guard()
	tmp := make([]f64, ref.dofs * ref.cmpnts, scratch())
	for point in 0 ..< ref.points {
		src := bvec_at_point(ref, point).data
		if ob.n > 0 {
			orient_point(ob, bvec_at_point(ref, point), tmp)
			src = tmp
		}
		op := bvec_at_point(out, point)

		#partial switch kind {
		case .Density:
			s := 1 / element_map_measure(em, point)
			for &x, dof in op.data { x = T(src[dof]) * s }
		case .Covariant, .Contravariant:
			m := element_map_ctng(em, point)
			if kind == .Contravariant {
				m = element_map_tng(em, point)
				small_mat_scale_inplace(&m, 1 / element_map_measure(em, point))
			}
			for dof in 0 ..< ref.dofs {
				bvec_dof_vec(op, dof, A)^ = small_mat_vec_mul(m, small_vec_from_slice(T, src[dof * I:][:I], I))
			}
		case:
			panic("unity quantities are not pushed")
		}
	}
	return out
}
