package fe

import "base:runtime"
import "core:mem/virtual"

Ref_Basis_Tbl :: Bvec(f64)
Basis_Entry   :: [Basis_Quantity]Ref_Basis_Tbl

Tbl_Key :: struct {
	bd:        Basis_Desc,
	dim:       Dimension, // dimension of the entity the rule lives on (element dim = interior)
	entity:    int,
	point_key: u8, // orientation applied to the rule points before lifting
	points:    rawptr,
}

Basis_Store :: struct {
	tables: map[Tbl_Key]Basis_Entry,
	arena:  virtual.Arena,
}

// Rule on the element itself.
bstore_interior :: proc(bd: Basis_Desc, rule: Rule) -> Basis_Entry {
	assert(rule.element == bd.element)
	return bstore_get({bd, element_dim(bd.element), 0, 0, raw_data(rule.points)}, rule)
}

// Rule on a sub-entity (e.g. an edge or facet), lifted into the element. `point_key` orients the rule points on
// the entity first, to line them up with a neighbour's (DG facet terms). Keep 0 otherwise.
bstore_sub_entity :: proc(bd: Basis_Desc, dim: Dimension, index: int, rule: Rule, point_key: u8 = 0) -> Basis_Entry {
	assert(rule.element == element_sub_entity(bd.element, dim, index).type)
	return bstore_get({bd, dim, index, point_key, raw_data(rule.points)}, rule)
}

// Get the basis where the rule was tabulated over the facet (surface quadrature). point_key can be used to orient
// the quadrature points, keep 0 unless doing some cross-element intgeral.
bstore_facet :: proc(bd: Basis_Desc, facet: int, rule: Rule, point_key: u8 = 0) -> Basis_Entry {
	return bstore_sub_entity(bd, element_dim(bd.element) - Dimension(1), facet, rule, point_key)
}


@(private = "file")
bstore :: proc() -> ^Basis_Store {
	@(thread_local)
	_bstore: Basis_Store

	if _bstore.arena == {} {
		virtual.arena_init_growing(&_bstore.arena) or_else panic("Failed to create arena.")
		_bstore.tables = make(map[Tbl_Key]Basis_Entry, virtual.arena_allocator(&_bstore.arena))
		runtime.add_thread_local_cleaner(proc "contextless" () {
			context = runtime.default_context()
			virtual.arena_destroy(&_bstore.arena)
		})
	}
	return &_bstore
}

@(private = "file")
bstore_get :: proc(key: Tbl_Key, rule: Rule) -> Basis_Entry {
	store := bstore()
	if tbl, ok := store.tables[key]; ok { return tbl }

	tbl := basis_tab(key.bd, rule, key.dim, key.entity, key.point_key, virtual.arena_allocator(&store.arena))
	store.tables[key] = tbl
	return tbl
}

// Tabulate every quantity of the basis at the rule's points, which live on sub-entity (dim, entity).
basis_tab :: proc(
	bd: Basis_Desc,
	rule: Rule,
	dim: Dimension,
	entity: int,
	point_key: u8 = 0,
	alloc := context.allocator,
) -> (
	tbl: Basis_Entry,
) {
	n_dofs := basis_num_dofs(bd)
	for qty in BASIS_QUANTITIES[bd.family] {
		tbl[qty] = bvec_create(f64, len(rule.points), n_dofs, basis_quantity_cmpnts(bd, qty), alloc)
	}

	for p, i in rule.points {
		r := element_lift_to_parent_reference(
			bd.element,
			dim,
			entity,
			element_orient_point(rule.element, point_key, p),
		)
		for qty in BASIS_QUANTITIES[bd.family] {
			if tbl[qty].cmpnts > 0 { basis_eval(bd, qty, r, bvec_at_point(tbl[qty], i)) }
		}
	}
	return
}

//== Oriented basis

// Oriented copy of a reference table, if basis is unchanged from reference basis it is not copied.
basis_orient :: proc(bd: Basis_Desc, ref: Bvec($T), keys: Entity_Orientation, alloc := context.allocator) -> Bvec(T) {
	ob := orient_blocks(bd, keys)
	if ob.n == 0 { return ref }
	out := bvec_create(T, ref.points, ref.dofs, ref.cmpnts, alloc)
	for p in 0 ..< ref.points { orient_point(&ob, bvec_at_point(ref, p), bvec_at_point(out, p).data) }
	return out
}

// Oriented and pushed forward with a Piola map.
basis_orient_push :: proc {
	basis_orient_push_cov,
	basis_orient_push_con,
	basis_orient_push_density,
}

basis_orient_push_cov :: proc(
	bd: Basis_Desc,
	ref: Bvec($BT),
	keys: Entity_Orientation,
	p: Piola_Cov($A, $I, $T),
	alloc := context.allocator,
) -> Bvec(T) {
	return orient_push_mat(bd, ref, keys, p, A, I, T, alloc)
}

basis_orient_push_con :: proc(
	bd: Basis_Desc,
	ref: Bvec($BT),
	keys: Entity_Orientation,
	p: Piola_Con($A, $I, $T),
	alloc := context.allocator,
) -> Bvec(T) {
	return orient_push_mat(bd, ref, keys, p, A, I, T, alloc)
}

basis_orient_push_density :: proc(
	bd: Basis_Desc,
	ref: Bvec($BT),
	keys: Entity_Orientation,
	p: Piola_Den($A, $I, $T),
	alloc := context.allocator,
) -> Bvec(T) {
	assert(ref.cmpnts == 1, "density piola expects a scalar basis")
	ob := orient_blocks(bd, keys)
	out := bvec_create(T, ref.points, ref.dofs, 1, alloc)

	scratch_guard()
	tmp := make([]BT, ref.dofs, scratch())
	for point in 0 ..< ref.points {
		src := bvec_at_point(ref, point).data
		if ob.n > 0 {
			orient_point(&ob, bvec_at_point(ref, point), tmp)
			src = tmp
		}
		s := piola_at(p, point)
		for &x, dof in bvec_at_point(out, point).data { x = T(src[dof]) * s }
	}
	return out
}

// Reference dof values (e.g. from interpolation) -> dof values of the oriented basis.
basis_orient_dofs :: proc(bd: Basis_Desc, keys: Entity_Orientation, vals: Cvec($T)) {
	ob := orient_blocks(bd, keys)
	f := vals.fields

	scratch_guard()
	for &b in ob.blocks[:ob.n] {
		dst := vals.data[b.start * f:][:b.n * f]
		src := make([]T, len(dst), scratch())
		copy(src, dst)
		#partial switch b.t.kind {
		case .Signed_Perm: // M^-T == M: dof i = sign[i] * dof src[i]
			for i in 0 ..< b.n {
				sg := T(b.t.sign[i])
				for k in 0 ..< f { dst[i * f + k] = sg * src[b.t.src[i] * f + k] }
			}
		case .Dense: // dof i = sum_l M^-1[l, i] dof l
			for i in 0 ..< b.n {
				for k in 0 ..< f {
					acc: T
					for l in 0 ..< b.n { acc += T(dn_get(b.t.m_inv, l, i)^) * src[l * f + k] }
					dst[i * f + k] = acc
				}
			}
		}
	}
}

@(private = "file")
orient_push_mat :: proc(
	bd: Basis_Desc,
	ref: Bvec($BT),
	keys: Entity_Orientation,
	p: $P,
	$A, $I: int,
	$T: typeid,
	alloc := context.allocator,
) -> Bvec(T) {
	assert(ref.cmpnts == I, "basis component count must match the intrinsic dimension")
	ob := orient_blocks(bd, keys)
	out := bvec_create(T, ref.points, ref.dofs, A, alloc)

	scratch_guard()
	tmp := make([]BT, ref.dofs * I, scratch())
	for point in 0 ..< ref.points {
		src := bvec_at_point(ref, point).data
		if ob.n > 0 {
			orient_point(&ob, bvec_at_point(ref, point), tmp)
			src = tmp
		}
		m := piola_at(p, point)
		op := bvec_at_point(out, point)
		for dof in 0 ..< ref.dofs {
			outv := bvec_dof_vec(op, dof, A)
			outv^ = small_mat_vec_mul(m, small_vec_from_slice(T, src[dof * I:][:I], I))
		}
	}
	return out
}

// Sub-entity blocks of an element that actually transform under the given keys.
@(private = "file")
Orient_Blocks :: struct {
	blocks: [MAX_EDGES + MAX_FACES]struct {
		start, n: int,
		t:        Orientation_Transform,
	},
	n:      int,
}

@(private = "file")
orient_blocks :: proc(bd: Basis_Desc, keys: Entity_Orientation) -> (ob: Orient_Blocks) {
	for d in Dimension {
		if d == .D0 || d >= element_dim(bd.element) { continue } 	// vertices and the element itself never orient
		n := basis_dofs_per_entity(bd, d)
		if n == 0 { continue }
		for e in 0 ..< element_num_sub_entities(bd.element, d) {
			key := keys[d][e] if e < len(keys[d]) else 0
			t := basis_orientation(bd, element_sub_entity(bd.element, d, e).type, key)
			if t.kind == .Identity { continue }
			start, _ := basis_entity_dof_range(bd, d, e)
			ob.blocks[ob.n] = {start, n, t}
			ob.n += 1
		}
	}
	return
}

@(private = "file")
orient_point :: proc(ob: ^Orient_Blocks, src: Bvec_Point($T), dst: []T) {
	c := src.cmpnts
	copy(dst, src.data)
	for &b in ob.blocks[:ob.n] {
		s := src.data[b.start * c:][:b.n * c]
		d := dst[b.start * c:][:b.n * c]
		#partial switch b.t.kind {
		case .Signed_Perm: // row i = sign[i] * row src[i]
				for i in 0 ..< b.n {
					sg := T(b.t.sign[i])
					for k in 0 ..< c { d[i * c + k] = sg * s[b.t.src[i] * c + k] }
				}
		case .Dense: // row i = sum_l M[i, l] row l, column-major M[i, l] = values[l * n + i]
				for i in 0 ..< b.n {
					for k in 0 ..< c {
						acc: T
						for l in 0 ..< b.n { acc += T(dn_get(b.t.m, i, l)^) * s[l * c + k] }
						d[i * c + k] = acc
					}
				}
		}
	}
}
