package fe

/*
 Tabulation and retrieval of basis at different rules.

 The rules points ptr (raw_data(rule.points)) is used as a key. While this is sketchy, most
 rules are fixed tables, quadrature, dof functionals, etc so its quite natural and it
 works well when two different conceptual things (quadrature and RT functional rules) actually share the same points.

 The cache is thread local and lazy. Warming every type of basis needed is annoying an error-prone.
 Thread local to avoid any sync overhead.
*/

import "base:runtime"
import "core:mem/virtual"

Ref_Basis_Tbl :: Bvec(f64)

Basis_Entry :: [Basis_Quantity]Ref_Basis_Tbl

Tbl_Key :: struct {
	bd:     Basis_Desc,
	facet:  int, // -1 = interior/cell
	points: rawptr,
}

Basis_Store :: struct {
	tables: map[Tbl_Key]Basis_Entry,
	arena:  virtual.Arena,
}

@(thread_local, private)
_bstore: Basis_Store

@(private)
bstore :: proc() -> ^Basis_Store {
	if _bstore.arena == {} {
		virtual.arena_init_growing(&_bstore.arena) or_else panic("Failed to create arena.")
		_bstore.tables = make(map[Tbl_Key][Basis_Quantity]Ref_Basis_Tbl, virtual.arena_allocator(&_bstore.arena))
		runtime.add_thread_local_cleaner(proc "contextless" () {
			context = runtime.default_context()
			virtual.arena_destroy(&_bstore.arena)
		})
	}
	return &_bstore
}

// Use when the basis element type is of the same type as the rule.
bstore_get_interior :: proc(bd: Basis_Desc, rule: Rule) -> Basis_Entry {
	assert(element_dim(bd.element) == element_dim(rule.element))
	return bstore_get(bd, -1, rule)
}

// For when the rule is on the facet element type (surface quadrature)
bstore_get_facet :: proc(bd: Basis_Desc, facet: int, rule: Rule) -> Basis_Entry {
	assert(element_facet_dim(bd.element) == element_dim(rule.element))
	return bstore_get(bd, facet, rule)
}

bstore_get :: proc(bd: Basis_Desc, facet: int, rule: Rule) -> Basis_Entry {
	store := bstore()
	key := Tbl_Key{bd, facet, raw_data(rule.ref_points)}
	if tbl, ok := store.tables[key]; ok { return tbl }

	tbl := basis_tab(bd, rule, facet if facet > -1 else nil, virtual.arena_allocator(&store.arena))
	store.tables[key] = tbl
	return tbl
}

//== tabulation

basis_tab :: proc(bd: Basis_Desc, rule: Rule, facet: Maybe(int), alloc := context.allocator) -> Basis_Entry {
	n_dofs := len(basis_support(bd))
	n_points := len(rule.ref_points)

	tbl: Basis_Entry
	for qty in BASIS_QUANTITIES[bd.family] {
		tbl[qty] = bvec_create(f64, n_points, n_dofs, basis_quantity_components(bd.element, qty), alloc)
	}

	facet_idx, has_facet := facet.?

	for rule_point, point_idx in rule.ref_points {
		point := rule_point if !has_facet else lift_to_parent_reference(bd.element, facet_idx, rule_point)
		for dof in 0 ..< n_dofs {

			blocks: [Basis_Quantity][]f64
			for qty in BASIS_QUANTITIES[bd.family] {
				bt_p := bvec_at_point(tbl[qty], point_idx)
				blocks[qty] = bt_p.data[dof * bt_p.cmpnts:][:bt_p.cmpnts]
			}

			switch bd.family {
			case .Lagrange:
				ref := REFERENCE_ELEMENTS[bd.element].lagrange[bd.order]
				g := ref.grads(dof, point)
				copy(blocks[.Scalar], []f64{ref.vals(dof, point)})
				copy(blocks[.Scalar_Gradient], g[:tbl[.Scalar_Gradient].cmpnts]) // gradient always a 3-vector, need to slice off
			case .Raviart_Thomas:
				ref := REFERENCE_ELEMENTS[bd.element].rt[bd.order]
				v := ref.vals(dof, point)
				copy(blocks[.Vector], v[:tbl[.Vector].cmpnts]) // value always a 3-vector, need to slice off
				copy(blocks[.Vector_Divergence], []f64{ref.divs(dof, point)})
			}
		}
	}
	return tbl
}
