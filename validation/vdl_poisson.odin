package validation

import "core:log"
import "core:mem/virtual"
import "core:slice"
import "core:testing"

import fe "../fe"
import "../fe/fio"

import "core:math"

/*
 Various forms of poisson type problems over different geometries.
*/

// Element integrals of one cell: the sigma mass block, the two coupling blocks and the source.
// A: ambient dimension (coordinates per node), I: element dimension.
assemble_cell :: proc(
	$A, $I: int,
	sys: fe.Sys,
	geo: fe.Space_Vector,
	sigma, u: ^fe.Space,
	K: fe.Sparse_Matrix,
	rhs: fe.Vector,
	inhom: fe.State,
	cell: ^fe.Cell,
	diffusivity, source: f64,
) {
	sigma_bd := fe.space_bd(sigma, cell.type)
	quad, weights := fe.basis_quad_rule(sigma_bd)
	n_points := len(quad.points)

	// Geometry: the cell's tangent (Jacobian) at each quadrature point.
	geo_basis := fe.bstore_interior(fe.space_bd(geo, cell.type), quad)
	nodes := fe.space_gather(f64, geo, cell.id)
	tangent := fe.tangent_from_nodes(A, I, geo_basis[.S_Grd], nodes, cell.affine)

	// Reference bases, straight from the cache.
	sigma_ref := fe.bstore_interior(sigma_bd, quad)
	u_ref := fe.bstore_interior(fe.space_bd(u, cell.type), quad)

	// sigma needs the cell's orientation (shared facets must agree on the flux direction) and the contravariant
	// Piola map: sigma = J sigma_ref / det J. Its divergence only needs orienting: div sigma = div_ref / det J,
	// and that det J cancels against the one in dx, so it stays in reference form.
	// u is discontinuous P0, so it needs neither.
	keys := fe.cell_entity_keys(cell)
	sigma_val := fe.basis_orient_push(sigma_bd, sigma_ref[.V_Val], keys, fe.piola_contravariant(tangent))
	sigma_div := fe.basis_orient(sigma_bd, sigma_ref[.V_Div], keys)
	u_val := u_ref[.S_Val]

	// Point-wise coefficients, quadrature weights (and det J where needed) folded in.
	inv_k := fe.pmat_create(f64, n_points, .DIAGONAL, A, 1, A, 1) // k^-1 dx
	plus_dx := fe.pmat_create(f64, n_points, .DIAGONAL, 1, 1, 1, 1) // dx, det J already cancelled
	minus_dx := fe.pmat_create(f64, n_points, .DIAGONAL, 1, 1, 1, 1)
	source_dx := fe.pvec_create(f64, n_points, 1, 1) // f dx

	for p in 0 ..< n_points {
		dx := fe.tangent_measure(tangent, p) * weights[p]
		for &e in fe.pmat_at_point(inv_k, p).data { e = dx / diffusivity }
		for &e in fe.pmat_at_point(plus_dx, p).data { e = weights[p] }
		for &e in fe.pmat_at_point(minus_dx, p).data { e = -weights[p] }
		fe.pvec_at_point(source_dx, p).data[0] = source * dx
	}

	mass := fe.cmat_create_for(sigma_val, inv_k, sigma_val) //  int k^-1 sigma . tau
	div_sigma := fe.cmat_create_for(u_val, plus_dx, sigma_div) //  int v div sigma
	grad_u := fe.cmat_create_for(sigma_div, minus_dx, u_val) // -int u div tau
	load := fe.cvec_create_for(u_val, source_dx) //  int f v

	fe.contract_bilinear(A, 1, A, 1, mass, sigma_val, inv_k, sigma_val)
	fe.contract_bilinear(1, 1, 1, 1, div_sigma, u_val, plus_dx, sigma_div)
	fe.contract_bilinear(1, 1, 1, 1, grad_u, sigma_div, minus_dx, u_val)
	fe.contract_linear(1, 1, load, u_val, source_dx)

	fe.sys_scatter_mat(sys, K, rhs, inhom, .Linear, mass, sigma, sigma, cell.id)
	fe.sys_scatter_mat(sys, K, rhs, inhom, .Linear, div_sigma, u, sigma, cell.id)
	fe.sys_scatter_mat(sys, K, rhs, inhom, .Linear, grad_u, sigma, u, cell.id)
	fe.sys_scatter_vec(sys, rhs, load, u, cell.id)
}

// The Dirichlet term -int u_D tau . n on one boundary facet of a cell.
assemble_dirichlet_facet :: proc(
	$A, $I: int,
	sys: fe.Sys,
	geo: fe.Space_Vector,
	sigma: ^fe.Space,
	rhs: fe.Vector,
	cell: ^fe.Cell,
	local_facet: int,
	u_d: f64, // constant here;
) {
	facet := fe.element_facet(cell.type, local_facet)
	sigma_bd := fe.space_bd(sigma, cell.type)

	quad, weights := fe.basis_facet_quad_rule(sigma_bd, local_facet)
	n_points := len(quad.points)

	// Facet quadrature points are lifted into the cell, so the tangent here is the cell's Jacobian at them.
	geo_basis := fe.bstore_facet(fe.space_bd(geo, cell.type), local_facet, quad)
	nodes := fe.space_gather(f64, geo, cell.id)
	tangent := fe.tangent_from_nodes(A, I, geo_basis[.S_Grd], nodes, cell.affine)
	covariant := fe.piola_covariant(tangent)

	sigma_ref := fe.bstore_facet(sigma_bd, local_facet, quad)
	sigma_val := fe.basis_orient_push(
		sigma_bd,
		sigma_ref[.V_Val],
		fe.cell_entity_keys(cell),
		fe.piola_contravariant(tangent),
	)


	n_ref := fe.small_vec_from_slice(f64, facet.normal, I)
	boundary_value := fe.pvec_create(f64, n_points, A, 1) // -u_D n dA
	for p in 0 ..< n_points {
		n_da := fe.piola_apply_cofactor(fe.piola_at(covariant, p), fe.tangent_measure(tangent, p), n_ref) // n dA
		out := fe.pvec_at_point(boundary_value, p)
		for c in 0 ..< A { out.data[c] = -u_d * weights[p] * n_da.data[c] }
	}

	load := fe.cvec_create_for(sigma_val, boundary_value)
	fe.contract_linear(A, 1, load, sigma_val, boundary_value)
	fe.sys_scatter_vec(sys, rhs, load, sigma, cell.id)
}


@(test)
mixed_poisson :: proc(t: ^testing.T) {
	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)

	SOURCE :: -10.0
	DIFFUSIVITY :: 4
	U_D :: 10.0 // boundary value of u, everywhere on the boundary

	//== Mesh and geometry

	mesh := fio.load_mesh("./validation/meshes/2d_channel.msh", .GMSH_V2_BINARY) or_else testing.fail_now(t)
	defer fe.mesh_destroy(&mesh)

	// The geometry is itself an FE field: the mesh's own Lagrange space, holding node coordinates in the xy-plane.
	geo_space := fe.space_new_isoparemetric(&mesh, 2)
	geo := fe.Space_Vector{geo_space, fe.mesh_coord_coeffs(mesh, fe.XY_PLANE_FRAME)}

	//== Spaces and system

	sigma_space := fe.space_new(&mesh, {.Raviart_Thomas, .O0, .Continuous, fe.ALL_REGIONS}, 1)
	u_space := fe.space_new(&mesh, {.Lagrange, .O0, .Discontinuous, fe.ALL_REGIONS}, 1)

	// No constraints: u_D enters weakly through sigma's equation, and u (P0) has no boundary dofs anyway.
	ms := fe.ms_create(.Eliminate, {space = sigma_space}, {space = u_space})
	defer fe.ms_destroy(&ms)

	// Every block the weak form touches. The u-u block is empty, but declaring it keeps a diagonal for the solver.
	sys := fe.sys_create(
	ms,
	{test = sigma_space, trial = sigma_space},
	{test = u_space, trial = sigma_space},
	{test = sigma_space, trial = u_space},
	// no u-u saddle point
	)
	defer fe.sys_destroy(&sys)

	state := fe.ms_state_alloc(ms)
	inhom := fe.ms_state_alloc(ms) // essential values; all zero, there are none
	rhs, soln := fe.ms_soln_vector(ms), fe.ms_soln_vector(ms)
	K := fe.sys_soln_matrix(sys)

	//== Assemble

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()

		assemble_cell(2, 2, sys, geo, sigma_space, u_space, K, rhs, inhom, &cell, DIFFUSIVITY, SOURCE)

		for facet_id, local_facet in cell.facets {
			if !fe.facet_is_boundary(mesh.facets[facet_id]) { continue }
			assemble_dirichlet_facet(2, 2, sys, geo, sigma_space, rhs, &cell, local_facet, U_D)
		}
	}

	//== Solve

	precond := fe.amgcl_precond_create(K, fe.relaxation_params(.ILU0)) or_else testing.fail_now(t, string(fe.amgcl_last_error()))
	defer fe.amgcl_precond_destroy(precond)

	res := fe.amgcl_solve(precond, rhs, soln, fe.bicgstab_params(tol = 1e-12, maxiter = 700))
	log.info(res)
	testing.expect(t, res.status == .Converged)

	fe.ms_apply_soln(ms, state, inhom, soln)

	//== Output: evaluate both fields at the writer's points

	writer, out_rules := fio.output_setup(mesh, geo_space, geo.coeffs, .O1, fio.VTU_Config{})
	defer fio.output_takedown(writer)

	u_out := fio.output_field_create(mesh, "U", u_space.fields, out_rules)
	sigma_out := fio.output_field_create(mesh, "Sigma", 2, out_rules) // a 2D vector per point
	div_out := fio.output_field_create(mesh, "DivSigma", 1, out_rules)

	u_vec := fe.ms_space_vec(ms, state, u_space)
	sigma_vec := fe.ms_space_vec(ms, state, sigma_space)

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		rule := out_rules[cell.type]
		n_points := len(rule.points)

		// u: plain evaluation.
		u_ref := fe.bstore_interior(fe.space_bd(u_space, cell.type), rule)
		u_at := fe.pvec_create(f64, n_points, 1, 1)
		fe.contract_eval(1, 1, u_at, fe.space_gather(f64, u_vec, cell.id), u_ref[.S_Val])
		copy(u_out.data[cell.id], u_at.data)

		// sigma: same orientation and Piola push as in assembly.
		geo_basis := fe.bstore_interior(fe.space_bd(geo_space, cell.type), rule)
		tangent := fe.tangent_from_nodes(2, 2, geo_basis[.S_Grd], fe.space_gather(f64, geo, cell.id), cell.affine)

		sigma_bd := fe.space_bd(sigma_space, cell.type)
		sigma_ref := fe.bstore_interior(sigma_bd, rule)
		sigma_val := fe.basis_orient_push(
			sigma_bd,
			sigma_ref[.V_Val],
			fe.cell_entity_keys(&cell),
			fe.piola_contravariant(tangent),
		)

		sigma_at := fe.pvec_create(f64, n_points, 2, 1)
		sigma_coeffs := fe.space_gather(f64, sigma_vec, cell.id)
		fe.contract_eval(2, 1, sigma_at, sigma_coeffs, sigma_val)
		copy(sigma_out.data[cell.id], sigma_at.data)

		// div sigma = div_ref / det J: oriented, then the density Piola map.
		sigma_div := fe.basis_orient_push(
			sigma_bd,
			sigma_ref[.V_Div],
			fe.cell_entity_keys(&cell),
			fe.piola_density(tangent),
		)
		div_at := fe.pvec_create(f64, n_points, 1, 1)
		fe.contract_eval(1, 1, div_at, sigma_coeffs, sigma_div)
		copy(div_out.data[cell.id], div_at.data)
	}

	max: f64 = 0
	min: f64 = 10000
	for e in div_out.data {
		if abs(slice.max(e)) > max { max = abs(slice.max(e)) }
		if abs(slice.min(e)) < min { min = abs(slice.min(e)) }
	}
	log.info(max, min)

	fio.output_write(writer, {fields = {u_out, sigma_out, div_out}, path = "./validation/output/mixed_poisson"})
}


TAU :: 1.0 // stabilization

// One component of a vector-valued basis table, as a scalar table.
basis_component :: proc(b: fe.Bvec(f64), c: int) -> fe.Bvec(f64) {
	out := fe.bvec_create(f64, b.points, b.dofs, 1)
	for p in 0 ..< b.points {
		for d in 0 ..< b.dofs { out.data[p * b.dofs + d] = b.data[(p * b.dofs + d) * b.cmpnts + c] }
	}
	return out
}

// Per-point scalar weights, as a diagonal point matrix.
point_weights :: proc(values: []f64) -> fe.Pmat(f64) {
	m := fe.pmat_create(f64, len(values), .DIAGONAL, 1, 1, 1, 1)
	for v, p in values {
		for &e in fe.pmat_at_point(m, p).data { e = v }
	}
	return m
}

// Adds int (weights * test * trial) between two scalar bases to the condenser.
add_form :: proc(
	c: ^fe.Condenser,
	test: ^fe.Space,
	test_basis: fe.Bvec(f64),
	weights: fe.Pmat(f64),
	trial: ^fe.Space,
	trial_basis: fe.Bvec(f64),
	facet := -1,
) {
	m := fe.cmat_create_for(test_basis, weights, trial_basis)
	fe.contract_bilinear(1, 1, 1, 1, m, test_basis, weights, trial_basis)
	fe.cond_add_mat(c, m, test, trial, facet)
}

Spaces :: struct {
	q:      [2]^fe.Space, // q_x, q_y
	u, lam: ^fe.Space,
}

// Volume terms of one cell.
assemble_cell_hdg :: proc(
	c: ^fe.Condenser,
	geo: fe.Space_Vector,
	s: Spaces,
	cell: ^fe.Cell,
	diffusivity, source: f64,
) {
	u_bd := fe.space_bd(s.u, cell.type)
	quad, weights := fe.basis_quad_rule(u_bd)
	n_points := len(quad.points)

	geo_basis := fe.bstore_interior(fe.space_bd(geo, cell.type), quad)
	tangent := fe.tangent_from_nodes(2, 2, geo_basis[.S_Grd], fe.space_gather(f64, geo, cell.id), cell.affine)
	covariant := fe.piola_covariant(tangent)

	// q_x, q_y and u share one scalar DG basis. Gradients are pushed with the covariant map (J^-T).
	ref := fe.bstore_interior(u_bd, quad)
	phi := ref[.S_Val]
	grad := fe.piola_push_bvec(ref[.S_Grd], covariant)
	d_phi := [2]fe.Bvec(f64){basis_component(grad, 0), basis_component(grad, 1)}

	dx := make([]f64, n_points)
	for p in 0 ..< n_points { dx[p] = fe.tangent_measure(tangent, p) * weights[p] }

	inv_k_dx := make([]f64, n_points)
	minus_dx := make([]f64, n_points)
	source_dx := fe.pvec_create(f64, n_points, 1, 1)
	for p in 0 ..< n_points {
		inv_k_dx[p] = dx[p] / diffusivity
		minus_dx[p] = -dx[p]
		fe.pvec_at_point(source_dx, p).data[0] = source * dx[p]
	}

	for i in 0 ..< 2 {
		add_form(c, s.q[i], phi, point_weights(inv_k_dx), s.q[i], phi) //  (k^-1 q, r)
		add_form(c, s.q[i], d_phi[i], point_weights(minus_dx), s.u, phi) // -(u, div r)
		add_form(c, s.u, d_phi[i], point_weights(minus_dx), s.q[i], phi) // -(q, grad v)
	}

	load := fe.cvec_create_for(phi, source_dx) // (f, v)
	fe.contract_linear(1, 1, load, phi, source_dx)
	fe.cond_add_vec(c, load, s.u)
}

// Facet terms of one cell, on its local facet `lf`.
assemble_facet :: proc(c: ^fe.Condenser, mesh: ^fe.Mesh, geo: fe.Space_Vector, s: Spaces, cell: ^fe.Cell, lf: int) {
	u_bd := fe.space_bd(s.u, cell.type)
	quad, weights := fe.basis_facet_quad_rule(u_bd, lf)
	n_points := len(quad.points)

	key := cell.edge_orientation[lf] // 2D: facets are edges

	// Cell Jacobian at the facet points, and Nanson's n dA from the scaled reference facet normal.
	geo_basis := fe.bstore_facet(fe.space_bd(geo, cell.type), lf, quad, key)
	tangent := fe.tangent_from_nodes(2, 2, geo_basis[.S_Grd], fe.space_gather(f64, geo, cell.id), cell.affine)
	covariant := fe.piola_covariant(tangent)
	n_ref := fe.small_vec_from_slice(f64, fe.element_facet(cell.type, lf).normal, 2)

	n_da: [2][]f64 = {make([]f64, n_points), make([]f64, n_points)} // n_i dA
	tau_da, minus_tau_da := make([]f64, n_points), make([]f64, n_points)

	for p in 0 ..< n_points {
		n := fe.piola_apply_cofactor(fe.piola_at(covariant, p), fe.tangent_measure(tangent, p), n_ref)
		n_da[0][p], n_da[1][p] = n.data[0] * weights[p], n.data[1] * weights[p]
		da := math.sqrt(n.data[0] * n.data[0] + n.data[1] * n.data[1]) * weights[p]
		tau_da[p], minus_tau_da[p] = TAU * da, -TAU * da
	}

	// Cell-side basis at the facet points (the cell's view of the facet).
	phi := fe.bstore_facet(u_bd, lf, quad, key)[.S_Val]

	// Trace basis: lives on the facet element in the facet's own vertex order, so the cell's facet points are
	// re-parametrised with the cell's orientation key for that facet.
	fid := cell.facets[lf]
	ft := mesh.facets[fid].info.type
	mu := fe.bstore_sub_entity(fe.space_bd(s.lam, ft), .D1, 0, quad, 0)[.S_Val]

	for i in 0 ..< 2 {
		add_form(c, s.q[i], phi, point_weights(n_da[i]), s.lam, mu, lf) //  <lambda, r.n>
		add_form(c, s.u, phi, point_weights(n_da[i]), s.q[i], phi, lf) //  <q.n, v>
		add_form(c, s.lam, mu, point_weights(n_da[i]), s.q[i], phi, lf) //  <q.n, mu>
	}
	add_form(c, s.u, phi, point_weights(tau_da), s.u, phi, lf) //  <tau u, v>
	add_form(c, s.u, phi, point_weights(minus_tau_da), s.lam, mu, lf) // -<tau lambda, v>
	add_form(c, s.lam, mu, point_weights(tau_da), s.u, phi, lf) //  <tau u, mu>
	add_form(c, s.lam, mu, point_weights(minus_tau_da), s.lam, mu, lf) // -<tau lambda, mu>
}

@(test)
hdg_poisson :: proc(t: ^testing.T) {
	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)

	SOURCE :: -10
	DIFFUSIVITY :: 4
	U_D :: 10.0
	ORDER :: fe.Order.O1

	//== Mesh and geometry

	mesh := fio.load_mesh("./validation/meshes/2d_channel.msh", .GMSH_V2_BINARY) or_else testing.fail_now(t)
	defer fe.mesh_destroy(&mesh)

	geo_space := fe.space_new_isoparemetric(&mesh, 2)
	geo := fe.Space_Vector{geo_space, fe.mesh_coord_coeffs(mesh, fe.XY_PLANE_FRAME)}

	//== Spaces: q and u per cell (eliminated), lambda per facet (solved)

	s := Spaces {
		q   = {
			fe.space_new(&mesh, {.Lagrange, ORDER, .Discontinuous, fe.ALL_REGIONS}, 1),
			fe.space_new(&mesh, {.Lagrange, ORDER, .Discontinuous, fe.ALL_REGIONS}, 1),
		},
		u   = fe.space_new(&mesh, {.Lagrange, ORDER, .Discontinuous, fe.ALL_REGIONS}, 1),
		lam = fe.space_new(&mesh, {.Lagrange, ORDER, .Continuous, fe.ALL_REGIONS}, 1, .Facet),
	}

	cond := fe.cond_create(&mesh, {s.q[0], s.q[1], s.u}, {s.lam})
	defer fe.cond_destroy(&cond)

	// Only lambda is solved for. u_D on every boundary facet, as an essential constraint on lambda.
	ms := fe.ms_create(
		.Eliminate,
		{space = s.lam, constraints = {fe.Constraint_Essential{boundaries = ~fe.Boundary_Set{}}}},
	)
	defer fe.ms_destroy(&ms)

	// Condensation couples every pair of facets of a cell.
	sys := fe.sys_create(ms, {test = s.lam, trial = s.lam, through_cells = true})
	defer fe.sys_destroy(&sys)

	state := fe.ms_state_alloc(ms)
	inhom := fe.ms_state_alloc(ms)
	rhs, soln := fe.ms_soln_vector(ms), fe.ms_soln_vector(ms)
	K := fe.sys_soln_matrix(sys)

	// Constant u_D: a nodal (Lagrange) trace takes u_D at every constrained dof.
	for entry, gidx in ms.dof_map {
		if entry.role == .Constrained {
			inhom[gidx] = U_D
		}
	}


	//== Assemble: build each cell's local system, condense it onto lambda

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()

		fe.cond_begin(&cond, &cell)
		assemble_cell_hdg(&cond, geo, s, &cell, DIFFUSIVITY, SOURCE)
		for _, lf in cell.facets { assemble_facet(&cond, &mesh, geo, s, &cell, lf) }
		fe.cond_end(&cond, sys, K, rhs, inhom, .Linear)
	}

	//== Solve for lambda, then recover q and u

	precond := fe.amgcl_precond_create(K, fe.relaxation_params(.ILUK)) or_else testing.fail_now(t, string(fe.amgcl_last_error()))
	defer fe.amgcl_precond_destroy(precond)

	res := fe.amgcl_solve(precond, rhs, soln, fe.cg_params(tol = 1e-8, maxiter = 700))
	log.info(res)
	testing.expect(t, res.status == .Converged)

	fe.ms_apply_soln(ms, state, inhom, soln)

	//log.info(state)

	q_vecs := [2]fe.Space_Vector {
		{s.q[0], make([]f64, s.q[0].total_coeffs)},
		{s.q[1], make([]f64, s.q[1].total_coeffs)},
	}
	u_vec := fe.Space_Vector{s.u, make([]f64, s.u.total_coeffs)}
	fe.cond_reconstruct(&cond, {fe.ms_space_vec(ms, state, s.lam)}, {q_vecs[0], q_vecs[1], u_vec})

	//== Output

	writer, out_rules := fio.output_setup(mesh, geo_space, geo.coeffs, .O1, fio.VTU_Config{})
	defer fio.output_takedown(writer)

	u_out := fio.output_field_create(mesh, "U", 1, out_rules)
	q_out := fio.output_field_create(mesh, "Q", 2, out_rules)

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		rule := out_rules[cell.type]
		n_points := len(rule.points)
		phi := fe.bstore_interior(fe.space_bd(s.u, cell.type), rule)[.S_Val]

		u_at := fe.pvec_create(f64, n_points, 1, 1)
		fe.contract_eval(1, 1, u_at, fe.space_gather(f64, u_vec, cell.id), phi)
		copy(u_out.data[cell.id], u_at.data)

		for i in 0 ..< 2 {
			qi := fe.pvec_create(f64, n_points, 1, 1)
			fe.contract_eval(1, 1, qi, fe.space_gather(f64, q_vecs[i], cell.id), phi)
			for p in 0 ..< n_points { q_out.data[cell.id][p * 2 + i] = qi.data[p] }
		}
	}


	fio.output_write(writer, {fields = {u_out, q_out}, path = "./validation/output/hdg_poisson"})
}
