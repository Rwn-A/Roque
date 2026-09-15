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

//@(test)
standard_poisson :: proc(t: ^testing.T) {
	wf :: proc(
		$AMB, $INT: int,
		sys: fe.Sys,
		geo: fe.Space_Vector,
		phi: ^fe.Space,
		k: fe.Sparse_Matrix,
		f: fe.Vector,
		inhom: fe.State,
		cell: fe.Cell,
		diffusivity: f64,
		forcing: proc(x: fe.Small_Vec(AMB, f64)) -> f64,
	) {
		quad := fe.basis_infer_quad(fe.space_bd(phi, cell.type))
		n_p := len(quad.ref_points)

		GR_DIMS :: fe.Contraction_Dims {
			.CMPNTS = AMB,
			.FIELDS = 1,
		}
		VL_DIMS :: fe.Contraction_Dims {
			.CMPNTS = 1,
			.FIELDS = 1,
		}
		PT_DIMS :: fe.Contraction_Dims {
			.CMPNTS = 1,
			.FIELDS = AMB,
		}

		geo_basis := fe.bstore_interior(fe.space_bd(geo, cell.type), quad)
		geo_coeffs := fe.space_gather(f64, geo, cell.id)
		tng := fe.geom_tng_from_nodes(AMB, INT, geo_basis[.S_Grd], geo_coeffs, cell.affine)
		j_inv_t := fe.geom_ctng(tng)

		phi_basis := fe.bstore_interior(fe.space_bd(phi, cell.type), quad)
		phys_grads := fe.frame_push_bvec(phi_basis[.S_Grd], j_inv_t)

		phys_points := fe.pvec_create(f64, n_p, PT_DIMS)
		fe.contract_eval(PT_DIMS, phys_points, geo_coeffs, geo_basis[.S_Val])

		l_load := fe.space_cvec(f64, phi, cell.type)
		l_stiffness := fe.space_cmat(f64, phi, phi, cell.type, cell.type)

		diff_p := fe.pmat_create(f64, n_p, .DIAGONAL, GR_DIMS, GR_DIMS)
		force_p := fe.pvec_create(f64, n_p, VL_DIMS)

		for i in 0 ..< n_p {
			measure := fe.geom_measure(tng, i)
			pt := fe.small_vec_view_from_slice(fe.pvec_at_point(phys_points, i).data, AMB)^

			for &e in fe.pmat_at_point(diff_p, i).data { e = diffusivity * measure * quad.weights[i] }
			fe.pvec_at_point(force_p, i).data[0] = forcing(pt) * measure * quad.weights[i]
		}

		fe.contract_linear(VL_DIMS, l_load, phi_basis[.S_Val], force_p)
		fe.contract_bilinear(GR_DIMS, GR_DIMS, l_stiffness, phys_grads, diff_p, phys_grads)

		fe.sys_scatter_mat(sys, k, f, inhom, .Linear, l_stiffness, phi.id, phi.id, cell.id)
		fe.sys_scatter_vec(sys, f, l_load, phi.id, cell.id)
	}

	mms_l2_error :: proc(
		$AMB, $INT: int,
		mesh: fe.Mesh,
		geo: fe.Space_Vector,
		phi: fe.Space_Vector,
		exact: proc(x: fe.Small_Vec(AMB, f64)) -> f64,
	) -> f64 {
		PT_DIMS :: fe.Contraction_Dims {
			.CMPNTS = 1,
			.FIELDS = AMB,
		}
		VL_DIMS :: fe.Contraction_Dims {
			.CMPNTS = 1,
			.FIELDS = 1,
		}

		sum_sq: f64
		for cell in mesh.cells {
			fe.scratch_guard()
			context.allocator = fe.scratch()

			quad := fe.element_quad_rule(cell.type, .Q5)
			n_p := len(quad.ref_points)

			geo_basis := fe.bstore_interior(fe.space_bd(geo.space, cell.type), quad)
			geo_c := fe.space_gather(f64, geo, cell.id)
			tng := fe.geom_tng_from_nodes(AMB, INT, geo_basis[.S_Grd], geo_c, cell.affine)

			phys_points := fe.pvec_create(f64, n_p, PT_DIMS)
			fe.contract_eval(PT_DIMS, phys_points, geo_c, geo_basis[.S_Val])

			phi_basis := fe.bstore_interior(fe.space_bd(phi.space, cell.type), quad)
			phi_c := fe.space_gather(f64, phi, cell.id)
			computed := fe.pvec_create(f64, n_p, VL_DIMS)
			fe.contract_eval(VL_DIMS, computed, phi_c, phi_basis[.S_Val])

			for i in 0 ..< n_p {
				measure := fe.geom_measure(tng, i)
				pt := fe.small_vec_view_from_slice(fe.pvec_at_point(phys_points, i).data, AMB)^
				diff := computed.data[i] - exact(pt)
				sum_sq += diff * diff * measure * quad.weights[i]
			}
		}
		return math.sqrt(sum_sq)
	}

	run_mms_poisson :: proc(
		$AMB, $INT: int,
		t: ^testing.T,
		mesh_path: string,
		frame: fe.Small_Mat(3, AMB, f64),
		boundary: fe.Boundary_Set,
		order: fe.Order,
		diffusivity: f64,
		exact: proc(x: fe.Small_Vec(AMB, f64)) -> f64,
		forcing: proc(x: fe.Small_Vec(AMB, f64)) -> f64,
	) -> f64 {
		fe.scratch_guard()

		mesh := fio.load_mesh(mesh_path, .GMSH_V2_BINARY) or_else testing.fail_now(t)
		defer fe.mesh_destroy(&mesh)

		geo_space := fe.space_new_isoparemetric(mesh, AMB)
		geo_coeffs := fe.mesh_coord_coeffs(mesh, frame)
		geo := fe.Space_Vector{geo_space, geo_coeffs}

		phi_space := fe.space_new(mesh, {.Lagrange, order, .Continuous, fe.ALL_REGIONS}, 1)

		ms := fe.ms_create(mesh, {space = phi_space, constraints = {fe.Constraint_Essential{boundaries = boundary}}})
		defer fe.ms_destroy(&ms)

		sys := fe.sys_create(ms, {test = phi_space.id, trial = phi_space.id})
		defer fe.sys_destroy(&sys)

		state := fe.ms_state_alloc(ms)
		inhom := fe.ms_state_alloc(ms)
		f, soln := fe.ms_soln_vector(ms), fe.ms_soln_vector(ms)
		k := fe.sys_soln_matrix(sys)

		bc_vec := fe.ms_space_vec(ms, inhom, phi_space)
		for cell in mesh.cells {
			for bnd_facet in fe.cell_boundary_facet_set_of(mesh, cell, boundary) {
				restriction := fe.basis_facet_restriction(fe.space_bd(phi_space, cell.type), bnd_facet)
				ip := fe.interpolator(AMB, INT, geo, bc_vec, cell.type, cell.id, restriction)
				for jac, point, out in fe.interpolator_next(&ip) { out[0] = exact(point) }
				fe.interpolator_flush(&ip)
			}
		}
		fe.ms_enforce_constraints(ms, state, inhom)

		for cell in mesh.cells {
			fe.scratch_guard()
			context.allocator = fe.scratch()
			wf(AMB, INT, sys, geo, phi_space, k, f, inhom, cell, diffusivity, forcing)
		}

		prc := fe.amgcl_precond_create(k) or_else testing.fail_now(t)
		defer fe.amgcl_precond_destroy(prc)
		fe.amgcl_solve(prc, f, soln)

		fe.ms_apply_soln(ms, state, inhom, soln)

		return mms_l2_error(AMB, INT, mesh, geo, fe.ms_space_vec(ms, state, phi_space), exact)
	}

	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)
	fe.init_default_rank_ctx()

	DIFFUSIVITY :: 10.0

	exact_1d :: proc(x: fe.Small_Vec(1, f64)) -> f64 { return math.sin(math.PI * x.data[0]) }
	forcing_1d :: proc(x: fe.Small_Vec(1, f64)) -> f64 { return DIFFUSIVITY * 1 * math.PI * math.PI * exact_1d(x) }

	exact_2d :: proc(x: fe.Small_Vec(2, f64)) -> f64 { return(
			math.sin(math.PI * x.data[0]) *
			math.sin(math.PI * x.data[1]) \
		) }
	forcing_2d :: proc(x: fe.Small_Vec(2, f64)) -> f64 { return DIFFUSIVITY * 2 * math.PI * math.PI * exact_2d(x) }

	mms_linear_2d :: proc(x: fe.Small_Vec(2, f64)) -> f64 {
		return 1.0 + 2.0 * x.data[0] + 3.0 * x.data[1]
	}
	zero_forcing_2d :: proc(x: fe.Small_Vec(2, f64)) -> f64 { return 0 }

	exact_3d :: proc(x: fe.Small_Vec(3, f64)) -> f64 {
		return math.sin(math.PI * x.data[0]) * math.sin(math.PI * x.data[1]) * math.sin(math.PI * x.data[2])
	}
	forcing_3d :: proc(x: fe.Small_Vec(3, f64)) -> f64 { return DIFFUSIVITY * 3 * math.PI * math.PI * exact_3d(x) }

	err_2d := run_mms_poisson(
		2,
		2,
		t,
		"./validation/meshes/2d_square.msh",
		fe.XY_PLANE_FRAME,
		~{},
		.O2,
		DIFFUSIVITY,
		mms_linear_2d,
		zero_forcing_2d,
	)


	testing.expectf(t, err_2d < 1e-6, "2D MMS error too large: %v", err_2d)
}

mixed_poisson_wf :: proc(
	$AMB, $INT: int,
	sys: fe.Sys,
	geo: fe.Space_Vector,
	sigma: ^fe.Space,
	u: ^fe.Space,
	k: fe.Sparse_Matrix,
	f: fe.Vector,
	inhom: fe.State,
	cell: fe.Cell,
	diffusivity: f64,
	forcing: f64,
) {
	quad := fe.basis_infer_quad(fe.space_bd(sigma, cell.type))
	n_p := len(quad.ref_points)

	VEC_DIMS :: fe.Contraction_Dims{.CMPNTS = AMB, .FIELDS = 1} // sigma's value: vector via cmpnts
	SCL_DIMS :: fe.Contraction_Dims{.CMPNTS = 1, .FIELDS = 1}   // u
	DIV_DIMS :: fe.Contraction_Dims{.CMPNTS = 1, .FIELDS = 1}   // sigma's divergence: scalar per dof

	geo_basis := fe.bstore_interior(fe.space_bd(geo, cell.type), quad)
	geo_coeffs := fe.space_gather(f64, geo, cell.id)
	tng := fe.geom_tng_from_nodes(AMB, INT, geo_basis[.S_Grd], geo_coeffs, cell.affine)

	sigma_basis := fe.bstore_interior(fe.space_bd(sigma, cell.type), quad)
	u_basis := fe.bstore_interior(fe.space_bd(u, cell.type), quad)

	phys_sigma := fe.frame_push_bvec(sigma_basis[.V_Val], tng)
	ref_div := sigma_basis[.V_Div]

	l_mass := fe.space_cmat(f64, sigma, sigma, cell.type, cell.type)
	l_coupling := fe.space_cmat(f64, u, sigma, cell.type, cell.type)
	l_coupling_t := fe.space_cmat(f64, sigma, u, cell.type, cell.type)
	l_load := fe.space_cvec(f64, u, cell.type)

	inv_k := fe.pmat_create(f64, n_p, .DIAGONAL, VEC_DIMS, VEC_DIMS)
	one := fe.pmat_create(f64, n_p, .DIAGONAL, SCL_DIMS, SCL_DIMS)
	neg_one := fe.pmat_create(f64, n_p, .DIAGONAL, SCL_DIMS, SCL_DIMS)
	force_p := fe.pvec_create(f64, n_p, SCL_DIMS)



	for i in 0 ..< n_p {
		measure := fe.geom_measure(tng, i)
		for &e in fe.pmat_at_point(inv_k, i).data { e = (1.0 / diffusivity) / measure * quad.weights[i] }
		for &e in fe.pmat_at_point(one, i).data { e = quad.weights[i] }
		for &e in fe.pmat_at_point(neg_one, i).data { e = -quad.weights[i] }

		fe.pvec_at_point(force_p, i).data[0] = forcing * measure * quad.weights[i]
	}

	fe.contract_bilinear(VEC_DIMS, VEC_DIMS, l_mass, phys_sigma, inv_k, phys_sigma)
	fe.contract_bilinear(SCL_DIMS, DIV_DIMS, l_coupling, u_basis[.S_Val], one, ref_div)
	fe.contract_bilinear(DIV_DIMS, SCL_DIMS, l_coupling_t, ref_div, neg_one, u_basis[.S_Val])
	fe.contract_linear(SCL_DIMS, l_load, u_basis[.S_Val], force_p)

	fe.sys_scatter_mat(sys, k, f, inhom, .Linear, l_mass, sigma.id, sigma.id, cell.id)
	fe.sys_scatter_mat(sys, k, f, inhom, .Linear, l_coupling, u.id, sigma.id, cell.id)
	fe.sys_scatter_mat(sys, k, f, inhom, .Linear, l_coupling_t, sigma.id, u.id, cell.id)
	fe.sys_scatter_vec(sys, f, l_load, u.id, cell.id)
}

mixed_poisson_bc :: proc(
	$AMB, $INT: int,
	sys: fe.Sys,
	geo: fe.Space_Vector,
	sigma: ^fe.Space,
	f: fe.Vector,
	cell: fe.Cell,
	lf_idx: int,
	facet: fe.Facet,
	u_d: f64, // Dirichlet value on this facet - constant here; swap for proc(x)->f64 if spatially varying
) {
	VEC_DIMS :: fe.Contraction_Dims{.CMPNTS = AMB, .FIELDS = 1}
	fquad := fe.element_quad_rule(facet.info.type, .Q3)
	n_fp := len(fquad.ref_points)

	geo_fbasis := fe.bstore_facet(fe.space_bd(geo, cell.type), lf_idx, fquad)
	sigma_fbasis := fe.bstore_facet(fe.space_bd(sigma, cell.type), lf_idx, fquad)

	geo_coeffs := fe.space_gather(f64, geo, cell.id)

	ftng := fe.geom_tng_from_nodes(AMB, INT, geo_fbasis[.S_Grd], geo_coeffs, cell.affine)
	fctng := fe.geom_ctng(ftng)

	phys_sigma_f := fe.frame_push_bvec(sigma_fbasis[.V_Val], ftng)

	ref_n := fe.element_facet_ref_normal(f64, INT, cell.type, lf_idx)

	l_bc := fe.space_cvec(f64, sigma, cell.type)
	neg_ud_n := fe.pvec_create(f64, n_fp, VEC_DIMS)


	for i in 0 ..< n_fp {
		tng_i := fe.frame_at(ftng, i)
		ctng_i := fe.frame_at(fctng, i)

		normal := fe.geom_facet_normal(tng_i^, ctng_i^, ref_n)

		p := fe.pvec_at_point(neg_ud_n, i)
		for c in 0 ..< AMB { p.data[c] = -u_d * normal.data[c] * fquad.weights[i] / fe.small_mat_measure(tng_i^) }
	}

	fe.contract_linear(VEC_DIMS, l_bc, phys_sigma_f, neg_ud_n)

	fe.sys_scatter_vec(sys, f, l_bc, sigma.id, cell.id)
}

@(test)
mixed_poisson :: proc(t: ^testing.T) {
	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)
	fe.init_default_rank_ctx()

	DIFFUSIVITY :: 10.0
	FORCING :: -10
	U_D :: 5.0 // constant Dirichlet value on u, imposed weakly via the boundary term - direct write, no MMS

	mesh := fio.load_mesh("./validation/meshes/2d_channel.msh", .GMSH_V2_BINARY) or_else testing.fail_now(t)
	defer fe.mesh_destroy(&mesh)

	geo_space := fe.space_new_isoparemetric(mesh, 2)
	geo_coeffs := fe.mesh_coord_coeffs(mesh, fe.XY_PLANE_FRAME)
	geo := fe.Space_Vector{geo_space, geo_coeffs}

	sigma_space := fe.space_new(mesh, {.Raviart_Thomas, .O0, .Continuous, fe.ALL_REGIONS}, 1)
	u_space := fe.space_new(mesh, {.Lagrange, .O0, .Discontinuous, fe.ALL_REGIONS}, 1)


	all_boundaries := ~fe.Boundary_Set{} // every boundary Dirichlet, imposed weakly on sigma's equation

	// No essential constraints: u has none (L2 has no facet dofs to constrain), sigma has none here
	// either since every boundary is Dirichlet-on-u (natural), not flux-on-sigma (essential).
	ms := fe.ms_create(mesh, {space = sigma_space}, {space = u_space})
	defer fe.ms_destroy(&ms)

	sys := fe.sys_create(
		ms,
		{test = sigma_space.id, trial = sigma_space.id}, // mass block
		{test = u_space.id, trial = sigma_space.id},     // coupling
		{test = sigma_space.id, trial = u_space.id},     // coupling transpose
		{test = u_space.id, trial = u_space.id},
	)
	defer fe.sys_destroy(&sys)

	state := fe.ms_state_alloc(ms)
	inhom := fe.ms_state_alloc(ms) // unused (zeroed) - no essential constraints, nothing reads this
	f, soln := fe.ms_soln_vector(ms), fe.ms_soln_vector(ms)
	k := fe.sys_soln_matrix(sys)

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()

		geo_coeffs := fe.space_gather(f64, geo, cell.id)

		x_tot, y_tot: f64
		for d, i in geo_coeffs.data{
			if i % 2 == 0 {y_tot += d} else {x_tot += d}
		}
		x := x_tot / cast(f64)(len(geo_coeffs.data) / 2)
		y := y_tot / cast(f64)(len(geo_coeffs.data) / 2)

		mixed_poisson_wf(2, 2, sys, geo, sigma_space, u_space, k, f, inhom, cell, x + y, FORCING)

		for facet_id, lf_idx in cell.facets {
			facet := mesh.facets[facet_id]
			if facet.info.boundary not_in all_boundaries { continue }
			mixed_poisson_bc(2, 2, sys, geo, sigma_space, f, cell, lf_idx, facet, U_D)
		}
	}

	// prc := fe.amgcl_precond_create(k, fe.Precond_Params{kind = .ILU0}) or_else testing.fail_now(t)
	// defer fe.amgcl_precond_destroy(prc)
	// log.info(fe.amgcl_solve(prc, f, soln, fe.bicgstab_params(max_iters = 700)))
	pmask := fe.ms_schur_mask(ms, u_space.id)

	prc := fe.amgcl_precond_create(k, fe.schur_params(pmask)) or_else testing.fail_now(t)
	fe.amgcl_precond_print(prc)
	defer fe.amgcl_precond_destroy(prc)
	log.info(fe.amgcl_solve(prc, f, soln, fe.bicgstab_params(max_iters = 700)))


	fe.ms_apply_soln(ms, state, inhom, soln)

	// Output both fields.
	output_w, out_rules := fio.output_setup(mesh, geo_space, geo_coeffs, .O1, fio.VTU_Config{})
	defer fio.output_takedown(output_w)

	u_out := fio.output_field_create(mesh, "U", u_space.fields, out_rules)
	sigma_out := fio.output_field_create(mesh, "Sigma", 2, out_rules) // 2 = AMB, sigma is vector-valued via cmpnts

	u_vec := fe.ms_space_vec(ms, state, u_space)
	sigma_vec := fe.ms_space_vec(ms, state, sigma_space)

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()

		u_basis := fe.bstore_interior(fe.space_bd(u_space, cell.type), out_rules[cell.type])
		u_coeffs := fe.space_gather(f64, u_vec, cell.id, fe.scratch())
		u_result := fe.pvec_create(f64, len(out_rules[cell.type].ref_points), {.CMPNTS = 1, .FIELDS = 1})
		fe.contract_eval({.CMPNTS = 1, .FIELDS = 1}, u_result, u_coeffs, u_basis[.S_Val])
		copy(u_out.data[cell.id], u_result.data)

		geo_basis := fe.bstore_interior(fe.space_bd(geo_space, cell.type), out_rules[cell.type])
		geo_c := fe.space_gather(f64, geo, cell.id, fe.scratch())
		tng := fe.geom_tng_from_nodes(2, 2, geo_basis[.S_Grd], geo_c, cell.affine)

		sigma_basis := fe.bstore_interior(fe.space_bd(sigma_space, cell.type), out_rules[cell.type])
		sigma_coeffs := fe.space_gather(f64, sigma_vec, cell.id, fe.scratch())
		phys_sigma := fe.frame_push_bvec(sigma_basis[.V_Val], tng)
		fe.frame_scale_bvec_by_measure(phys_sigma, tng)
		sigma_result := fe.pvec_create(f64, len(out_rules[cell.type].ref_points), {.CMPNTS = 2, .FIELDS = 1})
		fe.contract_eval({.CMPNTS = 2, .FIELDS = 1}, sigma_result, sigma_coeffs, phys_sigma)
		copy(sigma_out.data[cell.id], sigma_result.data)
	}

	fio.output_write(output_w, {fields = {u_out, sigma_out}, path = "./validation/output/mixed_poisson"})
}
