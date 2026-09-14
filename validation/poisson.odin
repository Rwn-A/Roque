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
