package validation

import "core:testing"

import "../fe"
import "../fe/fio"
import "core:mem/virtual"
import "core:log"


@(test)
elasticity_cantilever :: proc(t: ^testing.T) {
	wf :: proc(
		sys: fe.Sys,
		geo: fe.Space_Vector,
		u: ^fe.Space,
		k: fe.Sparse_Matrix,
		f: fe.Vector,
		inhom: fe.State,
		cell: fe.Cell,
		lambda, mu: f64,
		body_force: fe.Small_Vec(3, f64),
	) {
		DIM :: 3

		quad := fe.basis_infer_quad(fe.space_bd(u, cell.type))
		n_p := len(quad.ref_points)

		GR_DIMS :: fe.Contraction_Dims{.CMPNTS = DIM, .FIELDS = DIM} // displacement gradient: 3 spatial x 3 field
		VL_DIMS :: fe.Contraction_Dims{.CMPNTS = 1, .FIELDS = DIM}   // body force load: scalar basis val x 3 field

		geo_basis := fe.bstore_interior(fe.space_bd(geo, cell.type), quad)
		geo_coeffs := fe.space_gather(f64, geo, cell.id)
		tng := fe.geom_tng_from_nodes(DIM, DIM, geo_basis[.S_Grd], geo_coeffs, cell.affine)
		j_inv_t := fe.geom_ctng(tng)

		u_basis := fe.bstore_interior(fe.space_bd(u, cell.type), quad)
		phys_grads := fe.frame_push_bvec(u_basis[.S_Grd], j_inv_t)

		l_load := fe.space_cvec(f64, u, cell.type)
		l_stiffness := fe.space_cmat(f64, u, u, cell.type, cell.type)

		elast := fe.pmat_create_symmetric(f64, n_p, GR_DIMS)
		force_p := fe.pvec_create(f64, n_p, VL_DIMS)

		for i in 0 ..< n_p {
			measure := fe.geom_measure(tng, i)
			w := measure * quad.weights[i]

			pp := fe.pmat_at_point(elast, i)
			for rc in 0 ..< DIM {
				for cc in 0 ..= rc {
					block, status := fe.pmat_cmpnt_matrix(pp, rc, cc, DIM, DIM)
					assert(status == .FOUND)
					for rf in 0 ..< DIM {
						for cf in 0 ..< DIM {
							val: f64
							if rc == rf && cc == cf { val += lambda }
							if rf == cf && rc == cc { val += mu }
							if rc == cf && cc == rf { val += mu }
							block.data[cf][rf] = val * w // column-major: [col][row]
						}
					}
				}
			}

			fp := fe.pvec_at_point(force_p, i)
			for cf in 0 ..< DIM { fp.data[cf] = body_force.data[cf] * w }
		}

		fe.contract_linear(VL_DIMS, l_load, u_basis[.S_Val], force_p)
		fe.contract_bilinear(GR_DIMS, GR_DIMS, l_stiffness, phys_grads, elast, phys_grads)

		fe.sys_scatter_mat(sys, k, f, inhom, .Linear, l_stiffness, u.id, u.id, cell.id)
		fe.sys_scatter_vec(sys, f, l_load, u.id, cell.id)
	}

	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)
	fe.init_default_rank_ctx()

	// Material properties (E, nu) converted to Lamé parameters.
	E :: 200e9    // steel-ish, Pa
	NU :: 0.3
	LAMBDA :: E * NU / ((1 + NU) * (1 - 2 * NU))
	MU :: E / (2 * (1 + NU))

	BODY_FORCE := fe.Small_Vec(3, f64){data = {0, 0, -7800.0 * 9.81}}

	mesh := fio.load_mesh("./validation/meshes/3d_beam.msh", .GMSH_V2_BINARY) or_else testing.fail_now(t)
	defer fe.mesh_destroy(&mesh)

	geo_space := fe.space_new_isoparemetric(mesh, 3)
	geo_coeffs := fe.mesh_coord_coeffs(mesh, fe.MESH_FRAME)
	geo := fe.Space_Vector{geo_space, geo_coeffs}

	u_space := fe.space_new(mesh, {.Lagrange, .O1, .Continuous, fe.ALL_REGIONS}, 3)

	fixed := mesh.boundary_names["end_left"] or_else testing.fail_now(t)

	ms := fe.ms_create(mesh, {space = u_space, constraints = {fe.constraint_essential(fixed)}})
	defer fe.ms_destroy(&ms)

	sys := fe.sys_create(ms, {test = u_space.id, trial = u_space.id})
	defer fe.sys_destroy(&sys)

	state := fe.ms_state_alloc(ms)
	inhom := fe.ms_state_alloc(ms)
	f, soln := fe.ms_soln_vector(ms), fe.ms_soln_vector(ms)
	k := fe.sys_soln_matrix(sys)

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		wf(sys, geo, u_space, k, f, inhom, cell, LAMBDA, MU, BODY_FORCE)
	}

	prc := fe.amgcl_precond_create(k) or_else testing.fail_now(t)
	defer fe.amgcl_precond_destroy(prc)
	result := fe.amgcl_solve(prc, f, soln)

	log.info(result)

	fe.ms_apply_soln(ms, state, inhom, soln)

	output_w, out_rules := fio.output_setup(mesh, geo_space, geo_coeffs, .O1, fio.VTU_Config{})
	defer fio.output_takedown(output_w)

	u_out := fio.output_field_create(mesh, "Displacement", u_space.fields, out_rules)
	u_vec := fe.ms_space_vec(ms, state, u_space)

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()

		u_basis := fe.bstore_interior(fe.space_bd(u_space, cell.type), out_rules[cell.type])
		coeffs := fe.space_gather(f64, u_vec, cell.id, fe.scratch())
		result := fe.pvec_create(f64, len(out_rules[cell.type].ref_points), {.CMPNTS = 1, .FIELDS = 3})
		fe.contract_eval({.CMPNTS = 1, .FIELDS = 3}, result, coeffs, u_basis[.S_Val])
		copy(u_out.data[cell.id], result.data)
	}

	fio.output_write(output_w, {fields = {u_out}, path = "./validation/output/elasticity_cantilever"})
}
