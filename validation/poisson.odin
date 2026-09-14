package validation

import "core:log"
import "core:mem/virtual"
import "core:slice"
import "core:testing"

import fe "../fe"
import "../fe/fio"

/*
 Various forms of poisson type problems over different geometries.
*/

standard_poisson_wf :: proc(
	$AMB, $INT: int,
	ms: fe.Multi_Space,
	geo: fe.Space_Vector,
	phi: ^fe.Space,
	k: fe.Sparse_Matrix,
	f: fe.Vector,
	cell: fe.Cell,
) {
	quad := fe.element_quad_rule(cell.type, .Q3)
	n_p := len(quad.ref_points)

	GR_DIMS :: fe.Contraction_Dims {
		.CMPNTS = AMB,
		.FIELDS = 1,
	} // contracting against physical gradients so amb space.
	VL_DIMS :: fe.Contraction_Dims {
		.CMPNTS = 1,
		.FIELDS = 1,
	} // lagrange is scalar-valued space, only one field

	DIFFUSIVITY :: 10 // arbitrary material prop
	FORCE :: -4 //arbitrary forcing term

	geo_basis := fe.bstore_get_interior(fe.space_bd(geo, cell.type), quad)
	geo_coeffs := fe.space_gather(f64, geo, cell.id)
	jacobian := fe.frame_tangent_from_nodes(
		{.CMPNTS = INT, .FIELDS = AMB},
		geo_basis[.Scalar_Gradient],
		geo_coeffs,
		cell.affine,
	)
	j_inv_t := fe.frame_cotangent(jacobian)

	phi_basis := fe.bstore_get_interior(fe.space_bd(phi, cell.type), quad)

	// move grads to physical space, values are scalar so they dont transform between spaces.
	phys_grads := fe.frame_push_bvec(phi_basis[.Scalar_Gradient], j_inv_t)

	l_load := fe.space_cvec(f64, phi, cell.type)
	l_stiffness := fe.space_cmat(f64, phi, phi, cell.type, cell.type)

	// actual physics-tensors
	diffusivity := fe.pmat_create(f64, n_p, .DIAGONAL, GR_DIMS, GR_DIMS)
	force := fe.pvec_create(f64, n_p, VL_DIMS)

	// weight the physics tensors by integration measure
	for point, i in 0 ..< n_p {
		measure := fe.small_mat_measure(fe.frame_at(jacobian, point)^)

		d_p := fe.pmat_at_point(diffusivity, point)
		force_p := fe.pvec_at_point(force, point)

		for &e in d_p.data { e = DIFFUSIVITY * measure * quad.weights[i] }
		for &e in force_p.data { e = FORCE * measure * quad.weights[i] }
	}

	fe.contract_linear(VL_DIMS, l_load, phi_basis[.Scalar], force)
	fe.contract_bilinear(GR_DIMS, GR_DIMS, l_stiffness, phys_grads, diffusivity, phys_grads)

	fe.ms_scatter_mat(ms, k, f, .Linear, l_stiffness, phi.id, phi.id, cell.id)
	fe.ms_scatter_vec(ms, f, l_load, phi.id, cell.id)
}

@(test)
standard_poisson :: proc(t: ^testing.T) {
	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)

	// 2D
	{
		FIXED_VALUE :: 10

		mesh := fio.load_mesh("./validation/meshes/2d_channel.msh", .GMSH_V2_BINARY) or_else testing.fail_now(t)
		defer fe.mesh_destroy(&mesh) // internal arena, has to be cleaned up manually

		geo_space := fe.space_new_isoparemetric(mesh, 2)
		geo_coeffs := fe.mesh_coord_coeffs(mesh, fe.XY_PLANE_FRAME)

		phi := fe.space_new(mesh, {.Lagrange, .O1, .Continuous, fe.ALL_REGIONS}, 1)

		//output at whatever order phi is, doesnt have to be that, but matches viz order to soln order
		output_w, out_rules := fio.output_setup(mesh, geo_space, geo_coeffs, phi.order, fio.VTU_Config{})
		defer fio.output_takedown(output_w) //internal arena

		phi_out := fio.output_field_create(mesh, "Phi", phi.fields, out_rules)

		fixed := mesh.boundary_names["top"]
		periodic := fe.Constraint_Periodic{
			periodicity = fe.mesh_periodicity_from_names(mesh, "left", "right") or_else testing.fail_now(t)
		}

		ms := fe.ms_create(mesh, {space = phi, constraints = {fe.constraint_essential(fixed), periodic}} )
		defer fe.ms_destroy(&ms) // internal arena

		state := fe.ms_state(ms)
		f, soln, k := fe.ms_problem_data(ms)

		bc_vec := fe.ms_inhomogeneity(ms, phi.id)
		for cell in mesh.cells {
			for bnd_facet in fe.cell_boundary_facet_set_of(mesh, cell, {fixed}) {
				restriction := fe.basis_facet_restriction(fe.space_bd(phi, cell.type), bnd_facet)
				ip := fe.interpolator(2, 2, {geo_space, geo_coeffs}, {phi, bc_vec}, cell.type, cell.id, restriction)
				for jac, point, out in fe.interpolator_next(&ip) { out[0] = FIXED_VALUE * point.data.x }
				fe.interpolator_flush(&ip)
			}
		}

		fe.ms_enforce_constraints(ms, state) // state now is compatible with defined constraints and inhomogeneity

		for cell in mesh.cells{
			fe.scratch_guard()
			context.allocator = fe.scratch()
			standard_poisson_wf(2, 2, ms, {geo_space, geo_coeffs}, phi, k, f, cell)
		}

		prc := fe.amgcl_precond_create(k) or_else testing.fail_now(t)
		defer fe.amgcl_precond_destroy(prc) // allocated by 3rd party lib
		fe.amgcl_solve(prc, f, soln)

		fe.ms_apply_soln(ms, state, soln)

		for cell in mesh.cells{
			fe.scratch_guard()

			phi_basis := fe.bstore_get_interior(fe.space_bd(phi, cell.type), out_rules[cell.type])
			coeffs := fe.space_gather(f64, {phi, state[phi.id]}, cell.id, fe.scratch())
			result := fe.pvec_create(f64, len(out_rules[cell.type].ref_points), {.CMPNTS = 1, .FIELDS = 1})
			fe.contract_eval({.CMPNTS = 1, .FIELDS = 1}, result, coeffs, phi_basis[.Scalar])
			copy(phi_out.data[cell.id], result.data)
		}

		fio.output_write(output_w, {fields = {phi_out}, path = "./validation/output/2d_poisson"})
	}

}


@(test)
mixed_poisson :: proc(t: ^testing.T) {
}

@(test)
hdg_poisson :: proc(t: ^testing.T) {

}
