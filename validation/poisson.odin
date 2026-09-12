package validation

import "core:log"
import "core:testing"
import "core:slice"

import fe "../fe"
import "../fe/fio"

/*
 Various forms of poisson type problems over different geometries.
*/

@(test)
simple_poisson_1D :: proc(t: ^testing.T) {
	context.allocator = fe.scratch()
	fe.scratch_guard()

	mesh := fe.segment_mesh(10, 0, 1)
	defer fe.mesh_destroy(&mesh) // internal arena

	fixed_end := mesh.boundary_names["left"]

	geo_space := fe.space_isoparemetric(mesh, 1)
	geo_coeffs := fe.mesh_coord_coeffs(mesh, fe.X_SEGMENT_FRAME)

	phi := fe.space_isoparemetric(mesh, 1)

	output_w, out_rules := fio.output_setup(mesh, geo_space, geo_coeffs, phi.order, fio.VTU_Config{})
	defer fio.output_takedown(output_w)

	phi_out := fio.output_field_create(mesh, "Phi", phi.fields, out_rules)


	ms := fe.ms_create(mesh, {space = phi, constraints = {fe.Constraint_Essential{boundaries = {fixed_end}}}})
	defer fe.ms_destroy(&ms) // internal arena

	state := fe.ms_state(ms)
	residual, update, tangent := fe.ms_problem_data(ms)

	bc_vec := fe.ms_inhomogeneity(ms, phi.id)
	for cell in mesh.cells {
		for bnd_facet in fe.cell_boundary_facet_set_of(mesh, cell, {fixed_end}) {
			restriction := fe.basis_facet_restriction(fe.space_bd(phi, cell.type), bnd_facet)
			ip := fe.interpolator(1, 1, {geo_space, geo_coeffs}, {phi, bc_vec}, cell.id, cell.type, restriction)
			for jac, point, out in fe.interpolator_next(&ip) { out[0] = 10 }
			fe.interpolator_flush(&ip)
		}
	}

	fe.ms_enforce_constraints(ms, state) // state now is compatible with defined constraints and inhomogeneity

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()

		GRAD_GRAD :: fe.Contraction_Dims{.CMPNTS = 1, .FIELDS = 1}
		VAL_VAL :: GRAD_GRAD // cause 1D
		DIFFUSIVITY :: 2

		quad := fe.element_quad_rule(cell.type, .Q3)
		n_p := len(quad.ref_points)

		geo_basis_grads := fe.bstore_get_interior(fe.space_bd(geo_space, cell.type), quad, .Scalar_Gradient)
		geo_coeffs := fe.space_gather(f64, {geo_space, geo_coeffs}, cell.id)
		jacobian := fe.frame_tangent_from_nodes(GRAD_GRAD, geo_basis_grads, geo_coeffs, cell.affine)
		j_inv_t := fe.frame_cotangent(jacobian)

		phi_basis_vals := fe.bstore_get_interior(fe.space_bd(phi, cell.type), quad, .Scalar)
		phi_basis_grads := fe.bstore_get_interior(fe.space_bd(phi, cell.type), quad, .Scalar_Gradient)
		phi_coeffs := fe.space_gather(f64, {phi, state[phi.id]}, cell.id)
		phi_grad_lcl := fe.pvec_create(f64, n_p, GRAD_GRAD)

		l_residual := fe.cvec_create(f64, phi_basis_grads.dofs, phi_coeffs.fields)
		l_stiffness := fe.cmat_create(f64, phi_basis_grads.dofs, phi_basis_grads.dofs, phi_coeffs.fields, phi_coeffs.fields)

		diffusivity := fe.pmat_create(f64, n_p, .DIAGONAL, GRAD_GRAD, GRAD_GRAD)
		resid_stiffness := fe.pvec_create(f64, n_p, GRAD_GRAD)
		load := fe.pvec_create(f64, n_p, VAL_VAL)

		phys_grads := fe.frame_push_bvec(phi_basis_grads, j_inv_t)

		fe.contract_eval(GRAD_GRAD, phi_grad_lcl, phi_coeffs, phys_grads)

		for point, i in 0..<n_p{
			measure := fe.small_mat_measure(fe.frame_at(jacobian, point)^)

			d_p := fe.pmat_at_point(diffusivity, point)
			phi_p := fe.pvec_at_point(phi_grad_lcl, point)
			r_p := fe.pvec_at_point(resid_stiffness, point)
			load_p := fe.pvec_at_point(load, point)

			for &e in d_p.data{ e = DIFFUSIVITY * measure * quad.weights[i] }
			for &e, i in r_p.data{ e = -DIFFUSIVITY * phi_p.data[i] * measure * quad.weights[i] }
			for &e in load_p.data {e = -4 * measure * quad.weights[i]}
		}



		fe.contract_linear(GRAD_GRAD, l_residual, phys_grads, resid_stiffness)
		fe.contract_linear(VAL_VAL, l_residual, phi_basis_vals, load)
		fe.contract_bilinear(GRAD_GRAD, GRAD_GRAD, l_stiffness, phys_grads, diffusivity, phys_grads)

		fe.ms_scatter_mat(ms, tangent, l_stiffness, phi.id, phi.id, cell.id)
		fe.ms_scatter_vec(ms, residual, l_residual, phi.id, cell.id)
	}

	prc := fe.amgcl_precond_create(tangent) or_else panic("zoinks")
	defer fe.amgcl_precond_destroy(prc) // allocated by 3rd party lib

	fe.amgcl_solve(prc, residual, update)

	fe.ms_apply_update(ms, state, update)

	for cell in mesh.cells{
		fe.scratch_guard()

		phi_basis_vals := fe.bstore_get_interior(fe.space_bd(phi, cell.type), out_rules[cell.type], .Scalar)
		coeffs := fe.space_gather(f64, {phi, state[phi.id]}, cell.id, fe.scratch())
		result := fe.pvec_create(f64, len(out_rules[cell.type].ref_points), {.CMPNTS = 1, .FIELDS = 1})
		fe.contract_eval({.CMPNTS = 1, .FIELDS = 1}, result, coeffs, phi_basis_vals)
		copy(phi_out.data[cell.id], result.data)
	}

	fio.output_write(output_w, {fields = {phi_out}, path = "./validation/output/1d_poisson"})

}

@(test)
mixed_poisson :: proc(t: ^testing.T) {
}

@(test)
hdg_poisson :: proc(t: ^testing.T) {

}
