/*
 Curl-curl + mass, the smallest end-to-end check of Nedelec (H(curl)) elements.

     Find E in H(curl):  int curl E curl v  +  int E . v  =  int g . v   for all v

 With g curl-free, the natural boundary condition (curl E = 0) holds and E = g exactly. g = (x, y) lies in
 Nedelec O1 and up, so the error should be at round-off, not just small. For O0 use a constant g, e.g. (1, 2).

 Exercised: edge orientation (tangential signs/permutations), the covariant map for values and the density map
 for the 2D scalar curl. Symmetric positive definite, no constraints, no boundary terms.
*/
package validation

import "core:testing"
import "core:mem/virtual"
import "core:log"

import "../fe"
import "../fe/fio"
import "core:math"

// g = (x, y)
exact_field :: proc(x: []f64) -> [2]f64 {
	return {x[0], x[1]}
}

assemble_curl_cell :: proc(
	$A, $I: int,
	sys: fe.Sys,
	geo: fe.Space_Vector,
	e_space: ^fe.Space,
	K: fe.Sparse_Matrix,
	rhs: fe.Vector,
	inhom: fe.State,
	cell: ^fe.Cell,
) {
	bd := fe.space_bd(e_space, cell.type)
	quad, weights := fe.basis_quad_rule(bd, extra = 1) // + 1: g is linear
	n_points := len(quad.points)

	// Geometry: tangent and physical position at each quadrature point.
	geo_basis := fe.bstore_interior(fe.space_bd(geo, cell.type), quad)
	nodes := fe.space_gather(f64, geo, cell.id)
	tangent := fe.tangent_from_nodes(A, I, geo_basis[.S_Grd], nodes, cell.affine)
	x := fe.pvec_create(f64, n_points, 1, A)
	fe.contract_eval(1, A, x, nodes, geo_basis[.S_Val])

	// Nedelec: oriented, values through the covariant map (J^-T), the 2D scalar curl through the density map (1/det J).
	e_ref := fe.bstore_interior(bd, quad)
	keys := fe.cell_entity_keys(cell)
	e_val := fe.basis_orient_push(bd, e_ref[.V_Val], keys, fe.piola_covariant(tangent))
	e_curl := fe.basis_orient_push(bd, e_ref[.V_Curl], keys, fe.piola_density(tangent))

	vec_dx := fe.pmat_create(f64, n_points, .DIAGONAL, A, 1, A, 1) // dx for E . v
	scl_dx := fe.pmat_create(f64, n_points, .DIAGONAL, 1, 1, 1, 1) // dx for curl E curl v
	g_dx := fe.pvec_create(f64, n_points, A, 1) // g dx

	for p in 0 ..< n_points {
		dx := fe.tangent_measure(tangent, p) * weights[p]
		for &e in fe.pmat_at_point(vec_dx, p).data { e = dx }
		for &e in fe.pmat_at_point(scl_dx, p).data { e = dx }
		g := exact_field(fe.pvec_at_point(x, p).data)
		out := fe.pvec_at_point(g_dx, p).data
		out[0], out[1] = g.x * dx, g.y * dx
	}

	mass := fe.cmat_create_for(e_val, vec_dx, e_val) //  int E . v
	curl_curl := fe.cmat_create_for(e_curl, scl_dx, e_curl) //  int curl E curl v
	load := fe.cvec_create_for(e_val, g_dx) //  int g . v

	fe.contract_bilinear(A, 1, A, 1, mass, e_val, vec_dx, e_val)
	fe.contract_bilinear(1, 1, 1, 1, curl_curl, e_curl, scl_dx, e_curl)
	fe.contract_linear(A, 1, load, e_val, g_dx)

	fe.sys_scatter_mat(sys, K, rhs, inhom, .Linear, mass, e_space, e_space, cell.id)
	fe.sys_scatter_mat(sys, K, rhs, inhom, .Linear, curl_curl, e_space, e_space, cell.id)
	fe.sys_scatter_vec(sys, rhs, load, e_space, cell.id)
}

//@(test)
curl_curl :: proc(t: ^testing.T) {
	arena := virtual.Arena{}
	virtual.arena_init_growing(&arena) or_else testing.fail_now(t, "unable to create arena")
	context.allocator = virtual.arena_allocator(&arena)
	defer virtual.arena_destroy(&arena)

	//== Mesh and geometry

	mesh := fio.load_mesh("./validation/meshes/2d_channel.msh", .GMSH_V2_BINARY) or_else testing.fail_now(t)
	defer fe.mesh_destroy(&mesh)

	geo_space := fe.space_new_isoparemetric(&mesh, 2)
	geo := fe.Space_Vector{geo_space, fe.mesh_coord_coeffs(mesh, fe.XY_PLANE_FRAME)}

	//== Space and system: one Nedelec field, no constraints

	e_space := fe.space_new(&mesh, {.Nedelec, .O1, .Continuous, fe.ALL_REGIONS}, 1)

	ms := fe.ms_create(.Eliminate, {space = e_space})
	defer fe.ms_destroy(&ms)

	sys := fe.sys_create(ms, {test = e_space, trial = e_space})
	defer fe.sys_destroy(&sys)

	state := fe.ms_state_alloc(ms)
	inhom := fe.ms_state_alloc(ms)
	rhs, soln := fe.ms_soln_vector(ms), fe.ms_soln_vector(ms)
	K := fe.sys_soln_matrix(sys)

	//== Assemble and solve

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		assemble_curl_cell(2, 2, sys, geo, e_space, K, rhs, inhom, &cell)
	}

	precond := fe.amgcl_precond_create(K, fe.Precond_Params{kind = .ILU0}) or_else testing.fail_now(t)
	defer fe.amgcl_precond_destroy(precond)
	log.info(fe.amgcl_solve(precond, rhs, soln, fe.cg_params(max_iters = 700, tolerance = 1e-3)))

	fe.ms_apply_soln(ms, state, inhom, soln)

	//== Check E against g and write both

	writer, out_rules := fio.output_setup(mesh, geo_space, geo.coeffs, .O1, fio.VTU_Config{})
	defer fio.output_takedown(writer)

	e_out := fio.output_field_create(mesh, "E", 2, out_rules)
	err_out := fio.output_field_create(mesh, "Error", 1, out_rules) // |E - g|

	e_vec := fe.ms_space_vec(ms, state, e_space)
	max_err: f64

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		rule := out_rules[cell.type]
		n_points := len(rule.points)

		geo_basis := fe.bstore_interior(fe.space_bd(geo_space, cell.type), rule)
		nodes := fe.space_gather(f64, geo, cell.id)
		tangent := fe.tangent_from_nodes(2, 2, geo_basis[.S_Grd], nodes, cell.affine)
		x := fe.pvec_create(f64, n_points, 1, 2)
		fe.contract_eval(1, 2, x, nodes, geo_basis[.S_Val])

		bd := fe.space_bd(e_space, cell.type)
		e_ref := fe.bstore_interior(bd, rule)
		e_val := fe.basis_orient_push(bd, e_ref[.V_Val], fe.cell_entity_keys(&cell), fe.piola_covariant(tangent))

		e_at := fe.pvec_create(f64, n_points, 2, 1)
		fe.contract_eval(2, 1, e_at, fe.space_gather(f64, e_vec, cell.id), e_val)
		copy(e_out.data[cell.id], e_at.data)

		for p in 0 ..< n_points {
			e := fe.pvec_at_point(e_at, p).data
			g := exact_field(fe.pvec_at_point(x, p).data)
			err := math.sqrt((e[0] - g.x) * (e[0] - g.x) + (e[1] - g.y) * (e[1] - g.y))
			err_out.data[cell.id][p] = err
			max_err = max(max_err, err)
		}
	}

	log.info("max |E - g| =", max_err)
	testing.expect(t, max_err < 1e-8, "Nedelec O1 should reproduce a linear field exactly")

	fio.output_write(writer, {fields = {e_out, err_out}, path = "./validation/output/curl_curl"})
}
