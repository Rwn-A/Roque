package poisson

/*
 Steady heat conduction through a channel (5 x 0.5), the smallest complete Roque program.

     -div(K grad T) = 0,   T = 1 on the left wall,   T = 0 on the right wall,   top and bottom insulated.

 With an isotropic K the temperature would fall off linearly from left to right. Here K conducts ten times better
 along a direction tilted 30 degrees from the channel axis, so heat prefers to travel diagonally and the isotherms lean
 over instead of standing upright. No heat may cross the insulated walls, so there the flux K grad T runs along them,
 and with K tilted that puts the isotherms at a slant to the walls rather than square on.

 Continuous P2 Lagrange, the wall temperatures are essential constraints, one AMG preconditioned CG solve.

 Run from the repository root:  odin run demos/poisson -o:speed -out:build/poisson
 Open demos/output/poisson.vtu in ParaView.
*/

import "core:log"
import "core:math"

import fe "../../fe"
import "../../fe/fio"

MESH :: "./validation/meshes/2d_channel.msh"
OUTPUT :: "./demos/output/poisson"

T_HOT :: 1.0

// Conductivity along its principal axes, and the angle of the major axis from the channel axis (x).
K_MAJOR :: 10.0
K_MINOR :: 1.0
ANGLE :: math.PI / 6

main :: proc() {
	context.logger = log.create_console_logger(.Info)

	//== Mesh and geometry

	// The mesh is topology and node positions. The geometry turns the nodes into a continuous field, the map from
	// every reference cell to the physical one. XY_PLANE_FRAME keeps the x and y coordinates of the 2D mesh.
	mesh, ok := fio.load_mesh(MESH, .Gmsh_V2_Binary)
	if !ok { log.panic("could not load", MESH) }
	defer fe.mesh_destroy(&mesh)
	geo := fe.geo_create(&mesh, fe.XY_PLANE_FRAME)
	defer fe.geo_destroy(geo)

	// Boundary facets are tagged in Gmsh, look the walls up by name.
	hot := fe.mesh_tags_from_names(mesh, .D1, "left") or_else log.panic("no left tag")
	walls := fe.mesh_tags_from_names(mesh, .D1, "left", "right") or_else log.panic("no left / right tags")

	//== Space and system

	// One scalar field, continuous P2 Lagrange, over every cell of the mesh.
	T := fe.space_create(&mesh, {.Lagrange, .O2, .Continuous, fe.ALL_TAGS}, 1)
	defer fe.space_destroy(T)

	// The system numbers the space's dofs and applies its constraints: every dof on the two walls is fixed to its
	// value in `inhom`. Insulated walls need nothing, zero flux is the natural boundary condition.
	sys := fe.sys_create({space = T, constraints = {fe.Constraint_Essential{tags = walls}}})
	defer fe.sys_destroy(&sys)

	// state holds every dof, inhom the constrained values, rhs and x are sized for the free dofs that get solved.
	state, inhom := fe.sys_state(sys), fe.sys_state(sys)
	rhs, x := fe.sys_vector(sys), fe.sys_vector(sys)
	K := fe.sys_matrix(sys)

	// The hot wall's temperature into inhom. The right wall stays at the zero inhom starts with.
	fe.interpolate_facets(2, 2, 1, geo, fe.sys_space_coeffs(sys, inhom, T), hot, hot_wall)

	//== Assembly

	for cell in mesh.cells {
		// Everything allocated for one cell goes in the scratch arena and is freed at the end of the iteration.
		fe.scratch_guard()
		context.allocator = fe.scratch()

		// (K grad T, grad v) on the cell, see conduction.
		site := fe.cell_site(cell)
		local := conduction(geo, T, site)

		// Adding with rhs and inhom moves the known wall temperatures' columns to the right hand side.
		fe.sys_scatter_mat(sys, K, local, T, T, site, rhs, inhom)
	}
	A := fe.sys_matrix_finalize(&K)

	//== Solve

	precond := fe.amgcl_precond_create(fe.bsp_to_sp(A), fe.amg_params()) or_else log.panic(string(fe.amgcl_last_error()))
	defer fe.amgcl_precond_destroy(precond)
	res := fe.amgcl_solve(precond, rhs, x, fe.cg_params(tol = 1e-10))
	if res.status != .Converged { log.panicf("solve: %v", res) }
	log.infof("solved %v dofs in %v CG iterations", sys.n_soln, res.iters)

	// The solved free dofs into state, the constrained ones from inhom.
	fe.sys_apply_soln(sys, state, inhom, x)

	//== Output

	// Evaluates T at the P2 nodes of every cell and writes a .vtu for ParaView.
	writer := fio.output_create(mesh, geo, .O2, fio.VTU_Config{})
	defer fio.output_destroy(writer)
	T_out := fio.output_field_from_space(2, 2, 1, writer, geo, fe.sys_space_coeffs(sys, state, T), .S_Val, "T")
	if !fio.output_write(writer, {path = OUTPUT, fields = {T_out}}) { log.panic("could not write", OUTPUT) }
	log.infof("wrote %v.vtu", OUTPUT)
}

// (K grad T, grad v), the conduction stiffness of one cell.
conduction :: proc(geo: fe.Geometry, T: ^fe.Space, site: fe.Site) -> fe.Cmat(f64) {
	// A quadrature rule exact for the cell's P2 products, and the geometry map at its points.
	rule, w := fe.space_quad_rule(T, site)
	m := fe.geo_map(2, 2, geo, site, rule)

	// Physical gradients of the basis functions at every quadrature point.
	grad := fe.basis_tab(fe.space_local_basis(T, site), rule, m, fe.Quantity_Set{.S_Grd})[.S_Grd]

	// Quadrature weights times the map's area scaling: dx at every point.
	dx := fe.element_map_weights(m, w)

	// The conductivity tensor at every point, weighted by dx. K = R diag(K_MAJOR, K_MINOR) R^T for the rotation R by
	// ANGLE. It is symmetric, so only the lower triangle of its 2 x 2 component blocks is stored.
	c, s := math.cos(ANGLE), math.sin(ANGLE)
	k := [2][2]f64 {
		{K_MAJOR * c * c + K_MINOR * s * s, (K_MAJOR - K_MINOR) * c * s},
		{(K_MAJOR - K_MINOR) * c * s, K_MAJOR * s * s + K_MINOR * c * c},
	}
	k_dx := fe.pmat_create_symmetric(f64, len(dx), 2, 1)
	for d, p in dx {
		pp := fe.pmat_at_point(k_dx, p)
		for i in 0 ..< 2 {
			for j in 0 ..= i {
				blk, _ := fe.pmat_cmpnt_matrix(pp, i, j, 1, 1) // one field, so each block is a single value
				blk.data[0][0] = k[i][j] * d
			}
		}
	}

	// local[i, j] = sum over points of grad phi_i . K grad phi_j dx.
	local := fe.cmat_create_for(grad, k_dx, grad)
	fe.contract_bilinear_same(2, 1, local, grad, k_dx)
	return local
}

// The hot wall's temperature at the points `x`.
hot_wall :: proc(x, out: fe.Pvec(f64), _: rawptr) {
	for p in 0 ..< out.points { fe.pvec_at_point(out, p).data[0] = T_HOT }
}
