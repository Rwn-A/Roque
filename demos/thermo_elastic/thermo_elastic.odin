package thermo_elastic

/*
 One-way thermoelasticity on a 3D beam (1 x 0.05 x 0.05, along x).

 Heat: transient conduction, rho_c dT/dt = k lap T, with the top face (+y) pulsing between 0 and T_HOT every PERIOD
 and the bottom face held at 0, stepped with BDF2.

 Elasticity: at every output time the displacement is solved quasi-statically for the thermal strain alpha T I, the
 left end clamped. The hot top expands, bending the beam down (tip ~ -0.01 in y) as the heat soaks in, and it
 straightens as the top cools. Diffusion through the thickness is much faster than PERIOD, so the temperature follows
 the top almost quasi-statically.

 Both are continuous P1. Heat eliminates its constraints and rebuilds its AMG preconditioner only when the BDF shift
 changes. Elasticity keeps its constraints as identity rows, so AMG can take node blocks of 3 and the rigid body modes,
 and builds its stiffness and preconditioner once: only the thermal load changes.

 The time loop runs on every core (widen), assembling colour by colour. Solves and output narrow to one rank.

 Run from the repository root:  odin run demos/thermo_elastic -o:speed -out:build/thermo_elastic
 With a threaded amgcl (./build_amgcl.sh --openmp) add -define:AMGCL_OPENMP=true and run with OMP_WAIT_POLICY=passive,
 otherwise OpenMP's idle threads spin against the widened ranks.
 Open demos/output/thermo_elastic.pvd in ParaView.
*/

import "core:log"
import "core:math"
import "core:os"

import fe "../../fe"
import "../../fe/fio"

MESH :: "./validation/meshes/3d_beam.msh"
OUTPUT :: "./demos/output/thermo_elastic"

// Heat. The time scale through the thickness, thickness^2 rho_c / k, is 1.
RHO_C :: 400.0
K_T :: 1.0
T_HOT :: 100.0
PERIOD :: 3.0

// Elasticity. Displacements scale with ALPHA, not with E.
E :: 1.0
NU :: 0.3
ALPHA :: 1e-5
LAMBDA :: E * NU / ((1 + NU) * (1 - 2 * NU))
MU :: E / (2 * (1 + NU))

DT :: 0.03
T_END :: 2 * PERIOD
OUT_DT :: 0.1

// Everything the time loop shares between ranks.
Demo :: struct {
	mesh:                 ^fe.Mesh,
	geo:                  fe.Geometry,
	cells:                fe.Colouring, // no two cells of a colour share a dof, so ranks assemble a colour together
	top:                  fe.Tag_Set,
	T, u:                 ^fe.Space,
	heat, elastic:        fe.Sys,
	T_state, T_inhom:     fe.State,
	u_state, u_inhom:     fe.State,
	T_rhs, T_x:           fe.Vector,
	u_rhs, u_x:           fe.Vector,
	T_K:                  fe.System_Matrix, // reassembled every step
	u_A:                  fe.Block_Sparse, // assembled once
	integ:                fe.Integrator,
	T_precond, u_precond: ^fe.Precond,
	writer:               ^fio.Output_Writer,
	T_out, u_out:         fio.Output_Field,

	// written by rank 0 only
	shift:                f64, // BDF shift the heat preconditioner was built for
	heat_its, u_its:      int,
}

main :: proc() {
	context.logger = log.create_console_logger(.Info)

	mesh, ok := fio.load_mesh(MESH, .Gmsh_V2_Binary)
	if !ok { log.panic("could not load", MESH) }
	defer fe.mesh_destroy(&mesh)

	d := Demo{mesh = &mesh}
	d.geo = fe.geo_create(&mesh, fe.MESH_FRAME)
	defer fe.geo_destroy(d.geo)
	d.cells = fe.mesh_colour_cells(mesh)

	d.top = fe.mesh_tags_from_names(mesh, .D2, "top") or_else log.panic("no top tag")
	cooled := fe.mesh_tags_from_names(mesh, .D2, "top", "bottom") or_else log.panic("no top / bottom tags")
	clamped := fe.mesh_tags_from_names(mesh, .D2, "end_left") or_else log.panic("no end_left tag")

	//== Heat

	// Both faces are constrained, the top follows the pulse through T_inhom, the bottom stays at its zero.
	d.T = fe.space_create(&mesh, {.Lagrange, .O1, .Continuous, fe.ALL_TAGS}, 1)
	defer fe.space_destroy(d.T)

	d.heat = fe.sys_create({space = d.T, constraints = {fe.Constraint_Essential{tags = cooled}}})
	defer fe.sys_destroy(&d.heat)

	d.T_state, d.T_inhom = fe.sys_state(d.heat), fe.sys_state(d.heat)
	d.T_rhs, d.T_x = fe.sys_vector(d.heat), fe.sys_vector(d.heat)
	d.T_K = fe.sys_matrix(d.heat)

	// The integrator keeps T's history, the beam starts at T = 0 everywhere.
	d.integ = fe.integ_create(.BDF2, d.T_state, 0)
	defer fe.integ_destroy(&d.integ)
	defer if d.T_precond != nil { fe.amgcl_precond_destroy(d.T_precond) }

	//== Elasticity

	// Clamped at zero displacement, u_inhom stays zero.
	d.u = fe.space_create(&mesh, {.Lagrange, .O1, .Continuous, fe.ALL_TAGS}, 3)
	defer fe.space_destroy(d.u)
	d.elastic = fe.sys_create({space = d.u, constraints = {fe.Constraint_Essential{tags = clamped}}}, mode = .Identity)
	defer fe.sys_destroy(&d.elastic)
	d.u_state, d.u_inhom = fe.sys_state(d.elastic), fe.sys_state(d.elastic)
	d.u_rhs, d.u_x = fe.sys_vector(d.elastic), fe.sys_vector(d.elastic)

	// The stiffness never changes: assemble it once, before the time loop.
	u_K := fe.sys_matrix(d.elastic)
	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		site := fe.cell_site(cell)
		fe.sys_scatter_mat(d.elastic, u_K, elastic_stiffness(d.geo, d.u, site), d.u, d.u, site)
	}
	d.u_A = fe.sys_matrix_finalize(&u_K)
	fe.sys_constrain_matrix(d.elastic, d.u_A)

	// AMG on node blocks of 3 with the six rigid body modes as the near null space. The default strength threshold
	// drops the weak couplings a thin beam bends through, so keep every coupling when aggregating.
	modes := fe.sys_rigid_body_modes(d.elastic, d.geo, d.u)
	nns, cols := fe.sys_near_null_space(d.elastic, ..modes)
	u_params := fe.amg_params(block_size = 3, nullspace = nns, nullspace_cols = cols)
	u_params.amg.coarsening.aggr.eps_strong = 0
	d.u_precond = fe.amgcl_precond_create(fe.bsp_to_sp(d.u_A), u_params) or_else log.panic(string(fe.amgcl_last_error()))
	defer fe.amgcl_precond_destroy(d.u_precond)

	//== Output

	d.writer = fio.output_create(mesh, d.geo, .O2, fio.VTU_Config{pvd_path = OUTPUT + ".pvd"})
	defer fio.output_destroy(d.writer)
	d.T_out = fio.output_field_create(d.writer, "T", 1)
	d.u_out = fio.output_field_create(d.writer, "U", 3)
	write(&d, 0, 0)

	fe.widen(time_loop, &d, os.get_processor_core_count())
}

// Run by every rank. Solves, output and anything else serial go through narrow.
time_loop :: proc(d: ^Demo) {
	clock := fe.clock_create(0, T_END, DT, out_dt = OUT_DT) // plain data, every rank steps its own
	for dt in fe.clock_next(&clock) {
		// 1. BDF coefficients and history for the step to the new time
		step := fe.integ_step(&d.integ, dt)

		// 2. boundary values at the new time: the top pulses, the bottom stays 0
		t := step.t
		fe.interpolate_facets(3, 3, 1, d.geo, fe.sys_space_coeffs(d.heat, d.T_inhom, d.T), d.top, top_temperature, &t, d.cells)

		// 3. assemble shift rho_c (T - hist, v) + k (grad T, grad v) = 0, lifting the new boundary values into rhs
		fe.vec_zero_rank(d.T_rhs)
		fe.sys_matrix_zero(d.T_K)
		hist := fe.sys_space_coeffs(d.heat, step.hist, d.T)
		for it := fe.colour_iterator(d.cells); id in fe.colour_iterator_next(&it) {
			fe.scratch_guard()
			context.allocator = fe.scratch()
			site := fe.cell_site(d.mesh.cells[id])
			local, load := heat_cell(d.geo, d.T, hist, site, step.shift)
			fe.sys_scatter_mat(d.heat, d.T_K, local, d.T, d.T, site, d.T_rhs, d.T_inhom)
			fe.sys_scatter_vec(d.heat, d.T_rhs, load, d.T, site)
		}
		T_A := fe.sys_matrix_finalize(&d.T_K)

		// 4. solve on one rank, the matrix only changes with the shift: BDF1 on the first step, BDF2 after
		if fe.narrow() {
			if step.shift != d.shift {
				if d.T_precond != nil { fe.amgcl_precond_destroy(d.T_precond) }
				d.T_precond = fe.amgcl_precond_create(fe.bsp_to_sp(T_A), fe.amg_params()) or_else log.panic(string(fe.amgcl_last_error()))
				d.shift = step.shift
			}
			res := fe.amgcl_solve(d.T_precond, d.T_rhs, d.T_x, fe.cg_params(tol = 1e-8, maxiter = 500))
			if res.status != .Converged { log.panicf("heat solve at t = %v: %v", step.t, res) }
			d.heat_its = res.iters
		}
		fe.sys_apply_soln(d.heat, d.T_state, d.T_inhom, d.T_x)

		// 5. the solved T becomes history
		fe.integ_accept(&d.integ, step, d.T_state)

		index := fe.clock_output_due(&clock) or_continue

		// Quasi-static elasticity for the current temperature, only the thermal load is reassembled.
		fe.vec_zero_rank(d.u_rhs)
		T_now := fe.sys_space_coeffs(d.heat, d.T_state, d.T)
		for it := fe.colour_iterator(d.cells); id in fe.colour_iterator_next(&it) {
			fe.scratch_guard()
			context.allocator = fe.scratch()
			site := fe.cell_site(d.mesh.cells[id])
			fe.sys_scatter_vec(d.elastic, d.u_rhs, thermal_load(d.geo, d.u, T_now, site), d.u, site)
		}
		fe.sys_constrain_rhs(d.elastic, d.u_rhs, d.u_inhom)
		if fe.narrow() {
			res := fe.amgcl_solve(d.u_precond, d.u_rhs, d.u_x, fe.cg_params(tol = 1e-8, maxiter = 500))
			if res.status != .Converged { log.panicf("elastic solve at t = %v: %v", step.t, res) }
			d.u_its = res.iters
		}
		fe.sys_apply_soln(d.elastic, d.u_state, d.u_inhom, d.u_x)

		if fe.narrow() {
			log.infof("t = %.2f  heat %v its, elastic %v its", step.t, d.heat_its, d.u_its)
			write(d, step.t, index)
		}
	}
}

// Writes T and U at time t as output `index` of the time series.
write :: proc(d: ^Demo, t: f64, index: int) {
	fio.output_field_fill_space(3, 3, 1, d.writer, d.geo, fe.sys_space_coeffs(d.heat, d.T_state, d.T), .S_Val, d.T_out)
	fio.output_field_fill_space(3, 3, 3, d.writer, d.geo, fe.sys_space_coeffs(d.elastic, d.u_state, d.u), .S_Val, d.u_out)
	fio.output_write(d.writer, {path = OUTPUT, fields = {d.T_out, d.u_out}, step = index, time = t})
}

// shift rho_c (T, v) + k (grad T, grad v), and the history's part of the time derivative, shift rho_c (hist, v).
heat_cell :: proc(
	geo: fe.Geometry,
	T: ^fe.Space,
	hist: fe.Space_Coeffs,
	site: fe.Site,
	shift: f64,
) -> (
	local: fe.Cmat(f64),
	load: fe.Cvec(f64),
) {
	rule, w := fe.space_quad_rule(T, site)
	m := fe.geo_map(3, 3, geo, site, rule)
	tab := fe.basis_tab(fe.space_local_basis(T, site), rule, m)
	phi, grad := tab[.S_Val], tab[.S_Grd]
	dx := fe.element_map_weights(m, w)

	mass := fe.pmat_create_identity(dx, 1, 1, shift * RHO_C)
	conduction := fe.pmat_create_identity(dx, 3, 1, K_T)

	h := fe.space_eval(1, hist, site, rule, m, .S_Val)
	h_dx := fe.pvec_create(f64, len(dx), 1, 1)
	for d, p in dx { fe.pvec_at_point(h_dx, p).data[0] = shift * RHO_C * fe.pvec_at_point(h, p).data[0] * d }

	local = fe.cmat_create_for(phi, mass, phi)
	load = fe.cvec_create_for(phi, h_dx)
	fe.contract_bilinear_same(1, 1, local, phi, mass)
	fe.contract_bilinear_same(3, 1, local, grad, conduction)
	fe.contract_linear(1, 1, load, phi, h_dx)
	return
}

// (grad v : C : grad u), C isotropic: C_ijkl = lambda d_ij d_kl + mu (d_ik d_jl + d_il d_jk).
elastic_stiffness :: proc(geo: fe.Geometry, u: ^fe.Space, site: fe.Site) -> fe.Cmat(f64) {
	rule, w := fe.space_quad_rule(u, site)
	m := fe.geo_map(3, 3, geo, site, rule)
	grad := fe.basis_tab(fe.space_local_basis(u, site), rule, m, fe.Quantity_Set{.S_Grd})[.S_Grd]
	dx := fe.element_map_weights(m, w)

	// Block (j, l) over derivative components holds C_ijkl over (field i of v, field k of u). C is symmetric in the
	// block pairs, so only the lower triangle is stored.
	D := fe.pmat_create_symmetric(f64, len(dx), 3, 3)
	for d, p in dx {
		pp := fe.pmat_at_point(D, p)
		for j in 0 ..< 3 {
			for l in 0 ..= j {
				blk, _ := fe.pmat_cmpnt_matrix(pp, j, l, 3, 3)
				for i in 0 ..< 3 {
					for k in 0 ..< 3 {
						c := MU * (delta(i, k) * delta(j, l) + delta(i, l) * delta(j, k)) + LAMBDA * delta(i, j) * delta(k, l)
						blk.data[k][i] = c * d
					}
				}
			}
		}
	}
	local := fe.cmat_create_for(grad, D, grad)
	fe.contract_bilinear_same(3, 3, local, grad, D)
	return local

	delta :: proc(a, b: int) -> f64 { return 1 if a == b else 0 }
}

// (sigma_th : grad v), the thermal stress sigma_th = (3 lambda + 2 mu) alpha T I.
thermal_load :: proc(geo: fe.Geometry, u: ^fe.Space, T: fe.Space_Coeffs, site: fe.Site) -> fe.Cvec(f64) {
	rule, w := fe.space_quad_rule(u, site)
	m := fe.geo_map(3, 3, geo, site, rule)
	grad := fe.basis_tab(fe.space_local_basis(u, site), rule, m, fe.Quantity_Set{.S_Grd})[.S_Grd]
	dx := fe.element_map_weights(m, w)
	temp := fe.space_eval(1, T, site, rule, m, .S_Val)

	// derivative component j against field i, only the diagonal i == j is non-zero
	sigma_dx := fe.pvec_create(f64, len(dx), 3, 3)
	for d, p in dx {
		s := fe.pvec_point_matrix(sigma_dx, p, 3, 3)
		for i in 0 ..< 3 { s.data[i][i] = (3 * LAMBDA + 2 * MU) * ALPHA * fe.pvec_at_point(temp, p).data[0] * d }
	}
	load := fe.cvec_create_for(grad, sigma_dx)
	fe.contract_linear(3, 3, load, grad, sigma_dx)
	return load
}

// Top face temperature at the time `data` points to, a smooth pulse from 0 up to T_HOT and back every PERIOD.
top_temperature :: proc(x, out: fe.Pvec(f64), data: rawptr) {
	t := (cast(^f64)data)^
	for p in 0 ..< out.points { fe.pvec_at_point(out, p).data[0] = T_HOT * 0.5 * (1 - math.cos(2 * math.PI * t / PERIOD)) }
}
