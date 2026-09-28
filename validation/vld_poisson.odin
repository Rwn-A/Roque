package validation

/*
 Poisson, -lap u = f on the unit square with u = cos(pi x) sin(pi y) on the whole boundary, in four formulations at
 polynomial degree p = 1, 2, 3. Each logs the L2 errors of u and of its flux -grad u on meshes of 4 to 32 divisions,
 with the rates between them, and expects the asymptotic rate:
*/

import "core:math"
import "core:testing"

import fe "../fe"

@(test)
poisson_cg :: proc(t: ^testing.T) {
	convergence(t, "CG", solve_cg, u_rate = 1, flux_rate = 0)
}

@(test)
poisson_dg :: proc(t: ^testing.T) {
	convergence(t, "DG", solve_dg, u_rate = 1, flux_rate = 0)
}

@(test)
poisson_hdg :: proc(t: ^testing.T) {
	convergence(t, "HDG", solve_hdg, u_rate = 1, flux_rate = 1)
}

@(test)
poisson_mixed :: proc(t: ^testing.T) {
	convergence(t, "mixed", solve_mixed, u_rate = 0, flux_rate = 0)
}

//== Exact solution

@(private = "file")
exact_u :: proc(x, out: fe.Pvec(f64), _: rawptr) {
	for p in 0 ..< x.points {
		xp := fe.pvec_at_point(x, p).data
		fe.pvec_at_point(out, p).data[0] = math.cos(math.PI * xp[0]) * math.sin(math.PI * xp[1])
	}
}

// -grad u, what q (HDG) and sigma (mixed) approximate.
@(private = "file")
exact_flux :: proc(x, out: fe.Pvec(f64), _: rawptr) {
	for p in 0 ..< x.points {
		xp, q := fe.pvec_at_point(x, p).data, fe.pvec_at_point(out, p).data
		q[0] = math.PI * math.sin(math.PI * xp[0]) * math.sin(math.PI * xp[1])
		q[1] = -math.PI * math.cos(math.PI * xp[0]) * math.cos(math.PI * xp[1])
	}
}

// f = -lap u = 2 pi^2 u
@(private = "file")
source :: proc(x, out: fe.Pvec(f64), _: rawptr) {
	exact_u(x, out, nil)
	for p in 0 ..< x.points { fe.pvec_at_point(out, p).data[0] *= 2 * math.PI * math.PI }
}

// (grad u, grad v) and (f, v) on one cell, the cell terms of CG and DG.
@(private = "file")
laplace_cell :: proc(geo: fe.Geometry, u: ^fe.Space, site: fe.Site) -> (stiff: fe.Cmat(f64), load: fe.Cvec(f64)) {
	rule, w := fe.space_quad_rule(u, site)
	m := fe.geo_map(2, 2, geo, site, rule)
	tab := fe.basis_tab(fe.space_local_basis(u, site), rule, m)
	phi, grad := tab[.S_Val], tab[.S_Grd]
	dx := fe.element_map_weights(m, w)

	grad_dx := fe.pmat_create_identity(dx, 2, 1) // grad u . grad v dx
	f_dx := weighted_values(fe.geo_points(2, geo, site, rule), 1, source, dx) // f v dx

	stiff = fe.cmat_create_for(grad, grad_dx, grad)
	load = fe.cvec_create_for(phi, f_dx)
	fe.contract_bilinear_same(2, 1, stiff, grad, grad_dx)
	fe.contract_linear(1, 1, load, phi, f_dx)
	return
}

//== CG

@(private = "file")
solve_cg :: proc(t: ^testing.T, mesh: ^fe.Mesh, geo: fe.Geometry, degree: int) -> Errors {
	u := fe.space_create(mesh, {.Lagrange, fe.Order(degree), .Continuous, fe.ALL_TAGS}, 1)
	defer fe.space_destroy(u)
	sys := fe.sys_create({space = u, constraints = {fe.Constraint_Essential{tags = fe.ALL_TAGS}}})
	defer fe.sys_destroy(&sys)
	state, inhom := fe.sys_state(sys), fe.sys_state(sys)
	rhs := fe.sys_vector(sys)
	K := fe.sys_matrix(sys)

	// boundary values straight into the constrained dofs, lifted into rhs by the scatter
	fe.interpolate_facets(2, 2, 1, geo, fe.sys_space_coeffs(sys, inhom, u), fe.ALL_TAGS, exact_u)

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		site := fe.cell_site(cell)
		stiff, load := laplace_cell(geo, u, site)
		fe.sys_scatter_mat(sys, K, stiff, u, u, site, rhs, inhom)
		fe.sys_scatter_vec(sys, rhs, load, u, site)
	}
	solve(t, sys, fe.sys_matrix_finalize(&K), rhs, state, inhom)

	uh := fe.sys_space_coeffs(sys, state, u)
	return {l2_error(1, geo, uh, .S_Val, exact_u), l2_error(1, geo, uh, .S_Grd, exact_flux, scale = -1)}
}

//== DG

@(private = "file")
PENALTY :: 10.0

// One side of a facet: its site and u's values and physical gradients there.
@(private = "file")
Side :: struct {
	site:      fe.Site,
	phi, grad: fe.Bvec(f64),
}

@(private = "file")
solve_dg :: proc(t: ^testing.T, mesh: ^fe.Mesh, geo: fe.Geometry, degree: int) -> Errors {
	u := fe.space_create(mesh, {.Lagrange, fe.Order(degree), .Discontinuous, fe.ALL_TAGS}, 1)
	defer fe.space_destroy(u)
	sys := fe.sys_create({space = u})
	defer fe.sys_destroy(&sys)
	state, inhom := fe.sys_state(sys), fe.sys_state(sys)
	rhs := fe.sys_vector(sys)
	K := fe.sys_matrix(sys)

	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		site := fe.cell_site(cell)
		stiff, load := laplace_cell(geo, u, site)
		fe.sys_scatter_mat(sys, K, stiff, u, u, site)
		fe.sys_scatter_vec(sys, rhs, load, u, site)
	}

	// Facets, n outward from side 0, [w] = w0 - w1 and {w} = (w0 + w1) / 2, one sided on the boundary:
	//   -<{grad u . n}, [v]> - <[u], {grad v . n}> + s <[u], [v]>,   boundary rhs -<g, grad v . n> + s <g, v>
	// Side a's part of [v] is sign[a] v_a, so the (a, b) block is each term with its sides' signs in the coefficient.
	sign := [2]f64{1, -1}
	for facet in mesh.facets {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		n_sides := len(facet.cofaces)
		site0 := fe.coface_site(facet, facet.cofaces[0])
		rule, w := fe.space_quad_rule(u, site0)
		np := len(w)

		sides: [2]Side
		m0: fe.Element_Map(2, 2, f64)
		for cf, side in facet.cofaces {
			site := fe.coface_site(facet, cf)
			m := fe.geo_map(2, 2, geo, site, rule)
			tab := fe.basis_tab(fe.space_local_basis(u, site), rule, m)
			sides[side] = {site, tab[.S_Val], tab[.S_Grd]}
			if side == 0 { m0 = m }
		}
		normal, da := fe.element_map_facet_weights(m0, mesh.cells[site0.cell].type, facet.cofaces[0].local, w)

		h := 0.0 // facet size
		for d in da { h += d }
		s := PENALTY * f64((degree + 1) * (degree + 1)) / h
		avg := 0.5 if n_sides == 2 else 1.0

		for a in 0 ..< n_sides {
			for b in 0 ..< n_sides {
				test, trial := sides[a], sides[b]
				// one field, so each component block of these is a single value
				consistency := fe.pmat_create(f64, np, .Dense, 1, 1, 2, 1) // -{grad u . n} [v]
				symmetry := fe.pmat_create(f64, np, .Dense, 2, 1, 1, 1) // -[u] {grad v . n}
				penalty := fe.pmat_create_identity(da, 1, 1, s * sign[a] * sign[b]) //  s [u] [v]
				for p in 0 ..< np {
					for c in 0 ..< 2 {
						fe.pmat_at_point(consistency, p).data[c] = -avg * sign[a] * normal[p].data[c] * da[p]
						fe.pmat_at_point(symmetry, p).data[c] = -avg * sign[b] * normal[p].data[c] * da[p]
					}
				}

				local := fe.cmat_create_for(test.phi, penalty, trial.phi)
				fe.contract_bilinear(1, 1, 2, 1, local, test.phi, consistency, trial.grad)
				fe.contract_bilinear(2, 1, 1, 1, local, test.grad, symmetry, trial.phi)
				fe.contract_bilinear(1, 1, 1, 1, local, test.phi, penalty, trial.phi)
				fe.sys_scatter_mat_pair(sys, K, local, u, u, test.site, trial.site)
			}
		}

		if n_sides == 1 {
			g_da := weighted_values(fe.geo_points(2, geo, site0, rule), 1, exact_u, da)
			g_dn := fe.pvec_create(f64, np, 2, 1) // -g grad v . n
			g_pen := fe.pvec_create(f64, np, 1, 1) //  s g v
			for p in 0 ..< np {
				g := fe.pvec_at_point(g_da, p).data[0]
				for c in 0 ..< 2 { fe.pvec_at_point(g_dn, p).data[c] = -g * normal[p].data[c] }
				fe.pvec_at_point(g_pen, p).data[0] = s * g
			}

			load := fe.cvec_create_for(sides[0].phi, g_pen)
			fe.contract_linear(2, 1, load, sides[0].grad, g_dn)
			fe.contract_linear(1, 1, load, sides[0].phi, g_pen)
			fe.sys_scatter_vec(sys, rhs, load, u, site0)
		}
	}
	solve(t, sys, fe.sys_matrix_finalize(&K), rhs, state, inhom)

	uh := fe.sys_space_coeffs(sys, state, u)
	return {l2_error(1, geo, uh, .S_Val, exact_u), l2_error(1, geo, uh, .S_Grd, exact_flux, scale = -1)}
}

//== HDG

@(private = "file")
TAU :: 1.0

@(private = "file")
solve_hdg :: proc(t: ^testing.T, mesh: ^fe.Mesh, geo: fe.Geometry, degree: int) -> Errors {
	// q (2 fields) and u per cell are condensed out, lambda per facet is solved for
	q := fe.space_create(mesh, {.Lagrange, fe.Order(degree), .Discontinuous, fe.ALL_TAGS}, 2)
	u := fe.space_create(mesh, {.Lagrange, fe.Order(degree), .Discontinuous, fe.ALL_TAGS}, 1)
	lam := fe.space_create(mesh, {.Lagrange, fe.Order(degree), .Discontinuous, fe.ALL_TAGS}, 1, .Facets)
	defer fe.space_destroy(q, u, lam)
	cond := fe.cond_create(mesh, {q, u}, {lam})
	defer fe.cond_destroy(&cond)
	sys := fe.sys_create({space = lam})
	defer fe.sys_destroy(&sys)
	state, inhom := fe.sys_state(sys), fe.sys_state(sys)
	rhs := fe.sys_vector(sys)
	K := fe.sys_matrix(sys)

	for &cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		work := fe.cond_begin(&cond, &cell)

		// Cells: (q, r) - (u, div r) = 0,   -(q, grad v) = (f, v).
		// q is two fields of the scalar basis, so div r pairs gradient component c with field c of r.
		{
			site := fe.cell_site(cell)
			rule, w := fe.space_quad_rule(u, site)
			m := fe.geo_map(2, 2, geo, site, rule)
			tab := fe.basis_tab(fe.space_local_basis(u, site), rule, m) // q and u share this basis
			phi, grad := tab[.S_Val], tab[.S_Grd]
			dx := fe.element_map_weights(m, w)
			np := len(dx)

			mass := fe.pmat_create_identity(dx, 1, 2) //  q . r
			div_r := fe.pmat_create(f64, np, .Dense, 2, 2, 1, 1) // -u div r
			grad_v := fe.pmat_create(f64, np, .Dense, 2, 1, 1, 2) // -q . grad v
			for p in 0 ..< np {
				for c in 0 ..< 2 {
					dr, _ := fe.pmat_cmpnt_matrix(fe.pmat_at_point(div_r, p), c, 0, 2, 1)
					dr.data[0][c] = -dx[p] // field c of r, gradient component c
					gv, _ := fe.pmat_cmpnt_matrix(fe.pmat_at_point(grad_v, p), c, 0, 1, 2)
					gv.data[c][0] = -dx[p] // gradient component c of v, field c of q
				}
			}
			f_dx := weighted_values(fe.geo_points(2, geo, site, rule), 1, source, dx) //  f v

			qq := fe.cmat_create_for(phi, mass, phi)
			qu := fe.cmat_create_for(grad, div_r, phi)
			uq := fe.cmat_create_for(grad, grad_v, phi)
			load := fe.cvec_create_for(phi, f_dx)
			fe.contract_bilinear_same(1, 2, qq, phi, mass)
			fe.contract_bilinear(2, 2, 1, 1, qu, grad, div_r, phi)
			fe.contract_bilinear(2, 1, 1, 2, uq, grad, grad_v, phi)
			fe.contract_linear(1, 1, load, phi, f_dx)
			fe.cond_add_mat(&work, qq, q, q, site)
			fe.cond_add_mat(&work, qu, q, u, site)
			fe.cond_add_mat(&work, uq, u, q, site)
			fe.cond_add_vec(&work, load, u, site)
		}

		// Facets, n outward, numerical flux q^ . n = q . n + tau (u - lambda):
		//   <lambda, r . n>,   <q . n + tau (u - lambda), v>,   <q . n + tau (u - lambda), mu> = 0
		// On the boundary lambda is the L2 projection of g instead, <lambda, mu> = <g, mu>: interpolating g loses half an
		// order in q.
		for f, lf in cell.conn[.D1] {
			site := fe.cell_facet_site(cell, lf)
			rule, w := fe.space_quad_rule(u, site)
			m := fe.geo_map(2, 2, geo, site, rule)
			phi := fe.basis_tab(fe.space_local_basis(u, site), rule)[.S_Val]
			mu := fe.basis_tab(fe.space_local_basis(lam, site), rule)[.S_Val]
			normal, da := fe.element_map_facet_weights(m, cell.type, lf, w)
			np := len(da)

			q_n := fe.pmat_create(f64, np, .Dense, 1, 1, 1, 2) // q . n against a scalar test
			r_n := fe.pmat_create(f64, np, .Dense, 1, 2, 1, 1) // r . n against a scalar trial
			for p in 0 ..< np {
				qn, _ := fe.pmat_cmpnt_matrix(fe.pmat_at_point(q_n, p), 0, 0, 1, 2)
				rn, _ := fe.pmat_cmpnt_matrix(fe.pmat_at_point(r_n, p), 0, 0, 2, 1)
				for c in 0 ..< 2 { qn.data[c][0], rn.data[0][c] = normal[p].data[c] * da[p], normal[p].data[c] * da[p] }
			}
			tau := fe.pmat_create_identity(da, 1, 1, TAU)
			minus_tau := fe.pmat_create_identity(da, 1, 1, -TAU)

			q_lam := fe.cmat_create_for(phi, r_n, mu) //  <lambda, r . n>
			u_q := fe.cmat_create_for(phi, q_n, phi) //  <q . n, v>
			u_u := fe.cmat_create_for(phi, tau, phi) //  <tau u, v>
			u_lam := fe.cmat_create_for(phi, minus_tau, mu) // -<tau lambda, v>
			fe.contract_bilinear(1, 2, 1, 1, q_lam, phi, r_n, mu)
			fe.contract_bilinear(1, 1, 1, 2, u_q, phi, q_n, phi)
			fe.contract_bilinear_same(1, 1, u_u, phi, tau)
			fe.contract_bilinear(1, 1, 1, 1, u_lam, phi, minus_tau, mu)
			fe.cond_add_mat(&work, q_lam, q, lam, site)
			fe.cond_add_mat(&work, u_q, u, q, site)
			fe.cond_add_mat(&work, u_u, u, u, site)
			fe.cond_add_mat(&work, u_lam, u, lam, site)

			if fe.facet_is_boundary(mesh.facets[f]) {
				mass := fe.pmat_create_identity(da, 1, 1) //  <lambda, mu>
				g_da := weighted_values(fe.geo_points(2, geo, site, rule), 1, exact_u, da) //  <g, mu>

				lam_lam := fe.cmat_create_for(mu, mass, mu)
				lam_g := fe.cvec_create_for(mu, g_da)
				fe.contract_bilinear_same(1, 1, lam_lam, mu, mass)
				fe.contract_linear(1, 1, lam_g, mu, g_da)
				fe.cond_add_mat(&work, lam_lam, lam, lam, site)
				fe.cond_add_vec(&work, lam_g, lam, site)
				continue
			}

			lam_q := fe.cmat_create_for(mu, q_n, phi) //  <q . n, mu>
			lam_u := fe.cmat_create_for(mu, tau, phi) //  <tau u, mu>
			lam_lam := fe.cmat_create_for(mu, minus_tau, mu) // -<tau lambda, mu>
			fe.contract_bilinear(1, 1, 1, 2, lam_q, mu, q_n, phi)
			fe.contract_bilinear(1, 1, 1, 1, lam_u, mu, tau, phi)
			fe.contract_bilinear_same(1, 1, lam_lam, mu, minus_tau)
			fe.cond_add_mat(&work, lam_q, lam, q, site)
			fe.cond_add_mat(&work, lam_u, lam, u, site)
			fe.cond_add_mat(&work, lam_lam, lam, lam, site)
		}

		fe.cond_end(&work, sys, K, rhs)
	}
	solve(t, sys, fe.sys_matrix_finalize(&K), rhs, state, inhom)

	// q and u back from the solved trace, cell by cell
	qh, uh := fe.space_coeffs(q), fe.space_coeffs(u)
	fe.cond_apply_soln(&cond, sys, state, qh, uh)
	return {l2_error(1, geo, uh, .S_Val, exact_u), l2_error(2, geo, qh, .S_Val, exact_flux)}
}

//== Mixed

@(private = "file")
solve_mixed :: proc(t: ^testing.T, mesh: ^fe.Mesh, geo: fe.Geometry, degree: int) -> Errors {
	// sigma in block 0, u in block 1: the saddle point's two fields as separate blocks
	sigma := fe.space_create(mesh, {.Raviart_Thomas, fe.Order(degree - 1), .Continuous, fe.ALL_TAGS}, 1)
	u := fe.space_create(mesh, {.Lagrange, fe.Order(degree - 1), .Discontinuous, fe.ALL_TAGS}, 1)
	defer fe.space_destroy(sigma, u)
	sys := fe.sys_create({space = sigma}, {space = u, block = 1})
	defer fe.sys_destroy(&sys)
	state, inhom := fe.sys_state(sys), fe.sys_state(sys)
	rhs := fe.sys_vector(sys)
	K := fe.sys_matrix(sys)

	// Cells: (sigma, tau) - (u, div tau) = -<g, tau . n>,   -(div sigma, v) = -(f, v).
	// sigma's physical values are contravariant pushes and its divergence a density, both from basis_tab.
	for cell in mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		site := fe.cell_site(cell)
		rule, w := fe.space_quad_rule(sigma, site)
		m := fe.geo_map(2, 2, geo, site, rule)
		s_tab := fe.basis_tab(fe.space_local_basis(sigma, site), rule, m)
		val, div := s_tab[.V_Val], s_tab[.V_Div]
		phi := fe.basis_tab(fe.space_local_basis(u, site), rule, m)[.S_Val]
		dx := fe.element_map_weights(m, w)

		mass := fe.pmat_create_identity(dx, 2, 1) //  sigma . tau
		minus_dx := fe.pmat_create_identity(dx, 1, 1, -1) // -u div tau, -v div sigma
		f_dx := weighted_values(fe.geo_points(2, geo, site, rule), 1, source, dx, -1) // -f v

		ss := fe.cmat_create_for(val, mass, val)
		su := fe.cmat_create_for(div, minus_dx, phi)
		us := fe.cmat_create_for(phi, minus_dx, div)
		load := fe.cvec_create_for(phi, f_dx)
		fe.contract_bilinear_same(2, 1, ss, val, mass)
		fe.contract_bilinear(1, 1, 1, 1, su, div, minus_dx, phi)
		fe.contract_bilinear(1, 1, 1, 1, us, phi, minus_dx, div)
		fe.contract_linear(1, 1, load, phi, f_dx)
		fe.sys_scatter_mat(sys, K, ss, sigma, sigma, site)
		fe.sys_scatter_mat(sys, K, su, sigma, u, site)
		fe.sys_scatter_mat(sys, K, us, u, sigma, site)
		fe.sys_scatter_vec(sys, rhs, load, u, site)

		// boundary facets: -<g, tau . n>, u's Dirichlet data enters weakly through sigma's equation
		for f, lf in cell.conn[.D1] {
			if !fe.facet_is_boundary(mesh.facets[f]) { continue }
			fsite := fe.cell_facet_site(cell, lf)
			frule, fw := fe.space_quad_rule(sigma, fsite)
			fm := fe.geo_map(2, 2, geo, fsite, frule)
			tau := fe.basis_tab(fe.space_local_basis(sigma, fsite), frule, fm, fe.Quantity_Set{.V_Val})[.V_Val]
			normal, da := fe.element_map_facet_weights(fm, cell.type, lf, fw)
			g_da := weighted_values(fe.geo_points(2, geo, fsite, frule), 1, exact_u, da)

			g_n := fe.pvec_create(f64, len(da), 2, 1) // -g n
			for p in 0 ..< len(da) {
				g := fe.pvec_at_point(g_da, p).data[0]
				for c in 0 ..< 2 { fe.pvec_at_point(g_n, p).data[c] = -g * normal[p].data[c] }
			}
			bload := fe.cvec_create_for(tau, g_n)
			fe.contract_linear(2, 1, bload, tau, g_n)
			fe.sys_scatter_vec(sys, rhs, bload, sigma, fsite)
		}
	}
	solve(t, sys, fe.sys_matrix_finalize(&K), rhs, state, inhom)

	sh, uh := fe.sys_space_coeffs(sys, state, sigma), fe.sys_space_coeffs(sys, state, u)
	return {l2_error(1, geo, uh, .S_Val, exact_u), l2_error(1, geo, sh, .V_Val, exact_flux)}
}
