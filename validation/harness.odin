package validation

import "base:runtime"
import "core:fmt"
import "core:log"
import "core:math"
import "core:mem"
import "core:mem/virtual"
import "core:strings"
import "core:testing"

import fe "../fe"
import "../fe/fio"

// Divisions per side of the unit square meshes, see meshes/gen_square.py.
@(rodata)
MESH_SIZES := [?]int{4, 8, 16, 32}

MAX_DEGREE :: 3

// An observed rate may fall this far short of the expected one.
RATE_SLACK :: 0.25

// L2 errors of a solution and of its flux, -grad u.
Errors :: struct {
	u, flux: f64,
}

// Solves on `mesh` at polynomial `degree`, returns the errors against the exact solution.
Solve_Proc :: #type proc(t: ^testing.T, mesh: ^fe.Mesh, geo: fe.Geometry, degree: int) -> Errors

// Runs `solve` at degrees 1 to MAX_DEGREE on every mesh size, logs the errors and the rates between successive meshes,
// and expects the rate between the two finest to reach p + u_rate for u and p + flux_rate for the flux.
convergence :: proc(t: ^testing.T, name: string, solve: Solve_Proc, u_rate, flux_rate: int) {
	errors: [MAX_DEGREE][len(MESH_SIZES)]Errors
	for n, m in MESH_SIZES {
		context.allocator = test_arena()
		path := fmt.tprintf("./validation/meshes/2d_unit_square_%d.msh", n)
		mesh := fio.load_mesh(path, .Gmsh_V2_Binary) or_else testing.fail_now(t, path)
		defer fe.mesh_destroy(&mesh)
		geo := fe.geo_create(&mesh, fe.XY_PLANE_FRAME)
		defer fe.geo_destroy(geo)

		for p in 1 ..= MAX_DEGREE { errors[p - 1][m] = solve(t, &mesh, geo, p) }
	}

	for p in 1 ..= MAX_DEGREE {
		report(t, name, "u", p, p + u_rate, errors[p - 1][:], 0)
		report(t, name, "flux", p, p + flux_rate, errors[p - 1][:], 1)
	}

	report :: proc(t: ^testing.T, name, quantity: string, p, expected: int, errors: []Errors, which: int) {
		b := strings.builder_make(context.temp_allocator)
		fmt.sbprintf(&b, "%-5s p%d %-4s", name, p, quantity)
		rate: f64
		for e, m in errors {
			err := e.u if which == 0 else e.flux
			fmt.sbprintf(&b, "  %.2e", err)
			if m == 0 { continue }
			prev := errors[m - 1].u if which == 0 else errors[m - 1].flux
			rate = math.log2(prev / err)
			fmt.sbprintf(&b, " (%.2f)", rate)
		}
		log.info(strings.to_string(b))
		testing.expectf(t, rate > f64(expected) - RATE_SLACK, "%s p%d %s: rate %.2f, expected %d", name, p, quantity, rate, expected)
	}
}

// An arena for the calling scope, destroyed when it ends: context.allocator = test_arena().
@(deferred_out = test_arena_end)
test_arena :: proc() -> mem.Allocator {
	arena := new(virtual.Arena, runtime.heap_allocator())
	virtual.arena_init_growing(arena) or_else panic("Failed to create arena")
	return virtual.arena_allocator(arena)
}

@(private = "file")
test_arena_end :: proc(alloc: mem.Allocator) {
	arena := (^virtual.Arena)(alloc.data)
	virtual.arena_destroy(arena)
	free(arena, runtime.heap_allocator())
}

// Solves A x = rhs, ILU0 preconditioned BiCGStab for every method, expects convergence and applies the solution.
solve :: proc(t: ^testing.T, sys: fe.Sys, A: fe.Block_Sparse, rhs: fe.Vector, state, inhom: fe.State) {
	soln := fe.sys_vector(sys)
	p, ok := fe.amgcl_precond_create(fe.bsp_to_sp(A), fe.relaxation_params(.ILU0))
	if !ok { testing.fail_now(t, string(fe.amgcl_last_error())) }
	defer fe.amgcl_precond_destroy(p)

	res := fe.amgcl_solve(p, rhs, soln, fe.solver_params(.BiCGStab, tol = 1e-12, maxiter = 10000))
	testing.expectf(t, res.status == .Converged, "solve: %v", res)
	fe.sys_apply_soln(sys, state, inhom, soln)
}

// `f` at the points `x` times scale * weights[p], cmpnts values per point.
weighted_values :: proc(
	x: fe.Pvec(f64),
	cmpnts: int,
	f: fe.Interp_Proc,
	weights: []f64,
	scale := 1.0,
	alloc := context.allocator,
) -> fe.Pvec(f64) {
	out := fe.pvec_create(f64, x.points, cmpnts, 1, alloc)
	f(x, out, nil)
	for w, p in weights {
		for &v in fe.pvec_at_point(out, p).data { v *= scale * w }
	}
	return out
}

// L2 norm over the mesh of scale * (quantity q of the cell space) - exact, on a rule well past every degree used here.
l2_error :: proc(
	$F: int,
	geo: fe.Geometry,
	sc: fe.Space_Coeffs,
	q: fe.Basis_Quantity,
	exact: fe.Interp_Proc,
	scale := 1.0,
) -> f64 {
	sum := 0.0
	for cell in sc.space.mesh.cells {
		fe.scratch_guard()
		context.allocator = fe.scratch()
		site := fe.cell_site(cell)
		rule, w := fe.element_quadrature_rule(cell.type, .Q7)
		m := fe.geo_map(2, 2, geo, site, rule)
		uh := fe.space_eval(F, sc, site, rule, m, q)
		ue := fe.pvec_create(f64, uh.points, uh.cmpnts, uh.fields)
		exact(fe.geo_points(2, geo, site, rule), ue, nil)
		for dx, p in fe.element_map_weights(m, w) {
			for v, k in fe.pvec_at_point(uh, p).data {
				e := scale * v - fe.pvec_at_point(ue, p).data[k]
				sum += e * e * dx
			}
		}
	}
	return math.sqrt(sum)
}
