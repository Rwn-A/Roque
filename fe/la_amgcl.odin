package fe

/*
 Bindings for our AMGCL C wrapper.
*/

import "core:c"

BUILD_DIR :: #config(BUILD_DIR, "../build")

when ODIN_OS == .Windows {
	LIB_PATH :: BUILD_DIR + "/amgcl.lib"
} else {
	LIB_PATH :: BUILD_DIR + "/amgcl.a"
}

foreign import amgcl_lib {LIB_PATH}

// For some reason this isnt needed on windows, dunno about mac.
when ODIN_OS == .Linux {
	@(require) foreign import libm "system:m"
	@(require) foreign import libstdcpp "system:stdc++"
}

Amgcl_Status :: enum c.int {
	Ok            = 0,
	Not_Converged = 1,
	Error         = -1,
}

//== relaxation

Relax_Type :: enum c.int {
	Gauss_Seidel,
	ILU0,
	ILU0_Chow_Patel,
	ILUK,
	ILUP,
	ILUT,
	Damped_Jacobi,
	SPAI0,
	SPAI1,
	Chebyshev,
}

ILU_Solve :: struct {
	serial: b32,
}

Relax_Params :: struct {
	kind:            Relax_Type,
	gauss_seidel:    struct {
		serial: b32,
	},
	ilu0:            struct {
		damping: f64,
		solve:   ILU_Solve,
	},
	ilu0_chow_patel: struct {
		damping:           f64,
		sweeps:            c.int,
		omega:             f64,
		symmetric_scaling: b32,
		solve:             ILU_Solve,
	},
	iluk:            struct {
		k:       c.int,
		damping: f64,
		solve:   ILU_Solve,
	},
	ilup:            struct {
		k:       c.int,
		damping: f64,
		solve:   ILU_Solve,
	},
	ilut:            struct {
		p, tau, damping: f64,
		solve:           ILU_Solve,
	},
	damped_jacobi:   struct {
		damping: f64,
	},
	chebyshev:       struct {
		degree:        c.uint,
		higher, lower: f32,
		power_iters:   c.int,
		scale:         b32,
	},
}

//== coarsening

Coarsening_Type :: enum c.int {
	Ruge_Stuben,
	Aggregation,
	Smoothed_Aggregation,
	Smoothed_Aggr_Emin,
}

Coarsening_Params :: struct {
	kind:                 Coarsening_Type,
	ruge_stuben:          struct {
		eps_strong: f32,
		do_trunc:   b32,
		eps_trunc:  f32,
	},
	// Shared by the three aggregation coarsenings.
	aggr:                 struct {
		block_size:     c.uint,
		eps_strong:     f32,
		nullspace:      [^]f64,
		nullspace_cols: c.int,
	},
	aggregation:          struct {
		over_interp: f32,
	},
	smoothed_aggregation: struct {
		relax:                    f32,
		estimate_spectral_radius: b32,
		power_iters:              c.int,
	},
}

//== preconditioner

Precond_Class :: enum c.int {
	AMG,
	Relaxation,
	Dummy,
	Shell,
}

AMG_Params :: struct {
	coarsening:    Coarsening_Params,
	relax:         Relax_Params,
	coarse_enough: c.uint,
	direct_coarse: b32,
	max_levels:    c.uint,
	npre, npost:   c.uint,
	ncycle:        c.uint,
	pre_cycles:    c.uint,
}

// x = M^-1 rhs. Overwrite all of x, return 0 on success; anything else aborts
Shell_Apply :: #type proc "c" (ctx: rawptr, n: c.int, rhs: [^]f64, x: [^]f64) -> c.int

Precond_Params :: struct {
	class:      Precond_Class,
	amg:        AMG_Params,
	relaxation: Relax_Params,
	shell:      struct {
		apply: Shell_Apply,
		ctx:   rawptr, // passed through untouched; must outlive the preconditioner
	},
}

//== Krylov solver

Solver_Type :: enum c.int {
	CG,
	BiCGStab,
	BiCGStabL,
	GMRES,
	LGMRES,
	FGMRES,
	IDRS,
	Richardson,
	Preonly,
}

Solver_Params :: struct {
	kind:       Solver_Type,
	// Common to every kind except Preonly.
	tol:        f64,
	abstol:     f64,
	maxiter:    c.uint,
	ns_search:  b32,
	verbose:    b32,
	bicgstab:   struct {
		check_after: b32,
	},
	bicgstabl:  struct {
		L:      c.int,
		delta:  f64,
		convex: b32,
	},
	gmres:      struct {
		M: c.uint,
	},
	lgmres:     struct {
		M, K:         c.uint,
		always_reset: b32,
	},
	fgmres:     struct {
		M: c.uint,
	},
	idrs:       struct {
		s:                      c.uint,
		omega:                  f64,
		smoothing, replacement: b32,
	},
	richardson: struct {
		damping: f64,
	},
}

Conv_Info :: struct {
	iterations: c.int,
	residual:   f64,
}

Precond :: struct {} // opaque

@(default_calling_convention = "c")
foreign amgcl_lib {
	@(link_name = "amgcl_precond_params_default")
	amgcl_precond_params_default :: proc(out: ^Precond_Params) ---
	@(link_name = "amgcl_solver_params_default")
	amgcl_solver_params_default :: proc(out: ^Solver_Params) ---
	@(link_name = "amgcl_last_error")
	amgcl_last_error :: proc() -> cstring ---
	@(link_name = "amgcl_precond_create")
	_precond_create :: proc(n: c.int, row_ptr, col_ind: [^]c.int, values: [^]f64, prm: ^Precond_Params) -> ^Precond ---
	@(link_name = "amgcl_precond_destroy")
	_precond_destroy :: proc(p: ^Precond) ---
	@(link_name = "amgcl_precond_apply")
	_precond_apply :: proc(p: ^Precond, rhs, x: [^]f64) -> Amgcl_Status ---
	@(link_name = "amgcl_precond_print")
	_precond_print :: proc(p: ^Precond) ---
	@(link_name = "amgcl_precond_size")
	amgcl_precond_size :: proc(p: ^Precond) -> c.int ---
	@(link_name = "amgcl_solve")
	_solve :: proc(p: ^Precond, rhs, x: [^]f64, sp: ^Solver_Params, info: ^Conv_Info) -> Amgcl_Status ---
	@(link_name = "amgcl_solve_with")
	_solve_with :: proc(p: ^Precond, row_ptr, col_ind: [^]c.int, values: [^]f64, rhs, x: [^]f64, sp: ^Solver_Params, info: ^Conv_Info) -> Amgcl_Status ---
}

//== wrapper

Solve_Status :: enum {
	Converged,
	Not_Converged,
	Error,
}

Solve_Result :: struct {
	status:   Solve_Status,
	iters:    int,
	residual: f64,
}

@(private = "file")
check_csr :: proc(m: Sparse_Matrix) -> int {
	n := sp_n_rows(m)
	assert(n > 0, "row_ptrs must have at least 2 entries")
	assert(len(m.columns) == int(m.row_ptrs[n]), "columns length must match row_ptrs[n]")
	assert(len(m.values) == len(m.columns), "values length must match columns length")
	return n
}

@(private = "file")
to_result :: proc(st: Amgcl_Status, info: Conv_Info) -> Solve_Result {
	status: Solve_Status
	switch st {
	case .Ok: status = .Converged
	case .Not_Converged: status = .Not_Converged
	case .Error: status = .Error
	case: unreachable()
	}
	return {status = status, iters = int(info.iterations), residual = info.residual}
}

// Builds the preconditioner.
amgcl_precond_create :: proc(
	m: Sparse_Matrix,
	params: Maybe(Precond_Params) = nil,
) -> (
	precond: ^Precond,
	success: bool,
) {
	n := check_csr(m)
	p: Precond_Params
	if v, ok := params.?; ok { p = v } else { amgcl_precond_params_default(&p) }
	nsp := p.amg.coarsening.aggr
	assert(nsp.nullspace == nil || nsp.nullspace_cols > 0, "nullspace set but nullspace_cols is 0")
	assert(p.class != .Shell || p.shell.apply != nil, "shell preconditioner needs an apply proc")
	precond = _precond_create(c.int(n), raw_data(m.row_ptrs), raw_data(m.columns), raw_data(m.values), &p)
	return precond, precond != nil
}

// Free preconditioner
amgcl_precond_destroy :: proc(p: ^Precond) {
	_precond_destroy(p)
}

// Print preconditioner info from amgcl
amgcl_precond_print :: proc(p: ^Precond) {
	_precond_print(p)
}

// x = M^-1 rhs
amgcl_precond_apply :: proc(p: ^Precond, rhs, x: Vector) -> bool {
	assert(len(rhs) == len(x) && len(x) == int(amgcl_precond_size(p)))
	return _precond_apply(p, raw_data(rhs), raw_data(x)) == .Ok
}

// Solve with the matrix the preconditioner was built from. x is the initial guess.
amgcl_solve :: proc(p: ^Precond, rhs, x: Vector, params: Maybe(Solver_Params) = nil) -> Solve_Result {
	assert(len(rhs) == len(x) && len(x) == int(amgcl_precond_size(p)), "rhs and x must match the system size")
	sp: Solver_Params
	if v, ok := params.?; ok { sp = v } else { amgcl_solver_params_default(&sp) }
	info: Conv_Info
	st := _solve(p, raw_data(rhs), raw_data(x), &sp, &info)
	return to_result(st, info)
}

// Solve with a different (same-size) matrix, reusing the preconditioner.
amgcl_solve_with :: proc(
	p: ^Precond,
	m: Sparse_Matrix,
	rhs, x: Vector,
	params: Maybe(Solver_Params) = nil,
) -> Solve_Result {
	n := check_csr(m)
	assert(n == int(amgcl_precond_size(p)), "matrix size must match the preconditioner")
	assert(len(rhs) == n && len(x) == n, "rhs and x must match the system size")
	sp: Solver_Params
	if v, ok := params.?; ok { sp = v } else { amgcl_solver_params_default(&sp) }
	info: Conv_Info
	st := _solve_with(
		p,
		raw_data(m.row_ptrs),
		raw_data(m.columns),
		raw_data(m.values),
		raw_data(rhs),
		raw_data(x),
		&sp,
		&info,
	)
	return to_result(st, info)
}

//== preconditioner helpers

amg_params :: proc(
	relax := Relax_Type.SPAI0,
	coarsening := Coarsening_Type.Smoothed_Aggregation,
	block_size := 1,
	nullspace: []f64 = nil,
	nullspace_cols := 0,
) -> (
	p: Precond_Params,
) {
	amgcl_precond_params_default(&p)
	p.class = .AMG
	p.amg.relax.kind = relax
	p.amg.coarsening.kind = coarsening
	p.amg.coarsening.aggr.block_size = c.uint(block_size)
	if len(nullspace) > 0 {
		assert(
			nullspace_cols > 0 && len(nullspace) % nullspace_cols == 0,
			"nullspace length must be n * nullspace_cols",
		)
		p.amg.coarsening.aggr.nullspace = raw_data(nullspace)
		p.amg.coarsening.aggr.nullspace_cols = c.int(nullspace_cols)
	}
	return
}

// Single-level relaxation (ILU0, ILUT, SPAI0, ...) used directly as the preconditioner.
relaxation_params :: proc(kind := Relax_Type.ILU0) -> (p: Precond_Params) {
	amgcl_precond_params_default(&p)
	p.class = .Relaxation
	p.relaxation.kind = kind
	return
}

// No preconditioning.
identity_params :: proc() -> (p: Precond_Params) {
	amgcl_precond_params_default(&p)
	p.class = .Dummy
	return
}

// User preconditioner, e.g. a field split.
shell_params :: proc(apply: Shell_Apply, ctx: rawptr) -> (p: Precond_Params) {
	amgcl_precond_params_default(&p)
	p.class = .Shell
	p.shell.apply = apply
	p.shell.ctx = ctx
	return
}

//== solver helpers

solver_params :: proc(
	kind := Solver_Type.BiCGStab,
	tol := 1e-8,
	maxiter := 500,
	verbose := false,
) -> (
	p: Solver_Params,
) {
	amgcl_solver_params_default(&p)
	p.kind = kind
	p.tol = tol
	p.maxiter = c.uint(maxiter)
	p.verbose = b32(verbose)
	return
}

cg_params :: proc(tol := 1e-8, maxiter := 500, verbose := false) -> Solver_Params {
	return solver_params(.CG, tol, maxiter, verbose)
}

bicgstab_params :: proc(tol := 1e-8, maxiter := 500, verbose := false) -> Solver_Params {
	return solver_params(.BiCGStab, tol, maxiter, verbose)
}

gmres_params :: proc(tol := 1e-8, maxiter := 500, restart := 30, verbose := false) -> Solver_Params {
	p := solver_params(.GMRES, tol, maxiter, verbose)
	p.gmres.M = c.uint(restart)
	return p
}

fgmres_params :: proc(tol := 1e-8, maxiter := 500, restart := 30, verbose := false) -> Solver_Params {
	p := solver_params(.FGMRES, tol, maxiter, verbose)
	p.fgmres.M = c.uint(restart)
	return p
}

// Apply the preconditioner once, no iteration.
preonly_params :: proc() -> Solver_Params {
	return solver_params(.Preonly)
}
