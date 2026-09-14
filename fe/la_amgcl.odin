package fe

/*
 Bindings for the AMGCL library (more specifically bindings to our C wrapper of it).

 These bindings are temporary and will be expanded to support more features of AMGCL.
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

Precond :: struct {}

Precond_Kind :: enum c.int {
	SA   = 0,
	ILU0 = 1,
}

Solver_Kind :: enum c.int {
	CG       = 0,
	BICGSTAB = 1,
	FGMRES   = 2,
}

SA_Extra :: struct {
	block_size:           c.int,
	coarse_enough:        c.int,
	near_null_space:      [^]c.double,
	near_null_space_cols: c.int,
}

ILU0_Extra :: struct {
	_reserved: c.int,
}

Precond_Extra :: struct #raw_union {
	sa:   SA_Extra,
	ilu0: ILU0_Extra,
}

Precond_Params :: struct {
	kind:  Precond_Kind,
	extra: Precond_Extra,
}

Solver_Common_Params :: struct {
	tolerance: c.double,
	max_iters: c.int,
	verbose:   c.int,
}

CG_Extra :: struct {
	_reserved: c.int,
}

BiCGStab_Extra :: struct {
	_reserved: c.int,
}

FGMRES_Extra :: struct {
	gmres_m: c.int,
}

Solver_Extra :: struct #raw_union {
	cg:       CG_Extra,
	bicgstab: BiCGStab_Extra,
	fgmres:   FGMRES_Extra,
}

Solver_Params :: struct {
	kind:   Solver_Kind,
	common: Solver_Common_Params,
	extra:  Solver_Extra,
}

@(default_calling_convention = "c", link_prefix = "amgcl")
foreign amgcl_lib {
	_precond_params_default :: proc(kind: Precond_Kind, out: ^Precond_Params) ---
	_solver_params_default :: proc(kind: Solver_Kind, out: ^Solver_Params) ---
	_precond_create :: proc(n: c.int, row_ptr: [^]c.int, col_ind: [^]c.int, values: [^]c.double, p: ^Precond_Params) -> ^Precond ---
	_precond_destroy :: proc(p: ^Precond) ---
	_precond_print :: proc(p: ^Precond) ---
	@(link_name = "amgcl_solve")
	solve_raw :: proc(p: ^Precond, rhs: [^]c.double, x: [^]c.double, sp: ^Solver_Params, out_iters: ^c.int, out_residual: ^c.double) -> c.int ---
}

SA_DEFAULT := Precond_Params {
	kind = .SA,
	extra = {sa = {block_size = 1, coarse_enough = 50, near_null_space = nil, near_null_space_cols = 0}},
}

ILU0_DEFAULT := Precond_Params {
	kind = .ILU0,
	extra = {ilu0 = {}},
}

CG_DEFAULT := Solver_Params {
	kind = .CG,
	common = {tolerance = 1e-8, max_iters = 500, verbose = 0},
	extra = {cg = {}},
}

BICGSTAB_DEFAULT := Solver_Params {
	kind = .BICGSTAB,
	common = {tolerance = 1e-8, max_iters = 500, verbose = 0},
	extra = {bicgstab = {}},
}

FGMRES_DEFAULT := Solver_Params {
	kind = .FGMRES,
	common = {tolerance = 1e-8, max_iters = 500, verbose = 0},
	extra = {fgmres = {gmres_m = 30}},
}

// wrapper

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

sa_params :: proc(
	block_size := 1,
	coarse_enough := 50,
	near_null_space: []f64 = nil,
	near_null_space_cols := 0,
) -> Precond_Params {
	p := SA_DEFAULT
	p.extra.sa.block_size = clamp(c.int(block_size), 1, 4)
	p.extra.sa.coarse_enough = c.int(coarse_enough)
	if len(near_null_space) > 0 {
		p.extra.sa.near_null_space = cast([^]c.double)raw_data(near_null_space)
		p.extra.sa.near_null_space_cols = c.int(near_null_space_cols)
	}
	return p
}

cg_params :: proc(tolerance := 1e-8, max_iters := 500, verbose := false) -> Solver_Params {
	p := CG_DEFAULT
	p.common = {c.double(tolerance), c.int(max_iters), c.int(verbose)}
	return p
}

bicgstab_params :: proc(tolerance := 1e-8, max_iters := 500, verbose := false) -> Solver_Params {
	p := BICGSTAB_DEFAULT
	p.common = {c.double(tolerance), c.int(max_iters), c.int(verbose)}
	return p
}

fgmres_params :: proc(tolerance := 1e-8, max_iters := 500, gmres_m := 30, verbose := false) -> Solver_Params {
	p := FGMRES_DEFAULT
	p.common = {c.double(tolerance), c.int(max_iters), c.int(verbose)}
	p.extra.fgmres.gmres_m = c.int(gmres_m)
	return p
}

// Creates the preconditioner for the given matrix.
amgcl_precond_create :: proc(
	m: Sparse_Matrix,
	params: Precond_Params = SA_DEFAULT,
) -> (
	precond: ^Precond,
	success: bool,
) {
	n := sp_rows(m)
	assert(n > 0, "row_ptrs must have at least 2 entries")
	assert(len(m.columns) == int(m.row_ptrs[n]), "columns length must match row_ptrs[n]")
	assert(len(m.values) == len(m.columns), "values length must match columns length")
	p := params
	if p.kind == .SA && p.extra.sa.near_null_space != nil {
		assert(p.extra.sa.near_null_space_cols > 0, "near_null_space was set but near_null_space_cols is 0.")
	}
	precond = _precond_create(
		c.int(n),
		cast([^]c.int)raw_data(m.row_ptrs),
		cast([^]c.int)raw_data(m.columns),
		cast([^]c.double)raw_data(m.values),
		&p,
	)
	success = precond != nil
	return
}

amgcl_precond_destroy :: proc(p: ^Precond) {
	_precond_destroy(p)
}

amgcl_precond_print :: proc(p: ^Precond) {
	_precond_print(p)
}

// Solve the system using the preconditioner.
amgcl_solve :: proc(p: ^Precond, rhs, x: Vector, params: Solver_Params = CG_DEFAULT) -> Solve_Result {
	assert(len(rhs) == len(x), "rhs and x must be the same length")
	sp := params
	iters: c.int
	residual: c.double
	rc := solve_raw(p, cast([^]c.double)raw_data(rhs), cast([^]c.double)raw_data(x), &sp, &iters, &residual)
	status: Solve_Status
	switch rc {
	case 0: status = .Converged
	case 1: status = .Not_Converged
	case: status = .Error
	}
	return Solve_Result{status = status, iters = int(iters), residual = f64(residual)}
}
