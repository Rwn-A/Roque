#ifndef AMGCL_C_H
#define AMGCL_C_H

/*
 * C binding for amgcl, built with AMGCL_NO_BOOST (no Boost dependency).
 *
 * Every component is chosen at runtime: the Krylov solver, the preconditioner
 * class, and inside AMG the coarsening and the relaxation. Each choice is an
 * enum plus a params struct, and the C++ side dispatches through one variant
 * wrapper per axis. Instantiations grow with the SUM of options on each axis,
 * never the product, and all of amgcl's options on each axis are available.
 *
 * Block and field-split preconditioners are implemented outside the binding
 * through the SHELL preconditioner class (a C callback), with amgcl solving
 * the blocks.
 *
 * Usage: build a preconditioner once with amgcl_precond_create, then call
 * amgcl_solve as often as needed, choosing the Krylov solver per call.
 *
 * Params structs are plain data with no ownership. Always start from
 * amgcl_precond_params_default() / amgcl_solver_params_default(); the
 * defaults are copied from amgcl's own C++ defaults at runtime, so they can't
 * drift from the library. Each component struct holds settings for EVERY
 * variant (e.g. relax.ilut.tau exists even when relax.type is SPAI0); only
 * the one selected by `type` is used. You can switch type without re-filling
 * anything.
 *
 * Thread safety: distinct preconditioner handles may be used from different
 * threads at once. A single handle must not be used by two solves or applies
 * at the same time (AMG keeps per-level work vectors inside the hierarchy).
 * amgcl_last_error() is per-thread.
 */

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    AMGCL_OK            =  0,
    AMGCL_NOT_CONVERGED =  1,
    AMGCL_ERROR         = -1,
} amgcl_status_t;

/* ---- relaxation (smoothers) ------------------------------------------ */

typedef enum {
    AMGCL_RELAX_GAUSS_SEIDEL,
    AMGCL_RELAX_ILU0,
    AMGCL_RELAX_ILU0_CHOW_PATEL,
    AMGCL_RELAX_ILUK,
    AMGCL_RELAX_ILUP,
    AMGCL_RELAX_ILUT,
    AMGCL_RELAX_DAMPED_JACOBI,
    AMGCL_RELAX_SPAI0,
    AMGCL_RELAX_SPAI1,
    AMGCL_RELAX_CHEBYSHEV,
} amgcl_relax_type_t;

/* Triangular solve used by the ILU family. serial=0 uses amgcl's parallel
 * level-scheduled solve (only matters when amgcl is built with OpenMP). */
typedef struct { int serial; } amgcl_ilu_solve_t;

typedef struct {
    amgcl_relax_type_t type;

    struct { int serial; }                                    gauss_seidel;
    struct { double damping; amgcl_ilu_solve_t solve; }       ilu0;
    struct { double damping; int sweeps; double omega; int symmetric_scaling;
             amgcl_ilu_solve_t solve; }                       ilu0_chow_patel;
    struct { int k; double damping; amgcl_ilu_solve_t solve; } iluk;
    struct { int k; double damping; amgcl_ilu_solve_t solve; } ilup;
    struct { double p; double tau; double damping;
             amgcl_ilu_solve_t solve; }                       ilut;
    struct { double damping; }                                damped_jacobi;
    struct { unsigned degree; float higher; float lower; int power_iters;
             int scale; }                                     chebyshev;
    /* spai0 and spai1 have no parameters. */
} amgcl_relax_params_t;

/* ---- coarsening ------------------------------------------------------- */

typedef enum {
    AMGCL_COARSENING_RUGE_STUBEN,
    AMGCL_COARSENING_AGGREGATION,
    AMGCL_COARSENING_SMOOTHED_AGGREGATION,
    AMGCL_COARSENING_SMOOTHED_AGGR_EMIN,
} amgcl_coarsening_type_t;

typedef struct {
    amgcl_coarsening_type_t type;

    struct { float eps_strong; int do_trunc; float eps_trunc; } ruge_stuben;

    /* Shared by the three aggregation-based coarsenings. */
    struct {
        unsigned block_size;  /* unknowns per node, e.g. 3 for 3D elasticity */
        float    eps_strong;

        /* Optional near null-space (rigid body modes etc.). NULL/0 to skip.
         * ROW-MAJOR, rows x cols: mode k at unknown i is nullspace[i*cols + k],
         * which is how amgcl's tentative_prolongation reads it. rows is the
         * size of the system this AMG is built on (for a block solver inside a
         * field split, the size of that block).
         * Copied at solver creation; only needs to live until then. */
        const double *nullspace;
        int           nullspace_cols;
    } aggr;

    struct { float over_interp; }                                       aggregation;
    struct { float relax; int estimate_spectral_radius; int power_iters; } smoothed_aggregation;
    /* smoothed_aggr_emin uses only `aggr`. */
} amgcl_coarsening_params_t;

/* ---- preconditioner --------------------------------------------------- */

typedef enum {
    AMGCL_PRECOND_AMG,         /* multilevel: coarsening + relax           */
    AMGCL_PRECOND_RELAXATION,  /* single-level: `relaxation` applied as M^-1 */
    AMGCL_PRECOND_DUMMY,       /* identity (unpreconditioned Krylov)       */
    AMGCL_PRECOND_SHELL,       /* user callback, see amgcl_shell_apply_fn  */
} amgcl_precond_class_t;

/* User-implemented preconditioner: compute x = M^-1 rhs.
 *
 * This is the building block for field-split / block preconditioners written
 * outside the binding: extract the blocks yourself, build an amgcl solver per
 * block with amgcl_precond_create, and combine them in the callback using
 * amgcl_precond_apply (one cycle) or amgcl_solve (inner iterations).
 *
 * Contract:
 *  - n is the size of the system; rhs and x have n entries.
 *  - Overwrite all of x. Its contents on entry are unspecified.
 *  - Return 0 on success. Any other value aborts the solve: amgcl_solve
 *    returns AMGCL_ERROR and amgcl_last_error() reports the value.
 *  - Called on the thread that called amgcl_solve, once or more per
 *    Krylov iteration. Calling into other amgcl solver handles from inside the
 *    callback is fine; calling into the handle currently being solved is not.
 *  - The callback must not unwind (no C++ exceptions, longjmp or panics
 *    escaping into amgcl). Report failure through the return value.
 *  - The Krylov method sees a preconditioner that may not be a fixed linear
 *    operator (inner iterations, AMG cycles); use FGMRES unless you know the
 *    callback is linear, and CG only if it is also symmetric positive definite. */
typedef int (*amgcl_shell_apply_fn)(void *ctx, int n, const double *rhs, double *x);

typedef struct {
    amgcl_coarsening_params_t coarsening;
    amgcl_relax_params_t      relax;
    unsigned coarse_enough;   /* stop coarsening below this many unknowns */
    int      direct_coarse;   /* direct solve on the coarsest level */
    unsigned max_levels;
    unsigned npre, npost;     /* relaxation sweeps per level */
    unsigned ncycle;          /* 1 = V-cycle, 2 = W-cycle */
    unsigned pre_cycles;      /* cycles per preconditioner application */
} amgcl_amg_params_t;

typedef struct {
    amgcl_precond_class_t cls;
    amgcl_amg_params_t    amg;         /* used when cls == AMGCL_PRECOND_AMG */
    amgcl_relax_params_t  relaxation;  /* used when cls == AMGCL_PRECOND_RELAXATION */
    struct {
        amgcl_shell_apply_fn apply;
        void                *ctx;      /* passed through untouched; must outlive the solver */
    } shell;                           /* used when cls == AMGCL_PRECOND_SHELL */
} amgcl_precond_params_t;

/* ---- Krylov solver ---------------------------------------------------- */

typedef enum {
    AMGCL_SOLVER_CG,
    AMGCL_SOLVER_BICGSTAB,
    AMGCL_SOLVER_BICGSTABL,
    AMGCL_SOLVER_GMRES,
    AMGCL_SOLVER_LGMRES,
    AMGCL_SOLVER_FGMRES,
    AMGCL_SOLVER_IDRS,
    AMGCL_SOLVER_RICHARDSON,
    AMGCL_SOLVER_PREONLY,     /* apply the preconditioner once, no iteration */
} amgcl_solver_type_t;

typedef struct {
    amgcl_solver_type_t type;

    /* Common to every type except PREONLY. Converged when
     * |r| <= max(tol * |rhs|, abstol). */
    double   tol;
    double   abstol;
    unsigned maxiter;
    int      ns_search;  /* for singular systems: search the null space */
    int      verbose;    /* amgcl prints per-iteration residuals */

    struct { int check_after; }                            bicgstab;
    struct { int L; double delta; int convex; }            bicgstabl;
    struct { unsigned M; }                                 gmres;  /* restart */
    struct { unsigned M; unsigned K; int always_reset; }   lgmres;
    struct { unsigned M; }                                 fgmres;
    struct { unsigned s; double omega; int smoothing; int replacement; } idrs;
    struct { double damping; }                             richardson;
} amgcl_solver_params_t;

/* ---- defaults --------------------------------------------------------- */

/* Fill with amgcl's own defaults (copied from its C++ params at runtime).
 * Always start from these rather than zero-initialising.
 *   precond: smoothed-aggregation AMG with SPAI0 relaxation
 *   solver:  BiCGStab, tol 1e-8, maxiter 100 */
void amgcl_precond_params_default(amgcl_precond_params_t *out);
void amgcl_solver_params_default(amgcl_solver_params_t *out);

/* Message for the most recent failed call on this thread ("" if none). */
const char *amgcl_last_error(void);

/* ---- preconditioner handle -------------------------------------------- */

/* The expensive part (the AMG hierarchy, ILU factors, ...) lives in the
 * preconditioner handle, built once. The Krylov solver is chosen per solve and
 * is cheap to set up: it allocates a few work vectors of length n (M+1 of them
 * for the GMRES family). */
typedef struct amgcl_precond amgcl_precond_t;

typedef struct {
    int    iterations;
    double residual;  /* relative residual as reported by amgcl */
} amgcl_conv_info_t;

/* CSR matrix, 0-based, n x n, column indices sorted within each row. The
 * matrix and every array referenced from prm are copied; all of them can be
 * freed after this returns. prm may be NULL for the defaults.
 * Returns NULL on failure; see amgcl_last_error(). */
amgcl_precond_t *amgcl_precond_create(int n, const int *row_ptr, const int *col_ind,
                                      const double *values, const amgcl_precond_params_t *prm);
void             amgcl_precond_destroy(amgcl_precond_t *p);

/* x = M^-1 rhs, one application of the preconditioner. This is what block
 * solvers inside a SHELL field split call. */
amgcl_status_t amgcl_precond_apply(const amgcl_precond_t *p, const double *rhs, double *x);

/* Print the preconditioner hierarchy to stdout. */
void amgcl_precond_print(const amgcl_precond_t *p);

int amgcl_precond_size(const amgcl_precond_t *p);

/* ---- solving ---------------------------------------------------------- */

/* Solve A x = rhs with the matrix the preconditioner was built from.
 * x is the initial guess on entry and the solution on exit. sp may be NULL for
 * the defaults; info may be NULL. The same preconditioner can be used for any
 * number of solves, with different solver settings each time. */
amgcl_status_t amgcl_solve(const amgcl_precond_t *p, const double *rhs, double *x,
                           const amgcl_solver_params_t *sp, amgcl_conv_info_t *info);

/* Solve with a different matrix (same size) while reusing the preconditioner.
 * Useful across Newton or time steps where the matrix changes a little. The
 * arrays only need to live for the duration of the call. */
amgcl_status_t amgcl_solve_with(const amgcl_precond_t *p,
                                const int *row_ptr, const int *col_ind, const double *values,
                                const double *rhs, double *x,
                                const amgcl_solver_params_t *sp, amgcl_conv_info_t *info);

#ifdef __cplusplus
}
#endif

#endif /* AMGCL_C_H */
