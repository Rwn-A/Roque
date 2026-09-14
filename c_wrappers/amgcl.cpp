// Entirely AI generated, not catching me writing a lick of C++.
#define AMGCL_NO_BOOST
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <functional>
#include <memory>
#include <tuple>
#include <cstddef>
#include <cstdio>
#include <new>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amgcl_precond amgcl_precond_t;

typedef enum {
    AMGCL_PRECOND_SA   = 0,
    AMGCL_PRECOND_ILU0 = 1,
} amgcl_precond_kind_t;

typedef enum {
    AMGCL_SOLVER_CG       = 0,
    AMGCL_SOLVER_BICGSTAB = 1,
    AMGCL_SOLVER_FGMRES   = 2,
} amgcl_solver_kind_t;

/* ---- precond params ---- */

typedef struct {
    int block_size;     /* aggregation block size */
    int coarse_enough;

    /* Near null-space hints (rigid body modes, for elasticity). Optional --
     * leave near_null_space NULL / near_null_space_cols 0 to skip.
     * Layout is "vector-major": mode k's value at unknown i is
     * near_null_space[k * n + i], i.e. near_null_space_cols contiguous
     * blocks of length n, where n is the row count passed to
     * amgcl_precond_create.
     *
     * CAVEAT: this layout is inferred from amgcl's smoothed_aggregation.hpp
     * (which only shows prm.nullspace.cols being read, not nullspace_params
     * itself or how tentative_prolongation indexes B) -- not confirmed
     * against amgcl/coarsening/detail/nullspace_params.hpp or the
     * tentative_prolongation implementation. Check those two before relying
     * on this in production; if the layout's wrong the preconditioner will
     * likely still build (B is just doubles) but silently fail to help
     * convergence, which is an easy thing to miss. */
    const double *near_null_space;
    int           near_null_space_cols;
} amgcl_sa_extra_t;

typedef struct {
    int _reserved;       /* room for ilut fill/drop later */
} amgcl_ilu0_extra_t;

typedef struct {
    amgcl_precond_kind_t kind;
    union {
        amgcl_sa_extra_t   sa;
        amgcl_ilu0_extra_t ilu0;
    } extra;
} amgcl_precond_params_t;

/* ---- solver params ---- */

typedef struct {
    double tolerance;
    int    max_iters;
    int    verbose;
} amgcl_solver_common_params_t;

typedef struct { int _reserved; } amgcl_cg_extra_t;
typedef struct { int _reserved; } amgcl_bicgstab_extra_t;
typedef struct { int gmres_m; }   amgcl_fgmres_extra_t;

typedef struct {
    amgcl_solver_kind_t           kind;
    amgcl_solver_common_params_t  common;
    union {
        amgcl_cg_extra_t       cg;
        amgcl_bicgstab_extra_t bicgstab;
        amgcl_fgmres_extra_t   fgmres;
    } extra;
} amgcl_solver_params_t;

void amgcl_precond_params_default(amgcl_precond_kind_t kind, amgcl_precond_params_t *out);
void amgcl_solver_params_default(amgcl_solver_kind_t kind, amgcl_solver_params_t *out);

amgcl_precond_t *amgcl_precond_create(int n, const int *row_ptr, const int *col_ind,
                                       const double *values, const amgcl_precond_params_t *p);
void amgcl_precond_destroy(amgcl_precond_t *p);
void amgcl_precond_print(amgcl_precond_t *p);

int amgcl_solve(amgcl_precond_t *p, const double *rhs, double *x,
                 const amgcl_solver_params_t *sp, int *out_iters, double *out_residual);

#ifdef __cplusplus
}
#endif

/* ======================================================================= */
/* Implementation                                                           */
/* ======================================================================= */

using Backend = amgcl::backend::builtin<double>;
using Range   = amgcl::iterator_range<const double *>;
using RangeX  = amgcl::iterator_range<double *>;

using Precond_SA   = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::gauss_seidel>;
using Precond_ILU0 = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilu0>;

using Solver_CG        = amgcl::solver::cg<Backend>;
using Solver_BiCGStab  = amgcl::solver::bicgstab<Backend>;
using Solver_FGMRES    = amgcl::solver::fgmres<Backend>;

/* Type-erased handle: holds whichever concrete preconditioner was built,
 * plus closures that know how to invoke each Krylov method against it.
 * Building the closures once at precond-creation time (instead of doing a
 * kind-based switch on every solve call) keeps amgcl_solve free of any
 * per-precond-type branching -- it only branches on solver kind. */
struct amgcl_precond {
    int n;

    std::function<std::tuple<std::size_t, double>(Range, RangeX, const Solver_CG::params&)>       run_cg;
    std::function<std::tuple<std::size_t, double>(Range, RangeX, const Solver_BiCGStab::params&)>  run_bicgstab;
    std::function<std::tuple<std::size_t, double>(Range, RangeX, const Solver_FGMRES::params&)>    run_fgmres;
    std::function<void()> print_fn;
};

void amgcl_precond_params_default(amgcl_precond_kind_t kind, amgcl_precond_params_t *out) {
    if (!out) return;
    *out = amgcl_precond_params_t{};
    out->kind = kind;
    switch (kind) {
    case AMGCL_PRECOND_SA:
        out->extra.sa.block_size          = 1;
        out->extra.sa.coarse_enough       = 50;
        out->extra.sa.near_null_space     = nullptr;
        out->extra.sa.near_null_space_cols = 0;
        break;
    case AMGCL_PRECOND_ILU0:
        out->extra.ilu0 = amgcl_ilu0_extra_t{};
        break;
    }
}

void amgcl_solver_params_default(amgcl_solver_kind_t kind, amgcl_solver_params_t *out) {
    if (!out) return;
    *out = amgcl_solver_params_t{};
    out->kind             = kind;
    out->common.tolerance = 1e-8;
    out->common.max_iters = 500;
    out->common.verbose   = 0;
    switch (kind) {
    case AMGCL_SOLVER_FGMRES:
        out->extra.fgmres.gmres_m = 30;
        break;
    default:
        break;
    }
}

amgcl_precond_t *amgcl_precond_create(int n, const int *row_ptr, const int *col_ind,
                                       const double *values, const amgcl_precond_params_t *params)
{
    amgcl_precond_params_t p;
    amgcl_precond_params_default(params ? params->kind : AMGCL_PRECOND_SA, &p);
    if (params) p = *params;

    amgcl_precond *h = new(std::nothrow) amgcl_precond();
    if (!h) return nullptr;
    h->n = n;

    auto make_A = [&]() {
        return std::make_tuple(
            n,
            amgcl::make_iterator_range(row_ptr, row_ptr + n + 1),
            amgcl::make_iterator_range(col_ind, col_ind + row_ptr[n]),
            amgcl::make_iterator_range(values,  values  + row_ptr[n]));
    };

    /* Krylov solvers only need P.system_matrix() + P itself to run, so once
     * we capture a shared_ptr to the concrete preconditioner in these
     * closures, run_cg/run_bicgstab/run_fgmres can each construct a fresh
     * lightweight Solver_XXX(n, params) per call and invoke solver(*P, r, x)
     * -- no need to keep the raw CSR arrays or a separate matrix copy alive. */
    try {
        switch (p.kind) {
        case AMGCL_PRECOND_SA: {
            int block_size = p.extra.sa.block_size;
            if (block_size < 1) block_size = 1;
            if (block_size > 4) block_size = 4;

            Precond_SA::params pp;
            pp.coarsening.aggr.block_size = block_size;
            pp.coarse_enough              = p.extra.sa.coarse_enough;

            /* Near null-space hints (rigid body modes for elasticity, etc).
             * amgcl's smoothed_aggregation coarsening consumes these via
             * params.nullspace.B / .cols; without them SA still works but
             * converges much more slowly (or not at all) on elasticity-type
             * systems, since plain aggregation has no way to know rigid
             * translations/rotations lie in the near-kernel of A. */
            if (p.extra.sa.near_null_space && p.extra.sa.near_null_space_cols > 0) {
                std::size_t total = static_cast<std::size_t>(n) *
                                     static_cast<std::size_t>(p.extra.sa.near_null_space_cols);
                pp.coarsening.nullspace.B.assign(
                    p.extra.sa.near_null_space, p.extra.sa.near_null_space + total);
                pp.coarsening.nullspace.cols = p.extra.sa.near_null_space_cols;
            }

            auto P = std::make_shared<Precond_SA>(make_A(), pp);

            h->run_cg = [P](Range r, RangeX x, const Solver_CG::params &sp) {
                Solver_CG S(amgcl::backend::rows(P->system_matrix()), sp);
                return S(*P, r, x);
            };
            h->run_bicgstab = [P](Range r, RangeX x, const Solver_BiCGStab::params &sp) {
                Solver_BiCGStab S(amgcl::backend::rows(P->system_matrix()), sp);
                return S(*P, r, x);
            };
            h->run_fgmres = [P](Range r, RangeX x, const Solver_FGMRES::params &sp) {
                Solver_FGMRES S(amgcl::backend::rows(P->system_matrix()), sp);
                return S(*P, r, x);
            };
            h->print_fn = [P]{ std::cout << *P << std::endl; };
            break;
        }
        case AMGCL_PRECOND_ILU0: {
            Precond_ILU0::params pp;
            auto P = std::make_shared<Precond_ILU0>(make_A(), pp);

            h->run_cg = [P](Range r, RangeX x, const Solver_CG::params &sp) {
                Solver_CG S(amgcl::backend::rows(P->system_matrix()), sp);
                return S(*P, r, x);
            };
            h->run_bicgstab = [P](Range r, RangeX x, const Solver_BiCGStab::params &sp) {
                Solver_BiCGStab S(amgcl::backend::rows(P->system_matrix()), sp);
                return S(*P, r, x);
            };
            h->run_fgmres = [P](Range r, RangeX x, const Solver_FGMRES::params &sp) {
                Solver_FGMRES S(amgcl::backend::rows(P->system_matrix()), sp);
                return S(*P, r, x);
            };
            h->print_fn = [P]{ std::cout << *P << std::endl; };
            break;
        }
        default:
            delete h;
            return nullptr;
        }
    } catch (...) {
        delete h;
        return nullptr;
    }

    return h;
}

void amgcl_precond_destroy(amgcl_precond_t *h) { delete h; }

void amgcl_precond_print(amgcl_precond_t *h) {
    if (h && h->print_fn) h->print_fn();
}

int amgcl_solve(amgcl_precond_t *h, const double *rhs, double *x,
                 const amgcl_solver_params_t *sp_in, int *out_iters, double *out_residual)
{
    if (!h || !sp_in) return -1;
    amgcl_solver_params_t sp = *sp_in;

    std::size_t iters = 0;
    double      resid = 0.0;

    try {
        auto r = amgcl::make_iterator_range(rhs, rhs + h->n);
        auto X = amgcl::make_iterator_range(x,   x   + h->n);

        switch (sp.kind) {
        case AMGCL_SOLVER_CG: {
            if (!h->run_cg) return -1;
            Solver_CG::params p;
            p.tol     = sp.common.tolerance;
            p.maxiter = sp.common.max_iters;
            std::tie(iters, resid) = h->run_cg(r, X, p);
            break;
        }
        case AMGCL_SOLVER_BICGSTAB: {
            if (!h->run_bicgstab) return -1;
            Solver_BiCGStab::params p;
            p.tol     = sp.common.tolerance;
            p.maxiter = sp.common.max_iters;
            std::tie(iters, resid) = h->run_bicgstab(r, X, p);
            break;
        }
        case AMGCL_SOLVER_FGMRES: {
            if (!h->run_fgmres) return -1;
            Solver_FGMRES::params p;
            p.tol     = sp.common.tolerance;
            p.maxiter = sp.common.max_iters;
            p.M       = sp.extra.fgmres.gmres_m;
            std::tie(iters, resid) = h->run_fgmres(r, X, p);
            break;
        }
        default:
            return -1;
        }
    } catch (...) {
        return -1;
    }

    if (out_iters)    *out_iters    = static_cast<int>(iters);
    if (out_residual) *out_residual = resid;
    if (sp.common.verbose)
        printf("amgcl: %d iters  residual = %e\n", static_cast<int>(iters), resid);

    return resid <= sp.common.tolerance ? 0 : 1;
}
