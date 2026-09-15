// Entirely AI generated, not catching me writing a lick of C++.
#define AMGCL_NO_BOOST
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>
#include <amgcl/preconditioner/schur_pressure_correction.hpp>
#include <amgcl/solver/preonly.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/backend/builtin.hpp>
#include <functional>
#include <memory>
#include <tuple>
#include <vector>
#include <cstddef>
#include <cstdio>
#include <new>
#include <iostream>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct amgcl_precond amgcl_precond_t;

typedef enum {
    AMGCL_PRECOND_SA    = 0,
    AMGCL_PRECOND_ILU0  = 1,
    AMGCL_PRECOND_SCHUR = 2,
} amgcl_precond_kind_t;

typedef enum {
    AMGCL_SOLVER_CG       = 0,
    AMGCL_SOLVER_BICGSTAB = 1,
    AMGCL_SOLVER_FGMRES   = 2,
} amgcl_solver_kind_t;

/* Relaxation (AMG smoother) choice. Only meaningful for precond kinds that
 * actually run AMG (SA, and the USolver half of SCHUR). Deliberately NOT
 * crossed against every axis -- ILU0-as-precond has no relaxation slot
 * (it IS the relaxation, single-level), and SCHUR's PSolver is fixed to
 * SPAI0 rather than exposed, since that pairing is what amgcl's own Stokes
 * docs recommend and there's little reason to vary it per problem class.
 * So the actual C++ type count stays linear in relax choices, not
 * multiplicative: ILU0 (1) + SA x {GS, ILU0, SPAI0} (3) +
 * Schur<USolver x {GS, ILU0, SPAI0}, PSolver_fixed> (3) = 7 instantiations
 * total. */
typedef enum {
    AMGCL_RELAX_GAUSS_SEIDEL = 0,
    AMGCL_RELAX_ILU0         = 1,
    AMGCL_RELAX_SPAI0        = 2,
} amgcl_relax_kind_t;

/* ---- precond params ---- */

typedef struct {
    int block_size;     /* aggregation block size */
    int coarse_enough;
    amgcl_relax_kind_t relax;

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

/* Schur complement pressure-correction preconditioner, for saddle-point
 * systems: mixed Poisson/Darcy, incompressible NS, anything with a
 * structural zero block. "pressure" here is amgcl's own vocabulary and
 * just means "the field sitting on the zero diagonal block" -- for mixed
 * Poisson that's your scalar u, not your flux sigma. */
typedef struct {
    /* Length n, one entry per row of the system passed to
     * amgcl_precond_create. 1 marks a "pressure" (zero-block) dof, 0 marks
     * a "flow" dof. Caller-owned; must be built to match whatever dof
     * ordering the assembled matrix actually used. */
    const char *pmask;

    /* 1 or 2, see amgcl docs for schur_pressure_correction::params::type.
     * 0 (unset) defaults to 1 at precond-create time. */
    int type;

    /* When true, approximate A^-1 as diag(A)^-1 when forming the
     * matrix-free Schur complement. Cheaper, often good enough. */
    int approx_schur;

    int adjust_p;

    /* Relaxation for the flow (USolver) AMG block. PSolver is fixed to
     * SPAI0 and not exposed -- see the comment on amgcl_relax_kind_t. */
    amgcl_relax_kind_t usolver_relax;
} amgcl_schur_extra_t;

typedef struct {
    amgcl_precond_kind_t kind;
    union {
        amgcl_sa_extra_t    sa;
        amgcl_ilu0_extra_t  ilu0;
        amgcl_schur_extra_t schur;
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

using IntRange   = amgcl::iterator_range<const int *>;
using MatrixTuple = std::tuple<int, IntRange, IntRange, Range>;

/* AMG, one instantiation per relaxation choice. Reused both directly (as
 * AMGCL_PRECOND_SA) and as the USolver half of AMGCL_PRECOND_SCHUR -- same
 * three typedefs serve both call sites, so this is the only place the
 * relaxation axis gets multiplied out. */
using SA_GaussSeidel = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::gauss_seidel>;
using SA_ILU0        = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::ilu0>;
using SA_SPAI0       = amgcl::amg<Backend, amgcl::coarsening::smoothed_aggregation, amgcl::relaxation::spai0>;

using Precond_ILU0 = amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::ilu0>;

/* Schur pressure-correction: USolver varies by relaxation, PSolver is fixed
 * to SPAI0 -- see the comment on amgcl_relax_kind_t for why. `preonly`
 * means "apply the AMG/relaxation cycle once as an approximate inverse",
 * not "run a nested Krylov loop"; that's the cheap default and matches
 * what amgcl's own Stokes example found worked well. */
using USolver_GS    = amgcl::make_solver<SA_GaussSeidel, amgcl::solver::preonly<Backend>>;
using USolver_ILU0  = amgcl::make_solver<SA_ILU0,        amgcl::solver::preonly<Backend>>;
using USolver_SPAI0 = amgcl::make_solver<SA_SPAI0,       amgcl::solver::preonly<Backend>>;
using PSolver       = amgcl::make_solver<
    amgcl::relaxation::as_preconditioner<Backend, amgcl::relaxation::spai0>,
    amgcl::solver::preonly<Backend>>;

using Precond_Schur_GS    = amgcl::preconditioner::schur_pressure_correction<USolver_GS,    PSolver>;
using Precond_Schur_ILU0  = amgcl::preconditioner::schur_pressure_correction<USolver_ILU0,  PSolver>;
using Precond_Schur_SPAI0 = amgcl::preconditioner::schur_pressure_correction<USolver_SPAI0, PSolver>;

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

/* Wires up the three Krylov closures + printer for any concrete
 * preconditioner type that exposes system_matrix() and is usable as
 * S(*P, r, x). Written once so adding a new relaxation/precond variant is
 * "instantiate the type, call this" rather than three more hand-copied
 * closures per variant. */
template <class PrecondT>
static void wire_closures(amgcl_precond *h, std::shared_ptr<PrecondT> P) {
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
}

void amgcl_precond_params_default(amgcl_precond_kind_t kind, amgcl_precond_params_t *out) {
    if (!out) return;
    *out = amgcl_precond_params_t{};
    out->kind = kind;
    switch (kind) {
    case AMGCL_PRECOND_SA:
        out->extra.sa.block_size          = 1;
        out->extra.sa.coarse_enough       = 50;
        out->extra.sa.relax               = AMGCL_RELAX_GAUSS_SEIDEL;
        out->extra.sa.near_null_space     = nullptr;
        out->extra.sa.near_null_space_cols = 0;
        break;
    case AMGCL_PRECOND_ILU0:
        out->extra.ilu0 = amgcl_ilu0_extra_t{};
        break;
    case AMGCL_PRECOND_SCHUR:
        out->extra.schur.pmask         = nullptr;
        out->extra.schur.type          = 1;
        out->extra.schur.approx_schur  = 0;
        out->extra.schur.usolver_relax = AMGCL_RELAX_ILU0; /* Stokes-shaped default */
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

/* Builds one of the three SA-relaxation variants (also reused for Schur's
 * USolver) and wires its closures. Shared by both call sites below so the
 * relax-kind switch isn't duplicated. */
template <class GS, class ILU0, class SPAI0>
static bool build_amg_variant(amgcl_precond *h, amgcl_relax_kind_t relax,
                               const MatrixTuple &A,
                               const typename GS::params &pp_gs,
                               const typename ILU0::params &pp_ilu0,
                               const typename SPAI0::params &pp_spai0) {
    try {
        switch (relax) {
        case AMGCL_RELAX_GAUSS_SEIDEL: { auto P = std::make_shared<GS>(A, pp_gs);       wire_closures(h, P); return true; }
        case AMGCL_RELAX_ILU0:         { auto P = std::make_shared<ILU0>(A, pp_ilu0);   wire_closures(h, P); return true; }
        case AMGCL_RELAX_SPAI0:        { auto P = std::make_shared<SPAI0>(A, pp_spai0); wire_closures(h, P); return true; }
        default: return false;
        }
    } catch (...) {
        return false;
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

    try {
        switch (p.kind) {
        case AMGCL_PRECOND_SA: {
            int block_size = p.extra.sa.block_size;
            if (block_size < 1) block_size = 1;
            if (block_size > 4) block_size = 4;

            /* Same params struct shape across all three relaxations
             * (coarsening/nullspace config lives in amg<>::params
             * regardless of the relaxation template parameter), so build
             * it once and reuse for whichever variant gets selected. */
            SA_GaussSeidel::params pp;
            pp.coarsening.aggr.block_size = block_size;
            pp.coarse_enough              = p.extra.sa.coarse_enough;

            if (p.extra.sa.near_null_space && p.extra.sa.near_null_space_cols > 0) {
                std::size_t total = static_cast<std::size_t>(n) *
                                     static_cast<std::size_t>(p.extra.sa.near_null_space_cols);
                pp.coarsening.nullspace.B.assign(
                    p.extra.sa.near_null_space, p.extra.sa.near_null_space + total);
                pp.coarsening.nullspace.cols = p.extra.sa.near_null_space_cols;
            }

            /* params structs are identically-shaped across the three
             * relaxation instantiations (relaxation-specific fields all
             * default fine here), so the same `pp` values transfer as-is
             * via the templated fields that exist on every variant. */
            SA_ILU0::params  pp_ilu0;
            pp_ilu0.coarsening = pp.coarsening;
            pp_ilu0.coarse_enough = pp.coarse_enough;
            SA_SPAI0::params pp_spai0;
            pp_spai0.coarsening = pp.coarsening;
            pp_spai0.coarse_enough = pp.coarse_enough;

            bool ok = build_amg_variant<SA_GaussSeidel, SA_ILU0, SA_SPAI0>(
                h, p.extra.sa.relax, make_A(), pp, pp_ilu0, pp_spai0);
            if (!ok) { delete h; return nullptr; }
            break;
        }
        case AMGCL_PRECOND_ILU0: {
            Precond_ILU0::params pp;
            auto P = std::make_shared<Precond_ILU0>(make_A(), pp);
            wire_closures(h, P);
            break;
        }
        case AMGCL_PRECOND_SCHUR: {
            if (!p.extra.schur.pmask) { delete h; return nullptr; }

            std::vector<char> pmask(p.extra.schur.pmask, p.extra.schur.pmask + n);
            int  schur_type   = p.extra.schur.type ? p.extra.schur.type : 1;
            bool approx_schur = p.extra.schur.approx_schur != 0;
            bool adjust_p     = p.extra.schur.adjust_p != 0;


            auto A = make_A();
            bool ok = false;
            switch (p.extra.schur.usolver_relax) {
            case AMGCL_RELAX_GAUSS_SEIDEL: {
                Precond_Schur_GS::params pp;
                pp.pmask = pmask; pp.type = schur_type; pp.approx_schur = approx_schur; pp.adjust_p = adjust_p;

                auto P = std::make_shared<Precond_Schur_GS>(A, pp);
                wire_closures(h, P);
                ok = true;
                break;
            }
            case AMGCL_RELAX_ILU0: {
                Precond_Schur_ILU0::params pp;
               pp.pmask = pmask; pp.type = schur_type; pp.approx_schur = approx_schur; pp.adjust_p = adjust_p;
                auto P = std::make_shared<Precond_Schur_ILU0>(A, pp);
                wire_closures(h, P);
                ok = true;
                break;
            }
            case AMGCL_RELAX_SPAI0: {
                Precond_Schur_SPAI0::params pp;
                pp.pmask = pmask; pp.type = schur_type; pp.approx_schur = approx_schur; pp.adjust_p = adjust_p;
                auto P = std::make_shared<Precond_Schur_SPAI0>(A, pp);
                wire_closures(h, P);
                ok = true;
                break;
            }
            default: break;
            }
            if (!ok) { delete h; return nullptr; }
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
