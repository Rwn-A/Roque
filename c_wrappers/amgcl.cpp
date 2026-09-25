// C binding for amgcl with AMGCL_NO_BOOST. See amgcl_c.h for the interface.
//
// amgcl's own runtime wrappers (amgcl/*/runtime.hpp) need Boost only because
// their params type is a property tree; the dispatch itself doesn't. This file
// provides the same four wrappers (relaxation, coarsening, preconditioner,
// solver) with typed params instead, using std::variant for the dispatch.
// Adding an option to an axis means adding one type to that axis's list and
// one line to the matching C struct mapping.

#define AMGCL_NO_BOOST

#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#include <amgcl/amg.hpp>

#include <amgcl/coarsening/ruge_stuben.hpp>
#include <amgcl/coarsening/aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/coarsening/smoothed_aggr_emin.hpp>

#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/relaxation/ilu0.hpp>
#include <amgcl/relaxation/ilu0_chow_patel.hpp>
#include <amgcl/relaxation/iluk.hpp>
#include <amgcl/relaxation/ilup.hpp>
#include <amgcl/relaxation/ilut.hpp>
#include <amgcl/relaxation/damped_jacobi.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/spai1.hpp>
#include <amgcl/relaxation/chebyshev.hpp>
#include <amgcl/relaxation/as_preconditioner.hpp>

#include <amgcl/preconditioner/dummy.hpp>

#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/solver/bicgstabl.hpp>
#include <amgcl/solver/gmres.hpp>
#include <amgcl/solver/lgmres.hpp>
#include <amgcl/solver/fgmres.hpp>
#include <amgcl/solver/idrs.hpp>
#include <amgcl/solver/richardson.hpp>
#include <amgcl/solver/preonly.hpp>

#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>

#include "amgcl_c.h"

namespace amgcl_c {

using Backend = amgcl::backend::builtin<double>;

//===========================================================================
// Generic "one of these types, picked at runtime" machinery
//===========================================================================

// A runtime choice between Ts. The handle owns one constructed alternative;
// params holds settings for every alternative plus the index to build.
template <class... Ts>
struct choice {
    using handle = std::variant<std::unique_ptr<Ts>...>;

    struct params {
        std::size_t type = 0;
        std::tuple<typename Ts::params...> of;
    };

    template <std::size_t I>
    using at = std::tuple_element_t<I, std::tuple<Ts...>>;

    // Builds alternative number `type`: make(integral_constant<I>) must return
    // unique_ptr<at<I>>. One branch is instantiated per alternative.
    template <std::size_t I = 0, class Make>
    static handle build(const char *what, std::size_t type, Make &&make) {
        if constexpr (I < sizeof...(Ts)) {
            if (type == I)
                return handle(std::in_place_index<I>, make(std::integral_constant<std::size_t, I>{}));
            return build<I + 1>(what, type, std::forward<Make>(make));
        } else {
            throw std::invalid_argument(std::string("invalid ") + what + " " + std::to_string(type) +
                                        " (valid: 0.." + std::to_string(sizeof...(Ts) - 1) + ")");
        }
    }
};

template <class H, class F>
decltype(auto) visit_ptr(H &h, F &&f) {
    return std::visit([&](auto &p) -> decltype(auto) { return f(*p); }, h);
}

//===========================================================================
// Relaxation. Order matches amgcl_relax_type_t.
//===========================================================================

template <class B>
struct any_relax {
    using C = choice<
        amgcl::relaxation::gauss_seidel<B>,
        amgcl::relaxation::ilu0<B>,
        amgcl::relaxation::ilu0_chow_patel<B>,
        amgcl::relaxation::iluk<B>,
        amgcl::relaxation::ilup<B>,
        amgcl::relaxation::ilut<B>,
        amgcl::relaxation::damped_jacobi<B>,
        amgcl::relaxation::spai0<B>,
        amgcl::relaxation::spai1<B>,
        amgcl::relaxation::chebyshev<B>>;

    using params         = typename C::params;
    using backend_params = typename B::params;

    typename C::handle h;

    template <class Matrix>
    any_relax(const Matrix &A, const params &p, const backend_params &bp)
        : h(C::build("relaxation type", p.type, [&](auto I) {
              using T = typename C::template at<decltype(I)::value>;
              return std::make_unique<T>(A, std::get<decltype(I)::value>(p.of), bp);
          })) {}

    template <class M, class R, class X, class T>
    void apply_pre(const M &A, const R &rhs, X &x, T &tmp) const {
        visit_ptr(h, [&](auto &r) { r.apply_pre(A, rhs, x, tmp); });
    }
    template <class M, class R, class X, class T>
    void apply_post(const M &A, const R &rhs, X &x, T &tmp) const {
        visit_ptr(h, [&](auto &r) { r.apply_post(A, rhs, x, tmp); });
    }
    template <class M, class R, class X>
    void apply(const M &A, const R &rhs, X &x) const {
        visit_ptr(h, [&](auto &r) { r.apply(A, rhs, x); });
    }
    std::size_t bytes() const {
        return visit_ptr(h, [](auto &r) { return amgcl::backend::bytes(r); });
    }
};

//===========================================================================
// Coarsening. Order matches amgcl_coarsening_type_t.
//===========================================================================

template <class B>
struct any_coarsening {
    using C = choice<
        amgcl::coarsening::ruge_stuben<B>,
        amgcl::coarsening::aggregation<B>,
        amgcl::coarsening::smoothed_aggregation<B>,
        amgcl::coarsening::smoothed_aggr_emin<B>>;

    using params = typename C::params;

    typename C::handle h;

    explicit any_coarsening(const params &p)
        : h(C::build("coarsening type", p.type, [&](auto I) {
              using T = typename C::template at<decltype(I)::value>;
              return std::make_unique<T>(std::get<decltype(I)::value>(p.of));
          })) {}

    template <class Matrix>
    std::tuple<std::shared_ptr<Matrix>, std::shared_ptr<Matrix>>
    transfer_operators(const Matrix &A) {
        return visit_ptr(h, [&](auto &c) { return c.transfer_operators(A); });
    }

    template <class Matrix>
    std::shared_ptr<Matrix>
    coarse_operator(const Matrix &A, const Matrix &P, const Matrix &R) const {
        return visit_ptr(h, [&](auto &c) { return c.coarse_operator(A, P, R); });
    }
};

//===========================================================================
// Shell preconditioner: x = M^-1 rhs computed by a user C callback. Keeps its
// own copy of the system matrix, which the Krylov solver needs for spmv.
//===========================================================================

template <class B>
struct shell_precond {
    using backend_type   = B;
    using value_type     = typename B::value_type;
    using matrix         = typename B::matrix;
    using vector         = typename B::vector;
    using backend_params = typename B::params;
    using build_matrix   = typename amgcl::backend::builtin<value_type>::matrix;

    struct params {
        amgcl_shell_apply_fn apply = nullptr;
        void                *ctx   = nullptr;
    };

    std::shared_ptr<matrix> A;
    params prm;

    template <class Matrix>
    shell_precond(const Matrix &M, const params &p, const backend_params &bp)
        : A(B::copy_matrix(std::make_shared<build_matrix>(M), bp)), prm(p)
    {
        if (!prm.apply) throw std::invalid_argument("shell preconditioner has no apply callback");
    }

    // Krylov solvers call this with backend vectors (numa_vector for builtin);
    // amgcl_precond_apply calls it with iterator ranges. Both are
    // contiguous and indexable, so &v[0] is the raw pointer in either case.
    template <class V1, class V2>
    void apply(const V1 &rhs, V2 &x) const {
        const std::size_t n = amgcl::backend::rows(*A);
        if (n == 0) return;
        int rc = prm.apply(prm.ctx, static_cast<int>(n), &rhs[0], &x[0]);
        if (rc != 0)
            throw std::runtime_error("shell preconditioner callback returned " + std::to_string(rc));
    }

    std::shared_ptr<matrix> system_matrix_ptr() const { return A; }
    const matrix &system_matrix() const { return *A; }
    std::size_t bytes() const { return amgcl::backend::bytes(*A); }

    friend std::ostream &operator<<(std::ostream &os, const shell_precond &p) {
        return os << "Shell preconditioner (user callback)\n"
                  << "  Unknowns: " << amgcl::backend::rows(*p.A) << "\n"
                  << "  Nonzeros: " << amgcl::backend::nonzeros(*p.A) << "\n";
    }
};

//===========================================================================
// Preconditioner class. Order matches amgcl_precond_class_t.
//===========================================================================

template <class B>
struct any_precond {
    using C = choice<
        amgcl::amg<B, any_coarsening, any_relax>,
        amgcl::relaxation::as_preconditioner<B, any_relax>,
        amgcl::preconditioner::dummy<B>,
        shell_precond<B>>;

    using backend_type   = B;
    using value_type     = typename B::value_type;
    using matrix         = typename B::matrix;
    using vector         = typename B::vector;
    using backend_params = typename B::params;
    using params         = typename C::params;

    typename C::handle h;

    template <class Matrix>
    any_precond(const Matrix &A, const params &p = params(),
                const backend_params &bp = backend_params())
        : h(C::build("preconditioner class", p.type, [&](auto I) {
              using T = typename C::template at<decltype(I)::value>;
              return std::make_unique<T>(A, std::get<decltype(I)::value>(p.of), bp);
          })) {}

    template <class V1, class V2>
    void apply(const V1 &rhs, V2 &x) const {
        visit_ptr(h, [&](auto &P) { P.apply(rhs, x); });
    }
    std::shared_ptr<matrix> system_matrix_ptr() const {
        return visit_ptr(h, [](auto &P) { return P.system_matrix_ptr(); });
    }
    const matrix &system_matrix() const { return *system_matrix_ptr(); }
    std::size_t size() const { return amgcl::backend::rows(system_matrix()); }
    std::size_t bytes() const {
        return visit_ptr(h, [](auto &P) { return amgcl::backend::bytes(P); });
    }
    friend std::ostream &operator<<(std::ostream &os, const any_precond &p) {
        return visit_ptr(p.h, [&](auto &P) -> std::ostream & { return os << P; });
    }
};

//===========================================================================
// Krylov solver. Order matches amgcl_solver_type_t.
//===========================================================================

template <class B>
struct any_solver {
    using C = choice<
        amgcl::solver::cg<B>,
        amgcl::solver::bicgstab<B>,
        amgcl::solver::bicgstabl<B>,
        amgcl::solver::gmres<B>,
        amgcl::solver::lgmres<B>,
        amgcl::solver::fgmres<B>,
        amgcl::solver::idrs<B>,
        amgcl::solver::richardson<B>,
        amgcl::solver::preonly<B>>;

    using backend_type   = B;
    using value_type     = typename B::value_type;
    using scalar_type    = typename amgcl::math::scalar_of<value_type>::type;
    using backend_params = typename B::params;
    using params         = typename C::params;

    typename C::handle h;

    any_solver(std::size_t n, const params &p = params(),
               const backend_params &bp = backend_params())
        : h(C::build("solver type", p.type, [&](auto I) {
              using T = typename C::template at<decltype(I)::value>;
              return std::make_unique<T>(n, std::get<decltype(I)::value>(p.of), bp);
          })) {}

    template <class Matrix, class Precond, class V1, class V2>
    std::tuple<std::size_t, scalar_type>
    operator()(const Matrix &A, const Precond &P, const V1 &rhs, V2 &&x) const {
        return visit_ptr(h, [&](auto &s) {
            return std::tuple<std::size_t, scalar_type>(s(A, P, rhs, x));
        });
    }
    template <class Precond, class V1, class V2>
    std::tuple<std::size_t, scalar_type>
    operator()(const Precond &P, const V1 &rhs, V2 &&x) const {
        return (*this)(P.system_matrix(), P, rhs, x);
    }
    std::size_t bytes() const {
        return visit_ptr(h, [](auto &s) { return amgcl::backend::bytes(s); });
    }
    friend std::ostream &operator<<(std::ostream &os, const any_solver &s) {
        return visit_ptr(s.h, [&](auto &S) -> std::ostream & { return os << S; });
    }
};

//===========================================================================
// The two runtime-dispatched types everything goes through: the handle owns a
// Precond, and each solve builds a Solver around it.
//===========================================================================

using Precond = any_precond<Backend>;
using Solver  = any_solver<Backend>;

//===========================================================================
// C struct <-> C++ params. Each mapping is written once and runs in both
// directions: to_c = true fills the C struct from amgcl's defaults,
// to_c = false fills amgcl params from the C struct.
//===========================================================================

template <class C, class P>
void x(bool to_c, C &c, P &p) {
    if (to_c) c = static_cast<C>(p);
    else      p = static_cast<P>(c);
}
template <class P>
void xb(bool to_c, int &c, P &p) {  // C int <-> C++ bool
    if (to_c) c = p ? 1 : 0;
    else      p = (c != 0);
}
template <class E>
void xe(bool to_c, E &c, std::size_t &p) {  // C enum <-> variant index
    // C callers may store any int in an enum field, and loading an
    // out-of-range value as a C++ enum is undefined. Read the raw integer
    // instead, so bad values reach choice::build and become an error message.
    static_assert(sizeof(E) == sizeof(int), "C enums are expected to be int-sized");
    if (to_c) {
        c = static_cast<E>(p);
    } else {
        int raw;
        std::memcpy(&raw, &c, sizeof raw);
        p = raw < 0 ? static_cast<std::size_t>(-1) : static_cast<std::size_t>(raw);
    }
}

template <class P>
void map_ilu_solve(bool to_c, amgcl_ilu_solve_t &c, P &p) { xb(to_c, c.serial, p.serial); }

void map_relax(bool to_c, amgcl_relax_params_t &c, any_relax<Backend>::params &p) {
    xe(to_c, c.type, p.type);
    auto &o = p.of;
    xb(to_c, c.gauss_seidel.serial, std::get<0>(o).serial);

    x (to_c, c.ilu0.damping, std::get<1>(o).damping);
    map_ilu_solve(to_c, c.ilu0.solve, std::get<1>(o).solve);

    auto &cp = std::get<2>(o);
    x (to_c, c.ilu0_chow_patel.damping, cp.damping);
    x (to_c, c.ilu0_chow_patel.sweeps, cp.sweeps);
    x (to_c, c.ilu0_chow_patel.omega, cp.omega);
    xb(to_c, c.ilu0_chow_patel.symmetric_scaling, cp.symmetric_scaling);
    map_ilu_solve(to_c, c.ilu0_chow_patel.solve, cp.solve);

    x (to_c, c.iluk.k, std::get<3>(o).k);
    x (to_c, c.iluk.damping, std::get<3>(o).damping);
    map_ilu_solve(to_c, c.iluk.solve, std::get<3>(o).solve);

    x (to_c, c.ilup.k, std::get<4>(o).k);
    x (to_c, c.ilup.damping, std::get<4>(o).damping);
    map_ilu_solve(to_c, c.ilup.solve, std::get<4>(o).solve);

    x (to_c, c.ilut.p, std::get<5>(o).p);
    x (to_c, c.ilut.tau, std::get<5>(o).tau);
    x (to_c, c.ilut.damping, std::get<5>(o).damping);
    map_ilu_solve(to_c, c.ilut.solve, std::get<5>(o).solve);

    x (to_c, c.damped_jacobi.damping, std::get<6>(o).damping);

    auto &ch = std::get<9>(o);
    x (to_c, c.chebyshev.degree, ch.degree);
    x (to_c, c.chebyshev.higher, ch.higher);
    x (to_c, c.chebyshev.lower, ch.lower);
    x (to_c, c.chebyshev.power_iters, ch.power_iters);
    xb(to_c, c.chebyshev.scale, ch.scale);
}

// Aggregation-family settings shared across the three aggregation coarsenings.
template <class P>
void map_aggr(bool to_c, amgcl_coarsening_params_t &c, P &p, std::size_t rows) {
    x(to_c, c.aggr.block_size, p.aggr.block_size);
    x(to_c, c.aggr.eps_strong, p.aggr.eps_strong);
    if (to_c) {
        c.aggr.nullspace = nullptr;
        c.aggr.nullspace_cols = 0;
    } else if (c.aggr.nullspace && c.aggr.nullspace_cols > 0) {
        if (rows == 0) throw std::invalid_argument("nullspace given for an empty system");
        p.nullspace.cols = c.aggr.nullspace_cols;
        p.nullspace.B.assign(c.aggr.nullspace,
                             c.aggr.nullspace + rows * static_cast<std::size_t>(c.aggr.nullspace_cols));
    }
}

void map_coarsening(bool to_c, amgcl_coarsening_params_t &c,
                    any_coarsening<Backend>::params &p, std::size_t rows) {
    xe(to_c, c.type, p.type);
    auto &o = p.of;

    auto &rs = std::get<0>(o);
    x (to_c, c.ruge_stuben.eps_strong, rs.eps_strong);
    xb(to_c, c.ruge_stuben.do_trunc, rs.do_trunc);
    x (to_c, c.ruge_stuben.eps_trunc, rs.eps_trunc);

    // The C struct has one shared `aggr` block. Reading defaults takes it from
    // smoothed aggregation; writing copies it into all three.
    if (to_c) {
        map_aggr(true, c, std::get<2>(o), rows);
    } else {
        map_aggr(false, c, std::get<1>(o), rows);
        map_aggr(false, c, std::get<2>(o), rows);
        map_aggr(false, c, std::get<3>(o), rows);
    }

    x (to_c, c.aggregation.over_interp, std::get<1>(o).over_interp);

    auto &sa = std::get<2>(o);
    x (to_c, c.smoothed_aggregation.relax, sa.relax);
    xb(to_c, c.smoothed_aggregation.estimate_spectral_radius, sa.estimate_spectral_radius);
    x (to_c, c.smoothed_aggregation.power_iters, sa.power_iters);
}

void map_precond(bool to_c, amgcl_precond_params_t &c, Precond::params &p, std::size_t rows) {
    xe(to_c, c.cls, p.type);

    auto &a = std::get<0>(p.of);
    map_coarsening(to_c, c.amg.coarsening, a.coarsening, rows);
    map_relax(to_c, c.amg.relax, a.relax);
    x (to_c, c.amg.coarse_enough, a.coarse_enough);
    xb(to_c, c.amg.direct_coarse, a.direct_coarse);
    x (to_c, c.amg.max_levels, a.max_levels);
    x (to_c, c.amg.npre, a.npre);
    x (to_c, c.amg.npost, a.npost);
    x (to_c, c.amg.ncycle, a.ncycle);
    x (to_c, c.amg.pre_cycles, a.pre_cycles);

    map_relax(to_c, c.relaxation, std::get<1>(p.of));

    auto &sh = std::get<3>(p.of);
    if (to_c) { c.shell.apply = nullptr; c.shell.ctx = nullptr; }
    else      { sh.apply = c.shell.apply; sh.ctx = c.shell.ctx; }
}

// Every Krylov params struct except preonly's has these five fields.
template <class P>
void map_common(bool to_c, amgcl_solver_params_t &c, P &p) {
    if constexpr (!std::is_same_v<P, amgcl::detail::empty_params>) {
        x (to_c, c.tol, p.tol);
        x (to_c, c.abstol, p.abstol);
        x (to_c, c.maxiter, p.maxiter);
        xb(to_c, c.ns_search, p.ns_search);
        xb(to_c, c.verbose, p.verbose);
    }
}

void map_solver(bool to_c, amgcl_solver_params_t &c, Solver::params &p) {
    xe(to_c, c.type, p.type);
    auto &o = p.of;

    // Common fields: read defaults from cg, write them into every type.
    if (to_c) map_common(true, c, std::get<0>(o));
    else std::apply([&](auto &...each) { (map_common(false, c, each), ...); }, o);

    xb(to_c, c.bicgstab.check_after, std::get<1>(o).check_after);

    auto &bl = std::get<2>(o);
    x (to_c, c.bicgstabl.L, bl.L);
    x (to_c, c.bicgstabl.delta, bl.delta);
    xb(to_c, c.bicgstabl.convex, bl.convex);

    x (to_c, c.gmres.M, std::get<3>(o).M);

    auto &lg = std::get<4>(o);
    x (to_c, c.lgmres.M, lg.M);
    x (to_c, c.lgmres.K, lg.K);
    xb(to_c, c.lgmres.always_reset, lg.always_reset);

    x (to_c, c.fgmres.M, std::get<5>(o).M);

    auto &id = std::get<6>(o);
    x (to_c, c.idrs.s, id.s);
    x (to_c, c.idrs.omega, id.omega);
    xb(to_c, c.idrs.smoothing, id.smoothing);
    xb(to_c, c.idrs.replacement, id.replacement);

    x (to_c, c.richardson.damping, std::get<7>(o).damping);
}

//===========================================================================
// Error handling and small helpers
//===========================================================================

thread_local std::string last_error;

// Runs f, converting any exception into AMGCL_ERROR + last_error.
template <class F>
amgcl_status_t guarded(F &&f) {
    try {
        last_error.clear();
        return f();
    } catch (const std::exception &e) {
        last_error = e.what();
    } catch (...) {
        last_error = "unknown C++ exception";
    }
    return AMGCL_ERROR;
}

template <class Ptr>
amgcl::iterator_range<Ptr> range(Ptr p, std::size_t n) {
    return amgcl::make_iterator_range(p, p + n);
}

auto csr_tuple(std::size_t n, const int *ptr, const int *col, const double *val) {
    return std::make_tuple(n, range(ptr, n + 1), range(col, ptr[n]), range(val, ptr[n]));
}

// amgcl's runtime-interface defaults: SA AMG with spai0, bicgstab.
Precond::params default_precond_params() {
    Precond::params p;
    p.type = 0;                                // amg
    std::get<0>(p.of).coarsening.type = 2;     // smoothed_aggregation
    std::get<0>(p.of).relax.type      = 7;     // spai0
    std::get<1>(p.of).type            = 7;     // spai0 for single-level relaxation
    return p;
}

Solver::params default_solver_params() {
    Solver::params p;
    p.type = 1;                                // bicgstab
    return p;
}

} // namespace amgcl_c

struct amgcl_precond {
    std::size_t n = 0;
    std::unique_ptr<amgcl_c::Precond> P;
};

using namespace amgcl_c;

namespace {

bool bad_args(const amgcl_precond *p, const double *rhs, const double *x) {
    if (p && p->P && rhs && x) return false;
    last_error = "invalid argument";
    return true;
}

// One Krylov solve against matrix A (the preconditioner's own, or a
// replacement) with a solver built from the per-call params.
template <class Matrix>
amgcl_status_t run_solve(const amgcl_precond *p, const Matrix &A,
                         const double *rhs, double *x,
                         const amgcl_solver_params_t *sp_in, amgcl_conv_info_t *info) {
    amgcl_solver_params_t c;
    if (sp_in) c = *sp_in; else amgcl_solver_params_default(&c);

    Solver::params prm;
    map_solver(false, c, prm);
    Solver S(p->n, prm);

    auto X = range(x, p->n);
    std::size_t iters;
    double resid;
    std::tie(iters, resid) = S(A, *p->P, range(rhs, p->n), X);

    if (info) {
        info->iterations = static_cast<int>(iters);
        info->residual   = resid;
    }
    if (std::isnan(resid)) {
        last_error = "solve produced NaN";
        return AMGCL_ERROR;
    }
    if (c.type == AMGCL_SOLVER_PREONLY) return AMGCL_OK;
    // amgcl stops at max(tol*|b|, abstol) or at maxiter; running out of
    // iterations above tol is the only non-converged outcome.
    bool converged = resid <= c.tol || iters < c.maxiter;
    return converged ? AMGCL_OK : AMGCL_NOT_CONVERGED;
}

} // namespace

extern "C" {

const char *amgcl_last_error(void) { return last_error.c_str(); }

void amgcl_precond_params_default(amgcl_precond_params_t *out) {
    if (!out) return;
    *out = amgcl_precond_params_t{};
    Precond::params p = default_precond_params();
    map_precond(true, *out, p, 0);
}

void amgcl_solver_params_default(amgcl_solver_params_t *out) {
    if (!out) return;
    *out = amgcl_solver_params_t{};
    Solver::params p = default_solver_params();
    map_solver(true, *out, p);
}

amgcl_precond_t *amgcl_precond_create(int n, const int *row_ptr, const int *col_ind,
                                      const double *values, const amgcl_precond_params_t *params) {
    if (n <= 0 || !row_ptr || !col_ind || !values) {
        last_error = "invalid argument: need n > 0 and non-null CSR arrays";
        return nullptr;
    }
    amgcl_precond_params_t c;
    if (params) c = *params; else amgcl_precond_params_default(&c);

    amgcl_precond *result = nullptr;
    guarded([&] {
        const std::size_t N = static_cast<std::size_t>(n);
        Precond::params prm;
        map_precond(false, c, prm, N);

        auto h = std::make_unique<amgcl_precond>();
        h->n = N;
        h->P = std::make_unique<Precond>(csr_tuple(N, row_ptr, col_ind, values), prm);
        result = h.release();
        return AMGCL_OK;
    });
    return result;
}

void amgcl_precond_destroy(amgcl_precond_t *p) { delete p; }

amgcl_status_t amgcl_precond_apply(const amgcl_precond_t *p, const double *rhs, double *x) {
    if (bad_args(p, rhs, x)) return AMGCL_ERROR;
    return guarded([&] {
        auto X = range(x, p->n);
        p->P->apply(range(rhs, p->n), X);
        return AMGCL_OK;
    });
}

void amgcl_precond_print(const amgcl_precond_t *p) {
    if (!p || !p->P) return;
    try { std::cout << *p->P << std::endl; } catch (...) {}
}

int amgcl_precond_size(const amgcl_precond_t *p) { return p ? static_cast<int>(p->n) : 0; }

amgcl_status_t amgcl_solve(const amgcl_precond_t *p, const double *rhs, double *x,
                           const amgcl_solver_params_t *sp, amgcl_conv_info_t *info) {
    if (bad_args(p, rhs, x)) return AMGCL_ERROR;
    return guarded([&] { return run_solve(p, p->P->system_matrix(), rhs, x, sp, info); });
}

amgcl_status_t amgcl_solve_with(const amgcl_precond_t *p,
                                const int *row_ptr, const int *col_ind, const double *values,
                                const double *rhs, double *x,
                                const amgcl_solver_params_t *sp, amgcl_conv_info_t *info) {
    if (bad_args(p, rhs, x)) return AMGCL_ERROR;
    if (!row_ptr || !col_ind || !values) { last_error = "invalid argument"; return AMGCL_ERROR; }
    return guarded([&] {
        // The builtin backend's Krylov kernels want its own CSR type.
        Backend::matrix A(csr_tuple(p->n, row_ptr, col_ind, values));
        return run_solve(p, A, rhs, x, sp, info);
    });
}

} // extern "C"
