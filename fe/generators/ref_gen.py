#!/usr/bin/env python3
"""
Reference element table generator -> reference_tables.odin

    python3 gen_ref.py [out_path]          # build, self-test, emit

Pipeline per (cell, family, order):
  space (monomial spanning set)  ->  dof functionals (entity blocks, quadrature form)
  -> V = l_i(span_b)  ->  C = V^-T  ->  monomial coefficients  ->  eval procs
  -> orientations: rebuild each entity block from the canonically ordered vertex list, A = l~(phi), M = A^-T

Self-tests (all run before emitting): quadrature exactness, dof counts, duality, emitted-code evaluation,
orientation consistency across entities, two-cell patch test (trace continuity under every orientation).
"""
import itertools
import math
import sys
from fractions import Fraction

import numpy as np
from numpy.polynomial import legendre as npleg

TOL = 1e-10
MAXD = 4  # max per-direction monomial exponent carried
QSETS = [("Q1", 1), ("Q3", 3), ("Q5", 5), ("Q7", 7)]
ORDERS = ["O0", "O1", "O2", "O3"]
FAMILIES = ["Lagrange", "Raviart_Thomas", "Nedelec"]
ELEMENT_TYPES = ["Point", "Line", "Tri", "Quad", "Hex", "Tet"]
DIMS = ["D0", "D1", "D2", "D3"]
QUANTITIES = ["S_Val", "V_Val", "S_Grd", "V_Div", "V_Curl"]
FAMILY_QUANTITIES = {"Lagrange": ["S_Val", "S_Grd"], "Raviart_Thomas": ["V_Val", "V_Div"], "Nedelec": ["V_Val", "V_Curl"]}
VEC_ORDERS = [0, 1, 2]  # RT / Nedelec O0..O2 (k = 1..3)

# ============================================================================ cells

def _hex_verts():
    return [(-1.0 + 2 * (i & 1), -1.0 + 2 * ((i >> 1) & 1), -1.0 + 2 * ((i >> 2) & 1)) for i in range(8)]

CELLS = {
    "Point": dict(dim=0, verts=[(0.0, 0.0, 0.0)], ents={}),
    "Line": dict(dim=1, verts=[(-1.0, 0, 0), (1.0, 0, 0)], ents={}),
    "Tri": dict(dim=2, verts=[(0.0, 0, 0), (1.0, 0, 0), (0.0, 1, 0)], ents={1: [[1, 2], [0, 2], [0, 1]]}),
    "Quad": dict(dim=2, verts=[(-1.0, -1, 0), (1.0, -1, 0), (-1.0, 1, 0), (1.0, 1, 0)],
                 ents={1: [[0, 1], [0, 2], [1, 3], [2, 3]]}),
    "Tet": dict(dim=3, verts=[(0.0, 0, 0), (1.0, 0, 0), (0.0, 1, 0), (0.0, 0, 1)],
                ents={1: [[2, 3], [1, 3], [1, 2], [0, 3], [0, 2], [0, 1]],
                      2: [[1, 2, 3], [0, 2, 3], [0, 1, 3], [0, 1, 2]]}),
    "Hex": dict(dim=3, verts=_hex_verts(),
                ents={1: [[0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3], [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7]],
                      2: [[0, 1, 2, 3], [0, 1, 4, 5], [0, 2, 4, 6], [1, 3, 5, 7], [2, 3, 6, 7], [4, 5, 6, 7]]}),
}
for _c in CELLS.values():
    _c["verts"] = np.array(_c["verts"], dtype=float)
    _c["ents"][0] = [[i] for i in range(len(_c["verts"]))]
    _c["ents"][_c["dim"]] = [list(range(len(_c["verts"])))]

SIMPLEX = {"Tri", "Tet"}
TENSOR = {"Line", "Quad", "Hex"}


def ent_type(dim, nverts):
    return {0: "Point", 1: "Line", 2: "Tri" if nverts == 3 else "Quad", 3: "Tet" if nverts == 4 else "Hex"}[dim]


def cell_ent_type(cell, d, e):
    return ent_type(d, len(CELLS[cell]["ents"][d][e]))


def closure(cell, d, e):
    """[Dimension] -> parent indices of the entity's own sub-entities, in the entity's reference order.
    closure[dd][k] = parent index of the entity's k-th sub-entity of dim dd. D0 = local vertex order, own dim = {e}."""
    ents = CELLS[cell]["ents"]
    vs = ents[d][e]
    et = ent_type(d, len(vs))
    out = [[] for _ in range(4)]
    out[0] = list(vs)
    for dd in range(1, d):
        for local in CELLS[et]["ents"][dd]:  # the entity type's own reference sub-entities
            want = {vs[v] for v in local}
            out[dd].append(next(i for i, ev in enumerate(ents[dd]) if set(ev) == want))
    out[d] = [e]
    return out


def p1(et, s):
    x, y, z = s[:, 0], s[:, 1], s[:, 2]
    if et == "Point":
        return np.ones((len(s), 1))
    if et == "Line":
        return np.stack([(1 - x) / 2, (1 + x) / 2], 1)
    if et == "Tri":
        return np.stack([1 - x - y, x, y], 1)
    if et == "Tet":
        return np.stack([1 - x - y - z, x, y, z], 1)
    if et == "Quad":
        return np.stack([(1 - x) * (1 - y) / 4, (1 + x) * (1 - y) / 4, (1 - x) * (1 + y) / 4, (1 + x) * (1 + y) / 4], 1)
    if et == "Hex":
        cols = []
        for i in range(8):
            sx, sy, sz = (2 * (i & 1) - 1), (2 * ((i >> 1) & 1) - 1), (2 * ((i >> 2) & 1) - 1)
            cols.append((1 + sx * x) * (1 + sy * y) * (1 + sz * z) / 8)
        return np.stack(cols, 1)
    raise ValueError(et)


def lift(et, vc, s):
    return p1(et, s) @ vc


def axes(et, vc):
    """Constant Jacobian columns of the entity -> parent map (unnormalized)."""
    if et == "Point":
        return []
    if et == "Line":
        return [(vc[1] - vc[0]) / 2]
    if et in SIMPLEX:
        return [vc[i] - vc[0] for i in range(1, len(vc))]
    if et == "Quad":
        return [(vc[1] - vc[0]) / 2, (vc[2] - vc[0]) / 2]
    if et == "Hex":
        return [(vc[1] - vc[0]) / 2, (vc[2] - vc[0]) / 2, (vc[4] - vc[0]) / 2]
    raise ValueError(et)


def facet_normals(cell):
    c = CELLS[cell]
    d = c["dim"]
    if d == 0:
        return []
    ctr = c["verts"].mean(0)
    out = []
    for fv in c["ents"][d - 1]:
        vc = c["verts"][fv]
        et = ent_type(d - 1, len(fv))
        if d == 1:
            n = np.array([1.0, 0, 0])
        elif d == 2:
            a = axes(et, vc)[0]
            n = np.array([a[1], -a[0], 0.0])
        else:
            a = axes(et, vc)
            n = np.cross(a[0], a[1])
        if np.dot(n, vc.mean(0) - ctr) < 0:
            n = -n
        out.append(n)
    return out


# orientation perms: perm[i] = position, in the canonical (global) order, of the entity's local vertex i
def _quad_perms():
    vc = CELLS["Quad"]["verts"][:, :2]
    mats = []
    for p in [(0, 1), (1, 0)]:
        for sx in (1, -1):
            for sy in (1, -1):
                m = np.zeros((2, 2))
                m[0, p[0]] = sx
                m[1, p[1]] = sy
                mats.append(m)
    # identity, rotations, then reflections
    mats.sort(key=lambda m: (not np.allclose(m, np.eye(2)), np.linalg.det(m) < 0))
    perms = []
    for m in mats:
        img = vc @ m.T
        perm = [int(np.argmin(np.linalg.norm(vc - img[i], axis=1))) for i in range(4)]
        perms.append(perm)
    return perms

PERMS = {
    "Line": [[0, 1], [1, 0]],
    "Tri": [[0, 1, 2], [1, 2, 0], [2, 0, 1], [0, 2, 1], [2, 1, 0], [1, 0, 2]],
    "Quad": _quad_perms(),
}

# ============================================================================ quadrature

def _gl(n):
    return npleg.leggauss(max(1, n))


def _n_for(deg):
    return max(1, math.ceil((deg + 1) / 2))


def quad_rule(et, deg):
    if et == "Point":
        return np.zeros((1, 3)), np.ones(1)
    if et in TENSOR:
        dim = {"Line": 1, "Quad": 2, "Hex": 3}[et]
        x, w = _gl(_n_for(deg))
        pts, wts = [], []
        for idx in itertools.product(range(len(x)), repeat=dim):
            idx = idx[::-1]  # x fastest
            p = [0.0, 0.0, 0.0]
            ww = 1.0
            for a, i in enumerate(idx):
                p[a] = x[i]
                ww *= w[i]
            pts.append(p)
            wts.append(ww)
        return np.array(pts), np.array(wts)

    def g01(n):
        x, w = _gl(n)
        return (x + 1) / 2, w / 2

    if et == "Tri":
        u, wu = g01(_n_for(deg + 1))
        v, wv = g01(_n_for(deg))
        pts, wts = [], []
        for i in range(len(u)):
            for j in range(len(v)):
                pts.append([u[i], v[j] * (1 - u[i]), 0.0])
                wts.append(wu[i] * wv[j] * (1 - u[i]))
        return np.array(pts), np.array(wts)
    if et == "Tet":
        u, wu = g01(_n_for(deg + 2))
        v, wv = g01(_n_for(deg + 1))
        t, wt = g01(_n_for(deg))
        pts, wts = [], []
        for i in range(len(u)):
            for j in range(len(v)):
                for k in range(len(t)):
                    pts.append([u[i], v[j] * (1 - u[i]), t[k] * (1 - u[i]) * (1 - v[j])])
                    wts.append(wu[i] * wv[j] * wt[k] * (1 - u[i]) ** 2 * (1 - v[j]))
        return np.array(pts), np.array(wts)
    raise ValueError(et)


def qset_for(deg):
    for name, d in QSETS:
        if d >= deg:
            return name, d
    raise RuntimeError(f"no quadrature set of degree {deg}")


# ============================================================================ polynomials

def monos_for(dim):
    out = []
    for e in itertools.product(range(MAXD + 1), repeat=3):
        if any(e[a] for a in range(dim, 3)):
            continue
        out.append(e)
    out.sort(key=lambda e: (sum(e), e[::-1]))
    return out


MONOS = {d: monos_for(d) for d in range(4)}
MIDX = {d: {e: i for i, e in enumerate(MONOS[d])} for d in range(4)}


def eval_monos(dim, pts):
    ms = np.array(MONOS[dim], dtype=float)
    return np.prod(pts[:, None, :] ** ms[None, :, :], axis=2)


def deriv_mat(dim, axis):
    n = len(MONOS[dim])
    D = np.zeros((n, n))
    for i, e in enumerate(MONOS[dim]):
        if e[axis] > 0:
            f = list(e)
            f[axis] -= 1
            D[MIDX[dim][tuple(f)], i] = e[axis]
    return D


def mono(dim, e):
    v = np.zeros(len(MONOS[dim]))
    v[MIDX[dim][tuple(e)]] = 1
    return v


def exps_total(dim, k):
    return [e for e in MONOS[dim] if sum(e) <= k] if k >= 0 else []


def exps_homog(dim, k):
    return [e for e in MONOS[dim] if sum(e) == k] if k >= 0 else []


def exps_aniso(dim, degs):
    if any(d < 0 for d in degs):
        return []
    return [e for e in MONOS[dim] if all(e[a] <= degs[a] for a in range(dim))]


def mul_axis(dim, poly, axis):
    out = np.zeros_like(poly)
    for i, e in enumerate(MONOS[dim]):
        if poly[i] != 0:
            f = list(e)
            f[axis] += 1
            out[MIDX[dim][tuple(f)]] += poly[i]
    return out


# ============================================================================ spaces

def space_span(cell, fam, order):
    """Spanning set, shape (n, vs, nm)."""
    dim = CELLS[cell]["dim"]
    nm = len(MONOS[dim])
    rows = []

    def vec(comps):
        return np.array(comps)

    zero = np.zeros(nm)
    if fam == "Lagrange":
        p = order
        exps = exps_total(dim, p) if cell in SIMPLEX else exps_aniso(dim, [p] * dim)
        for e in exps:
            rows.append(vec([mono(dim, e)]))
        return np.array(rows)
    k = order + 1
    if cell in SIMPLEX:
        for e in exps_total(dim, k - 1):
            for a in range(dim):
                comps = [zero] * dim
                comps[a] = mono(dim, e)
                rows.append(vec(comps))
        for e in exps_homog(dim, k - 1):
            p = mono(dim, e)
            if fam == "Raviart_Thomas":
                rows.append(vec([mul_axis(dim, p, a) for a in range(dim)]))
            elif dim == 2:
                rows.append(vec([-mul_axis(dim, p, 1), mul_axis(dim, p, 0)]))
            else:
                x, y, z = (mul_axis(dim, p, a) for a in range(3))
                rows.append(vec([zero, z, -y]))   # x cross e_x p
                rows.append(vec([-z, zero, x]))   # x cross e_y p
                rows.append(vec([y, -x, zero]))   # x cross e_z p
        return np.array(rows)
    # tensor cells
    for a in range(dim):
        if fam == "Raviart_Thomas":
            degs = [k - 1] * dim
            degs[a] = k
        else:
            degs = [k] * dim
            degs[a] = k - 1
        for e in exps_aniso(dim, degs):
            comps = [zero] * dim
            comps[a] = mono(dim, e)
            rows.append(vec(comps))
    return np.array(rows)


def reduce_span(S):
    n, vs, nm = S.shape
    flat = S.reshape(n, vs * nm)
    u, s, vt = np.linalg.svd(flat, full_matrices=False)
    r = int(np.sum(s > TOL * s[0]))
    return vt[:r].reshape(r, vs, nm)


def space_degree(cell, S):
    dim = CELLS[cell]["dim"]
    deg = 0
    for i, e in enumerate(MONOS[dim]):
        if np.any(np.abs(S[:, :, i]) > TOL):
            deg = max(deg, sum(e) if cell in SIMPLEX else max(e))
    return deg


# ============================================================================ dof specs

def leg(j):
    c = np.zeros(j + 1)
    c[j] = 1
    return c


def q_leg_line(k):
    return [((lambda s, j=j: npleg.legval(s[:, 0], leg(j))), j) for j in range(k)]


def q_leg_tensor(degs):
    if any(d < 0 for d in degs):
        return []
    out = []
    for idx in itertools.product(*[range(d + 1) for d in degs[::-1]]):
        idx = idx[::-1]  # first axis fastest

        def f(s, idx=idx):
            v = np.ones(len(s))
            for a, i in enumerate(idx):
                v = v * npleg.legval(s[:, a], leg(i))
            return v
        out.append((f, max(idx)))
    return out


def tri_lattice(m):
    if m == 0:
        return [(1 / 3, 1 / 3, 0.0)]
    pts = []
    for j in range(m + 1):
        for i in range(m + 1 - j):
            pts.append((i / m, j / m, 0.0))
    return pts


def q_tri_lagrange(m):
    if m < 0:
        return []
    pts = np.array(tri_lattice(m))
    ex = exps_total(2, m)
    Vm = eval_monos(2, pts)[:, [MIDX[2][e] for e in ex]]
    C = np.linalg.inv(Vm)  # columns: basis coefficients
    out = []
    for i in range(len(pts)):
        def f(s, i=i):
            return eval_monos(2, s)[:, [MIDX[2][e] for e in ex]] @ C[:, i]
        out.append((f, m))
    return out


def q_mono(dim, m):
    return [((lambda s, e=e: np.prod(s ** np.array(e, dtype=float), axis=1)), sum(e)) for e in exps_total(dim, m)]


def lagrange_nodes(et, p):
    if p == 0:
        return {"Point": [(0, 0, 0)], "Line": [(0, 0, 0)], "Tri": [(1 / 3, 1 / 3, 0)], "Quad": [(0, 0, 0)],
                "Tet": [(0.25, 0.25, 0.25)], "Hex": [(0, 0, 0)]}[et]
    if et == "Point":
        return [(0.0, 0.0, 0.0)]
    t = lambda i: -1 + 2 * i / p
    r = range(1, p)
    if et == "Line":
        return [(t(i), 0.0, 0.0) for i in r]
    if et == "Quad":
        return [(t(i), t(j), 0.0) for j in r for i in r]
    if et == "Hex":
        return [(t(i), t(j), t(k)) for k in r for j in r for i in r]
    if et == "Tri":
        return [(i / p, j / p, 0.0) for j in r for i in r if i + j <= p - 1]
    if et == "Tet":
        return [(i / p, j / p, k / p) for k in r for j in r for i in r if i + j + k <= p - 1]
    raise ValueError(et)


def dof_spec(cell, fam, order):
    """{dim: [per-entity spec]}. Lagrange spec: list of entity-local nodes. Moment spec: list of (dir, q, qdeg)."""
    c = CELLS[cell]
    cd = c["dim"]
    spec = {d: [[] for _ in c["ents"][d]] for d in range(cd + 1)}
    if fam == "Lagrange":
        if cell == "Point":
            spec[0] = [[(0.0, 0.0, 0.0)]]
            return spec
        if order == 0:
            spec[cd] = [lagrange_nodes(cell, 0)]
            return spec
        for d in range(cd + 1):
            for e in range(len(c["ents"][d])):
                spec[d][e] = lagrange_nodes(cell_ent_type(cell, d, e), order)
        return spec

    k = order + 1
    X = ["ex", "ey", "ez"]

    def put(d, items):
        for e in range(len(c["ents"][d])):
            spec[d][e] = list(items)

    if fam == "Raviart_Thomas":
        if cell == "Tri":
            put(1, [("nrm",) + q for q in q_leg_line(k)])
            put(2, [(X[a],) + q for a in range(2) for q in q_mono(2, k - 2)])
        elif cell == "Tet":
            put(2, [("nrm",) + q for q in q_tri_lagrange(k - 1)])
            put(3, [(X[a],) + q for a in range(3) for q in q_mono(3, k - 2)])
        elif cell == "Quad":
            put(1, [("nrm",) + q for q in q_leg_line(k)])
            put(2, [("ex",) + q for q in q_leg_tensor([k - 2, k - 1])] + [("ey",) + q for q in q_leg_tensor([k - 1, k - 2])])
        elif cell == "Hex":
            put(2, [("nrm",) + q for q in q_leg_tensor([k - 1, k - 1])])
            put(3, [("ex",) + q for q in q_leg_tensor([k - 2, k - 1, k - 1])]
                + [("ey",) + q for q in q_leg_tensor([k - 1, k - 2, k - 1])]
                + [("ez",) + q for q in q_leg_tensor([k - 1, k - 1, k - 2])])
    else:
        put(1, [("tan",) + q for q in q_leg_line(k)])
        if cell == "Tri":
            put(2, [(X[a],) + q for a in range(2) for q in q_mono(2, k - 2)])
        elif cell == "Tet":
            put(2, [("ax0",) + q for q in q_tri_lagrange(k - 2)] + [("ax1",) + q for q in q_tri_lagrange(k - 2)])
            put(3, [(X[a],) + q for a in range(3) for q in q_mono(3, k - 3)])
        elif cell == "Quad":
            put(2, [("ex",) + q for q in q_leg_tensor([k - 1, k - 2])] + [("ey",) + q for q in q_leg_tensor([k - 2, k - 1])])
        elif cell == "Hex":
            put(2, [("ax0",) + q for q in q_leg_tensor([k - 1, k - 2])] + [("ax1",) + q for q in q_leg_tensor([k - 2, k - 1])])
            put(3, [("ex",) + q for q in q_leg_tensor([k - 1, k - 2, k - 2])]
                + [("ey",) + q for q in q_leg_tensor([k - 2, k - 1, k - 2])]
                + [("ez",) + q for q in q_leg_tensor([k - 2, k - 2, k - 1])])
    return spec


def direction(name, et, vc, cd):
    ax = axes(et, vc)
    if name == "tan":
        return ax[0]
    if name == "nrm":
        if cd == 2:
            return np.array([ax[0][1], -ax[0][0], 0.0])
        return np.cross(ax[0], ax[1])
    if name == "ax0":
        return ax[0]
    if name == "ax1":
        return ax[1]
    return np.eye(3)[{"ex": 0, "ey": 1, "ez": 2}[name]]


class Block:
    """Functionals for one entity: rule in entity space, points lifted to parent, W[dof][pt][cmpnt]."""

    def __init__(self, et, set_name, pts, wts, xp, W):
        self.et, self.set_name, self.pts, self.wts, self.xp, self.W = et, set_name, pts, wts, xp, W

    def apply(self, cd, coef):
        """coef (nb, vs, nm) -> (ndof, nb)."""
        vals = np.einsum("pm,bcm->pbc", eval_monos(cd, self.xp), coef)
        return np.einsum("p,ipc,pbc->ib", self.wts, self.W, vals)


def make_block(cell, fam, espec, et, vlist, sdeg):
    c = CELLS[cell]
    cd = c["dim"]
    vc = c["verts"][vlist]
    vs = 1 if fam == "Lagrange" else cd
    if fam == "Lagrange":
        pts = np.array(espec, dtype=float).reshape(-1, 3)
        n = len(pts)
        W = np.eye(n)[:, :, None].copy()
        return Block(et, None, pts, np.ones(n), lift(et, vc, pts), W)
    qdeg = max(q[2] for q in espec)
    set_name, sdeg_set = qset_for(sdeg + qdeg)
    pts, wts = quad_rule(et, sdeg_set)
    W = np.zeros((len(espec), len(pts), vs))
    for i, (dname, qf, _) in enumerate(espec):
        W[i] = qf(pts)[:, None] * direction(dname, et, vc, cd)[None, :vs]
    return Block(et, set_name, pts, wts, lift(et, vc, pts), W)


# ============================================================================ element build

def quantity_coefs(cell, fam, coef):
    cd = CELLS[cell]["dim"]
    D = [deriv_mat(cd, a) for a in range(cd)]
    d = lambda comp, a: np.einsum("mn,jn->jm", D[a], coef[:, comp])
    out = {}
    if fam == "Lagrange":
        out["S_Val"] = coef
        if cd > 0:
            out["S_Grd"] = np.stack([d(0, a) for a in range(cd)], 1)
    elif fam == "Raviart_Thomas":
        out["V_Val"] = coef
        out["V_Div"] = sum(d(a, a) for a in range(cd))[:, None, :]
    else:
        out["V_Val"] = coef
        if cd == 2:
            out["V_Curl"] = (d(1, 0) - d(0, 1))[:, None, :]
        else:
            out["V_Curl"] = np.stack([d(2, 1) - d(1, 2), d(0, 2) - d(2, 0), d(1, 0) - d(0, 1)], 1)
    return out


class Element:
    pass


def classify(M):
    n = M.shape[0]
    if np.allclose(M, np.eye(n), atol=1e-9):
        return None
    nz = np.abs(M) > 1e-9
    if np.all(nz.sum(1) == 1) and np.all(nz.sum(0) == 1) and np.allclose(np.abs(M[nz]), 1, atol=1e-9):
        src = [int(np.argmax(nz[i])) for i in range(n)]
        sign = [float(np.round(M[i, src[i]])) for i in range(n)]
        return ("perm", src, sign)
    return ("dense", M, np.linalg.inv(M))


def transform_matrix(t, n):
    if t is None:
        return np.eye(n)
    if t[0] == "perm":
        M = np.zeros((n, n))
        for i, (s, g) in enumerate(zip(t[1], t[2])):
            M[i, s] = g
        return M
    return t[1]


def build(cell, fam, order):
    c = CELLS[cell]
    cd = c["dim"]
    el = Element()
    el.cell, el.fam, el.order, el.cd = cell, fam, order, cd
    el.vs = 1 if fam == "Lagrange" else cd
    S = reduce_span(space_span(cell, fam, order))
    sdeg = space_degree(cell, S)
    spec = dof_spec(cell, fam, order)

    el.dpe = [0, 0, 0, 0]
    for d in range(cd + 1):
        counts = {len(s) for s in spec[d]}
        assert len(counts) == 1, (cell, fam, order, d, counts)
        el.dpe[d] = counts.pop()
    el.n = sum(el.dpe[d] * len(c["ents"][d]) for d in range(cd + 1))
    assert S.shape[0] == el.n, f"{cell} {fam} O{order}: space dim {S.shape[0]} != dofs {el.n}"

    el.blocks = {d: [] for d in range(cd + 1)}
    el.start = {}
    off = 0
    for d in range(cd + 1):
        for e, vl in enumerate(c["ents"][d]):
            el.start[(d, e)] = off
            off += el.dpe[d]
            if el.dpe[d]:
                el.blocks[d].append(make_block(cell, fam, spec[d][e], cell_ent_type(cell, d, e), vl, sdeg))

    V = np.vstack([b.apply(cd, S) for d in range(cd + 1) for b in el.blocks[d]])
    C = np.linalg.inv(V).T
    el.coef = np.einsum("jb,bcm->jcm", C, S)
    el.coef[np.abs(el.coef) < 1e-13] = 0.0
    el.quant = quantity_coefs(cell, fam, el.coef)

    # duality
    L = np.vstack([b.apply(cd, el.coef) for d in range(cd + 1) for b in el.blocks[d]])
    err = np.abs(L - np.eye(el.n)).max()
    assert err < 1e-9, f"duality {cell} {fam} O{order}: {err}"
    # same, through the emitted Bvec form (contract_linear: sum_p sum_c B[p][j][c] v_c(x_p))
    Le = np.vstack([np.einsum("pic,pbc->ib", functional_bvec(b),
                              np.einsum("pm,bcm->pbc", eval_monos(cd, b.xp), el.coef))
                    for d in range(cd + 1) for b in el.blocks[d]])
    assert np.abs(Le - np.eye(el.n)).max() < 1e-9, f"emitted functional form {cell} {fam} O{order}"

    # orientations: rebuild each proper sub-entity from its canonical vertex order
    el.orient = {}
    for d in range(1, cd):
        if not el.dpe[d]:
            continue
        et = cell_ent_type(cell, d, 0)
        table = []
        for perm in PERMS[et]:
            pinv = [perm.index(j) for j in range(len(perm))]
            M0 = None
            for e, vl in enumerate(c["ents"][d]):
                can = [vl[pinv[j]] for j in range(len(vl))]
                blk = make_block(cell, fam, spec[d][e], et, can, sdeg)
                Af = blk.apply(cd, el.coef)
                s0, n = el.start[(d, e)], el.dpe[d]
                off_block = np.delete(Af, range(s0, s0 + n), axis=1)
                assert np.abs(off_block).max(initial=0) < 1e-9, f"orientation leaks {cell} {fam} O{order} d{d}"
                M = np.linalg.inv(Af[:, s0:s0 + n]).T
                if M0 is None:
                    M0 = M
                assert np.allclose(M, M0, atol=1e-9), f"orientation differs across entities {cell} {fam} O{order} d{d}"
            table.append(classify(M0))
        el.orient[et] = table if any(t is not None for t in table) else None
    return el


def defined(cell, fam, order):
    if fam == "Lagrange":
        return True
    return cell in ("Tri", "Quad", "Tet", "Hex") and order in VEC_ORDERS


# ============================================================================ code generation

def snap(v):
    v = float(v)
    if abs(v) < 1e-14:
        return 0.0
    fr = Fraction(v).limit_denominator(100000)
    if abs(float(fr) - v) < 1e-13 * max(1.0, abs(v)):
        return float(fr)
    return v


def fnum(v):
    v = snap(v)
    if v == 0:
        return "0"
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


def mono_factors(dim, e):
    names = "xyz"
    f = []
    for a in range(dim):
        if e[a] == 1:
            f.append(names[a])
        elif e[a] > 1:
            f.append(f"{names[a]}{e[a]}")
    return f


def poly_expr(dim, coefs):
    terms = []
    for i, cval in enumerate(coefs):
        cv = snap(cval)
        if cv == 0:
            continue
        f = mono_factors(dim, MONOS[dim][i])
        if not f:
            terms.append(fnum(cv))
        elif cv == 1:
            terms.append("*".join(f))
        elif cv == -1:
            terms.append("-" + "*".join(f))
        else:
            terms.append(fnum(cv) + "*" + "*".join(f))
    if not terms:
        return "0"
    s = terms[0]
    for t in terms[1:]:
        s += (" - " + t[1:]) if t.startswith("-") else (" + " + t)
    return s


def proc_body(dim, arr):
    """arr (ndof, ncmp, nm) -> (var_lines [(name, expr)], assignments [(idx, expr)])."""
    nd, nc, nm = arr.shape
    maxe = [0, 0, 0]
    for i, e in enumerate(MONOS[dim]):
        if np.any(np.abs(arr[:, :, i]) > 1e-14):
            for a in range(3):
                maxe[a] = max(maxe[a], e[a])
    base = [("xyz"[a], f"r[{a}]") for a in range(dim) if maxe[a] >= 1]
    pows = []
    for a in range(dim):
        v = "xyz"[a]
        for p in range(2, maxe[a] + 1):
            prev = v if p == 2 else f"{v}{p - 1}"
            pows.append((f"{v}{p}", f"{prev}*{v}"))
    assigns = [(j * nc + c, poly_expr(dim, arr[j, c])) for j in range(nd) for c in range(nc)]
    return base, pows, assigns


def py_eval_fn(dim, arr):
    base, pows, assigns = proc_body(dim, arr)
    src = ["def f(r):"]
    for n, e in base + pows:
        src.append(f"    {n} = {e}")
    src.append(f"    d = [0.0]*{len(assigns)}")
    for i, e in assigns:
        src.append(f"    d[{i}] = {e}")
    src.append("    return d")
    ns = {}
    exec("\n".join(src), ns)
    return ns["f"]


def odin_proc(name, dim, arr):
    nd, nc, _ = arr.shape
    base, pows, assigns = proc_body(dim, arr)
    L = ["@(private)", f"{name} :: proc(r: Ref_Vec, out: Bvec_Point(f64)) {{",
         f"\tassert(out.dofs == {nd} && out.cmpnts == {nc})"]
    if base:
        L.append(f"\t{', '.join(n for n, _ in base)} := {', '.join(e for _, e in base)}")
    for n, e in pows:
        L.append(f"\t{n} := {e}")
    L.append("\td := out.data")
    for i, e in assigns:
        L.append(f"\td[{i}] = {e}")
    L.append("}")
    return "\n".join(L)


def proc_name(cell, fam, order, q):
    return f"ref_{cell.lower()}_{fam.lower()}_o{order}_{q.lower()}"


# ============================================================================ tests

def test_quadrature():
    fact = math.factorial
    for et in ("Line", "Tri", "Quad", "Tet", "Hex"):
        dim = {"Line": 1, "Tri": 2, "Quad": 2, "Tet": 3, "Hex": 3}[et]
        for _, deg in QSETS:
            pts, wts = quad_rule(et, deg)
            for e in MONOS[dim]:
                if et in SIMPLEX and sum(e) > deg or et in TENSOR and max(e) > deg:
                    continue
                num = np.sum(wts * np.prod(pts ** np.array(e, dtype=float), axis=1))
                if et in SIMPLEX:
                    exact = math.prod(fact(a) for a in e[:dim]) / fact(sum(e) + dim)
                else:
                    exact = math.prod(0.0 if a % 2 else 2.0 / (a + 1) for a in e[:dim])
                assert abs(num - exact) < 1e-12, (et, deg, e, num, exact)


def test_emitted_code(el):
    rng = np.random.default_rng(1)
    pts = rng.uniform(-1, 1, (4, 3))
    pts[:, el.cd:] = 0
    for q, arr in el.quant.items():
        f = py_eval_fn(el.cd, arr)
        ref = np.einsum("pm,jcm->pjc", eval_monos(el.cd, pts), arr)
        for p in range(len(pts)):
            got = np.array(f(pts[p])).reshape(arr.shape[0], arr.shape[1])
            assert np.allclose(got, ref[p], atol=1e-10), (el.cell, el.fam, el.order, q)


def _neighbour_maps(cell):
    """Affine maps F(x) = J x + b (det > 0) placing a second reference cell across a facet of the first."""
    c = CELLS[cell]
    cd = c["dim"]
    V = c["verts"][:, :cd]
    maps = []
    if cell in SIMPLEX:
        for fv in c["ents"][cd - 1]:
            opp = [v for v in range(len(V)) if v not in fv][0]
            P = 2 * V[fv].mean(0) - V[opp]
            for gv in c["ents"][cd - 1]:
                gopp = [v for v in range(len(V)) if v not in gv][0]
                for sig in itertools.permutations(fv):
                    T = np.zeros((len(V), cd))
                    for i, v in enumerate(gv):
                        T[v] = V[sig[i]]
                    T[gopp] = P
                    A = np.hstack([V, np.ones((len(V), 1))])
                    X = np.linalg.solve(A, T)
                    J, b = X[:cd].T, X[cd]
                    if np.linalg.det(J) > 0:
                        maps.append((J, b))
    else:
        rots = []
        for p in itertools.permutations(range(cd)):
            for sg in itertools.product((1, -1), repeat=cd):
                S = np.zeros((cd, cd))
                for a in range(cd):
                    S[a, p[a]] = sg[a]
                if np.linalg.det(S) > 0:
                    rots.append(S)
        for a in range(cd):
            for s in (1, -1):
                u = np.zeros(cd)
                u[a] = 2 * s
                for S in rots:
                    maps.append((S, u))
    return maps


def canonical(et, gids):
    if et in ("Line", "Tri"):
        return sorted(gids)
    nb = {0: (1, 2), 1: (0, 3), 2: (0, 3), 3: (1, 2)}
    p0 = int(np.argmin(gids))
    a, b = nb[p0]
    v1, v2 = sorted([gids[a], gids[b]])
    return [gids[p0], v1, v2, gids[3 - p0]]


def test_patch(el, mode="sorted"):
    """mode: 'sorted' (canonical = rule from global ids), 'first_A' / 'first_B' (canonical = the order the
    first cell to encounter the entity uses, as a mesh loader would store it)."""
    cell, cd = el.cell, el.cd
    c = CELLS[cell]
    V = c["verts"][:, :cd]
    rng = np.random.default_rng(7)
    fnorm = facet_normals(cell)
    for J, b in _neighbour_maps(cell):
        VB = V @ J.T + b
        gidB = []
        for v in range(len(V)):
            m = np.where(np.linalg.norm(V - VB[v], axis=1) < 1e-9)[0]
            gidB.append(int(m[0]) if len(m) else 100 + v)
        gidA = list(range(len(V)))
        shared = {g for g in gidB if g < 100}
        fA = [i for i, fv in enumerate(c["ents"][cd - 1]) if set(fv) == shared][0]
        fB = [i for i, fv in enumerate(c["ents"][cd - 1]) if {gidB[v] for v in fv} == shared][0]

        # sample points on the shared facet (physical == A reference)
        fet = cell_ent_type(cell, cd - 1, fA)
        s = rng.uniform(-1, 1, (6, 3)) if fet in TENSOR else rng.dirichlet(np.ones(cd), 6)[:, :cd - 1]
        s = np.hstack([s[:, :cd - 1], np.zeros((len(s), 3 - (cd - 1)))])
        x = lift(fet, c["verts"][c["ents"][cd - 1][fA]], s)[:, :cd]
        nh = fnorm[fA][:cd] / np.linalg.norm(fnorm[fA][:cd])

        stored = {}
        if mode != "sorted":
            first_gid, first_facet = (gidA, fA) if mode == "first_A" else (gidB, fB)
            fcl = closure(cell, cd - 1, first_facet)
            for d in range(1, cd):
                for e in fcl[d]:
                    g = [first_gid[vv] for vv in c["ents"][d][e]]
                    stored[frozenset(g)] = g

        def traces(gid, facet, Jc, bc):
            xr = np.linalg.solve(Jc, (x - bc).T).T
            xr3 = np.hstack([xr, np.zeros((len(xr), 3 - cd))])
            v = np.einsum("pm,jcm->pjc", eval_monos(cd, xr3), el.coef)
            if el.fam == "Nedelec":
                v = np.einsum("ab,pjb->pja", np.linalg.inv(Jc).T, v)
            elif el.fam == "Raviart_Thomas":
                v = np.einsum("ab,pjb->pja", Jc, v) / np.linalg.det(Jc)
            labels = [None] * el.n
            cl = closure(cell, cd - 1, facet)
            for d in range(cd):
                for e in cl[d]:
                    s0, n = el.start[(d, e)], el.dpe[d]
                    if not n:
                        continue
                    vl = c["ents"][d][e]
                    g = [gid[vv] for vv in vl]
                    if d > 0:
                        et = cell_ent_type(cell, d, e)
                        tgt = canonical(et, g) if mode == "sorted" else stored[frozenset(g)]
                        perm = [tgt.index(gg) for gg in g]
                        key = PERMS[et].index(perm)
                        t = el.orient.get(et)
                        M = transform_matrix(t[key] if t else None, n)
                        v[:, s0:s0 + n] = np.einsum("jl,plc->pjc", M, v[:, s0:s0 + n])
                    for j in range(n):
                        labels[s0 + j] = (d, frozenset(g), j)
            if el.fam == "Raviart_Thomas":
                tr = np.einsum("pjc,c->pj", v, nh)[:, :, None]
            elif el.fam == "Nedelec":
                tr = v - np.einsum("pjc,c->pj", v, nh)[:, :, None] * nh
            else:
                tr = v
            return tr, labels

        trA, lA = traces(gidA, fA, np.eye(cd), np.zeros(cd))
        trB, lB = traces(gidB, fB, J, b)
        assert {l for l in lA if l} == {l for l in lB if l}, "shared dof sets differ"
        iB = {l: i for i, l in enumerate(lB) if l}
        for i, l in enumerate(lA):
            if l is None:
                assert np.abs(trA[:, i]).max() < 1e-9, f"non-closure dof has trace ({cell} {el.fam} O{el.order})"
            else:
                err = np.abs(trA[:, i] - trB[:, iB[l]]).max()
                assert err < 1e-8, f"patch test {cell} {el.fam} O{el.order} dof {l}: {err}"
        for i, l in enumerate(lB):
            if l is None:
                assert np.abs(trB[:, i]).max() < 1e-9


# ============================================================================ Odin emission

def fl(vals, per_line=8, indent=""):
    vals = [fnum(v) for v in vals]
    if len(vals) <= per_line:
        return "{" + ", ".join(vals) + "}"
    lines = [", ".join(vals[i:i + per_line]) for i in range(0, len(vals), per_line)]
    return "{\n" + ",\n".join(indent + "\t" + l for l in lines) + ",\n" + indent + "}"


def vecs(pts, indent=""):
    items = ["{" + ", ".join(fnum(v) for v in p) + "}" for p in pts]
    if len(items) <= 4:
        return "{" + ", ".join(items) + "}"
    lines = [", ".join(items[i:i + 4]) for i in range(0, len(items), 4)]
    return "{\n" + ",\n".join(indent + "\t" + l for l in lines) + ",\n" + indent + "}"


def enum_arr(keys, vals, indent):
    """Fully specified enumerated array literal (no #partial)."""
    inner = ",\n".join(f"{indent}\t.{k} = {v}" for k, v in zip(keys, vals))
    return "{\n" + inner + ",\n" + indent + "}"


def emit_dense(M, indent):
    n = M.shape[0]
    return f"{{rows = {n}, cols = {n}, values = {fl(M.T.ravel(), indent=indent)}}}"  # column-major


def emit_orientation(t, indent):
    if t is None:
        return "{}"
    if t[0] == "perm":
        return f"{{kind = .Signed_Perm, src = {{{', '.join(map(str, t[1]))}}}, sign = {fl(t[2])}}}"
    return (f"{{\n{indent}\tkind = .Dense,\n{indent}\tm = {emit_dense(t[1], indent + chr(9))},\n"
            f"{indent}\tm_inv = {emit_dense(t[2], indent + chr(9))},\n{indent}}}")


FUNCTIONAL_RULE_FIELD = "rule"  # Entity_Functionals field holding the Rule {element, points}


def functional_bvec(blk):
    """[point][dof][cmpnt] with the quadrature weight folded in (the emitted form)."""
    return np.einsum("p,ipc->pic", blk.wts, blk.W)


def emit_block(blk, indent):
    i1 = indent + "\t"
    B = functional_bvec(blk)
    npt, nd, nc = B.shape
    return (f"{{\n{i1}{FUNCTIONAL_RULE_FIELD} = {{element = .{blk.et}, points = {vecs(blk.pts, i1)}}},\n"
            f"{i1}weights = {{points = {npt}, dofs = {nd}, cmpnts = {nc}, data = {fl(B.ravel(), indent=i1)}}},\n{indent}}}")


def emit_basis(el, indent):
    if el is None:
        return "{}"
    i1 = indent + "\t"
    i2 = i1 + "\t"
    evals = [proc_name(el.cell, el.fam, el.order, q) if q in el.quant else "nil" for q in QUANTITIES]
    orients = []
    for et in ELEMENT_TYPES:
        t = el.orient.get(et)
        if t is None:
            orients.append("nil")
        else:
            orients.append("{\n" + ",\n".join(i2 + "\t" + emit_orientation(x, i2 + "\t") for x in t) + ",\n" + i2 + "}")
    funcs = []
    for d in range(4):
        blks = el.blocks.get(d, [])
        if not blks:
            funcs.append("nil")
        else:
            funcs.append("{\n" + ",\n".join(i2 + "\t" + emit_block(b, i2 + "\t") for b in blks) + ",\n" + i2 + "}")
    return (f"{{\n{i1}n_dofs = {el.n},\n"
            f"{i1}dofs_per_entity = {{.D0 = {el.dpe[0]}, .D1 = {el.dpe[1]}, .D2 = {el.dpe[2]}, .D3 = {el.dpe[3]}}},\n"
            f"{i1}evals = {enum_arr(QUANTITIES, evals, i1)},\n"
            f"{i1}orientations = {enum_arr(ELEMENT_TYPES, orients, i1)},\n"
            f"{i1}functionals = {enum_arr(DIMS, funcs, i1)},\n{indent}}}")


def emit_cell(cell, elements):
    c = CELLS[cell]
    cd = c["dim"]
    t = "\t"
    subs = []
    for d in range(4):
        if d > cd:
            subs.append("nil")
            continue
        items = []
        for e, vl in enumerate(c["ents"][d]):
            cl = closure(cell, d, e)
            clv = ["{" + ", ".join(map(str, x)) + "}" if (dd <= d) else "nil" for dd, x in enumerate(cl)]
            items.append(f"{{type = .{cell_ent_type(cell, d, e)}, closure = {{.D0 = {clv[0]}, .D1 = {clv[1]}, .D2 = {clv[2]}, .D3 = {clv[3]}}}}}")
        subs.append("{\n" + ",\n".join(t * 4 + x for x in items) + ",\n" + t * 3 + "}")
    fn = facet_normals(cell)
    perms = PERMS.get(cell)
    perm_s = "nil" if not perms else "{" + ", ".join("{" + ", ".join(map(str, p)) + "}" for p in perms) + "}"
    quads = []
    for name, deg in QSETS:
        pts, wts = quad_rule(cell, deg)
        quads.append(f"{{\n{t * 3}points = {vecs(pts, t * 3)},\n{t * 3}weights = {fl(wts, indent=t * 3)},\n{t * 2}}}")
    fams = []
    for fam in FAMILIES:
        per = [emit_basis(elements.get((fam, o)), t * 3) for o in range(4)]
        fams.append(enum_arr(ORDERS, per, t * 2))
    return (f".{cell} = {{\n"
            f"\ttopology = {{\n\t\tdim = .D{cd},\n\t\tvertices = {vecs(c['verts'], t * 2)},\n"
            f"\t\tsub_entities = {enum_arr(DIMS, subs, t * 2)},\n"
            f"\t\tfacet_normals = {vecs(fn, t * 2) if fn else 'nil'},\n"
            f"\t\torientation_perms = {perm_s},\n\t}},\n"
            f"\tquadrature = {enum_arr([q for q, _ in QSETS], quads, t)},\n"
            f"\tbases = {enum_arr(FAMILIES, fams, t)},\n}}\n")


HEADER = """// GENERATED by gen_ref.py -- do not edit by hand.
//
// Reference cells:
//   Point: {0}.  Line: [-1, 1].  Quad: [-1, 1]^2.  Hex: [-1, 1]^3.  Tri / Tet: unit simplex, vertex 0 at the origin.
//   Quad / Hex vertices in tensor order: vertex i = (bit0, bit1, bit2) of i, 0 -> -1, 1 -> +1.
//   Sub-entity numbering and local vertex order: see sub_entities[d][e].closure[.D0].
//
// Orientation keys (Line, Tri, Quad):
//   orientation_perms[key][i] = position, in the entity's CANONICAL vertex order, of the entity's local vertex i.
//   This is exactly what element_orientation(et, local_order = cell-local entity vertices (global ids),
//   target_order = canonical order) returns. The canonical order can be ANY fixed vertex order of the entity
//   that every cell sharing it agrees on, provided it is a symmetry image of a local order (always true when
//   it is taken from a cell, e.g. "the local order of the first cell that encounters the entity", which then
//   gets key 0). Sorted-by-global-id canonical orders (Line, Tri) also work. Both are tested.
//   Key 0 is the identity.
//
// Dof functionals: weights are Bvecs [point][dof][cmpnt] with the quadrature weight folded in. Moments use Legendre q on lines and quads (tensor), Lagrange q on triangle faces, monomials
// on simplex interiors. Normal / tangent directions follow the entity's closure[.D0] vertex order (not outward).
//
// Verified before emission: quadrature exactness, duality l_i(phi_j) = delta_ij, emitted-code evaluation,
// orientation consistency across entities, two-cell trace continuity for every orientation of every shared entity.
package fe

"""


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "./fe/fe_ref_generated.odin"
    test_quadrature()
    print("quadrature ok")
    elements = {}
    max_block = 0
    for cell in CELLS:
        for fam in FAMILIES:
            for o in range(4):
                if not defined(cell, fam, o):
                    continue
                el = build(cell, fam, o)
                test_emitted_code(el)
                if cell not in ("Point", "Line") and not (fam == "Lagrange" and o == 0):  # P0 is discontinuous
                    for mode in ("sorted", "first_A", "first_B"):
                        test_patch(el, mode)
                elements[(cell, fam, o)] = el
                max_block = max([max_block] + el.dpe)
                kinds = {et: ("dense" if any(x and x[0] == "dense" for x in t) else "perm") for et, t in el.orient.items() if t}
                print(f"  {cell:5s} {fam:15s} O{o}: {el.n:4d} dofs, per entity {el.dpe}, orient {kinds}")
    procs = []
    for (cell, fam, o), el in elements.items():
        for q in QUANTITIES:
            if q in el.quant:
                procs.append(odin_proc(proc_name(cell, fam, o, q), el.cd, el.quant[q]))
    src = HEADER
    src += f"// Largest dof block on any single entity (including cell interiors): {max_block}\n\n"
    src += "@(rodata)\nREFERENCE_ELEMENTS := [Element_Type]Reference_Element{\n"
    for cell in ELEMENT_TYPES:
        per = {(f, o): elements[(cell, f, o)] for f in FAMILIES for o in range(4) if (cell, f, o) in elements}
        src += "\n".join("\t" + l if l else l for l in emit_cell(cell, per).rstrip("\n").splitlines()) + ",\n"
    src += "}\n\n"
    src += "//== Basis evaluation\n\n" + "\n\n".join(procs) + "\n"
    with open(out_path, "w") as f:
        f.write(src)
    print(f"wrote {out_path} ({len(src.splitlines())} lines), max entity block {max_block}")


if __name__ == "__main__":
    main()
