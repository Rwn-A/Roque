# Roque

A small finite element library for solving partial differential equations, written in Odin.

> [!NOTE]
> Roque is still in development.

## Purpose

Roque is a tool for exploring finite element methods without many abstractions. The design aims to be general and
hackable, so a variety of physics can be explored without large amounts of library code.

We may eventually wrap Roque in an application layer, similar to a previous project, to provide convenient
configuration for common PDEs.

## Current Capabilities

Roque aims to be low-level and general-purpose, so something missing from this list isn't necessarily impossible. It
might be a trivial table addition, need a bit of extra code, or need an abstraction that doesn't fit cleanly into the
library yet.

### Reference Elements

- Point, Line
- Triangle, Quadrilateral
- Tetrahedron, Hexahedron

### Function Spaces

- Lagrange, orders 0–3 (all elements)
- Raviart–Thomas, orders 0–2 (Tri, Quad, Tet, Hex)
- Nédélec, orders 0–2 (Tri, Quad, Tet, Hex)
- Continuous, discontinuous and trace (facet) spaces
- Mixed and multi-variate combined spaces

### Geometry

- Conforming 1D–3D, including embedded geometries
- Arbitrary coordinate frames
- Graph-like meshes (e.g. truss and frame structures)
- Linear and quadratic (isoparametric) cells

### Algebraic Constraints

- Dirichlet, including partial constraints
- Periodic
- Either eliminated from the system or kept as identity rows (1 on the diagonal)

_Note: Periodic boundary conditions aren't fully robust yet. Periodicity is defined on the mesh (via Gmsh), and the
associated function spaces must match exactly._

General multi-point constraints (MPCs) fall out of the constraint system naturally, but there's no friendly API for
them yet, so they're not considered supported.

### Discretization Methods

- CG
- DG (works, may be missing a few conveniences)
- HDG, EDG

### Linear Solvers

- FGMRES
- Conjugate Gradient
- BiCGSTAB

### Preconditioners

- ILU
- Multigrid (smoothed aggregation)

**Solvers aren't hand-rolled.** Roque links to AMGCL through custom, work-in-progress bindings, so what's available
is limited to what the bindings expose, not the full capabilities of AMGCL.


### Problem Types

- Linear problems
- Eigenvalue problems

_Note: Time stepping and nonlinear problems are mostly additive on top of the existing infrastructure. There are no dedicated helpers yet, but they're easy to build._

### I/O

- Gmsh (version 2 binary format only, linear and quadratic)
- VTU/PVD (output only, no readers yet)

## Installation / Usage

Will be provided once the project matures. See the `validation/` directory for usage examples.

## Validation

Ongoing. The examples above check against known solutions, and a proper convergence-rate harness is next on the list.

## Performance

Roque hasn't had a general optimization pass yet, but some care went into it and the design tries to avoid
unnecessary work.

Roque is set up for threading with a simple single-program-multiple-data model, but the graph colouring and iteration
helpers aren't implemented yet. SIMD is a similar story: the core infrastructure works naturally for SIMD across
elements, but building and iterating batches isn't implemented yet.
