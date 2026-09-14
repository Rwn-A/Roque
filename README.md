# Roque

A small finite element library for solving partial differential equations, written in Odin.

> [!NOTE]
> Roque is still in development.

## Purpose

Roque is a tool for exploring finite-element methods without many abstractions.
The design aims to be general and hackable, allowing a variety of physics to
be explored without requiring large amounts of library code.

We may eventually wrap Roque in an application layer, similar to a previous
project, to provide convenient configuration systems for common PDEs.

## Current Capabilities

Roque aims to be low-level and general-purpose. As a result, the fact that
something is not listed as a capability does not necessarily mean it is
impossible. It may be a trivial table addition (e.g. a higher Lagrange
order), require some additional code, or require an abstraction that does
not currently fit cleanly into the library's design.

### Reference Elements

- Point
- Line
- Triangle
- Quadrilateral
- Tetrahedron
- Hexahedron

### Function Spaces

- Lagrange (all elements)
- RT0 (2D & 3D elements)
- Mixed & multi-variate combined spaces.

### Geometry

- Conforming 1D–3D, including embedded geometries
- Arbitrary coordinate frames
- graph-like meshes (e.g. truss and frame structures)

### Algebraic Constraints

- Dirichlet, including partial constraints
- Periodic

_Note: Periodic boundary conditions are not yet robust. Periodicity is defined
on the mesh (via Gmsh), and the associated function spaces must match exactly._

General multi-point constraints (MPCs) are supported naturally by the underlying
constraint system, but there is currently no friendly API for defining them
without modifying the constraint structure directly. They are therefore not
considered supported at this time.

### Discretization Methods

- CG
- _DG_ (should work, but may be missing some conveniences)
- _HDG and related methods_ (almost)

### Linear Solvers

- FGMRES
- Conjugate Gradient
- BiCGSTAB

### Preconditioners

- ILU
- Multigrid (smoothed aggregation)

**Solvers are not hand-rolled.** Roque links to AMGCL through custom,
work-in-progress bindings. The available solvers and configurations are
therefore limited to what the bindings currently expose and do not represent
the full capabilities of AMGCL.

### Problem Types

- Linear problems
- Eigenvalue problems

_Note: Time stepping and nonlinear problems are largely additive on top of the
existing infrastructure. While Roque does not yet provide dedicated helpers
for them, they can be implemented easily_.

### I/O

- Gmsh (version 2 binary format only, linear and quadratic)
- VTU/PVD (output only, readers are not yet implemented)

## Installation / Usage
Will be provided once project matures, see `validation/` directory for usage examples.

## Validation

Validation is currently ongoing.

## Performance

Roque has yet to have a general optimization pass but some care was taken while writing and the overall design aims to avoid
unnecessary work.

Roque is setup for threading using a simple single-program-multiple-data paradigm but the graph colouring and iteration helpers have yet to be implemented. SIMD is a similar story, core infrastructure works naturally for SIMD across elements, but building / iterating batches currently is not implemented.
