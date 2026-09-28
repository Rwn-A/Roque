"""
Structured triangle meshes of the unit square for convergence studies, Gmsh 2.2 binary.

Every square of an n x n grid is split in two, the diagonal alternating like a checkerboard so vertex valence varies.
Physical groups: "domain" (cells, id 1) and "boundary" (every boundary line, id 2).

Run from the repository root:  python validation/meshes/gen_square.py
"""

import struct

SIZES = [4, 8, 16, 32]


def write(n, path):
    def node(i, j):
        return j * (n + 1) + i + 1

    nodes = [(node(i, j), i / n, j / n) for j in range(n + 1) for i in range(n + 1)]

    tris = []
    for j in range(n):
        for i in range(n):
            a, b, c, d = node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)
            if (i + j) % 2 == 0:
                tris += [(a, b, c), (a, c, d)]
            else:
                tris += [(a, b, d), (b, c, d)]

    lines = []
    for k in range(n):
        lines += [(node(k, 0), node(k + 1, 0)), (node(n, k), node(n, k + 1))]
        lines += [(node(k + 1, n), node(k, n)), (node(0, k + 1), node(0, k))]

    out = bytearray()
    out += b"$MeshFormat\n2.2 1 8\n" + struct.pack("<i", 1) + b"\n$EndMeshFormat\n"
    out += b'$PhysicalNames\n2\n1 2 "boundary"\n2 1 "domain"\n$EndPhysicalNames\n'

    out += b"$Nodes\n%d\n" % len(nodes)
    for id, x, y in nodes:
        out += struct.pack("<iddd", id, x, y, 0.0)
    out += b"\n$EndNodes\n"

    out += b"$Elements\n%d\n" % (len(lines) + len(tris))
    out += struct.pack("<iii", 1, len(lines), 2)  # type, count, tags (physical, elementary)
    for k, (a, b) in enumerate(lines):
        out += struct.pack("<iiiii", k + 1, 2, 1, a, b)
    out += struct.pack("<iii", 2, len(tris), 2)
    for k, (a, b, c) in enumerate(tris):
        out += struct.pack("<iiiiii", len(lines) + k + 1, 1, 1, a, b, c)
    out += b"\n$EndElements\n"

    with open(path, "wb") as f:
        f.write(out)


for n in SIZES:
    write(n, "validation/meshes/2d_unit_square_%d.msh" % n)
