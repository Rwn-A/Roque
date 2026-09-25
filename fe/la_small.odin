package fe

/*
 Small fixed-size matrices & vectors.

 Backed by arbitrary type T to support SIMD operations.

 Implementation has been limited to whats been needed by contractions & geometry.
*/

import "base:intrinsics"

Small_Vec :: struct(N: int, T: typeid) {
	data: [N]T,
}

// Reinterprets the first N element `data` as a ^Small_Vec(N,T), aliasing the same memory.
small_vec_view_from_slice :: proc(data: []$T, $N: int) -> ^Small_Vec(N, T) {
	assert(len(data) >= N)
	return cast(^Small_Vec(N, T))raw_data(data)
}

// Copies and converts the first N elements of data into a Small_Vec.
small_vec_from_slice :: proc($VT: typeid, data: []$ST, $N: int) -> (r: Small_Vec(N, VT)) {
	#unroll for i in 0 ..< N { r.data[i] = cast(VT)data[i] }
	return r
}

// a . b
small_vec_dot :: proc(a, b: Small_Vec($N, $T)) -> (r: T) {
	#unroll for i in 0 ..< N {
		r += a.data[i] * b.data[i]
	}
	return r
}

// r = a + b
small_vec_add :: proc(a, b: Small_Vec($N, $T)) -> (r: Small_Vec(N, T)) {
	#unroll for i in 0 ..< N {
		r.data[i] = a.data[i] + b.data[i]
	}
	return r
}

// r = s * r
small_vec_scale :: proc(a: Small_Vec($N, $T), s: T) -> (r: Small_Vec(N, T)) {
	#unroll for i in 0 ..< N {
		r.data[i] = a.data[i] * s
	}
	return r
}

// len(a)
small_vec_norm :: proc(a: Small_Vec($N, $T)) -> T {
	return intrinsics.sqrt(small_vec_dot(a, a))
}

// a / len(a)
small_vec_normalize :: proc(a: Small_Vec($N, $T)) -> Small_Vec(N, T) {
	return small_vec_scale(a, T(1) / small_vec_norm(a))
}

small_vec3_cross :: proc(a, b: Small_Vec(3, $T)) -> Small_Vec(3, T) {
	return Small_Vec(3, T) {
		data = {
			a.data.y * b.data.z - a.data.z * b.data.y,
			a.data.z * b.data.x - a.data.x * b.data.z,
			a.data.x * b.data.y - a.data.y * b.data.x,
		},
	}
}

small_vec2_cross :: proc(a, b: Small_Vec(2, $T)) -> T {
	return (a.data.y * b.data.x) - (a.data.x * b.data.y)
}

// a x b (for vectors where the cross product is well defined)
small_vec_cross :: proc {
	small_vec3_cross,
	small_vec2_cross,
}


// column-major
Small_Mat :: struct(R, C: int, T: typeid) {
	data: [C][R]T,
}

// Directly alias the slice data as a small matrix.
small_mat_view_from_slice :: proc(data: []$T, $R, $C: int) -> ^Small_Mat(R, C, T) {
	assert(len(data) >= R * C)
	return cast(^Small_Mat(R, C, T))raw_data(data)
}

// Odin matrix representation of the small matrix, only usable if T is supported by Odins matrix.
small_mat_view_as_matrix :: proc(m: ^Small_Mat($R, $C, $T)) -> ^matrix[R, C]T {
	return cast(^matrix[R, C]T)m.data
}

// Generally prefer to handle transpose inline an operation, but here if needed.
small_mat_transpose :: proc(m: Small_Mat($R, $C, $T)) -> (r: Small_Mat(C, R, T)) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			r.data[row][col] = m.data[col][row]
		}
	}
	return r
}

// Return columns as small vectors
small_mat_columns :: proc(m: Small_Mat($R, $C, $T)) -> (r: [C]Small_Vec(R, T)) {
	#unroll for col in 0 ..< C {
		r[col].data = m.data[col]
	}
	return r
}

// Perform the standard matrix vector product, where `v` is a column vector.
small_mat_vec_mul :: proc(m: Small_Mat($R, $C, $T), v: Small_Vec(C, T)) -> (r: Small_Vec(R, T)) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			r.data[row] += m.data[col][row] * v.data[col]
		}
	}
	return r
}

// Matrix vector product of where `v` is a row vector. (M * v^T)
small_mat_vec_mul_t :: proc(m: Small_Mat($R, $C, $T), v: Small_Vec(R, T)) -> (r: Small_Vec(C, T)) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			r.data[col] += m.data[col][row] * v.data[row]
		}
	}
	return r
}

// dst *= scale
small_mat_scale_inplace :: proc(m: ^Small_Mat($R, $C, $T), scale: T) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			m.data[col][row] *= scale
		}
	}
}

// dst += src * scale
small_mat_add_inplace :: proc(dst: ^Small_Mat($R, $C, $T), src: Small_Mat(R, C, T), scale: T) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			dst.data[col][row] += scale * src.data[col][row]
		}
	}
}

// dst += src^T * scale
small_mat_add_inplace_t :: proc(dst: ^Small_Mat($C, $R, $T), src: Small_Mat(R, C, T), scale: T) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			dst.data[col][row] += scale * src.data[row][col]
		}
	}
}

// Accumulates the outer product of `a` and `b` into dst.
small_vec_outer :: proc(dst: ^Small_Mat($R, $C, $T), a: Small_Vec(R, T), b: Small_Vec(C, T)) {
	#unroll for col in 0 ..< C {
		#unroll for row in 0 ..< R {
			dst.data[col][row] += a.data[row] * b.data[col]
		}
	}
}

// Perform a matrix multiply, (single contraction), between a & b.
small_mat_mul :: proc(a: Small_Mat($R, $K, $T), b: Small_Mat(K, $C, T)) -> (r: Small_Mat(R, C, T)) {
	#unroll for col in 0 ..< C {
		#unroll for k in 0 ..< K {
			#unroll for row in 0 ..< R {
				r.data[col][row] += a.data[k][row] * b.data[col][k]
			}
		}
	}
	return r
}

// Frobenius inner product (double contraction) between a & b.
small_mat_frob :: proc(a: Small_Mat($R, $K, $T), b: Small_Mat(R, K, T)) -> (r: T) {
	#unroll for col in 0 ..< K {
		#unroll for row in 0 ..< R {
			r += a.data[col][row] * b.data[col][row]
		}
	}
	return r
}

// The Gram matrix M^T * M.
small_mat_gram :: proc(m: Small_Mat($R, $C, $T)) -> (g: Small_Mat(C, C, T)) {
	#unroll for j in 0 ..< C {
		for i in 0 ..= j {
			sum: T
			#unroll for row in 0 ..< R { sum += m.data[i][row] * m.data[j][row] }
			g.data[j][i], g.data[i][j] = sum, sum
		}
	}
	return g
}

//== Shape specific routines.

small_mat1x1_det :: proc(m: Small_Mat(1, 1, $T)) -> T {
	return m.data[0][0]
}

small_mat2x2_det :: proc(m: Small_Mat(2, 2, $T)) -> T {
	return m.data[0][0] * m.data[1][1] - m.data[1][0] * m.data[0][1]
}

small_mat3x3_det :: proc(m: Small_Mat(3, 3, $T)) -> T {
	c0, c1, c2 := m.data[0], m.data[1], m.data[2]
	return(
		c0[0] * (c1[1] * c2[2] - c1[2] * c2[1]) -
		c0[1] * (c1[0] * c2[2] - c1[2] * c2[0]) +
		c0[2] * (c1[0] * c2[1] - c1[1] * c2[0]) \
	)
}

// Standard determinant
small_mat_det :: proc {
	small_mat1x1_det,
	small_mat2x2_det,
	small_mat3x3_det,
}

small_mat1x1_measure :: proc(m: Small_Mat(1, 1, $T)) -> T {
	return abs(small_mat1x1_det(m))
}

small_mat2x1_measure :: proc(m: Small_Mat(2, 1, $T)) -> T {
	return intrinsics.sqrt(small_mat1x1_det(small_mat_gram(m)))
}

small_mat3x1_measure :: proc(m: Small_Mat(3, 1, $T)) -> T {
	return intrinsics.sqrt(small_mat1x1_det(small_mat_gram(m)))
}

small_mat2x2_measure :: proc(m: Small_Mat(2, 2, $T)) -> T {
	return abs(small_mat2x2_det(m))
}

small_mat3x2_measure :: proc(m: Small_Mat(3, 2, $T)) -> T {
	return intrinsics.sqrt(small_mat2x2_det(small_mat_gram(m)))
}

small_mat3x3_measure :: proc(m: Small_Mat(3, 3, $T)) -> T {
	return abs(small_mat3x3_det(m))
}

small_mat1x1_inv_t :: proc(m: Small_Mat(1, 1, $T)) -> (r: Small_Mat(1, 1, T)) {
	r.data[0][0] = T(1) / m.data[0][0]
	return r
}

small_mat2x2_inv_t :: proc(m: Small_Mat(2, 2, $T)) -> (r: Small_Mat(2, 2, T)) {
	inv_d := T(1) / small_mat2x2_det(m)
	r.data[0][0] = m.data[1][1] * inv_d
	r.data[0][1] = -m.data[1][0] * inv_d
	r.data[1][0] = -m.data[0][1] * inv_d
	r.data[1][1] = m.data[0][0] * inv_d
	return r
}

small_mat3x3_inv_t :: proc(m: Small_Mat(3, 3, $T)) -> (r: Small_Mat(3, 3, T)) {
	c0, c1, c2 := **small_mat_columns(m)
	e0 := small_vec_cross(c1, c2)
	e1 := small_vec_cross(c2, c0)
	e2 := small_vec_cross(c0, c1)
	inv_d := T(1) / small_vec_dot(c0, e0)
	r.data[0] = small_vec_scale(e0, inv_d).data
	r.data[1] = small_vec_scale(e1, inv_d).data
	r.data[2] = small_vec_scale(e2, inv_d).data
	return r
}

small_mat2x1_inv_t :: proc(m: Small_Mat(2, 1, $T)) -> Small_Mat(2, 1, T) {
	return small_mat_mul(m, small_mat1x1_inv_t(small_mat_gram(m)))
}

small_mat3x1_inv_t :: proc(m: Small_Mat(3, 1, $T)) -> Small_Mat(3, 1, T) {
	return small_mat_mul(m, small_mat1x1_inv_t(small_mat_gram(m)))
}

small_mat3x2_inv_t :: proc(m: Small_Mat(3, 2, $T)) -> Small_Mat(3, 2, T) {
	return small_mat_mul(m, small_mat2x2_inv_t(small_mat_gram(m)))
}

// Generalized "scaling" of the transformation, equivalent to abs(determinant) for square cases.
small_mat_measure :: proc {
	small_mat1x1_measure,
	small_mat2x1_measure,
	small_mat3x1_measure,
	small_mat2x2_measure,
	small_mat3x2_measure,
	small_mat3x3_measure,
}

// Inverse (or pseudo-inverse) transpose.
small_mat_inv_t :: proc {
	small_mat1x1_inv_t,
	small_mat2x1_inv_t,
	small_mat3x1_inv_t,
	small_mat2x2_inv_t,
	small_mat3x2_inv_t,
	small_mat3x3_inv_t,
}

// Inverse (or pesudo-inverse)
small_mat_inv :: proc(m: Small_Mat($R, $C, $T)) -> Small_Mat(C, R, T) {
	return small_mat_transpose(small_mat_inv_t(m))
}
