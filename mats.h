#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define __FAST__
#define TEST_MODULES

#include <cmath>
#include <array>
#include <functional>

typedef long unsigned int	dim;
typedef double			type;

template <dim N>
using vec = std::array<type, N>;

template <dim N, dim M>
using mat = std::array<vec<M>, N>;

template<dim N>
void print_vec(vec<N> v)
{
	dim i;
	for (i=0;i<v.size();i++)
	{
		fprintf(stdout, "%0.2f ,", v[i]);
	}
}
template<dim N, dim M>
void print_mat(mat<N,M> A)
{
	dim i;

	for (i=0; i<A.size(); i++)
	{
		print_vec(A[i]);
		fprintf(stdout, "\n\0");
	}
	fprintf(stdout, "\n\0");
}

/* copy data */
template<dim I> void _copy(const vec<I>& src, vec<I>& dst)										{ dst = src; }
template<dim N, dim M> void _copy(const mat<N,M>& src, mat<N,M>& dst)							{ dst = src; }

/* generator functions */
type zero()											{return 0.0;}
type one()											{return 1.0;}
type rando()										{return ((type)rand() / (type)RAND_MAX);}
type sigmoid(type x)								{return 1.0 / (1.0 + exp(-x));}
type sigmoid_derivative(type x)						{return x * (1.0 - x);}

template<dim N>					void _fill(vec<N>& a, type x)
{
	dim i;
	for (i=0; i<a.size (); i++)
	{
		a[i] = x;
	}
}
template<dim N, dim M>			void _fill(mat<N,M>& A, type x)
{
	dim i;
	for (i = 0; i<A.size (); i++)
	{
		_fill(A[i], x);
	}
}

/* Apply a function to all values in the vector */
template<dim N>					void _map(const vec<N>& a, std::function<type()> fn, vec<N>& o)
{
	dim i;

	for (i=0; i<a.size(); i++)
	{
		o[i] = fn();
	}
}
template<dim N>					void _map(const vec<N>& a, std::function<type(type)> fn, vec<N>& o)
{
	dim i;

	for (i=0; i<a.size(); i++)
	{
		o[i] = fn(a[i]);
	}
}
/* Apply a function f to all values in the matrix */
template<dim N, dim M>			void _map(const mat<N,M>& A, std::function<type()> fn, mat<N,M>& O)
{
	dim i;
	for (i=0; i<A.size(); i++)
	{
		_map (A[i], fn, O[i]);
	}
}

/* Apply a function f to all values in the matrix */
template<dim N, dim M>			void _map(const mat<N,M>& A, std::function<type(type)> fn, mat<N,M>& O)
{
	dim i;
	for (i=0; i<A.size(); i++)
	{
		_map (A[i], fn, O[i]);
	}
}

/* fills a matrix like A with zeros */
template<dim N>					void _zeros(vec<N>& a)								{_map(a, zero, a);}
template<dim N, dim M>			void _zeros(mat<N,M>& A)							{_map(A, zero, A);}
/* fills a vector like a with unit values */
template<dim N>					void _ones(vec<N>& a)								{_map(a, one, a);}
/* fills a matrix like A with unit values */
template<dim N, dim M>			void _ones(mat<N,M>& A)								{_map(A, one, A);}
/* fills a matrix like A with random numbers (0..1] */
//mat random(mat A)								{ return map(A, rando); }
/* Returns a square identity matrix of given size */
template<dim N>					void _identity(mat<N,N>& A)
{
	dim i;
	_zeros(A);

	for (i=0; i<A.size(); i++)
	{
		A[i][i] = 1.0;
	}
}
/* implementation of matlab command: "A = gallery('lehmer', size) */
template<dim N, dim M>			void _lehmer(mat<N,M>& A)
{
	dim i, j;

	for (i=0; i<A.size (); i++)
	{
		for (j=0; j<A[0].size(); j++)
		{
			A[i][j] = std::min((type)(i + 1), (type)(j + 1)) / std::max((type)(i + 1), (type)(j + 1));
		}
	}
}
/* returns a hilbert matrix. */
template<dim N, dim M>			void _hilbert(mat<N,M>& A)
{
	dim i, j;
	for (i=0; i<A.size(); i++)
	{
		for (j=0; j<A[0].size(); i++)
		{
			A[i][j] = 1.0/(type)(i+j-1);
		}
	}
}
/* return a counted matrix */
template<dim N>					void _counter(vec<N>& a)
{
	dim i;

	for (i = 0; i < a.size(); i++)
	{
		a[i] = (type)i + 1;
	}
}
template<dim N, dim M>			void _counter(mat<N,M>& A)
{
	dim i, j;

	for (i=0; i<A.size(); i++)
	{
		for (j=0; j<A[0].size(); j++)
		{
			A[i][j] = (i*A.size())+j;
		}
	}
}
/* swap two vectors using a local copy */
#if false
template<dim N>					void swap(vec<N>& a, vec<N>& b)
{
#ifndef __FAST__
	auto c = copy(a);
	a = copy(b);
	b = copy(c);
#else
	type c = 0.0;
	dim i;

	for (i=0; i<a.size(); i++)
	{
		c = a[i];
		a[i] = b[i];
		b[i] = c;
	}
#endif
}
#endif
/* Element-wise negation of a vector a (i.e., a = -a) */
template<dim N>					void _negate(const vec<N>& a, vec<N>& o)
{
	dim i;
	for (i=0; i<a.size(); i++)
	{
		o[i] = -a[i];
	}
}
/* Element-wise negation of a matrix A */
template<dim N, dim M>			void _negate(const mat<N,M>& A, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		_negate(A[i], O[i]);
	}
}
/* Transposes a matrix */
template<dim N, dim M>			void _transpose(const mat<N,M>& A, mat<M,N>& O)
{
	for (auto i=0; i<M; i++)
	{
		for (auto j=0; j<N; j++)
		{
			O[i][j] = A[j][i];
		}
	}
}
template<dim N>					void _scale(const vec<N>& a, type scalar, vec<N>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = a[i]*scalar;
	}
}
/* Element-wise sum of a vector */
/* FIXME: if len(a) > X split the vector and sum in parallel
   to avoid rounding errors */
template<dim N>					type _sum(const vec<N>& a)
{
	auto c = zero();

	for (auto i=0; i<N; i++)
	{
		c += a[i];
	}
	return c;
}

/* Element-wise sum of a matrix */
template<dim N, dim M>			type _sum(const mat<N,M>& A)
{
	dim i;
	type c = zero();

	for (auto i=0; i<M; i++)
	{
		c += _sum (A[i]);
	}

	return c;
}
/* Element-wise product of a vector */
template<dim N>					type product_sum(const vec<N>& a)
{
	type c = one();

	for (auto i : a)
	{
		c*=i;
	}

	return c;
}
template<dim N>					void _add(const vec<N>& a, const vec<N>& b, vec<N>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = a[i] + b[i];
	}
}
template<dim N, dim M>			void _add(const mat<N,M>& A, const mat<N,M>& B, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		_add(A[i], B[i], O[i]);
	}
}
/* Multiply then add operator. (a*scalar)+b */
template<dim N>					void _mad(vec<N>& o, const vec<N>& a, const vec<N>& b, const type scalar)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = (a[i]*scalar)+b[i];
	}
}
template<dim N, dim M>			void _scale(const mat<N,M>& A, const type scalar, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		_scale(A[i], scalar, O[i]);
	}
}
/* multiply two given vectors */
template<dim N>					void _mult(const vec<N>& a, const vec<N>& b, vec<N>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i]=a[i]*b[i];
	}
}
template<dim N>					type _dot(const vec<N>& a, const vec<N>& b)
{
	type d = zero();
	for (auto i=0; i<N; i++)
	{
		d+=a[i]*b[i];
	}
	return d;
}
template<dim N>					type inner(const vec<N>& a, const vec<N>& b)						{ return _dot(a, b); }
/* multiply a matrix by a vector */
template<dim N, dim M>			void _mult(const mat<N,M>& A, const vec<M>& a, vec<M>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = _dot(A[i], a);
	}
}
/* multiplication of two matrices */
template<dim N, dim M, dim I>	void _mult(const mat<N,M>& A, const mat<M,I>& B, mat<N,I>& O)
{
	auto BT = mat<I,M>();
	_transpose(B, BT);

	for (auto i=0; i<N; i++)
	{
		for (auto j=0; j<I; j++)
		{
			O[i][j] = _dot(A[i], BT[j]);
		}
	}
}
template<dim N, dim M, dim I>	mat<M,M> mult(const mat<N,M>& A, const mat<M,I>& B)
{
	auto c = mat<N, I>();
	_mult(A, B, c);
	return c;
}
/* Element-wise multiplication of two matrices */
template<dim N, dim M>			void _hadamard(const mat<N,M>& A, const mat<N,M>& B, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		_mult(A[i], B[i], O[i]);
	}
}
template<dim N>					void _sub(const vec<N>& a, const vec<N>& b, vec<N>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = a[i]-b[i];
	}
}
template<dim N, dim M>			void _sub(const mat<N,M>& A, const mat<N,M>& B, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		_sub(A[i], B[i], O[i]);
	}
}
template<dim N, dim M>			void _sub(const mat<N,M>& A, const type v, mat<N,M>& O)
{
	auto st = vec<M>();
	_fill(st, v);

	for (auto i=0;i<N;i++)
	{
		_sub(A[i], st, O[i]);
	}
}
/* the minor of a vector rooted at x (all elements except index x) */
template<dim N>					void _minor(const vec<N>& a, const dim x, vec<N-1>& o)
{
	for (auto i=0, m=0; i<N; i++)
	{
		if (i == x)
		{
			continue;
		}

		o[m++] = a[i];
	}
}
/* Returns the minor of A rooted at ->v[x,y] */
template<dim N, dim M>			void _minor(const mat<N,M>& A, const dim x, const dim y, mat<N-1,M-1>& O)
{
	for (auto i=0, m=0; i<N; i++)
	{
		if (i == x)
		{
			continue;
		}

		_minor(A[i], y, O[m++]);
	}
}
/* Get the diagonal elements of a matrix as a vector */
template<dim N>					void _diag(const mat<N,N>& A, vec<N>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = A[i][i];
	}
}
/* Computes the inverse of vector a, where the inverse is 1.0/a */
template<dim N>					void _inverse(const vec<N>& a, vec<N>& o)
{
	for (auto i=0; i<N; i++)
	{
		o[i] = 1.0/a[i];
	}
}
/**
 * Computes the inverse and determinant of a matrix by gaussian elimination.
 * This method used for matrices larger than 3x3.
 * This method is O(n^2).
 * </summary>
 * <param name="A">input matrix</param>
 * <param name="inverse">holder for the inverse of A</param>
 * <param name="determinant">holder for the determinant of A</param>
 * <returns>identity matrix if A is non-singular</returns>
 */
template<dim N>					void _gaussian_elimination(const mat<N,N>& A, mat<N,N>& inverse, type& determinant)
{
	auto C = mat<N,N>();							// output matrix to return (will be just the identity)
	auto I = mat<N,N>();	_identity(I);			// inverse matrix
	auto diags = vec<N>();							// diagonal vector
	auto c = zero();								// coefficent for column zeroing

	_copy(A, C);
	determinant = one();							// determinant (sign is important)

	// put the matrix in row echelon form
	for (auto i=0; i<N; i++)
	{
		auto k=i;

		// find the largest diagonal
		for (auto j=i+1; j<N; j++)
		{
			if (abs(C[j][i]) > abs(C[k][i]))
				k = j;
		}

		// pivot the matrix around the largest diagonal
		if (k != i)								// if max is not current, swap the rows
		{
			determinant *= -1.0;				// everytime we swap a row, flip the sign of the determinant
			std::swap(C[i], C[k]);	// swap the max diagonal row with the current row
			std::swap(I[i], I[k]);	// apply the same method to the inverse matrix
		}

		// zero this column
		for (auto j=i+1; j<N; j++)
		{
			c = C[j][i] / C[i][i];			// get the diagonal of this row
#ifndef __FAST__
			C[j] = add(C[j], scale(C[i], -c));	// subtract and divide this diagonal from all the row
			I[j] = add(I[j], scale(I[i], -c));
#else
			_mad(C[j], C[i], C[j], -c);
			_mad(I[j], I[i], I[j], -c);
#endif
		}
	}

	_diag(C, diags);
	determinant *= product_sum(diags);		// compute the determinant as the product of the diagonals

	// transform the diagonals to 1 by dividing each row by the row diagonal
	for (auto i=0; i<N; i++)
	{
		c = C[i][i];
#ifndef __FAST__
		C[i] = scale(C[i], 1.0 / c);
		I[i] = scale(I[i], 1.0 / c);
#else
		_scale(C[i], 1.0 / c, C[i]);
		_scale(I[i], 1.0 / c, I[i]);
#endif
	}

	// transform the input to the identity matrix (reduced row echelon form)
	for (auto i=N; i>0; i--)
	{
		for (auto j=N-1; j>i-1; j--)
		{
			c = -C[i-1][j];
#ifndef __FAST__
			C[i-1] = add(scale(C[j], c), C[i-1]);
			I[i-1] = add(scale(I[j], c), I[i-1]);
#else
			_mad(C[i-1], C[j], C[i-1], c);
			_mad(I[i-1], I[j], I[i-1], c);
#endif
		}
	}

	// return the inverse
	_copy(I, inverse);
}
/* Wrapper to compute the determinant of a matrix using the gaussian elimination method */
template<dim N>					type _determinant_gaussian_elimination(const mat<N,N>& A)
{
	type sum = 1.0;
	auto I = mat<N,N>();
	_copy(A, I);
	_gaussian_elimination(A, I, sum);
	return sum;
}
/*
 * Computes the determinant of a square matrix.
 * 2x2 and 3x3 are direct, from then on the method is based on
 * gaussian elimiation via the gaussian_elimination function.
 * </summary>
 * <param name="A">input matrix</param>
 * <returns>determinant of A</returns>
 */
template<dim N>					type _determinant(const mat<N,N>& A)
{
	return _determinant_gaussian_elimination(A);
}
template<>						type _determinant(const mat<2,2>& A)
{
	return ((A[0][0]*A[1][1]) - (A[0][1]*A[1][0]));
}
template<>						type _determinant(const mat<3,3>& A)
{
	return ((A[0][0] * A[1][1] * A[2][2]) -
			(A[0][0] * A[1][2] * A[2][1]) -
			(A[0][1] * A[1][0] * A[2][2]) +
			(A[0][1] * A[1][2] * A[2][0]) +
			(A[0][2] * A[1][0] * A[2][1]) -
			(A[0][2] * A[1][1] * A[2][0]));
}
/*
 * Computes the determinant of a matrix by recursion of the cofactors.
 * This method is only here to illustrate the naive method of finding
 * a determinant and should not be used for matrices larger than 5x5.
 * </summary>
 * <param name="A">input matrix</param>
 * <returns>determinant of A</returns>
 */
template<dim N, dim M>			type determinant_recursive(const mat<N,M>& A)
{
	type sum = zero();
	auto Am = mat<N-1,M-1>();

	for (auto j=0; j<N; j++)
	{
		_minor(A, 0, j, Am);
		sum += A[0][j] * std::pow(-1.0, j) * _determinant(Am);
	}

	return sum;
}
/* cofactor matrix of A, i.e., the matrix of minor determinants
 * matrix where each element i,j is determinant (minor(A, i,j))
 */
template<dim N, dim M>			void _cofactor(const mat<N,M>& A, mat<N,M>& O)
{
	auto Am = mat<N-1,M-1>();

	for (auto i=0; i<N; i++)
	{
		for (auto j=0; j<M; j++)
		{
			_minor(A, 0, j, Am);
			O[i][j] = std::pow(-1.0, i + j) * _determinant(Am);
		}
	}
}
/**
 * Returns the inverse of a square matrix A.
 * Does not check of the matrix is invertible.
 * 2x2 inverses are computed directly,
 * 3x3 inverses are calculated recursively using cofactors
 * nxn inverses are calculated using gaussian elimination
 * </summary>
 * <param name="A">input matrix</param>
 * <returns>matrix which when multiplied with A yields the identity matrix</returns>
 */
template<dim N, dim M>			void _inverse(const mat<N,M>& A, mat<N,M>& O)
{
	type det = 1.0;
	_gaussian_elimination(A, O, det);
}
template<>						void _inverse(const mat<2,2>& A, mat<2,2>& O)
{
	O[0][0] =  A[1][1];
	O[0][1] = -A[0][1];
	O[1][0] = -A[1][0];
	O[1][1] =  A[0][0];

	_scale(O, 1.0 / _determinant(A), O);
}
/* Compute the pseudo-inverse as used in linear regression problems */
template<dim N, dim M>			void _pseudo_inverse(const mat<N,M>& A, mat<N,M>& O)
{
	auto AT = mat<M,N>();
	auto ATA = mat<M,M>();
	auto ATA_inv = mat<M,M>();

	_transpose(A, AT);
	_mult(AT, A, ATA);
	_inverse(ATA, ATA_inv);
	_mult(ATA_inv, AT, O);
}
/* Sum of diagonals of a matrix */
template<dim N, dim M>			type _trace(const mat<N,M>& A)
{
	auto diags = vec<N>();
	auto unit = vec<N>();
	type tr = zero();

	_diag(A, diags);
	_fill(unit, one());

	tr = _dot(diags, unit);

	return tr;
}
/* Outer product of two vectors, produces a matrix */
template<dim N, dim M>			void _outer(const vec<N>& a, const vec<M>& b, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		for (auto j=0; j<M; j++)
		{
			O[i][j] = a[i]*b[j];
		}
	}
}
template<dim N>					type _norm_squared(const vec<N>& a)
{
	auto a_sq = vec<N>();
	_mult(a, a, a_sq);
	return _sum(a_sq);
}
template<dim N, dim M>			type _norm_squared(const mat<N,M>& A)
{
	auto A_sq = mat<N,M>();
	_hadamard(A, A, A_sq);
	return _sum(A_sq);
}
template<dim N>					type _norm(const vec<N>& a)			{ return sqrt(_norm_squared(a)); }
template<dim N, dim M>			type _norm(const mat<N,M>& A)		{ return sqrt(_norm_squared(A)); }
template<dim N>					type _length(const vec<N>& a)		{ return _norm(a); }
template<dim N>					void _normalise(const vec<N>& a, vec<N>& o)
{
	auto a_inv = vec<N>();
	_fill(a_inv, _length(a));
	_inverse(a_inv, a_inv);
	_mult(a, a_inv, o);
}
/* cross product of two vectors */
#if 0
template<dim N>					void _cross(vec<N> a, vec<N> b, vec<N> o)
{
	check_size_match(a, b);
	type d = zero();

	auto A = make(4);	A[0] = a[1]; A[1] = a[2]; A[2] = a[0]; A[3] = d;
	auto B = make(4);	B[0] = b[2]; B[1] = b[0]; B[2] = b[1]; B[3] = d;
	auto C = make(4);	C[0] = a[2]; C[1] = a[0]; C[2] = a[1]; C[3] = d;
	auto D = make(4);	D[0] = b[1]; D[1] = b[2]; D[2] = b[0]; D[3] = d;

	/*
	return sub(mult(make(4, new vec_t{4, new type[4]{a->v[1], a->v[2], a->v[0], d}}),
	make(4, new vec_t{4, new type[4]{b->v[2], b->v[0], b->v[1], d}})),
	mult(make(4, new vec_t{4, new type[4]{a->v[2], a->v[0], a->v[1], d}}),
	make(4, new vec_t{4, new type[4]{b->v[1], b->v[2], b->v[0], d}})));
	*/
	auto AB = make(4); _mult(A, B, AB);
	auto CD = make(4); _mult(C, D, CD);
	auto AB_CD = make(4); _sub(AB, CD, AB_CD);
	
	o[0] = AB_CD[0];
	o[1] = AB_CD[1];
	o[2] = AB_CD[2];
	o[3] = AB_CD[3];

	//cake(A);
	//cake(B);
	//cake(C);
	//cake(D);
	//cake(AB);
	//cake(CD);
	//cake(AB_CD);
}
#endif

#if 0
/// <summary>
/// Apply a geometric translation to a 4x4 matrix
/// </summary>
/// <param name="M">input matrix</param>
/// <param name="v">translation vector</param>
/// <returns>matrix combining the input and the geometric translation</returns>
mat translate(mat M, vec v)
{
	check_size_match(M, vec_size(v), vec_size(v));
	mat c = identity(vec_size(M->v[0]));
	c->v[0]->v[3] = v->v[0];
	c->v[1]->v[3] = v->v[1];
	c->v[2]->v[3] = v->v[2];
	return mult(M, c);
}
/// <summary>
/// Apply geometric rotation to a 4x4 matrix
/// </summary>
/// <param name="M">input matrix</param>
/// <param name="xyz">angles of rotation about the x y and z axes</param>
/// <returns>matrix combining the input and the geometric rotation</returns>
mat rotate(mat M, vec xyz)
{
	check_size_match(M, 4, 4);
	check_size_match(M, xyz);

	var ca = (T)Math.Cos((double)xyz->v[0]);
	var sa = (T)Math.Sin((double)xyz->v[0]);
	var cb = (T)Math.Cos((double)xyz->v[1]);
	var sb = (T)Math.Sin((double)xyz->v[1]);
	var cc = (T)Math.Cos((double)xyz->v[2]);
	var sc = (T)Math.Sin((double)xyz->v[2]);

	var xa = make(4, cb * cc, -ca * sc + sa * sb * cc, ca * sb * cc + sa * sc, default(T));
	var ya = make(4, cb * sc, ca * cc + sa * sb * sc, -sa * cc + ca * sb * sc, default(T));
	var za = make(4, -sb, sa * cb, ca * cb, default(T));
	var wa = fill(copy(za), default(T));

	return mult(M, make(4, 4, xa, ya, za, wa));
}
/// <summary>
/// Creates a 4x4 geometric scaling matrix
/// </summary>
/// <param name="M">input matrix</param>
/// <param name="x">x axis scaling factor</param>
/// <param name="y">y axis scaling factor</param>
/// <param name="z">z axis scaling factor</param>
/// <returns>matrix combining the input and the given scaling factors</returns>
public static T->v[]->v[] scale(T->v[]->v[] M, T x, T y, T z)
{
	check_size_match(M, 4, 4);

	var d = default(T);

	return mult(M, make(4, 4, make(4, x, d, d, d),
		make(4, d, y, d, d),
		make(4, d, d, z, d),
		make(4, d, d, d, 1)));
}
#endif

#if 0
/**
* Symmetric Householder reduction to tridiagonal form.
*/
private static void tred2(T->v[]->v[] V, T->v[] d, T->v[] e)
{
	var n = V.Length;

	//  This is derived from the Algol procedures tred2 by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.
	int i, j, k;
	T	f, g, h, hh;

	for (j = 0; j < n; j++)							{ d->v[j] = V->v[n - 1]->v[j]; }

	// Householder reduction to tridiagonal form.
	for (i = n - 1; i > 0; i--)
	{
		// Scale to avoid under/overflow.
		double scale = 0.0;
		double h2 = 0.0;

		for (k = 0; k < i; k++)						{ scale = scale + Math.Abs(d->v[k]); }

		if (scale == 0.0)
		{
			e->v[i] = d->v[i - 1];

			for (j = 0; j < i; j++)
			{
				d->v[j] = V->v[i - 1]->v[j];
				V->v[i]->v[j] = 0.0;
				V->v[j]->v[i] = 0.0;
			}
		}
		else
		{
			// Generate Householder vector.
			for (k = 0; k < i; k++)					{ d->v[k] /= scale; h2 += d->v[k] * d->v[k]; }

			f = d->v[i - 1];
			g = Math.Sqrt(h2);

			g = (f > 0) ? (-g) : (g);

			e->v[i] = scale * g;
			h2 -= f * g;
			d->v[i - 1] = f - g;

			for (j = 0; j < i; j++)					{ e->v[j] = 0.0; }

			// Apply similarity transformation to remaining columns.
			for (j = 0; j < i; j++)
			{
				f = d->v[j];
				V->v[j]->v[i] = f;
				g = e->v[j] + V->v[j]->v[j] * f;

				for (k = j + 1; k <= i - 1; k++)
				{
					g += V->v[k]->v[j] * d->v[k];
					e->v[k] += V->v[k]->v[j] * f;
				}

				e->v[j] = g;
			}

			f = 0.0;

			for (j = 0; j < i; j++)					{ e->v[j] /= h2; f += e->v[j] * d->v[j]; }

			hh = f / (h2 + h2);

			for (j = 0; j < i; j++)					{ e->v[j] -= hh * d->v[j]; }

			for (j = 0; j < i; j++)
			{
				f = d->v[j];
				g = e->v[j];

				for (k = j; k <= i - 1; k++)			{ V->v[k]->v[j] -= (f * e->v[k] + g * d->v[k]); }

				d->v[j] = V->v[i - 1]->v[j];
				V->v[i]->v[j] = 0.0;
			}
		}

		d->v[i] = h2;
	}

	// Accumulate transformations
	for (i = 0; i < n - 1; i++)
	{
		V->v[n - 1]->v[i] = V->v[i]->v[i];
		V->v[i]->v[i] = 1.0;
		h = d->v[i + 1];

		if (h != 0.0)
		{
			for (k = 0; k < i + 1; k++)
			{
				d->v[k] = V->v[k]->v[i + 1] / h;
			}

			for (j = 0; j < i + 1; j++)
			{
				g = 0.0;

				for (k = 0; k < i + 1; k++)
				{
					g += V->v[k]->v[i + 1] * V->v[k]->v[j];
				}

				for (k = 0; k < i + 1; k++)
				{
					V->v[k]->v[j] -= g * d->v[k];
				}
			}
		}

		for (k = 0; k < i + 1; k++)
		{
			V->v[k]->v[i + 1] = 0.0;
		}
	}

	for (j = 0; j < n; j++)
	{
		d->v[j] = V->v[n - 1]->v[j];
		V->v[n - 1]->v[j] = 0.0;
	}

	V->v[n - 1]->v[n - 1] = 1.0;
	e->v[0] = 0.0;
}
/**
* Symmetric tridiagonal QL algorithm.
*/
private static void tql2(T->v[]->v[] V, T->v[] d, T->v[] e)
{
	var N = V.Length;

	//  This is derived from the Algol procedures tql2, by
	//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
	//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
	//  Fortran subroutine in EISPACK.
	int i, m, l, k;
	T	g, p, r, dl1, h, f, tst1, eps;
	T	c, c2, c3, el1, s, s2;

	for (i = 1; i < N; i++)
	{
		e->v[i - 1] = e->v[i];
	}

	e->v[N - 1] = 0.0;

	f = 0.0;
	tst1 = 0.0;
	eps = Math.Pow(2.0, -52.0);

	for (l = 0; l < N; l++)
	{
		// Find small subdiagonal element
		tst1 = Math.Max(tst1, Math.Abs(d->v[l]) + Math.Abs(e->v[l]));

		for (m = l; m < N; m++)
		{
			if (Math.Abs(e->v[m]) <= eps*tst1)
			{
				break;
			}
		}

		// If m == l, d->v[l] is an eigenvalue,
		// otherwise, iterate.
		if (m > l)
		{
			int iter = 0;

			do
			{
				iter = iter + 1;  // (Could check iteration count here.)

				// Compute implicit shift
				g = d->v[l];
				p = (d->v[l + 1] - g) / (2.0 * e->v[l]);
				//r = hypot2(p,1.0);
				r = length(make(2, p, 1.0));

				r = (p < 0) ? (-r) : (r);

				d->v[l] = e->v[l] / (p + r);
				d->v[l + 1] = e->v[l] * (p + r);
				dl1 = d->v[l + 1];
				h = g - d->v[l];

				for (i = l + 2; i < N; i++)		{ d->v[i] -= h; }

				f = f + h;

				// Implicit QL transformation.
				p = d->v[m];
				c = 1.0;
				c2 = c;
				c3 = c;
				el1 = e->v[l + 1];
				s = 0.0;
				s2 = 0.0;

				for (i = m - 1; i >= l; i--)
				{
					c3 = c2;
					c2 = c;
					s2 = s;
					g = c * e->v[i];
					h = c * p;
					//r = hypot2(p,e->v[i]);
					r = length(make(2, p, e->v[i]));
					e->v[i + 1] = s * r;
					s = e->v[i] / r;
					c = p / r;
					p = c * d->v[i] - s * g;
					d->v[i + 1] = h + s * (c * g + s * d->v[i]);

					// Accumulate transformation
					for (k = 0; k < N; k++)
					{
						h = V->v[k]->v[i + 1];
						V->v[k]->v[i + 1] = s * V->v[k]->v[i] + c * h;
						V->v[k]->v[i] = c * V->v[k]->v[i] - s * h;
					}
				}

				p = -s * s2 * c3 * el1 * e->v[l] / dl1;
				e->v[l] = s * p;
				d->v[l] = c * p;

				// Check for convergence
			} while (Math.Abs(e->v[l]) > eps*tst1);
		}

		d->v[l] = d->v[l] + f;
		e->v[l] = 0.0;
	}
}
/**
* Sort eigenvalues and corresponding vectors.
*/
private static void sortBy(ref T->v[]->v[] V, ref T->v[] d)
{
	check_size_match(V, d);

	var n = V.Length;
	int i, j, k;
	T p;

	for (i = 0; i < n - 1; i++)
	{
		k = i;
		p = d->v[i];

		for (j = i + 1; j < n; j++)
		{
			if (d->v[j] < p)
			{
				k = j;
				p = d->v[j];
			}
		}

		if (k != i)
		{
			d->v[k] = d->v[i];
			d->v[i] = p;

			// not sure why the sorting is based on these indices
			// but we cannot change it here, it is within the 
			// algorithm. Post-transpose?
			for (j = 0; j < n; j++)
			{
				p = V->v[j]->v[i];
				V->v[j]->v[i] = V->v[j]->v[k];
				V->v[j]->v[k] = p;
			}
		}
	}
}
/// <summary>
/// Compute the eigenvectors and values of a given matrix A.
/// Eigenvectors are stored in V,
/// Eigenvalues are stores in d
/// Currently returns the array sorted back to front.
/// NB: not sure this algorithm is ordered correctly.
/// </summary>
/// <param name="A">input matrix</param>
/// <param name="V">output matrix of eigenvectors</param>
/// <param name="d">output vector of eigenvalues</param>
public static void eig(T->v[]->v[] A, ref T->v[]->v[] V, ref T->v[] d)
{
	int n = sizeof_mat(A);

	int i, j;
	var e = new double->v[n];

	for (i = 0; i < n; i++)
	{
		for (j = 0; j < n; j++)
		{
			V->v[i]->v[j] = A->v[i]->v[j];
		}
	}

	tred2(V, d, e);
	tql2(V, d, e);

	sortBy(ref V, ref d);

	//V = transpose(V);
}
#endif

template<dim N, dim M>				void cholesky_decomposition(const mat<N,M>& A, mat<N,M>& O)
{
	for (auto i=0; i<N; i++)
	{
		for (auto j=0; j<i+1; j++)
		{
			auto s = 0.0;

			for (auto k=0; k<j; k++)
			{
				s += O[i][k] * O[j][k];
			}

			if (i == j)
			{
				O[i][j] = sqrt(A[i][i] - s);
			}
			else
			{
				O[i][j] = 1.0 / (O[j][j] * (A[i][j] - s));
			}
		}
	}
}

/// <summary>
/// Checks if a given matrix is the identity matrix
/// </summary>
/// <param name="A">matrix to test</param>
/// <param name="tolerance">tolerance for the value comparisons</param>
/// <returns>true if the matrix is the identity, false otherwise</returns>
template<dim N, dim M>				bool is_identity (const mat<N,M>& A, const type tolerance = 0.0001)
{
	// should only have 1.0 in the diagonals, so the sum is the
	// size of the matrix (allowing for rounding error)
	if (abs(_sum(A) - N) > tolerance)
	{
		return 0;
	}

	// check the actual diagonals
	for (auto i=0; i<N;i++)
	{
		if (abs(1.0 - A[i][i]) > tolerance)
		{
			return false;
		}
	}

	return true;
}
template<dim N, dim M>				void print_matrix (const mat<N,M>& A)
{
	for (auto i=0;i<M;i++)
	{
		for (auto j=0;j<N;j++)
		{
			fprintf(stdout, "%0.4f \0", A[i][j]);
		}
		fprintf (stdout, "\n\0");
	}
}
/// <summary>
/// checks if a matrix is a square matrix
/// </summary>
/// <param name="A">matrix to test</param>
/// <returns>true if the matrix is square, false otherwise</returns>
template<dim N>						bool is_square(const mat<N,N>& A)
{
	return true;
}
template<dim N, dim M>				bool is_square(const mat<N,M>& A)
{
	return false;
}
template<dim N, dim M>				bool is_transpose(const mat<N,M>& A, const mat<M,N>& B, type tolerance = 0.0001)
{
	dim i, j;

	if (A.size() != B[0].size())			{ return 0; }
	if (A[0].size() != B.size())			{ return 0; }

	for (i = 0; i < A.size(); i++)
	{
		for (j = 0; j < A[0].size(); j++)
		{
			if (B[j][i] - A[i][j] > tolerance)
			{
				return 0;
			}
		}
	}

	return 1;
}

template<dim N, dim M>				bool are_equal(const mat<N,M>& A, const mat<N,M>& B, type tolerance = 0.0001)
{
	for (auto i=0;i<N;i++)
	{
		for (auto j=0;j<M;j++)
		{
			if (abs(A[i][j] - B[i][j]) > tolerance)
			{
				return 0;
			}
		}
	}

	return 1;
}

template<dim N, dim M>				void _least_squares_regression(const mat<N,M>& design_matrix, const mat<N,1>& observations, mat<1,M>& O)
{
	auto ps_inv = mat<N,N>();

	_pseudo_inverse(design_matrix, ps_inv);
	_mult(ps_inv, observations, O);
}
