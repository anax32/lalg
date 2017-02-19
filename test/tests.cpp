#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define BIT_32		1

#include "mats.h"
#include "tests.h"

void vector_make_test()
{
	auto a = vec<5>();

	assert_is_true(a.size () == 5);
	assert_is_true(a.data () != NULL);
}
void vector_counter_test()
{
	auto a = vec<5>();
	_counter(a);
	assert_are_equal(a[0], 1);
	assert_are_equal(a[1], 2);
	assert_are_equal(a[2], 3);
	assert_are_equal(a[3], 4);
	assert_are_equal(a[4], 5);
}
void vector_sum_test()
{
	auto a = vec<5>();
	_counter(a);
	assert_are_equal(1 + 2 + 3 + 4 + 5, _sum(a));
}
void vector_add_test()
{
	dim i;
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	_counter(a);
	_ones(b);
	_add(a, b, c);

	for (i = 0; i < c.size(); i++)
	{
		assert_are_equal(a[i] + b[i], c[i]);
	}
}
void vector_sub_test()
{
	dim i;
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	_counter(a);
	_ones(b);
	_sub(a, b, c);

	for (i = 0; i < c.size(); i++)
	{
		assert_are_equal(a[i] - b[i], c[i]);
	}
}
void vector_mult_test()
{
	dim i;
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	_counter(a);
	_fill(b, 2.0);
	_mult(a, b, c);

	for (i=0; i<c.size(); i++)
	{
		assert_are_equal(a[i] * b[i], c[i]);
	}
}
void vector_product_sum_test()
{
	auto a = vec<5>();
	a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4; a[4] = 5;
	assert_are_equal(1 * 2 * 3 * 4 * 5, product_sum(a));
}
void vector_multiply_test()
{
	dim i;

	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	_counter(a);
	_fill(b, 2.0);
	a[0] = 1.0;
	a[1] = 2.0;
	a[2] = 3.0;
	a[3] = 4.0;
	a[4] = 5.0;

	_mult(a, b, c);

	assert_are_equal(a.size(), b.size());
	assert_are_equal(a.size(), c.size());
	assert_are_equal(b.size(), c.size());

	for (i = 0; i<a.size(); i++)
	{
		assert_are_equal(a[i]*b[i], c[i]);
	}
}
void vector_cross_test()
{
#if 0
	auto a = vec<4>();
	auto b = vec<4>();
	auto c = vec<4>();

	a[0] = 1; a[1] = 0; a[2] = 0; a[3] = 0;
	b[0] = 0; b[1] = 1; b[2] = 0; b[3] = 0;
	_cross(a, b, c);

	assert_are_equal(a.size(), b.size());
	assert_are_equal(a.size(), c.size());

	assert_are_equal_t(0.0, c[0], 0.00001);
	assert_are_equal_t(0.0, c[1], 0.00001);
	assert_are_equal_t(1.0, c[2], 0.00001);
	assert_are_equal_t(0.0, c[3], 0.00001);
#endif
}
void vector_minor_test()
{
	auto a = vec<5>();
	a[0] = 1; a[1] = 2; a[2] = 3; a[3] = 4; a[4] = 5;

	auto am = vec<4>();

	_minor(a, 2, am);

	assert_are_equal(a.size() - 1, am.size());
	assert_are_equal(12, _sum(am));
}
void vector_many_add_test()
{
	dim i, j;
	auto a = vec<ALL_SIZE>();
	auto b = vec<ALL_SIZE>();
	auto c = vec<ALL_SIZE>();

	for (i=0; i<ALL_SIZE; i++)
	{
		_counter(a);
		_fill(b, i);
		_add(a, b, c);

		for (j = 0; j < c.size(); j++)
		{
			assert_are_equal(a[j]+b[j], c[j]);
		}
	}
}
void vector_many_mult_test()
{
	dim i, j;
	auto a = vec<ALL_SIZE>();
	auto b = vec<ALL_SIZE>();
	auto c = vec<ALL_SIZE>();

	for (i = 0; i<ALL_SIZE; i++)
	{
		_counter(a);
		_fill(b, i);
		_mult(a, b, c);

		for (j = 0; j < c.size(); j++)
		{
			assert_are_equal(a[j]*b[j], c[j]);
		}
	}
}
void vector_function_tests()
{
	TEST(vector_counter_test);
	TEST(vector_sum_test);
	TEST(vector_add_test);
	TEST(vector_sub_test);
	TEST(vector_mult_test);
	TEST(vector_product_sum_test);
	TEST(vector_multiply_test);
	TEST(vector_cross_test);
	TEST(vector_minor_test);
}
void vector_stress_tests()
{
	TEST(vector_many_add_test);
	TEST(vector_many_mult_test);
}
void matrix_cake_test()
{
	auto A = mat<4,4>();

	assert_is_true(A.size () == 4);
	assert_is_true(A[0].size () == 4);
}
void matrix_smel_test()
{
	dim i, j, k;
	auto A = mat<3,5>();

	// FIXME: use the counter function
	A[0][0] = 1.0;	A[0][1] = 2.0;	A[0][2] = 3.0;	A[0][3] = 4.0; 	A[0][4] = 5.0;
	A[1][0] = 6.0;	A[1][1] = 7.0;	A[1][2] = 8.0;	A[1][3] = 9.0; 	A[1][4] = 10.0;
	A[2][0] = 11.0;	A[2][1] = 12.0;	A[2][2] = 13.0;	A[2][3] = 14.0; A[2][4] = 15.0;

	for (i=0,k=1; i<3; i++)
	{
		for (j=0; j<5; j++, k++)
		{
			assert_are_equal(k, A[i][j]);
		}
	}
}
void matrix_gmel_test()
{
	dim i, j, k;
	auto A = mat<3, 5>();

	// FIXME: use the counter function
	A[0][0] = 1.0;	A[0][1] = 2.0;	A[0][2] = 3.0;	A[0][3] = 4.0; 	A[0][4] = 5.0;
	A[1][0] = 6.0;	A[1][1] = 7.0;	A[1][2] = 8.0;	A[1][3] = 9.0; 	A[1][4] = 10.0;
	A[2][0] = 11.0;	A[2][1] = 12.0;	A[2][2] = 13.0;	A[2][3] = 14.0; A[2][4] = 15.0;

	for (i=0, k=1; i<3; i++)
	{
		for (j=0; j<5; j++, k++)
		{
			assert_are_equal(k, A[i][j]);
		}
	}
}
void matrix_set_row_test()
{
	dim i;
	auto A = mat<3, 5>();
	auto r1 = vec<5>();
	auto r2 = vec<5>();
	auto r3 = vec<5>();

	r1[0] =  1.0; r1[1] =  2.0; r1[2] =  3.0; r1[3] =  4.0; r1[4] =  5.0;
	r2[0] =  6.0; r2[1] =  7.0; r2[2] =  8.0; r2[3] =  9.0; r2[4] = 10.0;
	r3[0] = 11.0; r3[1] = 12.0; r3[2] = 13.0; r3[3] = 14.0; r3[4] = 15.0;

	A[0] = r1;
	A[1] = r2;
	A[2] = r3;

	for (i=0; i<5;i++)
	{
		assert_are_equal(A[0][i], r1[i]);
	}

	for (i=0; i<5; i++)
	{
		assert_are_equal(A[1][i], r2[i]);
	}

	for (i=0; i<5; i++)
	{
		assert_are_equal(A[2][i], r3[i]);
	}
}
void matrix_gmrow_test()
{
	dim i;
	auto A = mat<3, 5>();
	auto r1 = vec<5>();
	auto r2 = vec<5>();
	auto r3 = vec<5>();

	r1[0] =  1.0; r1[1] =  2.0; r1[2] =  3.0; r1[3] =  4.0; r1[4] =  5.0;
	r2[0] =  6.0; r2[1] =  7.0; r2[2] =  8.0; r2[3] =  9.0; r2[4] = 10.0;
	r3[0] = 11.0; r3[1] = 12.0; r3[2] = 13.0; r3[3] = 14.0; r3[4] = 15.0;

	A[0] = r1;
	A[1] = r2;
	A[2] = r3;

	auto A_r1 = A[0];
	for (i = 0; i<5; i++)
	{
		assert_are_equal(A_r1[i], r1[i]);
	}

	auto A_r2 = A[1];
	for (i = 0; i<5; i++)
	{
		assert_are_equal(A_r2[i], r2[i]);
	}

	auto A_r3 = A[2];
	for (i = 0; i<5; i++)
	{
		assert_are_equal(A_r3[i], r3[i]);
	}
}
void matrix_are_equal_test()
{
	auto A = mat<8, 5>();
	auto B = mat<8, 5>();

	_counter(A);
	_counter(B);

	assert_is_true(are_equal(A, B));
}
void matrix_are_equal_fails_test()
{
	auto A = mat<8, 5>();
	auto B = mat<8, 5>();

	_counter(A);
	_fill(B, zero());

	assert_is_true(are_equal(A, B) == false);
}
void matrix_transpose_2x2_test()
{
	auto A = mat<2, 2>(); _counter(A);
	auto B = mat<2, 2>();
	auto C = mat<2, 2>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_transpose_5x5_test()
{
	auto A = mat<5, 5>(); _counter(A);
	auto B = mat<5, 5>();
	auto C = mat<5, 5>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_transpose_1x2_test()
{
	auto A = mat<1, 2>(); _counter(A);
	auto B = mat<2, 1>();
	auto C = mat<2, 1>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_transpose_2x1_test()
{
	auto A = mat<2, 1>(); _counter(A);
	auto B = mat<1, 2>();
	auto C = mat<1, 2>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_transpose_3x5_test()
{
	auto A = mat<3, 5>(); _counter(A);
	auto B = mat<5, 3>();
	auto C = mat<5, 3>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_transpose_5x3_test()
{
	auto A = mat<5, 3>(); _counter(A);
	auto B = mat<3, 5>();
	auto C = mat<3, 5>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_transpose_200x200_test()
{
	auto A = mat<200, 200>(); _counter(A);
	auto B = mat<200, 200>();
	auto C = mat<200, 200>(); _counter(C);

	_transpose(A, B);

	assert_is_true(is_transpose(A, B));
}
void matrix_add_2x2_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();
	auto D = mat<2, 2>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 0.0);
	_fill(D, 2.0);

	_add(A, B, C);
	assert(are_equal(C, D));
}
void matrix_add_200x200_test()
{
	auto A = mat<200, 200>();
	auto B = mat<200, 200>();
	auto C = mat<200, 200>();
	auto D = mat<200, 200>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 0.0);
	_fill(D, 2.0);

	_add(A, B, C);
	assert(are_equal(C, D));
}
void matrix_add_1x3_test()
{
	auto A = mat<1, 3>();
	auto B = mat<1, 3>();
	auto C = mat<1, 3>();
	auto D = mat<1, 3>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 0.0);
	_fill(D, 2.0);

	_add(A, B, C);
	assert(are_equal(C, D));
}
void matrix_add_3x1_test()
{
	auto A = mat<3, 1>();
	auto B = mat<3, 1>();
	auto C = mat<3, 1>();
	auto D = mat<3, 1>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 0.0);
	_fill(D, 2.0);

	_add(A, B, C);
	assert(are_equal(C, D));
}
void matrix_sub_2x2_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();
	auto D = mat<2, 2>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 3.0);
	_fill(D, 0.0);

	_sub(A, B, C);
	assert(are_equal(C, D));
}
void matrix_sub_1x3_test()
{
	auto A = mat<1, 3>();
	auto B = mat<1, 3>();
	auto C = mat<1, 3>();
	auto D = mat<1, 3>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 3.0);
	_fill(D, 0.0);

	_sub(A, B, C);
	assert(are_equal(C, D));
}
void matrix_sub_3x1_test()
{
	auto A = mat<3, 1>();
	auto B = mat<3, 1>();
	auto C = mat<3, 1>();
	auto D = mat<3, 1>();

	_fill(A, 1.0);
	_fill(B, 1.0);
	_fill(C, 3.0);
	_fill(D, 0.0);

	_sub(A, B, C);
	assert(are_equal(C, D));
}
void matrix_mult_identity_identity_test()
{
	auto A = mat<2, 2>(); _identity(A);
	auto B = mat<2, 2>(); _identity(B);
	auto C = mat<2, 2>();

	_mult(A, B, C);

	assert_is_true(is_identity(C));
}
void matrix_mult_identity_lehmer_test()
{
	auto A = mat<5, 5>(); _identity(A);
	auto B = mat<5, 5>(); _lehmer(B);
	auto C = mat<5, 5>();
	_mult(A, B, C);

	assert_is_true(are_equal(B, C));
}
void matrix_mult_15x3_3x15_test()
{
	auto A = mat<15,  3>();
	auto B = mat< 3, 15>();
	auto C = mat<15, 15>();
	auto D = mat<15, 15>();

	_fill(A, 2.0);
	_fill(B, 3.0);
	_fill(C, 0.0);
	_fill(D, (2.0*3.0)*3.0);

	_mult(A, B, C);

	assert(C.size() == D.size());
	assert(C[0].size() == D[0].size());
	assert(are_equal(C, D));
}
void matrix_mult_3x15_15x3_test()
{
	auto A = mat< 3, 15>();
	auto B = mat<15,  3>();
	auto C = mat< 3,  3>();
	auto D = mat< 3,  3>();

	_fill(A, 3.0);
	_fill(B, 4.0);
	_fill(C, 0.0);
	_fill(D, (3.0*4.0)*15.0);

	_mult(A, B, C);

	assert(C.size() == D.size());
	assert(C[0].size() == D[0].size());
	assert(are_equal(C, D));
}
void matrix_hadamard_identity_identity_test()
{
	auto A = mat<2, 2>(); _identity(A);
	auto B = mat<2, 2>(); _identity(B);
	auto C = mat<2, 2>();

	_hadamard(A, B, C);

	assert_is_true(is_identity(C));
}
void matrix_diag_test()
{
	dim i;
	auto A = mat<5, 5>();
	auto diags = vec<5>();

	_map(A, rando, A);
	_diag(A, diags);

	assert_are_equal(A.size(), diags.size());

	for (i = 0; i<diags.size(); i++)
	{
		assert_are_equal(A[i][i], diags[i]);
	}
}
void matrix_identity_test()
{
	auto I = mat<6, 6>();
	_identity(I);
	assert_is_true(is_identity(I));
}
void matrix_minor_test()
{
	auto A = mat<5, 5>();
	auto Am = mat<4, 4>();

	_identity(A);
	_minor(A, 2, 2, Am);

	assert_are_equal(A.size() - 1, Am.size());
	assert_are_equal(A[0].size() - 1, Am[0].size());
	assert_is_true(is_identity(Am));
}
void matrix_determinant_2x2_test()
{
	auto A = mat<2, 2>();

	A[0][0] = 1;
	A[0][1] = 2;
	A[1][0] = -1;
	A[1][1] = 1;
	
	auto d = _determinant(A);

	assert_are_equal(3.0, d);
}
void matrix_determinant_3x3_test()
{
	auto A = mat<3, 3>();

	A[0][0] = 1; A[0][1] = 2; A[0][2] = 0;
	A[1][0] =-1; A[1][1] = 1; A[1][2] = 1;
	A[2][0] = 1; A[2][1] = 2; A[2][2] = 3;

	auto d = _determinant(A);

	assert_are_equal(9.0, d);
}
void matrix_determinant_4x4_test()
{
	auto A = mat<4, 4>();

	A[0][0]=1; A[0][1]= 3; A[0][2]=-2; A[0][3]= 1;
	A[1][0]=5; A[1][1]= 1; A[1][2]= 0; A[1][3]=-1;
	A[2][0]=0; A[2][1]= 2; A[2][2]= 0; A[2][3]=-2;
	A[3][0]=2; A[3][1]=-1; A[3][2]= 0; A[3][3]= 3;

	auto d = _determinant(A);
	assert_are_equal(-40.0, d);
}
void matrix_determinant_4x4_lehmer_test()
{
	auto A = mat<4, 4>();
	_lehmer(A);
	auto d = _determinant(A);
	assert_are_equal_t(0.1823, d, 0.0001);
}
void matrix_inverse_2x2_test()
{
	auto A = mat<2, 2>();
	auto A_inv = mat<2, 2>();
	auto I = mat<2, 2>();

	A[0][0] = 1.0;
	A[0][1] = 2.0;
	A[1][0] = -1.0;
	A[1][1] = 1.0;

	_inverse(A, A_inv);

	assert_are_equal_t( 0.33333, A_inv[0][0], 0.002);
	assert_are_equal_t(-0.66666, A_inv[0][1], 0.002);
	assert_are_equal_t( 0.33333, A_inv[1][0], 0.002);
	assert_are_equal_t( 0.33333, A_inv[1][1], 0.002);

	_mult(A, A_inv, I);
	assert_is_true(is_identity(I));
}
void matrix_inverse_3x3_test()
{
	auto A = mat<3, 3>();
	auto A_inv = mat<3, 3>();
	auto I = mat<3, 3>();

	A[0][0] = 1.0;	A[0][1] = 2.0;	A[0][2] = 0.0;
	A[1][0] =-1.0;	A[1][1] = 1.0;	A[1][2] = 1.0;
	A[2][0] = 1.0;	A[2][1] = 2.0;	A[2][2] = 3.0;

	_inverse(A, A_inv);
	_mult(A, A_inv, I);

	assert_is_true(is_identity(I));
}
void matrix_inverse_4x4_test()
{
	auto A = mat<4, 4>();
	auto A_inv = mat<4, 4>();
	auto I = mat<4, 4>();

	_lehmer(A);
	_inverse(A, A_inv);
	_mult(A, A_inv, I);

	assert_is_true(is_identity(I));
}
void matrix_inverse_10x10_test()
{
	auto A = mat<10, 10>();
	auto A_inv = mat<10, 10>();
	auto I = mat<10, 10>();

	_lehmer(A);
	_inverse(A, A_inv);
	_mult(A, A_inv, I);

	assert_is_true(is_identity(I));
}
void matrix_inverse_100x100_test()
{
	auto A = mat<100, 100>();
	auto A_inv = mat<100, 100>();
	auto I = mat<100, 100>();

	_lehmer(A);
	_inverse(A, A_inv);
	_mult(A, A_inv, I);

	assert_is_true(is_identity(I));
}
void matrix_cholesky_decompsition_3x3_test()
{
	auto A = mat<3, 3>();
	auto L = mat<3, 3>();

	// input
	// NB: this input comes from wikipedia:
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#Example
	A[0][0] =   4; A[0][1] =  12; A[0][2] = -16;
	A[1][0] =  12; A[1][1] =  37; A[1][2] = -43;
	A[2][0] = -16; A[2][1] = -43; A[2][2] =  98;

	cholesky_decomposition(A, L);

	// check the evecs
	assert_are_equal ( 2.0, L[0][0]);
	assert_are_equal ( 0.0, L[0][1]);
	assert_are_equal ( 0.0, L[0][2]);
	assert_are_equal ( 6.0, L[1][0]);
	assert_are_equal ( 1.0, L[1][1]);
	assert_are_equal ( 0.0, L[1][2]);
	assert_are_equal (-8.0, L[2][0]);
	assert_are_equal ( 5.0, L[2][1]);
	assert_are_equal ( 3.0, L[2][2]);
}
void matrix_many_transpose_test()
{
	unsigned int i;
	auto A = mat<50, 50>();
	auto B = mat<50, 50>();

	_counter(A);

	for (i = 0; i<ALL_SIZE; i++)
	{
		_transpose(A, B);
		assert_is_true(is_transpose(A, B));
		_copy(B, A);
	}
}
void matrix_many_add_test()
{
	unsigned int i;
	auto A = mat<50, 50>();
	auto B = mat<50, 50>();
	auto C = mat<50, 50>();

	_counter(A);
	_fill(B, one());

	for (i = 0; i<ALL_SIZE; i++)
	{
		_fill(B, (type)i);
		_add(A, B, C);
		// FIXME: what should the succint test be?
	}
}
void matrix_many_mult_test()
{
	unsigned int i;
	auto A = mat<50, 50>();
	auto B = mat<50, 50>();
	auto C = mat<50, 50>();

	_counter(A);
	_inverse(A, B);

	for (i=0; i<ALL_SIZE; i++)
	{
		_fill(C, (type)i);
		_mult(A, B, C);

		assert_is_true(is_identity(C));
	}
}
void matrix_many_hadamard_test()
{
	unsigned int i;
	auto A = mat<5, 5>();
	auto B = mat<5, 5>();
	auto C = mat<5, 5>();

	_counter(A);
	_ones(B);

	for (i=0; i<ALL_SIZE; i++)
	{
		_fill(C, (type)i);
		_hadamard(A, B, C);

		assert_is_true(are_equal(A, C));
	}
}
void matrix_function_tests()
{
	TEST(matrix_smel_test);
	TEST(matrix_gmel_test);
	TEST(matrix_set_row_test);
	TEST(matrix_gmrow_test);

	TEST(matrix_are_equal_test);
	TEST(matrix_are_equal_fails_test);

	TEST(matrix_transpose_2x2_test);
	TEST(matrix_transpose_5x5_test);
	TEST(matrix_transpose_1x2_test);
	TEST(matrix_transpose_2x1_test);
	TEST(matrix_transpose_3x5_test);
	TEST(matrix_transpose_5x3_test);
	TEST(matrix_transpose_200x200_test);

	TEST(matrix_add_2x2_test);
	TEST(matrix_add_1x3_test);
	TEST(matrix_add_3x1_test);
	TEST(matrix_add_200x200_test);

	TEST(matrix_sub_2x2_test);
	TEST(matrix_sub_1x3_test);
	TEST(matrix_sub_3x1_test);

	TEST(matrix_mult_identity_identity_test);
	TEST(matrix_mult_identity_lehmer_test);
	TEST(matrix_mult_15x3_3x15_test);
	TEST(matrix_mult_3x15_15x3_test);

	TEST(matrix_hadamard_identity_identity_test);

	TEST(matrix_diag_test);

	TEST(matrix_identity_test);

	TEST(matrix_minor_test);

	TEST(matrix_determinant_2x2_test);
	TEST(matrix_determinant_3x3_test);
	TEST(matrix_determinant_4x4_test);
	TEST(matrix_determinant_4x4_lehmer_test);

	TEST(matrix_inverse_2x2_test);
	TEST(matrix_inverse_3x3_test);
	TEST(matrix_inverse_4x4_test);
	TEST(matrix_inverse_10x10_test);
	TEST(matrix_inverse_100x100_test);

//	TEST(matrix_cholesky_decompsition_3x3_test);
}
void matrix_stress_tests()
{
	TEST(matrix_many_transpose_test);
	TEST(matrix_many_add_test);
	TEST(matrix_many_mult_test);
	TEST(matrix_many_hadamard_test);
}

#if 0
void EigenDecomposition_2x2_Test()
{
	var V = make(2, 2);
	var d = make(2);
	var A = make(2, 2);
	A->v[0]->v[0] = 2; A->v[0]->v[1] = 1;
	A->v[1]->v[0] = 1; A->v[1]->v[1] = 2;

	// matlab: "A = ->v[2 1; 1 2]"
	//         "->v[V, D] = eig(A)"
	eig(A, ref V, ref d);

	assert_are_equal(1.0, d->v[0], 0.00001);
	assert_are_equal(3.0, d->v[1], 0.00001);

	// NB: this does not match the matlab results
	assert_are_equal(0.70710678118654746, V->v[0]->v[0], 0.0000001);
	assert_are_equal(0.70710678118654746, V->v[0]->v[1], 0.0000001);
	assert_are_equal(-0.70710678118654746, V->v[1]->v[0], 0.0000001);
	assert_are_equal(0.70710678118654746, V->v[1]->v[1], 0.0000001);
}
void EigenDecomposition_3x3_Test()
{
	var A = make(3, 3);
	var V = make(3, 3);
	var d = make(3);

	// input
	// NB: this input comes from wikipedia:
	// https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors#Diagonalization_and_the_eigendecomposition
	// "Three-dimensional example"
	// The eigenvectors given on that page do not correspond to the results here.
	// First the scaling is off (their vectors are not normalised)
	// Second, the values are in the wrong locations in the vector.
	// The results here correspond with running the same input through matlab, so I assume
	// the wikipedia page is incorrect (wrong handedness maybe?)
	A->v[0]->v[0] = 2; A->v[0]->v[1] = 0; A->v[0]->v[2] = 0;
	A->v[1]->v[0] = 0; A->v[1]->v[1] = 3; A->v[1]->v[2] = 4;
	A->v[2]->v[0] = 0; A->v[2]->v[1] = 4; A->v[2]->v[2] = 9;

	// matlab input: "A = ->v[2 0 0; 0 3 4; 0 4 9]"
	//               "->v[V, D] = eig (A)"
	eig(A, ref V, ref d);

	// check the eigenvalues
	assert_are_equal(1.0, d->v[0]);
	assert_are_equal(2.0, d->v[1]);
	assert_are_equal(11.0, d->v[2]);

	// check the evecs
	assert_are_equal(0.0000, V->v[0]->v[0], 0.0001);
	assert_are_equal(1.0000, V->v[0]->v[1], 0.0001);
	assert_are_equal(0.0000, V->v[0]->v[2], 0.0001);
	assert_are_equal(-0.8944, V->v[1]->v[0], 0.0001);
	assert_are_equal(0.0000, V->v[1]->v[1], 0.0001);
	assert_are_equal(0.4472, V->v[1]->v[2], 0.0001);
	assert_are_equal(0.4472, V->v[2]->v[0], 0.0001);
	assert_are_equal(0.0000, V->v[2]->v[1], 0.0001);
	assert_are_equal(0.8944, V->v[2]->v[2], 0.0001);
}
void EigenDecomposition_4x4_Test()
{
	var A = make(4, 4);
	var V = make(4, 4);
	var d = make(4);

	// output from matlab "A = gallery('lehmer', 4)"
	A->v[0]->v[0] = 1.0 / 1.0; A->v[0]->v[1] = 1.0 / 2.0; A->v[0]->v[2] = 1.0 / 3.0; A->v[0]->v[3] = 1.0 / 4.0;
	A->v[1]->v[0] = 1.0 / 2.0; A->v[1]->v[1] = 2.0 / 2.0; A->v[1]->v[2] = 2.0 / 3.0; A->v[1]->v[3] = 2.0 / 4.0;
	A->v[2]->v[0] = 1.0 / 3.0; A->v[2]->v[1] = 2.0 / 3.0; A->v[2]->v[2] = 3.0 / 3.0; A->v[2]->v[3] = 3.0 / 4.0;
	A->v[3]->v[0] = 1.0 / 4.0; A->v[3]->v[1] = 2.0 / 4.0; A->v[3]->v[2] = 3.0 / 4.0; A->v[3]->v[3] = 4.0 / 4.0;

	eig(A, ref V, ref d);

	// output from matlab "->v[V,D] = eig (A)"
	assert_are_equal(0.2078, d->v[0], 0.001);
	assert_are_equal(0.4078, d->v[1], 0.001);
	assert_are_equal(0.8482, d->v[2], 0.001);
	assert_are_equal(2.5362, d->v[3], 0.001);

	// NB: rows in the matrix correspond to rows in matlab
	assert_are_equal(0.0693, V->v[0]->v[0], 0.001);
	assert_are_equal(-0.4422, V->v[0]->v[1], 0.001);
	assert_are_equal(-0.8105, V->v[0]->v[2], 0.001);
	assert_are_equal(0.3778, V->v[0]->v[3], 0.001);
	assert_are_equal(-0.3618, V->v[1]->v[0], 0.001);
	assert_are_equal(0.7420, V->v[1]->v[1], 0.001);
	assert_are_equal(-0.1877, V->v[1]->v[2], 0.001);
	assert_are_equal(0.5322, V->v[1]->v[3], 0.001);
	assert_are_equal(0.7694, V->v[2]->v[0], 0.001);
	assert_are_equal(0.0486, V->v[2]->v[1], 0.001);
	assert_are_equal(0.3010, V->v[2]->v[2], 0.001);
	assert_are_equal(0.5614, V->v[2]->v[3], 0.001);
	assert_are_equal(-0.5219, V->v[3]->v[0], 0.001);
	assert_are_equal(-0.5014, V->v[3]->v[1], 0.001);
	assert_are_equal(0.4662, V->v[3]->v[2], 0.001);
	assert_are_equal(0.5088, V->v[3]->v[3], 0.001);
}
void EigenDecomposition_GeometricIdentity_Test()
{
	var A = make(3, 3);
	var V = make(3, 3);
	var d = make(3);

	// diagonal
	A->v[0]->v[0] = 1; A->v[0]->v[1] = 0; A->v[0]->v[2] = 0;
	A->v[1]->v[0] = 0; A->v[1]->v[1] = 2; A->v[1]->v[2] = 0;
	A->v[2]->v[0] = 0; A->v[2]->v[1] = 0; A->v[2]->v[2] = 3;

	eig(A, ref V, ref d);

	assert_are_equal(1.0, d->v[0]);
	assert_are_equal(2.0, d->v[1]);
	assert_are_equal(3.0, d->v[2]);

	assert_are_equal(1.0, V->v[0]->v[0]);
	assert_are_equal(0.0, V->v[0]->v[1]);
	assert_are_equal(0.0, V->v[0]->v[2]);
	assert_are_equal(0.0, V->v[1]->v[0]);
	assert_are_equal(1.0, V->v[1]->v[1]);
	assert_are_equal(0.0, V->v[1]->v[2]);
	assert_are_equal(0.0, V->v[2]->v[0]);
	assert_are_equal(0.0, V->v[2]->v[1]);
	assert_are_equal(1.0, V->v[2]->v[2]);
}
void EigenDecomposition_Large_Matrix_Test()
{
	var T = DateTime.Now.Ticks;

	// this test should take under 1s
	var N = 2000;
	var A = identity(N);
	var V = make(N, N);
	var d = make(N);

	eig(A, ref V, ref d);

	assert_is_true(DateTime.Now.Ticks - T < TimeSpan.TicksPerSecond);
}
void EigenDecomposition_Many_Small_Matrix_Test()
{
	var T = DateTime.Now.Ticks;

	// this test should take under 1s
	var C = 500000;
	var N = 3;
	var A = identity(N);
	var V = make(N, N);
	var d = make(N);

	for (var i = 0; i<C; i++)
	{
		A = identity(N);
		eig(A, ref V, ref d);
	}

	assert_is_true(DateTime.Now.Ticks - T < TimeSpan.TicksPerSecond);
}
#endif

void multiple_linear_regression_test()
{
	auto X = mat<17, 3>();
	auto XT = mat<3, 17>();
	auto XTX = mat<3, 3>();
	auto XTX_inv = mat<3, 3>();
	auto B = mat<3, 17>();
	auto Y = mat<17, 1>();
	auto Bv = mat<3, 1>();

	{
		X[ 0][0] = 1.0; X[ 0][1] = 41.9; X[ 0][2] = 29.1;
		X[ 1][0] = 1.0; X[ 1][1] = 43.4; X[ 1][2] = 29.3;
		X[ 2][0] = 1.0; X[ 2][1] = 43.9; X[ 2][2] = 29.5;
		X[ 3][0] = 1.0; X[ 3][1] = 44.5; X[ 3][2] = 29.7;
		X[ 4][0] = 1.0; X[ 4][1] = 47.3; X[ 4][2] = 29.9;
		X[ 5][0] = 1.0; X[ 5][1] = 47.5; X[ 5][2] = 30.3;
		X[ 6][0] = 1.0; X[ 6][1] = 47.9; X[ 6][2] = 30.5;
		X[ 7][0] = 1.0; X[ 7][1] = 50.2; X[ 7][2] = 30.7;
		X[ 8][0] = 1.0; X[ 8][1] = 52.8; X[ 8][2] = 30.8;
		X[ 9][0] = 1.0; X[ 9][1] = 53.2; X[ 9][2] = 30.9;
		X[10][0] = 1.0; X[10][1] = 56.7; X[10][2] = 31.5;
		X[11][0] = 1.0; X[11][1] = 57.0; X[11][2] = 31.7;
		X[12][0] = 1.0; X[12][1] = 63.5; X[12][2] = 31.9;
		X[13][0] = 1.0; X[13][1] = 65.3; X[13][2] = 32.0;
		X[14][0] = 1.0; X[14][1] = 71.1; X[14][2] = 32.1;
		X[15][0] = 1.0; X[15][1] = 77.0; X[15][2] = 32.5;
		X[16][0] = 1.0; X[16][1] = 77.8; X[16][2] = 32.9;

		Y[ 0][0] = 251.3;
		Y[ 1][0] = 251.3;
		Y[ 2][0] = 248.3;
		Y[ 3][0] = 267.5;
		Y[ 4][0] = 273.0;
		Y[ 5][0] = 276.5;
		Y[ 6][0] = 270.3;
		Y[ 7][0] = 274.9;
		Y[ 8][0] = 285.0;
		Y[ 9][0] = 290.0;
		Y[10][0] = 297.0;
		Y[11][0] = 302.5;
		Y[12][0] = 304.5;
		Y[13][0] = 309.3;
		Y[14][0] = 321.7;
		Y[15][0] = 330.7;
		Y[16][0] = 349.0;
	}

	_transpose(X, XT);

	_mult(XT, X, XTX);
	{
		assert_are_equal(   17.0, round(XTX[0][0]));
		assert_are_equal(  941.0, round(XTX[0][1]));
		assert_are_equal(  525.0, round(XTX[0][2]));
		assert_are_equal(  941.0, round(XTX[1][0]));
		assert_are_equal(54270.0, round(XTX[1][1]));
		assert_are_equal(29286.0, round(XTX[1][2]));
		assert_are_equal(  525.0, round(XTX[2][0]));
		assert_are_equal(29286.0, round(XTX[2][1]));
		assert_are_equal(16254.0, round(XTX[2][2]));
	}

	_inverse(XTX, XTX_inv);
	{
		assert_are_equal_t(336.5123, XTX_inv[0][0], 0.1);
		assert_are_equal_t(  1.2282, XTX_inv[0][1], 0.1);
		assert_are_equal_t(-13.0890, XTX_inv[0][2], 0.1);
		assert_are_equal_t(  1.2282, XTX_inv[1][0], 0.1);
		assert_are_equal_t(  0.0051, XTX_inv[1][1], 0.1);
		assert_are_equal_t( -0.0489, XTX_inv[1][2], 0.1);
		assert_are_equal_t(-13.0890, XTX_inv[2][0], 0.1);
		assert_are_equal_t( -0.0489, XTX_inv[2][1], 0.1);
		assert_are_equal_t(  0.5113, XTX_inv[2][2], 0.1);
	}

	_mult(XTX_inv, XT, B);
	{
		assert_are_equal_t( 7.0950, B[0][ 0], 0.01);
		assert_are_equal_t( 6.3196, B[0][ 1], 0.01);
		assert_are_equal_t( 4.3160, B[0][ 2], 0.01);
		assert_are_equal_t( 2.4351, B[0][ 3], 0.01);
		assert_are_equal_t( 3.2565, B[0][ 4], 0.01);
		assert_are_equal_t(-1.7333, B[0][ 5], 0.01);
		assert_are_equal_t(-3.8598, B[0][ 6], 0.01);
		assert_are_equal_t(-3.6526, B[0][ 7], 0.01);
		assert_are_equal_t(-1.7680, B[0][ 8], 0.01);
		assert_are_equal_t(-2.5855, B[0][ 9], 0.01);
		assert_are_equal_t(-6.1400, B[0][10], 0.01);
		assert_are_equal_t(-8.3893, B[0][11], 0.01);
		assert_are_equal_t(-3.0233, B[0][12], 0.01);
		assert_are_equal_t(-2.1213, B[0][13], 0.01);
		assert_are_equal_t( 3.6938, B[0][14], 0.01);
		assert_are_equal_t( 5.7050, B[0][15], 0.01);
		assert_are_equal_t( 1.4520, B[0][16], 0.01);
	}

	_mult(B, Y, Bv);
	{
		assert_are_equal_t(-153.4500, Bv[0][0], 0.1);
		assert_are_equal_t(   1.2387, Bv[1][0], 0.1);
		assert_are_equal_t(  12.0823, Bv[2][0], 0.1);
	}
	/*
	mat Bv2 = least_squares_regression(X, Y);
	assert_are_equal_t(gmel(Bv,0,0), gmel(Bv2,0,0),0.1);
	assert_are_equal_t(gmel(Bv,1,0), gmel(Bv2,1,0),0.1);
	assert_are_equal_t(gmel(Bv,2,0), gmel(Bv2,2,0),0.1);
	*/
}
#if 0
void Posit_Test()
{
	var location = Assembly.GetExecutingAssembly().Location;
	var directoryName = Path.GetDirectoryName(location);
	var triangulation = STLFile.Read(directoryName + @"\\TestData\\sphere.stl");
}
#endif

void neural_network_2_layer_test()
{
	unsigned int i;

	// see:
	// https://iamtrask.github.io/2015/07/12/basic-python-network/
	auto X = mat<4, 3>();
	auto Y = mat<4, 1>();

	{
		// input data
		X[0][0]=0; X[0][1]=0; X[0][2]=1;
		X[1][0]=0; X[1][1]=1; X[1][2]=1;
		X[2][0]=1; X[2][1]=0; X[2][2]=1;
		X[3][0]=1; X[3][1]=1; X[3][2]=1;

		// expected output
		Y[0][0]=0;
		Y[1][0]=0;
		Y[2][0]=1;
		Y[3][0]=1;
	}

	// initial synapse weights
	auto S0 = mat<3, 1>();
	_map(S0, rando, S0);
	_scale(S0, 2.0, S0);
	_sub(S0, 1.0, S0);

	auto layer0 = mat<4, 3>();	// input layer
	auto layer1 = mat<4, 1>();	// hidden layer
	auto layer1_error = mat<4, 1>();
	auto layer1_delta = mat<4, 1>();

	auto layer0_S0 = mat<4, 1>();
	auto layer0T = mat<3, 4>();
	auto layer0T_layer1_delta = mat<3, 1>();
	auto layer1_sigmoid_map = mat<4, 1>();

	// adaptation loop
	for (i = 0; i<10000; i++)
	{
		_copy(X, layer0);
		_mult(layer0, S0, layer0_S0);
		_map(layer0_S0, sigmoid, layer1);
		_sub(Y, layer1, layer1_error);
		_map(layer1, sigmoid_derivative, layer1_sigmoid_map);
		_hadamard(layer1_error, layer1_sigmoid_map, layer1_delta);
		_transpose(layer0, layer0T);
		_mult(layer0T, layer1_delta, layer0T_layer1_delta);
		_add(S0, layer0T_layer1_delta, S0);
	}

	// output verification
	assert_are_equal_t(Y[0][0], round(layer1[0][0]), 0.001);
	assert_are_equal_t(Y[1][0], round(layer1[1][0]), 0.001);
	assert_are_equal_t(Y[2][0], round(layer1[2][0]), 0.001);
	assert_are_equal_t(Y[3][0], round(layer1[3][0]), 0.001);
}
void neural_network_3_layer_test()
{
	dim i;

	// see:
	// https://iamtrask.github.io/2015/07/12/basic-python-network/
	auto X = mat<4, 3>();
	auto Y = mat<4, 1>();

	// input data
	X[0][0] = 0; X[0][1] = 0; X[0][2] = 1;
	X[1][0] = 0; X[1][1] = 1; X[1][2] = 1;
	X[2][0] = 1; X[2][1] = 0; X[2][2] = 1;
	X[3][0] = 1; X[3][1] = 1; X[3][2] = 1;

	// expected output
	Y[0][0] = 0; Y[1][0] = 1; Y[2][0] = 1; Y[3][0] = 0;

	// initial synapse weights
	auto S0 = mat<3, 4>();
	auto S1 = mat<4, 1>();

	_map(S0, rando, S0);
	_map(S1, rando, S1);

	_scale(S0, 2.0, S0);	_sub(S0, 1.0, S0);
	_scale(S1, 2.0, S1);	_sub(S1, 1.0, S1);

	auto layer0 = mat<4, 3>();	// input layer
	auto layer1 = mat<4, 4>();
	auto layer2 = mat<4, 1>();

	auto layer0_S0 = mat<4, 4>();
	auto layer1_S1 = mat<4, 1>();
	auto layer2_error = mat<4, 1>();
	auto layer2_delta = mat<4, 1>();
	auto S1T = mat<1, 4>();
	auto layer1_error = mat<4, 4>();
	auto layer1_delta = mat<4, 4>();
	auto layer1T = mat<4, 4>();
	auto T1 = mat<4, 1>();
	auto layer0T = mat<3, 4>();
	auto layer0T_layer1_delta = mat<3, 4>();

	// adaptation loop
	for (i = 0; i<10000; i++)
	{
		_copy(X, layer0);

		_mult(layer0, S0, layer0_S0);
		_mult(layer1, S1, layer1_S1);

		_map(layer0_S0, sigmoid, layer1);
		_map(layer1_S1, sigmoid, layer2);

		_sub(Y, layer2, layer2_error);
		_map(layer2, sigmoid_derivative, layer2);
		_hadamard(layer2_error, layer2, layer2_delta);

		_transpose(S1, S1T);
		_mult(layer2_delta, S1T, layer1_error);
		_map(layer1, sigmoid_derivative, layer1);
		_hadamard(layer1_error, layer1, layer1_delta);

		_transpose(layer1, layer1T);
		_mult(layer1T, layer2_delta, T1);

		_add(S1, T1, S1);
		_transpose(layer0, layer0T);
		_mult(layer0T, layer1_delta, layer0T_layer1_delta);
		_add(S0, layer0T_layer1_delta, S0);
	}

	// output verification
	assert_are_equal_t(Y[0][0], round(layer2[0][0]), 0.001);
	assert_are_equal_t(Y[1][0], round(layer2[1][0]), 0.001);
	assert_are_equal_t(Y[2][0], round(layer2[2][0]), 0.001);
	assert_are_equal_t(Y[3][0], round(layer2[3][0]), 0.001);
}
void misc_tests()
{
	TEST(multiple_linear_regression_test);
	TEST(neural_network_2_layer_test);
//	TEST(neural_network_3_layer_test);
}

int main(int argc, char **argv)
{
	TEST_GROUP(vector_function_tests);
	TEST_GROUP(vector_stress_tests);
	TEST_GROUP(matrix_function_tests);
	TEST_GROUP(matrix_stress_tests);
	TEST_GROUP(misc_tests);
	return 0;
}
