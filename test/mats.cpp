#include "../mats.h"
#include "tests.h"

void alloc_test ()
{
    auto m = mat<4,4>();
    assert_are_equal(m.size(), 4);
	assert_are_equal(m[0].size(), 4);
}
void set_element_test()
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
void get_element_test()
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
void set_row_test()
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
void get_row_test()
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
void are_equal_test()
{
	auto A = mat<8, 5>();
	auto B = mat<8, 5>();

	counter(A);
	counter(B);

	assert_is_true(are_equal(A, B));
}
void are_equal_fails_test()
{
	auto A = mat<8, 5>();
	auto B = mat<8, 5>();

	counter(A);
	fill(B, zero());

	assert_is_false(are_equal(A, B));
}
void fill_test ()
{
    auto m = mat<4,4>();
    fill (m, 2.0);

    for (auto v : m)
    {
        for (auto i : v)
		{
			assert_are_equal(i, 2.0);
		}
    }
}
void map_fn_test ()
{
    auto m = mat<4,4>();
    map (m, one, m);

    for (auto v : m)
    {
        for (auto i : v)
        {
            assert_are_equal (i, 1.0);
        }
    }
}
void map_fn_x_test ()
{
    auto m = mat<4,4>();
    fill (m, 2.0);
    map (m, [](auto x){return x*2.0;}, m);

    for (auto v : m)
    {
        for (auto i : v)
        {
            assert_are_equal (i, 4.0);
        }
    }
}
void zeros_test ()
{
    auto m = mat<4,4>();
    zeros(m);

    for (auto v : m)
    {
        for (auto i : v)
        {
            assert_are_equal (i, 0.0);
        }
    }
}
void ones_test ()
{
    auto m = mat<4,4>();
    ones (m);

    for (auto v : m)
    {
        for (auto i : v)
        {
            assert_are_equal (i, 1.0);
        }
    }
}
void identity_test ()
{
    auto m = mat<4,4>();
    identity (m);
    assert_is_true (is_identity (m));
}
void lehmer_test ()
{
    auto m = mat<4,4>();
    lehmer (m);

    assert_are_equal_t(m[0][0], 1.0, 0.0001);
    assert_are_equal_t(m[0][1], 0.5, 0.0001);
    assert_are_equal_t(m[0][2], 0.3333333, 0.0001);
    assert_are_equal_t(m[0][3], 0.25, 0.0001);
    assert_are_equal_t(m[1][0], 0.5, 0.0001);
    assert_are_equal_t(m[1][1], 1.0, 0.0001);
    assert_are_equal_t(m[1][2], 0.6666666, 0.0001);
    assert_are_equal_t(m[1][3], 0.5, 0.0001);
    assert_are_equal_t(m[2][0], 0.3333333, 0.0001);
    assert_are_equal_t(m[2][1], 0.6666666, 0.0001);
    assert_are_equal_t(m[2][2], 1.0, 0.0001);
    assert_are_equal_t(m[2][3], 0.75, 0.0001);
    assert_are_equal_t(m[3][0], 0.25, 0.0001);
    assert_are_equal_t(m[3][1], 0.5, 0.0001);
    assert_are_equal_t(m[3][2], 0.75, 0.0001);
    assert_are_equal_t(m[3][3], 1.0, 0.0001);
}
void hilbert_test ()
{
    assert_are_equal(true, true);
}
void matrix_contents_tests ()
{
    TEST(alloc_test);
    TEST(set_element_test);
    TEST(get_element_test);
    TEST(set_row_test);
    TEST(get_row_test);
    TEST(are_equal_test);
    TEST(are_equal_fails_test);
    TEST(fill_test);
    TEST(map_fn_test);
    TEST(map_fn_x_test);
    TEST(zeros_test);
    TEST(ones_test);
    TEST(identity_test);
    TEST(lehmer_test);
}

void negate_inplace_test ()
{
    auto m = mat<4,4>();
    ones(m);
    negate(m);

    for (auto v : m)
    {
        for (auto i : v)
        {
            assert_are_equal(i, -1.0);
        }
    }
}
void negate_output_test ()
{
    auto m = mat<4,4>();
    auto o = mat<4,4>();
    ones(m);
    negate(m, o);

    for (auto v : m)
    {
        for (auto i : v)
        {
            assert_are_equal (i, 1.0);
        }
    }

    for (auto v : o)
    {
        for (auto i : v)
        {
            assert_are_equal (i, -1.0);
        }
    }
}
void transpose_2x2_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();

    counter (A);
    counter (C);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void transpose_5x5_test()
{
	auto A = mat<5, 5>();
	auto B = mat<5, 5>();
	auto C = mat<5, 5>();

    counter (A);
    counter (C);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void transpose_1x2_test()
{
	auto A = mat<1, 2>();
	auto B = mat<2, 1>();
	auto C = mat<2, 1>();

    counter (A);
    counter (C);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void transpose_2x1_test()
{
	auto A = mat<2, 1>();
	auto B = mat<1, 2>();
	auto C = mat<1, 2>();

    counter (A);
    counter (C);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void transpose_3x5_test()
{
	auto A = mat<3, 5>();
	auto B = mat<5, 3>();
	auto C = mat<5, 3>();

    counter (A);
    counter (C);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void transpose_5x3_test()
{
	auto A = mat<5, 3>();
	auto B = mat<3, 5>();
	auto C = mat<3, 5>();

    counter (A);
    counter (C);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void transpose_200x200_test()
{
	auto A = mat<200, 200>();
	auto B = mat<200, 200>();

    counter (A);
	transpose (A, B);

	assert_is_true (is_transpose (A, B));
}
void add_2x2_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();
	auto D = mat<2, 2>();

	ones (A);
	ones (B);
	zeros (C);
	fill (D, 2.0);

	add (A, B, C);
	assert (are_equal (C, D));
}
void add_200x200_test()
{
	auto A = mat<200, 200>();
	auto B = mat<200, 200>();
	auto C = mat<200, 200>();
	auto D = mat<200, 200>();

	ones (A);
	ones (B);
	zeros (C);
	fill (D, 2.0);

	add (A, B, C);
	assert (are_equal (C, D));
}
void add_1x3_test()
{
	auto A = mat<1, 3>();
	auto B = mat<1, 3>();
	auto C = mat<1, 3>();
	auto D = mat<1, 3>();

	ones (A);
	ones (B);
	zeros (C);
	fill (D, 2.0);

	add (A, B, C);
	assert (are_equal (C, D));
}
void add_3x1_test()
{
	auto A = mat<3, 1>();
	auto B = mat<3, 1>();
	auto C = mat<3, 1>();
	auto D = mat<3, 1>();

	ones (A);
	ones (B);
	zeros (C);
	fill (D, 2.0);

	add (A, B, C);
	assert (are_equal (C, D));
}
void mult_identity_identity_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();

    identity (A);
    identity (B);
	mult (A, B, C);

	assert_is_true (is_identity (C));
}
void mult_identity_lehmer_test()
{
	auto A = mat<5, 5>();
	auto B = mat<5, 5>();
	auto C = mat<5, 5>();

    identity (A);
    lehmer (B);
	mult (A, B, C);

	assert_is_true (are_equal (B, C));
}
void mult_15x3_3x15_test()
{
	auto A = mat<15,  3>();
	auto B = mat< 3, 15>();
	auto C = mat<15, 15>();
	auto D = mat<15, 15>();

	fill (A, 2.0);
	fill (B, 3.0);
	fill (C, 0.0);
	fill (D, (2.0*3.0)*3.0);

	mult (A, B, C);

	assert_is_true (are_equal (C, D));
}
void mult_3x15_15x3_test()
{
	auto A = mat< 3, 15>();
	auto B = mat<15,  3>();
	auto C = mat< 3,  3>();
	auto D = mat< 3,  3>();

	fill (A, 3.0);
	fill (B, 4.0);
	fill (C, 0.0);
	fill (D, (3.0*4.0)*15.0);

	mult (A, B, C);

	assert_is_true (are_equal (C, D));
}
void sub_2x2_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();
	auto D = mat<2, 2>();

	fill (A, 1.0);
	fill (B, 1.0);
	fill (C, 3.0);
	fill (D, 0.0);

	sub (A, B, C);
	assert_is_true (are_equal (C, D));
}
void sub_1x3_test()
{
	auto A = mat<1, 3>();
	auto B = mat<1, 3>();
	auto C = mat<1, 3>();
	auto D = mat<1, 3>();

	fill (A, 1.0);
	fill (B, 1.0);
	fill (C, 3.0);
	fill (D, 0.0);

	sub (A, B, C);
	assert_is_true (are_equal (C, D));
}
void sub_3x1_test()
{
	auto A = mat<3, 1>();
	auto B = mat<3, 1>();
	auto C = mat<3, 1>();
	auto D = mat<3, 1>();

	fill (A, 1.0);
	fill (B, 1.0);
	fill (C, 3.0);
	fill (D, 0.0);

	sub (A, B, C);
	assert (are_equal (C, D));
}
void sub_3x3_scalar_test ()
{
	auto m = mat<3,3>();
	auto o = mat<3,3>();
	ones (m);
	sub (m, 1.0, o);

	for (auto v : m)
	{
		for (auto i : v)
		{
			assert_are_equal (i, 1.0);
		}
	}

	for (auto v : o)
	{
		for (auto i : v)
		{
			assert_are_equal (i, 0.0);
		}
	}
}
void hadamard_identity_identity_test()
{
	auto A = mat<2, 2>();
	auto B = mat<2, 2>();
	auto C = mat<2, 2>();

    identity (A);
    identity (B);
	hadamard (A, B, C);

	assert_is_true (is_identity (C));
}
void matrix_operations_tests ()
{
    TEST(negate_output_test);
    TEST(transpose_2x2_test);
    TEST(transpose_5x5_test);
    TEST(transpose_1x2_test);
    TEST(transpose_3x5_test);
    TEST(transpose_5x3_test);
    TEST(transpose_200x200_test);
    TEST(add_2x2_test);
    TEST(add_200x200_test);
    TEST(add_1x3_test);
    TEST(add_3x1_test);
    TEST(mult_identity_identity_test);
    TEST(mult_identity_lehmer_test);
    TEST(mult_15x3_3x15_test);
    TEST(mult_3x15_15x3_test);
    TEST(sub_2x2_test);
    TEST(sub_1x3_test);
    TEST(sub_3x1_test);
	TEST(sub_3x3_scalar_test);
    TEST(hadamard_identity_identity_test);
}

void diag_test()
{
	auto A = mat<5, 5>();
	auto diags = vec<5>();

	map (A, rando, A);
	diag (A, diags);

	for (auto i=0u; i<diags.size(); i++)
	{
		assert_are_equal (A[i][i], diags[i]);
	}
}
void minor_test()
{
	auto A = mat<5, 5>();
	auto Am = mat<4, 4>();

	identity (A);
	minor (A, 2, 2, Am);

	assert_is_true (is_identity (Am));
}
void determinant_2x2_test()
{
	auto A = mat<2, 2>();

	A[0][0] = 1;
	A[0][1] = 2;
	A[1][0] = -1;
	A[1][1] = 1;
	
	assert_are_equal (3.0, determinant(A));
}
void determinant_3x3_test()
{
	auto A = mat<3, 3>();

	A[0][0] = 1; A[0][1] = 2; A[0][2] = 0;
	A[1][0] =-1; A[1][1] = 1; A[1][2] = 1;
	A[2][0] = 1; A[2][1] = 2; A[2][2] = 3;

	assert_are_equal(9.0, determinant(A));
}
void determinant_4x4_test()
{
	auto A = mat<4, 4>();

	A[0][0]=1; A[0][1]= 3; A[0][2]=-2; A[0][3]= 1;
	A[1][0]=5; A[1][1]= 1; A[1][2]= 0; A[1][3]=-1;
	A[2][0]=0; A[2][1]= 2; A[2][2]= 0; A[2][3]=-2;
	A[3][0]=2; A[3][1]=-1; A[3][2]= 0; A[3][3]= 3;

	assert_are_equal (-40.0, determinant(A));
}
void determinant_recursive_3x3_test ()
{
	auto A = mat<3, 3>();

	A[0][0] = 1; A[0][1] = 2; A[0][2] = 0;
	A[1][0] =-1; A[1][1] = 1; A[1][2] = 1;
	A[2][0] = 1; A[2][1] = 2; A[2][2] = 3;
	
	assert_are_equal (determinant(A), determinant_recursive(A));
}
void determinant_4x4_lehmer_test()
{
	auto A = mat<4, 4>();
	lehmer(A);
	assert_are_equal_t (0.1823, determinant(A), 0.0001);
}
void inverse_2x2_test()
{
	auto A = mat<2, 2>();
	auto A_inv = mat<2, 2>();

	A[0][0] = 1.0;
	A[0][1] = 2.0;
	A[1][0] = -1.0;
	A[1][1] = 1.0;

	inverse(A, A_inv);

	assert_are_equal_t ( 0.33333, A_inv[0][0], 0.002);
	assert_are_equal_t (-0.66666, A_inv[0][1], 0.002);
	assert_are_equal_t ( 0.33333, A_inv[1][0], 0.002);
	assert_are_equal_t ( 0.33333, A_inv[1][1], 0.002);
}
void inverse_2x2_identity_property_test ()
{
	auto A = mat<2, 2>();
	auto A_inv = mat<2, 2>();
	auto I = mat<2, 2>();

	A[0][0] = 1.0;
	A[0][1] = 2.0;
	A[1][0] = -1.0;
	A[1][1] = 1.0;

	inverse (A, A_inv);
	mult (A, A_inv, I);
	assert_is_true (is_identity(I));
}
void inverse_3x3_identity_property_test()
{
	auto A = mat<3, 3>();
	auto A_inv = mat<3, 3>();
	auto I = mat<3, 3>();

	A[0][0] = 1.0;	A[0][1] = 2.0;	A[0][2] = 0.0;
	A[1][0] =-1.0;	A[1][1] = 1.0;	A[1][2] = 1.0;
	A[2][0] = 1.0;	A[2][1] = 2.0;	A[2][2] = 3.0;

	inverse (A, A_inv);
	mult (A, A_inv, I);

	assert_is_true (is_identity (I));
}
void inverse_4x4_identity_property_test()
{
	auto A = mat<4, 4>();
	auto A_inv = mat<4, 4>();
	auto I = mat<4, 4>();

	lehmer (A);
	inverse (A, A_inv);
	mult (A, A_inv, I);

	assert_is_true (is_identity (I));
}
void inverse_10x10_identity_property_test()
{
	auto A = mat<10, 10>();
	auto A_inv = mat<10, 10>();
	auto I = mat<10, 10>();

	lehmer (A);
	inverse (A, A_inv);
	mult (A, A_inv, I);

	assert_is_true (is_identity (I));
}
void inverse_100x100_identity_property_test()
{
	auto A = mat<100, 100>();
	auto A_inv = mat<100, 100>();
	auto I = mat<100, 100>();

	lehmer (A);
	inverse (A, A_inv);
	mult (A, A_inv, I);

	assert_is_true (is_identity (I));
}
void cholesky_decompsition_3x3_test()
{
	auto A = mat<3, 3>();
	auto L = mat<3, 3>();

	// input
	// NB: this input comes from wikipedia:
	// https://en.wikipedia.org/wiki/Cholesky_decomposition#Example
	A[0][0] =   4; A[0][1] =  12; A[0][2] = -16;
	A[1][0] =  12; A[1][1] =  37; A[1][2] = -43;
	A[2][0] = -16; A[2][1] = -43; A[2][2] =  98;

	cholesky_decomposition (A, L);

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

void matrix_extraction_tests ()
{
    TEST(diag_test);
    TEST(minor_test);
    TEST(determinant_2x2_test);
    TEST(determinant_3x3_test);
    TEST(determinant_4x4_test);
    TEST(determinant_4x4_lehmer_test);
    TEST(determinant_recursive_3x3_test);
    TEST(inverse_2x2_test);
    TEST(inverse_2x2_identity_property_test);
    TEST(inverse_3x3_identity_property_test);
    TEST(inverse_4x4_identity_property_test);
    TEST(inverse_10x10_identity_property_test);
    TEST(inverse_100x100_identity_property_test);
    //TEST(cholesky_decompsition_3x3_test);
}

template<int MAT_ROW_SIZE=50, int MAT_COL_SIZE=50, int BATCH_SIZE=500>
void many_transpose_test()
{
	auto A = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto B = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();

	counter (A);

	for (auto i=0; i<BATCH_SIZE; i++)
	{
		transpose(A, B);
		assert_is_true (is_transpose (A, B));
		A = B;
	}
}
template<int MAT_ROW_SIZE=50, int MAT_COL_SIZE=50, int BATCH_SIZE=500>
void many_add_test()
{
	auto A = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto B = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto C = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();

	counter (A);
	fill (B, one ());

	for (auto i=0; i<BATCH_SIZE; i++)
	{
		fill (B, (type)i);
		add (A, B, C);
		// FIXME: what should the succint test be?
	}
}
template<int MAT_ROW_SIZE=50, int MAT_COL_SIZE=50, int BATCH_SIZE=50>
void many_mult_test()
{
	auto A = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto B = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto C = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();

	counter (A);
	inverse (A, B);

	for (auto i=0; i<BATCH_SIZE; i++)
	{
		fill (C, (type)i);
		mult (A, B, C);

		assert_is_true (is_identity (C));
	}
}
template<int MAT_ROW_SIZE=50, int MAT_COL_SIZE=50, int BATCH_SIZE=500>
void many_hadamard_test()
{
	auto A = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto B = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();
	auto C = mat<MAT_ROW_SIZE, MAT_COL_SIZE>();

	counter(A);
	ones(B);

	for (auto i=0; i<BATCH_SIZE; i++)
	{
		fill (C, (type)i);
		hadamard (A, B, C);

		assert_is_true (are_equal (A, C));
	}
}

void matrix_batch_tests ()
{
    TEST(many_transpose_test);
    TEST(many_add_test);
    TEST(many_mult_test);
    TEST(many_hadamard_test);
}

int main (int argc, char **argv)
{
    TEST_GROUP(matrix_contents_tests);
    TEST_GROUP(matrix_operations_tests);
    TEST_GROUP(matrix_extraction_tests);
    TEST_GROUP(matrix_batch_tests);
    return 0;
}