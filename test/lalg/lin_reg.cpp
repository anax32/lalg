#include "../../include/lalg/mats.h"
#include "../tests.h"

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

	transpose (X, XT);

	mult (XT, X, XTX);
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

	inverse (XTX, XTX_inv);
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

	mult (XTX_inv, XT, B);
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

	mult (B, Y, Bv);
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

void linear_regression_tests()
{
	TEST(multiple_linear_regression_test);
}

int main (int argc, char** argv)
{
	TEST_GROUP(linear_regression_tests);
	return 0;
}