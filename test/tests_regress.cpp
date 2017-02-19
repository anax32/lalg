#include "tests.h"

void multiple_linear_regression_test()
{
	mat X = make(17, 3);
	mat XT = make(3, 17);
	mat XTX = make(3, 3);
	mat XTX_inv = make(3, 3);
	mat B = make(3, 17);
	mat Y = make(17, 1);
	mat Bv = make(3, 1);

	{
		X->v[0]->v[0] = 1.0; X->v[0]->v[1] = 41.9; X->v[0]->v[2] = 29.1;
		X->v[1]->v[0] = 1.0; X->v[1]->v[1] = 43.4; X->v[1]->v[2] = 29.3;
		X->v[2]->v[0] = 1.0; X->v[2]->v[1] = 43.9; X->v[2]->v[2] = 29.5;
		X->v[3]->v[0] = 1.0; X->v[3]->v[1] = 44.5; X->v[3]->v[2] = 29.7;
		X->v[4]->v[0] = 1.0; X->v[4]->v[1] = 47.3; X->v[4]->v[2] = 29.9;
		X->v[5]->v[0] = 1.0; X->v[5]->v[1] = 47.5; X->v[5]->v[2] = 30.3;
		X->v[6]->v[0] = 1.0; X->v[6]->v[1] = 47.9; X->v[6]->v[2] = 30.5;
		X->v[7]->v[0] = 1.0; X->v[7]->v[1] = 50.2; X->v[7]->v[2] = 30.7;
		X->v[8]->v[0] = 1.0; X->v[8]->v[1] = 52.8; X->v[8]->v[2] = 30.8;
		X->v[9]->v[0] = 1.0; X->v[9]->v[1] = 53.2; X->v[9]->v[2] = 30.9;
		X->v[10]->v[0] = 1.0; X->v[10]->v[1] = 56.7; X->v[10]->v[2] = 31.5;
		X->v[11]->v[0] = 1.0; X->v[11]->v[1] = 57.0; X->v[11]->v[2] = 31.7;
		X->v[12]->v[0] = 1.0; X->v[12]->v[1] = 63.5; X->v[12]->v[2] = 31.9;
		X->v[13]->v[0] = 1.0; X->v[13]->v[1] = 65.3; X->v[13]->v[2] = 32.0;
		X->v[14]->v[0] = 1.0; X->v[14]->v[1] = 71.1; X->v[14]->v[2] = 32.1;
		X->v[15]->v[0] = 1.0; X->v[15]->v[1] = 77.0; X->v[15]->v[2] = 32.5;
		X->v[16]->v[0] = 1.0; X->v[16]->v[1] = 77.8; X->v[16]->v[2] = 32.9;

		Y->v[0]->v[0] = 251.3;
		Y->v[1]->v[0] = 251.3;
		Y->v[2]->v[0] = 248.3;
		Y->v[3]->v[0] = 267.5;
		Y->v[4]->v[0] = 273.0;
		Y->v[5]->v[0] = 276.5;
		Y->v[6]->v[0] = 270.3;
		Y->v[7]->v[0] = 274.9;
		Y->v[8]->v[0] = 285.0;
		Y->v[9]->v[0] = 290.0;
		Y->v[10]->v[0] = 297.0;
		Y->v[11]->v[0] = 302.5;
		Y->v[12]->v[0] = 304.5;
		Y->v[13]->v[0] = 309.3;
		Y->v[14]->v[0] = 321.7;
		Y->v[15]->v[0] = 330.7;
		Y->v[16]->v[0] = 349.0;
	}

	_transpose(X, XT);

	_mult(XT, X, XTX);
	{
		assert_are_equal(17.0, round(gmel(XTX, 0, 0)));
		assert_are_equal(941.0, round(gmel(XTX, 0, 1)));
		assert_are_equal(525.0, round(gmel(XTX, 0, 2)));
		assert_are_equal(941.0, round(gmel(XTX, 1, 0)));
		assert_are_equal(54270.0, round(gmel(XTX, 1, 1)));
		assert_are_equal(29286.0, round(gmel(XTX, 1, 2)));
		assert_are_equal(525.0, round(gmel(XTX, 2, 0)));
		assert_are_equal(29286.0, round(gmel(XTX, 2, 1)));
		assert_are_equal(16254.0, round(gmel(XTX, 2, 2)));
	}

	_inverse(XTX, XTX_inv);
	{
		assert_are_equal_t(336.5123, gmel(XTX_inv, 0, 0), 0.1);
		assert_are_equal_t(1.2282, gmel(XTX_inv, 0, 1), 0.1);
		assert_are_equal_t(-13.0890, gmel(XTX_inv, 0, 2), 0.1);
		assert_are_equal_t(1.2282, gmel(XTX_inv, 1, 0), 0.1);
		assert_are_equal_t(0.0051, gmel(XTX_inv, 1, 1), 0.1);
		assert_are_equal_t(-0.0489, gmel(XTX_inv, 1, 2), 0.1);
		assert_are_equal_t(-13.0890, gmel(XTX_inv, 2, 0), 0.1);
		assert_are_equal_t(-0.0489, gmel(XTX_inv, 2, 1), 0.1);
		assert_are_equal_t(0.5113, gmel(XTX_inv, 2, 2), 0.1);
	}

	_mult(XTX_inv, XT, B);
	{
		assert_are_equal_t(7.0950, gmel(B, 0, 0), 0.01);
		assert_are_equal_t(6.3196, gmel(B, 0, 1), 0.01);
		assert_are_equal_t(4.3160, gmel(B, 0, 2), 0.01);
		assert_are_equal_t(2.4351, gmel(B, 0, 3), 0.01);
		assert_are_equal_t(3.2565, gmel(B, 0, 4), 0.01);
		assert_are_equal_t(-1.7333, gmel(B, 0, 5), 0.01);
		assert_are_equal_t(-3.8598, gmel(B, 0, 6), 0.01);
		assert_are_equal_t(-3.6526, gmel(B, 0, 7), 0.01);
		assert_are_equal_t(-1.7680, gmel(B, 0, 8), 0.01);
		assert_are_equal_t(-2.5855, gmel(B, 0, 9), 0.01);
		assert_are_equal_t(-6.1400, gmel(B, 0, 10), 0.01);
		assert_are_equal_t(-8.3893, gmel(B, 0, 11), 0.01);
		assert_are_equal_t(-3.0233, gmel(B, 0, 12), 0.01);
		assert_are_equal_t(-2.1213, gmel(B, 0, 13), 0.01);
		assert_are_equal_t(3.6938, gmel(B, 0, 14), 0.01);
		assert_are_equal_t(5.7050, gmel(B, 0, 15), 0.01);
		assert_are_equal_t(1.4520, gmel(B, 0, 16), 0.01);
	}

	_mult(B, Y, Bv);
	{
		assert_are_equal_t(-153.4500, gmel(Bv, 0, 0), 0.1);
		assert_are_equal_t(1.2387, gmel(Bv, 1, 0), 0.1);
		assert_are_equal_t(12.0823, gmel(Bv, 2, 0), 0.1);
	}
	/*
	mat Bv2 = least_squares_regression(X, Y);
	assert_are_equal_t(gmel(Bv,0,0), gmel(Bv2,0,0),0.1);
	assert_are_equal_t(gmel(Bv,1,0), gmel(Bv2,1,0),0.1);
	assert_are_equal_t(gmel(Bv,2,0), gmel(Bv2,2,0),0.1);
	*/

	cake(X);
	cake(XT);
	cake(XTX);
	cake(XTX_inv);
	cake(B);
	cake(Y);
	cake(Bv);
}

int main(int argc, char **argv)
{
	TEST(multiple_linear_regression_test);

	return 0;
}
