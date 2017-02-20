#include "../../include/lalg/mats.h"
#include "../tests.h"

void alloc_test ()
{
    auto v = vec<4>();
    assert_are_equal(v.size(), 4);
    assert_are_not_equal (v.data(), NULL);
}

void fill_test ()
{
    auto v = vec<4>();
    fill (v, 2.0);

    for (auto i : v)
    {
        assert_are_equal(i, 2.0);
    }
}

void map_fn_test ()
{
    auto v = vec<4>();
    map(v, one, v);

    for (auto i : v)
    {
        assert_are_equal(i, 1.0);
    }
}

void map_fn_x_test ()
{
    auto v = vec<5>();
    fill (v, 2.0);
    map(v,[](auto x){return x*2.0;},v);

    for (auto i : v)
    {
        assert_are_equal(i, 4.0);
    }
}

void zeros_test ()
{
    auto v = vec<4>();
    zeros(v);

    for (auto i : v)
    {
        assert_are_equal (i, 0.0);
    }
}

void ones_test ()
{
    auto v = vec<4>();
    ones (v);

    for (auto i : v)
    {
        assert_are_equal (i, 1.0);
    }
}

void counter_test ()
{
	auto a = vec<5>();
	counter (a);

	assert_are_equal (a[0], 1);
	assert_are_equal (a[1], 2);
	assert_are_equal (a[2], 3);
	assert_are_equal (a[3], 4);
	assert_are_equal (a[4], 5);   
}

void vector_contents_tests ()
{
    TEST(alloc_test);
    TEST(fill_test);
    TEST(map_fn_test);
    TEST(map_fn_x_test);
    TEST(zeros_test);
    TEST(ones_test);
}

void negate_inplace_test()
{
    auto v = vec<4>();
    ones(v);
    negate(v);

    for (auto i : v)
    {
        assert_are_equal (i, -1.0);
    }
}
void negate_output_test()
{
    auto v = vec<4>();
    auto o = vec<4>();

    ones (v);
    negate(v, o);

    for (auto i : v)
    {
        assert_are_equal (i, 1.0);
    }

    for (auto i : o)
    {
        assert_are_equal (i, -1.0);
    }
}
void scale_inplace_test ()
{
    auto v = vec<4>();
    ones (v);
    scale (v, 3.0);

    for (auto i : v)
    {
        assert_are_equal (i, 3.0);
    }
}
void scale_output_test ()
{
    auto v = vec<4>();
    auto o = vec<4>();
    ones (v);
    scale (v, 3.0, o);
    
    for (auto i : v)
    {
        assert_are_equal (1.0, i);
    }

    for (auto i : o)
    {
        assert_are_equal (3.0, i);
    }
}
void sum_test()
{
	auto a = vec<5>();
	counter(a);
	assert_are_equal(1 + 2 + 3 + 4 + 5, sum(a));
}
void product_sum_test()
{
	auto a = vec<5>();
	counter (a);
	assert_are_equal (1*2*3*4*5, product_sum(a));
}
void add_test()
{
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	counter(a);
	ones(b);
	add(a, b, c);

	for (auto i=0u; i<c.size(); i++)
	{
		assert_are_equal (a[i]+b[i], c[i]);
	}
}
void mad_test ()
{
    auto a = vec<4>();
    auto b = vec<4>();
    auto o = vec<4>();

    ones (a);
    ones (b);
    zeros (o);

    mad (a, 2.0, b, o);

    for (auto i : o)
    {
        assert_are_equal (i, 3.0);
    }
}
void mult_inplace_test()
{
	auto a = vec<5>();
	auto b = vec<5>();

	counter (a);
	fill (b, 2.0);
	auto c = mult (a, b);

	for (auto i=0u; i<c.size(); i++)
	{
		assert_are_equal (a[i]*b[i], c[i]);
	}    
}
void mult_output_test()
{
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	counter (a);
	fill (b, 2.0);
	mult (a, b, c);

	for (auto i=0u; i<c.size(); i++)
	{
		assert_are_equal (a[i]*b[i], c[i]);
	}
}
void dot_ones_test ()
{
    auto a = vec<4>();
    auto b = vec<4>();

    ones (a);
    ones (b);

    assert_are_equal (4.0, dot (a, b));
}
void dot_twos_test ()
{
    auto a = vec<4>();
    auto b = vec<4>();

    fill (a, 2.0);
    fill (b, 2.0);

    assert_are_equal (16.0, dot (a, b));
}
void inner_test ()
{
    auto a = vec<4>();
    auto b = vec<4>();

    fill (a, 2.0);
    fill (b, 3.0);

    assert_are_equal (dot (a, b), inner (a, b));
}
void sub_test()
{
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	counter(a);
	ones(b);
	sub(a, b, c);

	for (auto i=0u; i<c.size(); i++)
	{
		assert_are_equal (a[i]-b[i], c[i]);
	}
}
void multiply_test()
{
	auto a = vec<5>();
	auto b = vec<5>();
	auto c = vec<5>();

	counter (a);
	fill (b, 2.0);
	mult (a, b, c);

	for (auto i=0u; i<a.size(); i++)
	{
		assert_are_equal (a[i]*b[i], c[i]);
	}
}
void minor_test()
{
	auto a = vec<5>();
	auto am = vec<4>();

    counter (a);
	minor(a, 2, am);

	assert_are_equal (a.size()-1, am.size());
	assert_are_equal (12, sum(am));
}
void inverse_ones_output_test ()
{
    auto v = vec<4>();
    auto o = vec<4>();
    ones (v);
    inverse (v, o);

    for (auto i : o)
    {
        assert_are_equal (i, 1.0);
    }
}
void inverse_twos_output_test ()
{
    auto v = vec<4>();
    auto o = vec<4>();
    fill (v, 2.0);
    inverse (v, o);

    for (auto i : o)
    {
        assert_are_equal (i, 0.5);
    }
}
void inverse_ones_inplace_test ()
{
    auto v = vec<4>();
    ones (v);
    inverse (v);

    for (auto i : v)
    {
        assert_are_equal (i, 1.0);
    }
}
void inverse_twos_inplace_test ()
{
    auto v = vec<4>();
    fill (v, 2.0);
    inverse (v);

    for (auto i : v)
    {
        assert_are_equal (i, 0.5);
    }
}
void vector_operations_tests ()
{
    TEST(negate_inplace_test);
    TEST(negate_output_test);
    TEST(scale_inplace_test);
    TEST(scale_output_test);
    TEST(sum_test);
    TEST(product_sum_test);
    TEST(add_test);
    TEST(mad_test);
    TEST(mult_inplace_test);
    TEST(mult_output_test);
    TEST(dot_ones_test);
    TEST(dot_twos_test);
    TEST(inner_test);
    TEST(sub_test);
    TEST(multiply_test);
    TEST(minor_test);
    TEST(inverse_ones_output_test);
    TEST(inverse_twos_output_test);
    TEST(inverse_ones_inplace_test);
    TEST(inverse_twos_inplace_test);
}

void norm_squared_test ()
{
    auto v = vec<4>();
    fill (v, 2.0);
    assert_are_equal (16.0, norm_squared (v));
}
void norm_test ()
{
    auto v = vec<4>();
    fill (v, 2.0);
    assert_are_equal (4.0, norm (v));
}
void length_test ()
{
    auto v = vec<4>();
    fill (v, 2.0);
    assert_are_equal (norm (v), length (v));
}
void normalise_output_test ()
{
    auto v = vec<4>();
    auto o = vec<4>();
    fill (v, 2.0);
    normalise (v, o);

    for (auto i : v)
    {
        assert_are_equal (2.0, i);
    }
    
    for (auto i : o)
    {
        assert_are_equal (0.5, i);
    }
}
void normalise_inplace_test ()
{
    auto v = vec<4>();

    fill (v, 2.0);
    normalise (v);

    for (auto i : v)
    {
        assert_are_equal (0.5, i);
    }
}

void vector_measures_tests ()
{
    TEST(norm_squared_test);
    TEST(norm_test);
    TEST(length_test);
    TEST(normalise_output_test);
    TEST(normalise_inplace_test);
}

template<int VECTOR_SIZE=50, int BATCH_SIZE=5000>
void many_add_test()
{
	auto a = vec<VECTOR_SIZE>();
	auto b = vec<VECTOR_SIZE>();
	auto c = vec<VECTOR_SIZE>();

	for (auto i=0; i<BATCH_SIZE; i++)
	{
		counter(a);
		fill(b, i);
		add(a, b, c);

		for (auto j=0u; j<c.size(); j++)
		{
			assert_are_equal (a[j]+b[j], c[j]);
		}
	}
}
template<int VECTOR_SIZE=50, int BATCH_SIZE=5000>
void many_mult_test()
{
	auto a = vec<VECTOR_SIZE>();
	auto b = vec<VECTOR_SIZE>();
	auto c = vec<VECTOR_SIZE>();

	for (auto i=0u; i<BATCH_SIZE; i++)
	{
		counter(a);
		fill(b, i);
		mult(a, b, c);

		for (auto j=0u; j<c.size(); j++)
		{
			assert_are_equal (a[j]*b[j], c[j]);
		}
	}
}

void vector_batch_tests ()
{
    TEST(many_add_test);
    TEST(many_mult_test);
}

void vector_tests ()
{
    TEST_GROUP(vector_contents_tests);
    TEST_GROUP(vector_operations_tests);
    TEST_GROUP(vector_measures_tests);
    TEST_GROUP(vector_batch_tests);
}

int main(int argc, char **argv)
{
	TEST_SUITE(vector_tests);
	return 0;
}