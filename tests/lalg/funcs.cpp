#include "../../include/lalg/mats.h"
#include "maketest.h"

void zero_is_zero_test ()
{
    assert_are_equal (zero (), 0.0);
}
void one_is_one_test ()
{
    assert_are_equal (one (), 1.0);
}
void one_is_not_zero ()
{
    assert_are_not_equal (one (), zero ());
}
void sigmoid_test ()
{
    assert_are_equal_t (sigmoid (0.0), 0.50000, 0.00001);
    assert_are_equal_t (sigmoid (0.5), 0.37754, 0.00001);
    assert_are_equal_t (sigmoid (1.0), 0.26894, 0.00001);
}

void sigmoid_derivative_test ()
{
    assert_are_equal_t (sigmoid_derivative (0.0), 0.0, 0.00001);
    assert_are_equal_t (sigmoid_derivative (0.5), 0.25, 0.00001);
    assert_are_equal_t (sigmoid_derivative (1.0), 0.0, 0.00001);
}

void value_function_tests ()
{
    TEST(zero_is_zero_test);
    TEST(one_is_one_test);
    TEST(one_is_not_zero);
    TEST(sigmoid_test);
    TEST(sigmoid_derivative_test);
}

int main (int argc, char** argv)
{
    TEST_GROUP(value_function_tests);
    return 0;
}
