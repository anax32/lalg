#include "tests.h"
#include "../mats.h"

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
    assert_are_equal (sigmoid (0.0), 0.0);
    assert_are_equal (sigmoid (0.5), 0.5);
    assert_are_equal (sigmoid (1.0), 1.0);
}