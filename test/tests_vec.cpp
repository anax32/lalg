#include "tests.h"
#include "mats.h"

void alloc_test ()
{
    auto v = lalg.vec<4>();
    assert_are_equal(v.size(), 4);
}

void fill_test ()
{
    auto v = lalg.vec<4>();
    lalg.fill (v, 2.0);

    for (auto i : v)
    {
        assert_are_equal(i, 2.0);
    }
}