#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#include <assert.h>
#include <time.h>

#define assert_are_equal(x,y)		(assert(x == y))
#define assert_are_equal_t(x,y,t)	(assert(abs(x-y)<t))
#define assert_are_not_equal(x,y)	(assert(x != y))
#define assert_is_true(x)			(assert(x != 0))
#define assert_is_false(x)			(assert(x == 0))

#if 0
#define TEST(x)			{fprintf(stdout, "  %-45s", #x);clock_t T=clock(); x(); fprintf(stdout," [%0.4fs]\n\0", (double)(clock()-T)/CLOCKS_PER_SEC);}
#define TEST_GROUP(x)	{fprintf(stdout, " %s\n\0", #x); clock_t T=clock(); x(); fprintf(stdout, " %-40s GROUP [%0.4fs]\n\0", #x, (double)(clock()-T)/CLOCKS_PER_SEC);}
#define TEST_SUITE(x)	{fprintf(stdout, "%-45s\n\0", #x); clock_t T=clock(); x(); fprintf(stdout, " %-39s SUITE [%0.4fs]\n\0", #x, (double)(clock()-T)/CLOCKS_PER_SEC);}
#else
void time_function (std::function<void()> x, const char* fname)
{
	fprintf(stdout, "  %-45s", fname);
	clock_t T=clock();
	x();
	fprintf(stdout," [%0.4fs]\n", (double)(clock()-T)/CLOCKS_PER_SEC);
}
void time_function (std::function<void()> x, const char* fname, const char* group)
{
	fprintf(stdout, " %s\n", fname);
	clock_t T=clock();
	x();
	fprintf(stdout, " %-40s %s [%0.4fs]\n", fname, group, (double)(clock()-T)/CLOCKS_PER_SEC);
}
#define TEST(x)			{time_function (x, #x);}
#define TEST_GROUP(x)	{time_function (x, #x, "GROUP");}
#define TEST_SUITE(x)	{time_function (x, #x, "SUITE");}
#endif

template<dim N>
void print (vec<N> v)
{
	for (auto i : v)
	{
		fprintf(stdout, "%0.2f ,", i);
	}
}
template<dim N, dim M>
void print (mat<N,M> A)
{
	for (auto v : A)
	{
		print (v);
		fprintf(stdout, "\n");
	}
	fprintf(stdout, "\n");
}