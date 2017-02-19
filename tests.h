#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

#define BIT_32		1

#include "mats.h"

#include <assert.h>
#include <time.h>

#define assert_are_equal(x,y)		(assert(x == y))
#define assert_are_equal_t(x,y,t)	(assert(abs(x-y)<t))
#define assert_is_true(x)			(assert(x != 0))

#define TEST(x)			{fprintf(stdout, "  %-45s", #x);clock_t T=clock(); x(); fprintf(stdout," [%0.4fs]\n\0", (double)(clock()-T)/CLOCKS_PER_SEC); assert(mem_state.alloc_count==0);}
#define TEST_GROUP(x)	{fprintf(stdout, " %s\n\0", #x); clock_t T=clock(); x(); fprintf(stdout, " %-40s GROUP [%0.4fs]\n\0", #x, (double)(clock()-T)/CLOCKS_PER_SEC);}
#define TEST_SUITE(x)	{fprintf(stdout, "%-45s\n\0", #x); clock_t T=clock(); x(); fprintf(stdout, " %-39s SUITE [%0.4fs]\n\0", #x, (double)(clock()-T)/CLOCKS_PER_SEC);}

#ifdef _DEBUG
#define ALL_SIZE		(30)
#else
#define ALL_SIZE		(80)
#endif

#if (ALL_SIZE > MAX_DIM_SIZE)
#pragma error(Test size exceeds maximum dimension size)
#endif
