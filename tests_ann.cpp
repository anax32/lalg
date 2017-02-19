#include "tests.h"

void neural_network_2_layer_test()
{
	unsigned int i;

	// see:
	// https://iamtrask.github.io/2015/07/12/basic-python-network/
	mat X = make(4, 3);
	mat Y = make(4, 1);

	{
		// input data
		smel(X, 0, 0, 0); smel(X, 0, 1, 0); smel(X, 0, 2, 1);
		smel(X, 1, 0, 0); smel(X, 1, 1, 1); smel(X, 1, 2, 1);
		smel(X, 2, 0, 1); smel(X, 2, 1, 0); smel(X, 2, 2, 1);
		smel(X, 3, 0, 1); smel(X, 3, 1, 1); smel(X, 3, 2, 1);

		// expected output
		smel(Y, 0, 0, 0);
		smel(Y, 1, 0, 0);
		smel(Y, 2, 0, 1);
		smel(Y, 3, 0, 1);
	}

	// initial synapse weights
	mat S0 = make(3, 1);
	_map(S0, rando, S0);
	_scale(S0, 2.0, S0);
	_sub(S0, 1.0, S0);

	mat layer0 = make(4, 3);	// input layer
	mat layer1 = make(4, 1);	// hidden layer
	mat layer1_error = make(4, 1);
	mat layer1_delta = make(4, 1);

	mat layer0_S0 = make(4, 1);
	mat layer0T = make(3, 4);
	mat layer0T_layer1_delta = make(3, 1);
	mat layer1_sigmoid_map = make(4, 1);

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
	assert_are_equal_t(gmel(Y,0,0), round(gmel(layer1,0,0)), 0.001);
	assert_are_equal_t(gmel(Y,1,0), round(gmel(layer1,1,0)), 0.001);
	assert_are_equal_t(gmel(Y,2,0), round(gmel(layer1,2,0)), 0.001);
	assert_are_equal_t(gmel(Y,3,0), round(gmel(layer1,3,0)), 0.001);

	// cleanup
	cake(X);
	cake(Y);
	cake(S0);
	cake(layer0);
	cake(layer1);
	cake(layer1_error);
	cake(layer1_delta);
	cake(layer0_S0);
	cake(layer0T);
	cake(layer0T_layer1_delta);
	cake(layer1_sigmoid_map);
}

typedef struct n_s
{
	dim	i,j;	// coordinates of value in the value matrix
	vec	u,v;	// coordinates of inputs in the value matrix
	vec	w;		// weights
	vec	x;		// input values from gather()
} n_t;

n_t* make(mat M, dim i, dim j)
{
	n_t *n = (n_t *)mem_allocate(sizeof(n_t));
	n->i = i;
	n->j = j;
	n->u = make(row_count(M));
	n->v = make(col_count(M));
	n->w = make(mat_size(M));
	n->x = make(mat_size(M));
	return n;
}

void cake(n_t **n)
{
	cake((*n)->u);
	cake((*n)->v);
	cake((*n)->w);
	cake((*n)->x);
	mem_free(*n);
	*n = NULL;
}

void gather(mat X, vec u, vec v, vec x)
{
	dim i;

	check_size_match(u,v);
	check_size_match(u,x);

	for (i=0;i<vec_size(x);i++)
	{
		svel(x, i, gmel(X, gvel(u,i), gvel(v,i)));
	}
}

void print(n_t* nn)
{
	fprintf(stdout, "NN.x: %i\n\0", vec_size(nn->x));
	fprintf(stdout, "NN.u: %i\n\0", vec_size(nn->u));
	fprintf(stdout, "NN.v: %i\n\0", vec_size(nn->v));
	fprintf(stdout, "NN.w: %i\n\0", vec_size(nn->w));
}

void neural_network_2_layer_vector_test()
{
	n_t*	NN[4];
	mat		X = make(4,3);

	// setup the input data matrix
	smel(X, 0, 0, 0); smel(X, 0, 1, 0); smel(X, 0, 2, 1);
	smel(X, 1, 0, 0); smel(X, 1, 1, 1); smel(X, 1, 2, 1);
	smel(X, 2, 0, 1); smel(X, 2, 1, 0); smel(X, 2, 2, 1);
	smel(X, 3, 0, 1); smel(X, 3, 1, 1); smel(X, 3, 2, 1);

	// setup the neurons
	NN[0] = make(X, 0, 0);

	print(NN[0]);

	_counter(NN[0]->v);
	_zeros(NN[0]->u);
	_ones(NN[0]->w);

	// get all the values
	gather(X, NN[0]->u, NN[0]->v, NN[0]->x);

	fprintf (stdout, "NN[0]: %0.4f\n\0", _sum(NN[0]->x));
	fprintf (stdout, "NN[0]: %0.4f\n\0", _dot(NN[0]->x, NN[0]->w));

	cake(X);
	cake(&NN[0]);
}

void neural_network_3_layer_test()
{
	dim i;

	// see:
	// https://iamtrask.github.io/2015/07/12/basic-python-network/
	mat X = make(4, 3);
	mat Y = make(4, 1);

	// input data
	smel (X,0,0,0);	smel (X,0,1,0);	smel(X,0,2,1);
	smel (X,1,0,0);	smel (X,1,1,1);	smel(X,1,2,1);
	smel (X,2,0,1);	smel (X,2,1,0);	smel(X,2,2,1);
	smel (X,3,0,1);	smel (X,3,1,1);	smel(X,3,2,1);

	// expected output
	smel (Y,0,0,0);	smel (Y,1,0,1);	smel (Y,2,0,1);	smel (Y,3,0,0);

	// initial synapse weights
	mat S0 = make(3, 4);
	mat S1 = make(4, 1);

	_map(S0, rando, S0);
	_map(S1, rando, S1);

	_scale(S0, 2.0, S0);	_sub(S0, 1.0, S0);
	_scale(S1, 2.0, S1);	_sub(S1, 1.0, S1);

	mat layer0 = make(4, 3);	// input layer
	mat layer1 = make(4, 4);
	mat layer2 = make(4, 1);

	mat layer0_S0 = make(4, 4);
	mat layer1_S1 = make(4, 1);
	mat layer2_error = make(4, 1);
	mat layer2_delta = make(4, 1);
	mat S1T = make(1, 4);
	mat layer1_error = make(4, 4);
	mat layer1_delta = make(4, 4);
	mat layer1T = make(4, 4);
	mat T1 = make(4, 1);
	mat layer0T = make(3, 4);
	mat layer0T_layer1_delta = make(3, 4);

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
	assert_are_equal_t(gmel(Y,0,0), round(gmel(layer2,0,0)), 0.001);
	assert_are_equal_t(gmel(Y,1,0), round(gmel(layer2,1,0)), 0.001);
	assert_are_equal_t(gmel(Y,2,0), round(gmel(layer2,2,0)), 0.001);
	assert_are_equal_t(gmel(Y,3,0), round(gmel(layer2,3,0)), 0.001);
}

int main(int argc, char **argv)
{
	TEST(neural_network_2_layer_test);
	TEST(neural_network_2_layer_vector_test)
	TEST(neural_network_3_layer_test);

	return 0;
}

