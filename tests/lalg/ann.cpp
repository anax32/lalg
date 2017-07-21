#include "../../include/lalg/mats.h"
#include "maketest.h"

void neural_network_2_layer_test()
{
  unsigned int i;

  // see:
  // https://iamtrask.github.io/2015/07/12/basic-python-network/
  auto X = mat<4, 3>();
  auto Y = mat<4, 1>();

  {
    // input data
    X[0][0]=0; X[0][1]=0; X[0][2]=1;
    X[1][0]=0; X[1][1]=1; X[1][2]=1;
    X[2][0]=1; X[2][1]=0; X[2][2]=1;
    X[3][0]=1; X[3][1]=1; X[3][2]=1;

    // expected output
    Y[0][0]=0;
    Y[1][0]=0;
    Y[2][0]=1;
    Y[3][0]=1;
  }

  // initial synapse weights
  auto S0 = mat<3, 1>();
  map (S0, rando, S0);
  scale (S0, 2.0, S0);
  sub (S0, 1.0, S0);

  auto layer0 = mat<4, 3>();  // input layer
  auto layer1 = mat<4, 1>();  // hidden layer
  auto layer1_error = mat<4, 1>();
  auto layer1_delta = mat<4, 1>();

  auto layer0_S0 = mat<4, 1>();
  auto layer0T = mat<3, 4>();
  auto layer0T_layer1_delta = mat<3, 1>();
  auto layer1_sigmoid_map = mat<4, 1>();

  // adaptation loop
  for (i = 0; i<10000; i++)
  {
    layer0 = X;
    mult(layer0, S0, layer0_S0);
    map(layer0_S0, sigmoid, layer1);
    sub(Y, layer1, layer1_error);
    map(layer1, sigmoid_derivative, layer1_sigmoid_map);
    hadamard(layer1_error, layer1_sigmoid_map, layer1_delta);
    transpose(layer0, layer0T);
    mult(layer0T, layer1_delta, layer0T_layer1_delta);
    add(S0, layer0T_layer1_delta, S0);
  }

  // output verification
  assert_are_equal_t (Y[0][0], round(layer1[0][0]), 0.001);
  assert_are_equal_t (Y[1][0], round(layer1[1][0]), 0.001);
  assert_are_equal_t (Y[2][0], round(layer1[2][0]), 0.001);
  assert_are_equal_t (Y[3][0], round(layer1[3][0]), 0.001);
}

void neural_network_3_layer_test()
{
  dim i;

  // see:
  // https://iamtrask.github.io/2015/07/12/basic-python-network/
  auto X = mat<4, 3>();
  auto Y = mat<4, 1>();

  // input data
  X[0][0]=0;  X[0][1]=0;  X[0][2]=1;
  X[1][0]=0;  X[1][1]=1;  X[1][2]=1;
  X[2][0]=1;  X[2][1]=0;  X[2][2]=1;
  X[3][0]=1;  X[3][1]=1;  X[3][2]=1;

  // expected output
  Y[0][0]=0;
  Y[1][0]=1;
  Y[2][0]=1;
  Y[3][0]=0;

  // initial synapse weights
  auto S0 = mat<3, 4>();
  auto S1 = mat<4, 1>();

  map(S0, rando, S0);
  map(S1, rando, S1);

  scale(S0, 2.0, S0);  sub(S0, 1.0, S0);
  scale(S1, 2.0, S1);  sub(S1, 1.0, S1);

  auto layer0 = mat<4, 3>();  // input layer
  auto layer1 = mat<4, 4>();
  auto layer2 = mat<4, 1>();

  auto layer0_S0 = mat<4, 4>();
  auto layer1_S1 = mat<4, 1>();
  auto layer2_error = mat<4, 1>();
  auto layer2_delta = mat<4, 1>();
  auto S1T = mat<1, 4>();
  auto layer1_error = mat<4, 4>();
  auto layer1_delta = mat<4, 4>();
  auto layer1T = mat<4, 4>();
  auto T1 = mat<4, 1>();
  auto layer0T = mat<3, 4>();
  auto layer0T_layer1_delta = mat<3, 4>();

  // adaptation loop
  for (i = 0; i<10000; i++)
  {
    layer0 = X;

    mult (layer0, S0, layer0_S0);
    mult (layer1, S1, layer1_S1);

    map (layer0_S0, sigmoid, layer1);
    map (layer1_S1, sigmoid, layer2);

    sub (Y, layer2, layer2_error);
    map (layer2, sigmoid_derivative, layer2);
    hadamard (layer2_error, layer2, layer2_delta);

    transpose (S1, S1T);
    mult (layer2_delta, S1T, layer1_error);
    map (layer1, sigmoid_derivative, layer1);
    hadamard (layer1_error, layer1, layer1_delta);

    transpose (layer1, layer1T);
    mult (layer1T, layer2_delta, T1);

    add (S1, T1, S1);
    transpose (layer0, layer0T);
    mult (layer0T, layer1_delta, layer0T_layer1_delta);
    add (S0, layer0T_layer1_delta, S0);
  }

  // output verification
  assert_are_equal_t (Y[0][0], round(layer2[0][0]), 0.001);
  assert_are_equal_t (Y[1][0], round(layer2[1][0]), 0.001);
  assert_are_equal_t (Y[2][0], round(layer2[2][0]), 0.001);
  assert_are_equal_t (Y[3][0], round(layer2[3][0]), 0.001);
}

void neural_network_tests ()
{
  TEST(neural_network_2_layer_test);
  //TEST(neural_network_3_layer_test);
}

int main(int argc, char **argv)
{
  TEST_GROUP(neural_network_tests);
  return 0;
}

