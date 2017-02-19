MKDIR = mkdir -p
CC = g++
FLAGS = -Wall -std=c++14
BIN_DIR = bin/
TEST_DIR = test/
TEST_BIN_DIR = bin/test/

$(BIN_DIR):
	 $(MKDIR) $(BIN_DIR)

$(TEST_BIN_DIR): $(BIN_DIR)
	$(MKDIR) $(TEST_BIN_DIR)

test_funcs: $(TEST_DIR)funcs.cpp mats.h $(TEST_BIN_DIR)
	$(CC) $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)funcs.cpp

test_vec_ops: $(TEST_DIR)vecs.cpp mats.h $(TEST_BIN_DIR)
	$(CC) $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)vecs.cpp

test_mat_ops: $(TEST_DIR)mats.cpp mats.h $(TEST_BIN_DIR)
	$(CC) $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)mats.cpp

test_lin_reg: $(TEST_DIR)lin_reg.cpp mats.h $(TEST_BIN_DIR)
	$(CC) $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)lin_reg.cpp

test_ann: $(TEST_DIR)ann.cpp mats.h $(TEST_BIN_DIR)
	$(CC) $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)ann.cpp

all_tests: test_funcs test_vec_ops test_mat_ops test_lin_reg test_ann

all: all_tests

clean:
	rm -drf $(BIN_DIR) $(TEST_BIN_DIR)