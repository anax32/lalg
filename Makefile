MKDIR = mkdir -p
FLAGS = -std=c++11
BIN_DIR = bin
TEST_DIR = test

$(BIN_DIR):
	 $(MKDIR) $(BIN_DIR)

$(TEST_DIR):
	$(MKDIR) $(TEST_DIR)

tests: tests.cpp mats.h $(TEST_DIR)
	g++ $(FLAGS) -o $(TEST_DIR)/$@ tests.cpp

tests_ann: tests_ann.cpp mats.h $(TEST_DIR)
	g++ $(FLAGS) -o $(TEST_DIR)/$@ tests_ann.cpp

tests_regress: tests_regress.cpp mats.h $(TEST_DIR)
	g++ $(FLAGS) -o $(TEST_DIR)/$@ tests_regress.cpp

tests_all: tests tests_ann tests_regress

clean:
	rm -drf $(BIN_DIR) $(TEST_DIR)