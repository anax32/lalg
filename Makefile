MKDIR = mkdir -p
FLAGS = -std=c++11
BIN_DIR = bin/
TEST_DIR = test/
TEST_BIN_DIR = test/bin/

$(BIN_DIR):
	 $(MKDIR) $(BIN_DIR)

$(TEST_DIR):
	$(MKDIR) $(TEST_DIR)

$(TEST_BIN_DIR): $(TEST_DIR)
	$(MKDIR) $(TEST_BIN_DIR)

tests: $(TEST_DIR)tests.cpp mats.h $(TEST_DIR)
	g++ $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)tests.cpp

tests_ann: $(TEST_DIR)tests_ann.cpp mats.h $(TEST_DIR)
	g++ $(FLAGS) -o $(TEST_BIN_DIR)$@ $(TEST_DIR)tests_ann.cpp

tests_regress: $(TEST_DIR)tests_regress.cpp mats.h $(TEST_DIR)
	g++ $(FLAGS) -o $(TEST_BIN_DIR)/$@ $(TEST_DIR)tests_regress.cpp

tests_all: tests tests_ann tests_regress

clean:
	rm -drf $(BIN_DIR) $(TEST_BIN_DIR)