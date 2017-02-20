MKDIR = mkdir -p
CXX = g++
CXXFLAGS = -std=c++14
INCLUDE = include/
LDLIBS = -lpng -lX11 -lGL -lGLEW

BIN_DIR = bin/
TEST_DIR = test/
TEST_BIN_DIR = bin/test/

PROJS = lalg/ gl/ img_io/
PROJS_BIN_DIR = $(PROJS:%=$(TEST_BIN_DIR)%)

TESTS = $(wildcard test/*/*.cpp)
TEST_OUTPUTS = $(TESTS:.cpp=)

$(BIN_DIR):
	 $(MKDIR) $(BIN_DIR)

$(TEST_BIN_DIR): $(BIN_DIR)
	$(MKDIR) $(TEST_BIN_DIR)

$(PROJS_BIN_DIR): $(TEST_BIN_DIR)
	$(MKDIR) $(PROJS_BIN_DIR)

test/% : test/%.cpp $(PROJS_BIN_DIR)
	$(CXX) $(CXXFLAGS) -I$(INCLUDE) $(LDLIBS) -o $(BIN_DIR)$@ $<

all_tests : $(TEST_OUTPUTS); $(info test_outputs is $(TEST_OUTPUTS))

all: all_tests run_all

run_all:
	@for dir in $(TEST_BIN_DIR)*; do run-parts $$dir; done

clean:
	rm -drf $(BIN_DIR) $(TEST_BIN_DIR)
