CXX = g++
CXXFLAGS = -std=c++14 $(INCLUDE)
INCLUDE = -Iinclude/ -I../maketest/
LDLIBS = 

TEST_DIR = tests/
include ../maketest/Makefile
