FLAGS=-std=c++11

tests: tests.cpp mats.h
	g++ -std=c++11 -o tests tests.cpp

tests_ann: tests_ann.cpp mats.h
	g++ -g -o tests_ann tests_ann.cpp

tests_regress: tests_regress.cpp mats.h
	g++ -o tests_regress tests_regress.cpp

tests_all: tests tests_ann tests_regress


clean:
	rm *.o tests tests_ann tests_regress
