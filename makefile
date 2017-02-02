CXX = g++
CXXFLAGS = -I/usr/local/include/eigen/ -std=c++11 -O3 -march=native -funroll-loops 
LDFLAGS = -lm

all: p2v

p2v : main.cpp sphePV.cpp
	$(CXX) main.cpp sphePV.cpp -o sphePV $(CXXFLAGS) $(LDFLAGS)
