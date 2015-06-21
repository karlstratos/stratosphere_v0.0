# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the SVDLIBC package.
SVDLIBC = ../third_party/SVDLIBC

# Where to find the Eigen package.
EIGEN = ../third_party/eigen-eigen-10219c95fe65

# Where to find the core files.
CORE = ../core

# Compiler.
CC = clang++

# Warning level.
WARN = -Wall

# Optimization level.
OPT = -O3

# Flags passed to the C++ compiler.
CFLAGS = -isystem $(GTEST)/include $(WARN) -std=c++11
ifeq ($(shell uname), Darwin)  # Apple clang version 4.0
	CFLAGS += -stdlib=libc++
endif

# Top-level commands.
TARGETS = hmm

all: $(TARGETS)

hmm: main.o hmm.o $(CORE)/util.o $(CORE)/evaluate.o
	$(CC) $(CFLAGS) $^ -o $@

main.o: main.cc hmm.o
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

hmm.o: hmm.cc hmm.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/evaluate.o: $(CORE)/evaluate.cc $(CORE)/evaluate.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(SVDLIBC)/libsvd.a:
	make -C $(SVDLIBC)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o $(CORE)/*.o
	make -C $(SVDLIBC) clean
