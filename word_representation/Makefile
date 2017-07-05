# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the SVDLIBC package.
SVDLIBC = ../third_party/SVDLIBC

# Where to find the Eigen package.
EIGEN = ../third_party/eigen-eigen-5a0156e40feb

# Where to find the core files.
CORE = ../core

# Compiler.
CC = clang++

# Warning level.
WARN = -Wall

# Optimization level.
OPT = -O3

# Flags passed to the C++ compiler.
CFLAGS = $(WARN) $(OPT) -std=c++11
ifeq ($(shell uname), Darwin)  # Apple clang version 4.0
	CFLAGS += -stdlib=libc++
endif

# Top-level commands.
TARGETS = wordrep

all: $(TARGETS)

wordrep: main.o wordrep.o $(CORE)/corpus.o $(CORE)/util.o $(CORE)/sparsesvd.o \
	$(CORE)/evaluate.o $(CORE)/cluster.o $(SVDLIBC)/libsvd.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

main.o: main.cc wordrep.o
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

wordrep.o: wordrep.cc wordrep.h $(CORE)/corpus.o $(CORE)/evaluate.o \
	$(CORE)/cluster.o
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/corpus.o: $(CORE)/corpus.cc $(CORE)/corpus.h $(CORE)/sparsesvd.o
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/sparsesvd.o: $(CORE)/sparsesvd.cc $(CORE)/sparsesvd.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/evaluate.o: $(CORE)/evaluate.cc $(CORE)/evaluate.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(CORE)/cluster.o: $(CORE)/cluster.cc $(CORE)/cluster.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

$(SVDLIBC)/libsvd.a:
	make -C $(SVDLIBC)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o $(CORE)/*.o
	rm -rf scripts/__pycache__ scripts/*.pyc
	make -C $(SVDLIBC) clean
