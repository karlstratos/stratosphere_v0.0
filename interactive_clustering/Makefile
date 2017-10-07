# Author: Karl Stratos (me@karlstratos.com)

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
TARGETS = icluster prune

all: $(TARGETS)

icluster: main_icluster.o $(CORE)/icluster.o $(CORE)/util.o
	$(CC) $(CFLAGS) $^ -o $@

main_icluster.o: main_icluster.cc $(CORE)/icluster.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/icluster.o: $(CORE)/icluster.cc $(CORE)/icluster.h $(CORE)/util.o
	$(CC) $(CFLAGS) -c $< -o $@

prune: main_prune.o $(CORE)/pruner.o $(CORE)/trees.o $(CORE)/util.o
	$(CC) $(CFLAGS) $^ -o $@

main_prune.o: main_prune.cc $(CORE)/pruner.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/pruner.o: $(CORE)/pruner.cc $(CORE)/pruner.h $(CORE)/trees.o \
	$(CORE)/util.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/trees.o: $(CORE)/trees.cc $(CORE)/trees.h $(CORE)/util.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o $(CORE)/*.o
