# Author: Karl Stratos (stratos@cs.columbia.edu)

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
TARGETS = grammar

all: $(TARGETS)

grammar: main.o grammar.o $(CORE)/util.o $(CORE)/trees.o
	$(CC) $(CFLAGS) $^ -o $@

main.o: main.cc grammar.o
	$(CC) $(CFLAGS) -c $< -o $@

grammar.o: grammar.cc grammar.h
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/trees.o: $(CORE)/trees.cc $(CORE)/trees.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o $(CORE)/*.o
	make -C $(SVDLIBC) clean
