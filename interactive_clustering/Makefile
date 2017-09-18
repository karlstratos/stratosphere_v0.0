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
TARGETS = icluster

all: $(TARGETS)

icluster: main.o $(CORE)/icluster.o $(CORE)/util.o
	$(CC) $(CFLAGS) $^ -o $@

main.o: main.cc $(CORE)/icluster.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/icluster.o: $(CORE)/icluster.cc $(CORE)/icluster.h $(CORE)/util.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o $(CORE)/*.o
