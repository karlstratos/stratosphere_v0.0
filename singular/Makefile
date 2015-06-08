# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the SVDLIBC package.
SVDLIBC = ../third_party/SVDLIBC

# Where to find the Eigen package.
EIGEN = ../third_party/eigen-eigen-36fd1ba04c12

# Where to find the core files.
CORE = ..

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
TARGETS = singular

all: $(TARGETS)

singular: main.o arguments.o #$(SVDLIBC)/libsvd.a
	$(CC) $(CFLAGS) $^ -o $@

main.o: main.cc arguments.o
	$(CC) $(CFLAGS) -c $< -o $@

arguments.o: arguments.cc arguments.h
	$(CC) $(CFLAGS) -c $< -o $@

$(SVDLIBC)/libsvd.a:
	make -C $(SVDLIBC)

.PHONY: clean
clean:
	rm -f $(TARGETS) *.o $(CORE)/*.o
	make -C $(SVDLIBC) clean
