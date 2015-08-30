# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the root of Google Test.
GTEST = ../../third_party/gtest-1.7.0

# Where to find the core files.
CORE = ../../core

# Where to find the project source files.
SRC = ..

# C++ compiler.
CC = clang++

# Warning level.
WARN = -Wall

# Flags passed to the C++ compiler.
CFLAGS = -isystem $(GTEST)/include $(WARN) -std=c++11
ifeq ($(shell uname), Darwin)  # Apple clang version 4.0
	CFLAGS += -stdlib=libc++
endif

# Top-level commands.
TESTS = grammar_test

all: $(TESTS)

.PHONY: clean
clean:
	rm -f $(TESTS) gtest.a gtest_main.a *.o $(CORE)/*.o $(SRC)/*.o
	make -C $(SVDLIBC) clean

############## Usually you shouldn't change this section. ######################
# All Google Test headers.
GTEST_HEADERS = $(GTEST)/include/gtest/*.h \
                $(GTEST)/include/gtest/internal/*.h

# Builds gtest.a and gtest_main.a.

# Usually you shouldn't tweak such internal variables, indicated by a
# trailing _.
GTEST_SRCS_ = $(GTEST)/src/*.cc $(GTEST)/src/*.h $(GTEST_HEADERS)

# For simplicity and to avoid depending on Google Test's
# implementation details, the dependencies specified below are
# conservative and not optimized.  This is fine as Google Test
# compiles fast and for ordinary users its source rarely changes.
gtest-all.o: $(GTEST_SRCS_)
	     $(CC) -I $(GTEST) $(CFLAGS) -c $(GTEST)/src/gtest-all.cc

gtest_main.o: $(GTEST_SRCS_)
	      $(CC) -I $(GTEST) $(CFLAGS) -c $(GTEST)/src/gtest_main.cc

gtest.a: gtest-all.o
	 $(AR) $(ARFLAGS) $@ $^

gtest_main.a: gtest-all.o gtest_main.o
	      $(AR) $(ARFLAGS) $@ $^
################################################################################

# Builds tests.  A test should link with either gtest.a or gtest_main.a,
# depending on whether it defines its own main() function.

grammar_test: grammar_test.cc $(SRC)/grammar.o $(CORE)/util.o $(CORE)/trees.o \
	gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

grammar_test.o: grammar_test.cc
	$(CC) $(CFLAGS) -c $< -o $@

$(SRC)/grammar.o: $(SRC)/grammar.cc $(SRC)/grammar.h $(CORE)/util.o \
	$(CORE)/trees.o
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/util.o: $(CORE)/util.cc $(CORE)/util.h
	$(CC) $(CFLAGS) -c $< -o $@

$(CORE)/trees.o: $(CORE)/trees.cc $(CORE)/trees.h $(CORE)/util.o
	$(CC) $(CFLAGS) -c $< -o $@
