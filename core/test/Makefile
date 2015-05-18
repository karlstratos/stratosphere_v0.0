# Author: Karl Stratos (stratos@cs.columbia.edu)

# Where to find the root of Google Test.
GTEST = ../../third_party/gtest-1.7.0

# Where to find the Eigen package.
EIGEN = ../../third_party/eigen-eigen-10219c95fe65

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
TESTS = util_test eigen_helper_test

all: $(TESTS)

.PHONY: clean
clean:
	rm -f $(TESTS) gtest.a gtest_main.a *.o ../*.o

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

util_test: util_test.o ../util.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

util_test.o: util_test.cc
	$(CC) $(CFLAGS) -c $< -o $@

../util.o: ../util.cc ../util.h
	$(CC) $(CFLAGS) -c $< -o $@

eigen_helper_test: eigen_helper_test.o ../eigen_helper.o gtest_main.a
	$(CC) $(CFLAGS) $^ -o $@ -pthread

eigen_helper_test.o: eigen_helper_test.cc
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@

../eigen_helper.o: ../eigen_helper.cc ../eigen_helper.h
	$(CC) -I $(EIGEN) $(CFLAGS) -c $< -o $@
