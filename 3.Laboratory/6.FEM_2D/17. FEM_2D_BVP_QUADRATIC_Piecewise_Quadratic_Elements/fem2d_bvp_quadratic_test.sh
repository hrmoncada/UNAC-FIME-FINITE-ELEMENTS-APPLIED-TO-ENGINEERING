#! /bin/bash
# Compile and build the executable 
# g++: Invokes the GNU C++ compiler.
# -c: Compiles and assembles the code into an object file (.o), skipping the linking phase.
# -g: Generates debugging information (for use with debuggers like gdb).
# -Wall: Enables all common compiler warning messages to help catch potential bugs.
# -o fem2d_heat_rectangle: Sets the name of the resulting output executable file

g++ -c -g -Wall fem2d_bvp_quadratic.cpp  

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "Object file fem2d_bvp_quadratic.o  create"

g++ -c -g -Wall fem2d_bvp_quadratic_test.cpp  

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "Object file fem2d_bvp_quadratic_test.o  create"

g++ fem2d_bvp_quadratic.o fem2d_bvp_quadratic_test.o 

if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi

echo "Executable fem2d_bvp_quadratic_test create"

mv a.out fem2d_bvp_quadratic_test

# Execute:

./fem2d_bvp_quadratic_test > fem2d_bvp_quadratic_test.txt

if [ $? -ne 0 ]; then
  echo "Run error."
  exit
fi

echo "Normal end of execution."

# rm fem2d_bvp_quadratic_test.txt fem2d_bvp_quadratic_test.o  fem2d_bvp_quadratic.o

