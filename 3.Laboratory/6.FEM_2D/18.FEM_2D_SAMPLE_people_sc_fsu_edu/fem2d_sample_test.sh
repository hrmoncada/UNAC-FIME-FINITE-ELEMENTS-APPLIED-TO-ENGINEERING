#! /bin/bash

g++ -c -g -Wall fem2d_sample.cpp

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

g++ fem2d_sample.o

if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi

chmod ugo+x a.out

mv a.out fem2d_sample

echo "Normal end of execution."

#-----------------------------------------------------------------
# ./fem2d_sample fem_prefix sample_prefix

# where fem_prefix is the common prefix for the FEM files:

# fem_prefix_nodes.txt, the node coordinates.
# fem_prefix_elements.txt, the nodes that make up each element;
# fem_prefix_values.txt, the values defined at each node.

# and sample_prefix is the common prefix for the SAMPLE files, for which the node file is input, and the values file is created by the program:

# sample_prefix_nodes.txt, the node coordinates where samples are desired.
# sample_prefix_values.txt, the values computed at each sample node.

# SQ is FEM data for the function f(x,y)=x^2+y^2, on a 5x5 grid of nodes, organized into linear triangles. The sample data seeks the values of this function on a 4x4 evenly spaced grid.
# fem_sq_nodes.txt, defines the FEM nodes.
# fem_sq_elements.txt, defines the FEM elements.
# fem_sq_values.txt, defines the FEM nodal values.
# sample_sq_nodes.txt, defines the sample nodes.
# sample_sq_values.txt, defines the sample nodal values.
#-----------------------------------------------------------------
# ./fem2d_sample fem_prefix sample_prefix
./fem2d_sample fem_sq sample_sq > fem2d_sample_test.txt

if [ $? -ne 0 ]; then
  echo "Run error."
  exit
fi

echo "Normal end of execution."

#rm fem2d_sample.o
