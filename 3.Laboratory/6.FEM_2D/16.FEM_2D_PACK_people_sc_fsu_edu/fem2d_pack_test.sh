#! /bin/bash

g++ -c -g -Wall fem2d_pack.cpp

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "Object file fem2d_pack.o create"

g++ -c -g -Wall fem2d_pack_test.cpp

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "Object file fem2d_pack_test.o create"

g++ fem2d_pack_test.o fem2d_pack.o -lm

if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi

echo "Executable fem2d_pack_test create"

mv a.out fem2d_pack_test

./fem2d_pack_test > fem2d_pack_test.txt

if [ $? -ne 0 ]; then
  echo "Run error."
  exit
fi

echo "Normal end of execution."

# OUTPUT: a few files are created
# fem2d_pack_prb_q4.eps
# fem2d_pack_prb_t3.eps
# fem2d_pack_prb_t4.eps
# fem2d_pack_prb_t6.eps

# sphere_q4_elements.txt
# sphere_q4_nodes.txt

# sphere_q9_elements.txt
# sphere_q9_nodes.txt

# sphere_t3_elements.txt
# sphere_t3_nodes.txt

# sphere_t6_elements.txt
# sphere_t6_nodes.txt

# sphere_q16_elements.txt
# sphere_q16_nodes.txt


#rm fem2d_pack_test.o fem2d_pack_test.o fem2d_pack_test sphere_*.txt, fem2d_pack_prb_*.eps

