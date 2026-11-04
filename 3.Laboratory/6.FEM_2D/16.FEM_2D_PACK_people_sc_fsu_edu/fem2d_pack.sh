#! /bin/bash

g++ -c -g -Wall fem2d_pack.cpp

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "Normal end of execution."
