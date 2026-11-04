#! /bin/bash
#----------------------------------------------------------------
# 1. Compile process to create the object file fem2d_heat.o
#----------------------------------------------------------------
g++ -c -g -Wall fem2d_heat.cpp

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "End execution fem2d_heat.cpp"

#----------------------------------------------------------------
# 2. Compile process to create the object file fem2d_heat_test.o
#----------------------------------------------------------------

g++ -c -g -Wall fem2d_heat_test.cpp

if [ $? -ne 0 ]; then
  echo "Compile error"
  exit
fi

echo "End execution fem2d_heat_test.cpp"

#----------------------------------------------------------------
# 3. Build executable fem2d_heat.o + fem2d_heat_test.o 
#----------------------------------------------------------------

g++ fem2d_heat.o fem2d_heat_test.o

if [ $? -ne 0 ]; then
  echo "Load error."
  exit
fi

chmod ugo+x a.out
mv a.out fem2d_heat_test

#rm fem2d_heat_test.o  fem2d_heat_test.o

echo "End build executable fem2d_heat_test"

#----------------------------------------------------------------
# 4. Execute fem2d_heat_test
#
#  Need:
#	square_elements.txt
#	square_nodes.txt
#   fem2d_heat_test
#----------------------------------------------------------------

./fem2d_heat_test square > fem2d_heat_test.txt

if [ $? -ne 0 ]; then
  echo "Run error."
  exit
fi

#rm fem2d_heat_test

#------------------------------------------------------
# The run create multiples files with first word square
# 10 file : 
#	square_u0000.txt
#	 ... 
#	square_u0005.txt
#	 ...
#	square_u0010.txt,
# 1 file :
#	square_times.txt
#------------------------------------------------------
echo "End Execution fem2d_heat_test"

#------------------------------------------------------
# 4. Plot using python, the run create 10 png files, one for each u0000#.txt : 
#-----------------------------------------------------

python contour_sequence4.py square_nodes.txt square_u0000.txt

#rm square_u00*.png, square_u00*.txt fem2d*.o fem2d_heat_test





